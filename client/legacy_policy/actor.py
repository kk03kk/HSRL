"""BC / PPO checkpoint 的在线 actor 策略（与搜索/启发式共用同一决策接口）。

与 `LiveValuePolicy` 的区别：

* `LiveValuePolicy` 只接受作者 localpair 的 **scalar** value 头，并用可见结构后态
  的 value 增量参与排序（“BC actor + 局部两两 value”）；
* 本类接受任意 `EntityTransformerPolicy` checkpoint（BC 的 rank 分布头、PPO 微调
  后的 actor），按候选 logits 的 masked 概率排序，**不**使用未经真实对局校准的
  rank 价值头参与打分；rank 预测只写入审计供分析。

实机边界与其它在线策略一致：只吃旧客户端白名单观测、只返回服务器授权的原始
`Action`、复杂交互（目标/分支/多选/饰品/英雄）仍交给原协议。
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import threading
import time

from bg_agent.heuristic import HeuristicPolicy
from bg_agent.model import Decision, Kind, Phase
from legacy_policy.spec import CANDIDATE_MAX
from legacy_policy.adapter import ObservationAdapter, BridgeBlocked
from legacy_policy.common import CachedLivePolicy
from legacy_policy.protocol import waiting_for_sync

#: 当前实测可用的 checkpoint 角色（preflight 会打印，缺失时启动即失败）。
CHECKPOINT_ROLES = ('bc', 'ppo')


@dataclass(frozen=True)
class LiveActorConfig:
    #: actor 概率项的权重；BC 的 logits 可能很尖，概率项本身被限制在 [0,1]。
    actor_weight: float = 1.0
    #: 原启发式先验的小权重（只用于同分/极端 OOD 时防止荒谬动作）。
    heuristic_weight: float = 0.05
    #: 只允许原策略认为非负收益的动作进入候选，避免 OOD 状态下网络卖核心随从。
    min_utility_fraction: float = 0.25
    max_candidates: int = 64
    torch_threads: int = 2
    cache_size: int = 4

    def __post_init__(self):
        if self.actor_weight <= 0 or self.heuristic_weight < 0:
            raise ValueError('invalid_actor_score_weights')
        if not 1 <= self.max_candidates <= CANDIDATE_MAX or self.torch_threads < 1 or self.cache_size < 1:
            raise ValueError('invalid_live_actor_budget')
        if not 0 <= self.min_utility_fraction <= 1:
            raise ValueError('invalid_productive_action_guard')


class LiveActorPolicy(CachedLivePolicy):
    """任意 EntityTransformerPolicy checkpoint 的在线 actor 排序。"""

    participation_fields = ('neural_used', 'actor_changed_selection', 'value_ranked')

    def __init__(self, checkpoint, *, role='bc', audit=None, config=None,
                 supported_builds=('254477', '255130'), legacy=None):
        import torch
        self.torch = torch
        self.role = role
        self.config = config or LiveActorConfig()
        torch.set_num_threads(self.config.torch_threads)
        self.path = Path(checkpoint).resolve()
        payload = torch.load(self.path, map_location='cpu', weights_only=True)
        self.value_mode = payload.get('policy_config', {}).get('value_mode', 'distribution')
        self.trainer = payload.get('meta', {}).get('trainer')
        from legacy_policy.model import EntityTransformerPolicy
        self.model = EntityTransformerPolicy.load(self.path).eval()
        self.encoder = self.model.make_encoder()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.legacy = legacy or HeuristicPolicy()
        self.adapter = ObservationAdapter(supported_builds=supported_builds)
        self.audit = audit
        self.lock = threading.RLock()
        self.cache = OrderedDict()
        self.counters = Counter()
        self.last_trace = self.last_adopted = None

    def metadata(self):
        return dict(policy='live_actor_' + str(self.role) + '_v1', role=self.role,
                    checkpoint=str(self.path), checkpoint_sha256=self.sha256,
                    trainer=self.trainer, value_mode=self.value_mode,
                    value_head_used=False, config=asdict(self.config),
                    supported_builds=list(self.adapter.supported_builds), legacy=self.legacy.metadata(),
                    note='rank/value head 不参与打分；只做候选 actor 排序')

    def _compute(self, obs):
        started = time.perf_counter()
        fallback = self.legacy.decide(obs)
        trace = dict(checkpoint_sha256=self.sha256, revision=obs.revision, phase=str(obs.phase),
                     chosen_key=fallback.action.key, neural_used=False, actor_changed_selection=False,
                     value_ranked=False, candidates=[], mode='legacy_protocol',
                     reason='没有模型可表达的当前决策')
        if waiting_for_sync(obs):
            trace['reason'] = '状态尚未同步，等待而不投递旧动作'
            wait = Decision(_wait_action(), obs.revision, trace['reason'])
            trace['chosen_key'] = wait.action.key
            return wait, trace
        ranked = tuple(r for r in self.legacy.rank(obs) if r.utility >= 0)
        if obs.phase not in (Phase.RECRUIT, Phase.CHOICE) or not ranked:
            return fallback, trace
        if obs.build not in self.adapter.supported_builds:
            raise ValueError('unreviewed_live_build:' + str(obs.build))
        protocol = obs.interaction and obs.interaction.mode != 'choice'
        if protocol or fallback.action.kind in (Kind.TARGET, Kind.CONFIRM, Kind.SELECT, Kind.CANCEL):
            trace['reason'] = '分步目标/确认/多选沿用原协议'
            return fallback, trace
        if fallback.action.sub_option_id is not None or fallback.action.interaction:
            trace['reason'] = '分支或磁力动作沿用原协议'
            return fallback, trace
        floor = ranked[0].utility * self.config.min_utility_fraction
        productive = tuple(r for r in ranked if r.utility >= floor)
        candidates = tuple(r.action for r in productive[:self.config.max_candidates])
        enriched = self._enrich(replace_actions(obs, candidates))
        try:
            bound = self.adapter.convert(enriched)
        except BridgeBlocked as exc:
            trace['reason'] = '模型观测不可用: ' + str(exc)
            return fallback, trace
        utilities = {r.action.key: r.utility for r in ranked}
        if fallback.action not in bound.actions:
            trace['reason'] = '优先动作无法无损表达，保留原协议'
            return fallback, trace
        batch = self.encoder.collate([bound.observation])
        torch = self.torch
        with torch.inference_mode():
            output = self.model.forward_full(batch)
        n = len(bound.actions)
        logits = output['logits'][0, :n]
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite_actor_logits')
        logp = torch.log_softmax(logits, dim=-1).tolist()
        expected_rank = output.get('expected_rank')
        rank_hint = None if expected_rank is None else float(expected_rank[0])
        rows = []
        for i, a in enumerate(bound.actions):
            probability = math.exp(logp[i])
            rule = self.config.heuristic_weight * math.tanh(utilities[a.key] / 25.0)
            rows.append(dict(index=i, action_key=a.key, kind=str(a.kind), utility=utilities[a.key],
                             actor_logp=logp[i], actor_probability=probability,
                             heuristic_term=rule, score=self.config.actor_weight * probability + rule))
        best = max(rows, key=lambda r: (r['score'], -r['index']))
        trace.update(chosen_key=best['action_key'], neural_used=True, mode='actor',
                     actor_changed_selection=best['action_key'] != fallback.action.key,
                     value_ranked=False, candidates=rows,
                     network_expected_rank_not_used=rank_hint,
                     excluded=list(bound.excluded),
                     utility_guard_omitted=len(ranked) - len(productive),
                     shortlist_omitted=max(0, len(productive) - len(candidates)),
                     elapsed_ms=(time.perf_counter() - started) * 1000,
                     reason='checkpoint actor 对服务器候选排序（value/rank 头不参与）')
        a = bound.actions[best['index']]
        return Decision(a, obs.revision,
                        f'{self.role}-actor: p={best["actor_probability"]:.3f}, {a.kind}'), trace


def _wait_action():
    from bg_agent.model import Action
    return Action(Kind.WAIT)


def replace_actions(obs, actions):
    from dataclasses import replace
    return replace(obs, actions=tuple(actions))
