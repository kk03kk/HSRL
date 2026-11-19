"""评测侧「动作契约 + 协议指纹 + 策略构建」的唯一入口。

为什么单独成模块（2026-11-17 审计 P0-1 / P2-2）
------------------------------------------------
1. **根因修复**：``workflows/eval_human_gap.py`` 在 commit ``2acfebf`` 的重构里把
   ``PPOEvaluator`` 的构造删掉了，却保留了 ``if evaluator is None:``，于是任何
   **非** ``thirdparty:`` 的 checkpoint 都会 ``UnboundLocalError``。构造策略这件事
   原来散落在这个 800 行脚本里，改一处漏一处；现在收敛成本模块的一个纯函数。
2. **契约统一**：训练（``deploy/slurm/stage_arena.sbatch``）与评测曾经各写一遍
   ``freeze_cap / forbid_unfreeze / forbid_cancel / move_guard / ban_trinkets /
   no_cancel_choose``，``c0_gate`` / ``human_panel_eval`` / pairwise Elo 三条评测路径
   三套取值（审计 P2-2）。现在只有 :data:`TRAINING_CONTRACT` 一份真值。
3. **协议可指纹化**：审计 P0-2 发现 gap 面板路径**根本没有** ``protocol_sha256``，
   于是 ``docs/EVAL_PROTOCOL.md`` §4 规则 5（"不同协议禁止混榜"）无法执行。
   :class:`EvalProtocol` 把"会影响分数的所有输入"收进一个 dataclass，
   :meth:`EvalProtocol.sha256` 给出稳定指纹。

设计约束
--------
* **不在 import 期加载引擎或 torch**：``make_eval_policy`` 的依赖通过可选参数注入，
  默认值在函数内部惰性 import。这样本模块可以在没有 ``hsbg_engine`` 的机器上单测。
* ``protocol`` 里的字段**只包含跨模型/跨分片必须相同的量**；``games/seed_start/
  seed_end/shard_index`` 这类"每个分片不同"的量放 :meth:`EvalProtocol.run_dict`，
  **不进 hash**（否则 8 个分片会得到 8 个不同的 ``protocol_sha256``，
  合并时必然误报"换榜"）。
* 纯函数、无 I/O、无全局状态。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from .lobby_filter import engine_rule_overrides
from .strict_mirror import HUMAN_ACTION_PROFILES, OPEN_ACTION_PROFILE, STANDARD_ACTION_PROFILE, human_tail_policy

__all__ = [
    "TRAINING_CONTRACT",
    "EVAL_CONTRACT_KEYS",
    "EvalProtocol",
    "protocol_from_args",
    "make_eval_policy",
    "assert_same_protocol",
    "protocol_sha256_of",
]

#: **训练侧实际生效**的冻结动作契约（唯一真值）。
#: 来源：``deploy/slurm/stage_arena.sbatch`` 的
#: ``--max-actions-per-turn 100 --move-guard 1 --move-cap 2 --move-repeat-limit 2
#:   --forbid-cancel 1 --forbid-unfreeze 1 --freeze-cap 1 --ban-trinkets 1``
#: 与 ``docs/TRAINING_RECIPE.md`` §2.7。
#: 评测必须使用**同一份**取值，否则被评策略会进入训练时从未见过的状态
#: （例如训练限每回合 2 次 Freeze、评测放开到引擎上限 10 次）。
#:
#: **2026-11-17 用户决定收紧三个上限**（D-070，实测依据见
#: ``docs/analysis/ACTION_CAP_20261117.md``）：
#:   * ``max_actions_per_turn`` 80 → **100**（原值几乎从不触发：评测 13,031 个回合里只 3 次；
#:     放宽是为了给中后期留余量，不影响可比性以外的行为）
#:   * ``move_cap`` 5 → **2**（换位每回合最多 2 次）
#:   * ``freeze_cap`` 2 → **1**（Freeze 键每回合只能按 1 次）
#: 这三项属于**冻结契约**，改动即换榜（``protocol_sha256`` 会变），历史面板不可与新面板混比。
TRAINING_CONTRACT: Dict[str, Any] = {
    "max_actions_per_turn": 100,
    "move_guard": True,
    "move_cap": 2,
    "move_repeat_limit": 2,
    "forbid_cancel": True,
    "forbid_unfreeze": True,
    "freeze_cap": 1,
    "ban_trinkets": True,
    "no_cancel_choose": False,
}

#: 参与协议指纹的契约键（顺序固定，写进报告便于人读）。
_profile = os.environ.get("HSAI_ACTION_PROFILE", "v14")
if _profile in HUMAN_ACTION_PROFILES:
    TRAINING_CONTRACT = dict(TRAINING_CONTRACT, move_cap=4, freeze_cap=2,
                             forbid_unfreeze=False, ban_trinkets=False)
    if _profile == OPEN_ACTION_PROFILE:
        TRAINING_CONTRACT.update(max_actions_per_turn=200, move_guard=False, move_cap=0, freeze_cap=0)
    elif _profile == STANDARD_ACTION_PROFILE:
        TRAINING_CONTRACT.update(max_actions_per_turn=200, move_cap=10, move_repeat_limit=0, freeze_cap=3)
elif _profile != "v14":
    raise ValueError(f"Unknown HSAI_ACTION_PROFILE: {_profile}")
EVAL_CONTRACT_KEYS: Tuple[str, ...] = tuple(TRAINING_CONTRACT)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        return bool(default)
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _race_names(spec: Any) -> Tuple[str, ...]:
    """解析 ``--lobby-exclude``（``"MECH,DRAGON"`` / list）成规范种族名。

    与 ``lobby_filter.parse_race_list`` 同一口径；这里**惰性 import**，避免把
    ``lobby_filter``（依赖 numpy 的位解码）拖进本模块的 import 期。
    """
    if not spec:
        return ()
    try:
        from .lobby_filter import parse_race_list  # noqa: PLC0415
        return tuple(parse_race_list(spec))
    except Exception:  # noqa: BLE001 - 解析失败就按"没写"处理（后续 assert 会再查）
        return tuple(str(x) for x in (spec if isinstance(spec, (list, tuple))
                                      else str(spec).split(",")) if str(x).strip())


@dataclass(frozen=True)
class EvalProtocol:
    """一次评测的**可比性协议**：契约 + 对手场 + 引擎口径。

    字段分三组：

    1. 动作契约（``EVAL_CONTRACT_KEYS``）——必须与训练一致；
    2. 对手场（``mirror_*`` / ``forced_tier_*`` / ``human_*``）——决定难度与注入方式；
    3. 引擎口径（``engine_*``）——换引擎/换卡表就是换榜。

    ``engine`` 子字典允许为空（离线单测），为空时 ``sha256`` 仍可用，只是可复现性
    依赖调用方自己保证。
    """

    # ── 1. 动作契约 ──────────────────────────────────────────────
    max_actions_per_turn: int = TRAINING_CONTRACT["max_actions_per_turn"]
    move_guard: bool = TRAINING_CONTRACT["move_guard"]
    move_cap: int = TRAINING_CONTRACT["move_cap"]
    move_repeat_limit: int = TRAINING_CONTRACT["move_repeat_limit"]
    forbid_cancel: bool = TRAINING_CONTRACT["forbid_cancel"]
    forbid_unfreeze: bool = TRAINING_CONTRACT["forbid_unfreeze"]
    freeze_cap: int = TRAINING_CONTRACT["freeze_cap"]
    ban_trinkets: bool = TRAINING_CONTRACT["ban_trinkets"]
    no_cancel_choose: bool = TRAINING_CONTRACT["no_cancel_choose"]
    #: 仅对第三方权重生效（``thirdparty:`` spec）。作者只要求 no_cancel_choose，
    #: 全局禁 Cancel 是比他的训练契约更强的约束，因此单独开关、默认关。
    tp_forbid_cancel: bool = False
    #: 第三方权重的 ``keep_start_board``（回合开始已在场的随从不可卖）。
    keep_start_board: bool = True

    # ── 2. 对手场 ────────────────────────────────────────────────
    mirror_snaps: str = ""
    mirror_sample: str = "p50band"
    mirror_strict: int = 0
    mirror_seats: int = 7
    mirror_p: float = 1.0
    human_boards: str = ""
    human_board_sample: str = "auto"
    forced_tier_plan: str = ""
    forced_tier_prob: float = 0.0
    #: **lobby 种族排除**（D-071；空 = 不过滤）。它改变了"环境里有哪些部族"，
    #: 属于**必须进指纹**的量：禁龙/机械与不禁是两张不同的榜。
    #: 2026-11-17 验证时发现它原先不在 protocol 里 ⇒ 两次评测会撞同一个 sha。
    lobby_exclude: Tuple[str, ...] = ()

    # ── 3. 引擎口径（空 dict = 未提供）───────────────────────────
    engine: Mapping[str, Any] = field(default_factory=dict)
    human_tail: str = "exact"
    engine_rules: Mapping[str, Any] = field(default_factory=engine_rule_overrides)

    # ---- 序列化 ----
    def contract_dict(self) -> Dict[str, Any]:
        """只含动作契约（顺序固定），便于人读与断言。"""
        return {k: getattr(self, k) for k in EVAL_CONTRACT_KEYS}

    def as_dict(self) -> Dict[str, Any]:
        """进 ``protocol_sha256`` 的全部内容（顺序稳定）。"""
        out: Dict[str, Any] = {
            "schema": "hsai.eval_protocol/v1",
            "contract": self.contract_dict(),
            "tp_forbid_cancel": bool(self.tp_forbid_cancel),
            "keep_start_board": bool(self.keep_start_board),
            "field": {
                "mirror_snaps": str(self.mirror_snaps or ""),
                "mirror_sample": str(self.mirror_sample or ""),
                "mirror_strict": int(self.mirror_strict),
                "mirror_seats": int(self.mirror_seats),
                "mirror_p": float(self.mirror_p),
                "human_boards": str(self.human_boards or ""),
                "human_board_sample": str(self.human_board_sample or ""),
                "forced_tier_plan": str(self.forced_tier_plan or ""),
                "forced_tier_prob": float(self.forced_tier_prob),
                "lobby_exclude": [str(x) for x in (self.lobby_exclude or ())],
            },
            "engine": {str(k): self.engine[k] for k in sorted(self.engine or {})},
        }
        if self.human_tail != "exact":
            out["field"]["human_tail"] = self.human_tail
        if self.engine_rules:
            out["engine_rules"] = dict(self.engine_rules)
        return out

    def sha256(self) -> str:
        """协议指纹（``docs/EVAL_PROTOCOL.md`` §4 规则 5 的 ``protocol_sha256``）。"""
        return protocol_sha256_of(self.as_dict())

    def run_dict(self, *, games: int, seed_start: int, seed_end: int,
                 shard_index: Optional[int] = None,
                 shard_count: Optional[int] = None) -> Dict[str, Any]:
        """**不进 hash** 的运行参数（各分片/各轮不同）。"""
        out = {
            "games": int(games),
            "seed_start": int(seed_start),
            "seed_end": int(seed_end),
            "shard_index": None if shard_index is None else int(shard_index),
            "shard_count": None if shard_count is None else int(shard_count),
        }
        return out

    # ---- 契约校验：必须与训练一致 ----
    def assert_training_contract(self) -> None:
        """契约与 :data:`TRAINING_CONTRACT` 不一致时抛 ``ValueError``。

        为什么用异常而不是警告：训练/评测契约不一致会让"同口径比较"这句话失效，
        而这种偏差是**静默**的（分数照样出来）。宁可让评测起不来。
        """
        bad = {k: (getattr(self, k), TRAINING_CONTRACT[k])
               for k in EVAL_CONTRACT_KEYS
               if getattr(self, k) != TRAINING_CONTRACT[k]}
        if bad:
            detail = ", ".join(f"{k}={got!r}(训练={want!r})"
                               for k, (got, want) in sorted(bad.items()))
            raise ValueError(f"评测动作契约与训练不一致：{detail}")


def protocol_sha256_of(payload: Mapping[str, Any]) -> str:
    """``payload`` 的稳定 sha256（``sort_keys`` + 紧凑分隔符，无空格差异）。

    2026-11-17：**忽略 payload 里的 ``protocol_sha256`` 字段**。以前不忽略，导致最自然的
    审计动作——"拿面板里存下来的 ``protocol`` 段重算一次指纹"——**必然 MISMATCH**
    （自己把自己算进去了），会让人误判"面板被篡改"。``EvalProtocol.as_dict()`` 本来
    就没有这个键，所以本改动**不改变任何既有指纹**。
    """
    body = {k: v for k, v in dict(payload).items() if k != "protocol_sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def protocol_from_args(args: Any, *, engine: Optional[Mapping[str, Any]] = None,
                       mirror_seats: Optional[int] = None) -> EvalProtocol:
    """从 ``argparse.Namespace`` 构造 :class:`EvalProtocol`（缺字段用训练契约补）。

    刻意**不**读 ``games/seed``：它们是分片级参数，不能进协议指纹。
    """
    get = lambda name, default=None: getattr(args, name, default)  # noqa: E731
    return EvalProtocol(
        max_actions_per_turn=_as_int(get("max_actions_per_turn"),
                                     TRAINING_CONTRACT["max_actions_per_turn"]),
        move_guard=_as_bool(get("move_guard"), TRAINING_CONTRACT["move_guard"]),
        move_cap=_as_int(get("move_cap"), TRAINING_CONTRACT["move_cap"]),
        move_repeat_limit=_as_int(get("move_repeat_limit"),
                                  TRAINING_CONTRACT["move_repeat_limit"]),
        forbid_cancel=_as_bool(get("forbid_cancel"),
                               TRAINING_CONTRACT["forbid_cancel"]),
        forbid_unfreeze=_as_bool(get("forbid_unfreeze"),
                                 TRAINING_CONTRACT["forbid_unfreeze"]),
        freeze_cap=_as_int(get("freeze_cap"), TRAINING_CONTRACT["freeze_cap"]),
        ban_trinkets=_as_bool(get("ban_trinkets"), TRAINING_CONTRACT["ban_trinkets"]),
        no_cancel_choose=_as_bool(get("no_cancel_choose"),
                                  TRAINING_CONTRACT["no_cancel_choose"]),
        tp_forbid_cancel=_as_bool(get("tp_forbid_cancel"), False),
        keep_start_board=_as_bool(get("keep_start_board"), True),
        mirror_snaps=str(get("mirror_snaps", "") or ""),
        mirror_sample=str(get("mirror_sample", "p50band") or "p50band"),
        mirror_strict=_as_int(get("mirror_strict"), 0),
        mirror_seats=(_as_int(mirror_seats, 7) if mirror_seats is not None
                      else _as_int(get("mirror_seats"), 7)),
        mirror_p=float(1.0 if get("mirror_p", 1.0) is None else get("mirror_p", 1.0)),
        human_boards=str(get("human_opponents", "") or ""),
        human_board_sample=str(get("human_opponent_sample", "auto") or "auto"),
        forced_tier_plan=str(get("forced_tier_plan", "") or ""),
        forced_tier_prob=float(get("forced_tier_prob", 0.0) or 0.0),
        lobby_exclude=tuple(_race_names(get("lobby_exclude", ""))),
        human_tail=human_tail_policy(),
        engine=dict(engine or {}),
    )


def make_eval_policy(
    spec: str,
    *,
    protocol: EvalProtocol,
    device: str = "cpu",
    load_checkpoint: Optional[Callable[[str, str], Tuple[Any, Dict[str, Any]]]] = None,
    evaluator_factory: Optional[Callable[..., Any]] = None,
    greedy_policy_cls: Optional[Callable[..., Any]] = None,
    thirdparty_policy_cls: Optional[Callable[..., Any]] = None,
    is_thirdparty_spec: Optional[Callable[[str], bool]] = None,
    split_thirdparty: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """**唯一的**评测策略构造入口：``spec`` → ``{model, meta, policy, kind}``。

    两条分支：

    * 自家 checkpoint（``load_checkpoint`` + ``PPOEvaluator`` + ``PPOGreedyPolicy``）；
    * 第三方权重（``thirdparty:/path``，``ThirdPartyGreedyPolicy``）。

    依赖默认惰性 import（``hsbg.az.train.model`` / ``.ppo`` / ``.thirdparty_policy``），
    因此单测可以注入假实现而不需要引擎或 torch。

    ``kind`` 是 ``"own"`` / ``"thirdparty"``，写进报告便于审计。
    """
    if load_checkpoint is None or evaluator_factory is None \
            or greedy_policy_cls is None:
        from .model import load_checkpoint as _lc  # noqa: PLC0415 - 惰性，单测可注入
        from .ppo import PPOEvaluator as _EV, PPOGreedyPolicy as _GP  # noqa: PLC0415
        load_checkpoint = load_checkpoint or _lc
        evaluator_factory = evaluator_factory or _EV
        greedy_policy_cls = greedy_policy_cls or _GP
    if is_thirdparty_spec is None or split_thirdparty is None \
            or thirdparty_policy_cls is None:
        from .thirdparty_policy import (ThirdPartyGreedyPolicy as _TP,  # noqa: PLC0415
                                        is_thirdparty_spec as _is,  # noqa: PLC0415
                                        split_thirdparty as _sp)  # noqa: PLC0415
        is_thirdparty_spec = is_thirdparty_spec or _is
        split_thirdparty = split_thirdparty or _sp
        thirdparty_policy_cls = thirdparty_policy_cls or _TP

    if is_thirdparty_spec(str(spec)):
        real = split_thirdparty(str(spec))
        policy = thirdparty_policy_cls(
            real, device=device,
            max_actions_per_turn=int(protocol.max_actions_per_turn),
            forbid_cancel=bool(protocol.tp_forbid_cancel),
            keep_start_board=bool(protocol.keep_start_board),
            ban_trinkets=bool(protocol.ban_trinkets),
            freeze_cap=int(protocol.freeze_cap),
            forbid_unfreeze=bool(protocol.forbid_unfreeze),
        )
        return {"model": getattr(policy, "net", None), "meta": {}, "policy": policy,
                "kind": "thirdparty", "real_path": real}

    model, meta = load_checkpoint(str(spec), device)
    evaluator = evaluator_factory(model, device=device,
                                  norm_mean=meta.get("norm_mean"),
                                  norm_std=meta.get("norm_std"))
    policy = greedy_policy_cls(
        evaluator,
        max_actions_per_turn=int(protocol.max_actions_per_turn),
        move_guard=bool(protocol.move_guard),
        move_cap=int(protocol.move_cap),
        move_repeat_limit=int(protocol.move_repeat_limit),
        forbid_cancel=bool(protocol.forbid_cancel),
        freeze_cap=int(protocol.freeze_cap),
        forbid_unfreeze=bool(protocol.forbid_unfreeze),
        ban_trinkets=bool(protocol.ban_trinkets),
        no_cancel_choose_on=bool(protocol.no_cancel_choose),
    )
    return {"model": model, "meta": meta, "policy": policy, "kind": "own",
            "real_path": str(spec)}


def assert_same_protocol(*panels: Mapping[str, Any], where: str = "面板") -> str:
    """校验若干 panel 的 ``protocol_sha256`` 相同，返回该指纹。

    任何一侧缺失或不同都抛 ``ValueError``（**不是警告**）：``docs/EVAL_PROTOCOL.md``
    §4 规则 5 规定"不一致就是不同榜，禁止混比"，而混比出来的数字看起来一样正常。
    """
    if not panels:
        raise ValueError(f'{where}: 没有输入面板')
    seen: Dict[str, int] = {}
    for i, panel in enumerate(panels):
        proto = (panel or {}).get("protocol") or {}
        sha = str(proto.get("protocol_sha256") or "").strip()
        if not sha:
            raise ValueError(
                f"{where}[{i}] 没有 protocol_sha256：该面板无法证明与其它面板同协议"
                "（旧面板请重跑评测；不要直接混比）")
        if any(k in proto for k in ('schema', 'contract', 'field', 'engine')):
            if protocol_sha256_of(proto) != sha:
                raise ValueError(f'{where}[{i}] protocol_sha256 与协议正文不一致')
        seen[sha] = seen.get(sha, 0) + 1
    if len(seen) > 1:
        detail = ", ".join(f"{k[:12]}…×{v}" for k, v in sorted(seen.items()))
        raise ValueError(f"{where}的 protocol_sha256 不一致（不同榜，禁止混比）：{detail}")
    return next(iter(seen))
