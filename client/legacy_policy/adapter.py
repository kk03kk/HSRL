"""真实日志观测 → 原生候选/模型 → 身份绑定旧动作 → 核验。默认无 OS 输入。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Protocol

from bg_agent.execution import Delivery, Dispatch
from bg_agent.model import Action, Decision, Kind, Observation as ClientObservation, Phase
from bg_agent.legal import LegalActionResolver
from bg_agent.verification import verify, Status
from legacy_policy.spec import CandidateAction, EntityToken, HeroToken, Observation, PendingChoice
from legacy_policy.protocol import waiting_for_sync


class BridgeBlocked(ValueError):
    """缺少确定的映射或当前状态不允许决策/发送。"""


class UnsupportedInteraction(BridgeBlocked):
    """状态有效，但模型动作空间无法无损表达当前交互。"""


def fingerprint(obs: ClientObservation) -> str:
    data = obs.to_dict()
    for key in ("captured_at", "source_updated_at", "log_event_at"):
        data.pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class BoundObservation:
    observation: Observation
    client: ClientObservation
    actions: tuple[Action, ...]
    stamp: str
    excluded: tuple[str, ...]

    def action_at(self, index: int) -> Action:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.actions):
            raise BridgeBlocked("策略返回非法候选下标")
        return self.actions[index]


class ObservationAdapter:
    def __init__(self, *, supported_builds=("255130", "37.0.0.255130"), resolver=None):
        self.supported_builds = tuple(supported_builds)
        self.resolver = resolver or LegalActionResolver()

    @staticmethod
    def _token(card, zone, slot):
        if not card.card_id:
            raise BridgeBlocked("可见实体缺少 card_id；不能猜测隐藏牌")
        keywords = tuple(k.lower() for k in ("TAUNT", "DIVINE_SHIELD", "POISONOUS", "VENOMOUS", "REBORN",
                                             "WINDFURY", "DEATHRATTLE", "BATTLECRY", "MAGNETIC", "STEALTH")
                         if str(card.tags.get(k, "0")) == "1")
        race_names = {11: "undead", 14: "murloc", 15: "demon", 17: "mech", 18: "elemental", 20: "beast",
                      23: "pirate", 24: "dragon", 26: "all", 43: "quilboar", 92: "naga"}
        raw_race = card.tags.get("CARDRACE", "")
        race = race_names.get(int(raw_race)) if str(raw_race).isdigit() else str(raw_race).lower()
        race = {"mechanical": "mech"}.get(race, race)
        races = (race,) if race in race_names.values() else ()
        return EntityToken(card_id=card.card_id, card_type={"MINION": "minion", "SPELL": "spell",
                           "BATTLEGROUND_SPELL": "spell", "HERO_POWER": "hero_power", "HERO": "hero",
                           "BATTLEGROUND_TRINKET": "trinket"}.get(card.card_type, "unknown"),
                           zone=zone, slot=slot, atk=card.attack or 0, health=card.health or 0,
                           tier=card.tier or 0, cost=card.cost or 0, keywords=keywords, races=races,
                           golden=str(card.tags.get("PREMIUM", "0")) == "1", entity_id=card.entity_id)

    def convert(self, obs: ClientObservation, *, now=None) -> BoundObservation:
        if obs.schema_version != 1:
            raise BridgeBlocked("未知客户端观测版本")
        if obs.build not in self.supported_builds:
            raise BridgeBlocked(f"未验证的客户端 build: {obs.build}")
        if obs.phase not in (Phase.RECRUIT, Phase.CHOICE) or waiting_for_sync(obs):
            raise BridgeBlocked("非决策阶段或状态尚未同步")
        if obs.warnings:
            raise BridgeBlocked("客户端状态警告: " + ",".join(obs.warnings))
        if obs.phase == Phase.CHOICE and obs.hero is None and obs.choices and all(c.card_type == "HERO" for c in obs.choices):
            raise UnsupportedInteraction("初始英雄选择由原控制器处理")
        if obs.gold is None or obs.tavern_tier is None or obs.turn is None or obs.hero is None or obs.hero.health is None:
            raise BridgeBlocked("英雄、经济或回合关键字段缺失")
        if not obs.hero.card_id:
            raise BridgeBlocked("英雄 card_id 缺失")
        if len(obs.board) > 7 or len(obs.hand) > 10 or len(obs.shop) > 7 or len(obs.choices) > 5:
            raise BridgeBlocked("实体槽位超出模型编码范围")
        groups = {"board": obs.board, "hand": obs.hand, "tavern": obs.shop, "choice": obs.choices}
        ids = [c.entity_id for cards in groups.values() for c in cards]
        if len(set(ids)) != len(ids):
            raise BridgeBlocked("实体重复或区域不同步")
        if obs.local_player_id is None or any(c.controller != obs.local_player_id for c in obs.board + obs.hand):
            raise BridgeBlocked("己方实体归属不一致")
        slots = {c.entity_id: (z, i) for z, cards in groups.items() for i, c in enumerate(cards)}
        slots[obs.hero.entity_id] = ("", -1)
        eligible = self.resolver.resolve(obs, now=now).for_policy(obs).actions
        pairs, excluded = [], []
        for action in eligible:
            try:
                pairs.append((self._candidate(action, obs, slots), action))
            except BridgeBlocked as exc:
                excluded.append(f"{action.key}: {exc}")
        # 候选签名不能损失分支、磁力或服务器选项身份；含混映射全部拒绝。
        counts = Counter(c.signature() for c, _ in pairs)
        kept = [(c, a) for c, a in pairs if counts[c.signature()] == 1]
        excluded.extend(f"{a.key}: 候选签名含混" for c, a in pairs if counts[c.signature()] > 1)
        if not kept:
            if excluded:
                raise UnsupportedInteraction("没有可唯一映射的合法候选")
            raise BridgeBlocked("没有确认合法的候选")
        if len(kept) > 64:
            raise UnsupportedInteraction("候选超过 64")
        power = next((c for c in obs.controls if c.card_type == "HERO_POWER"), None)
        hero = HeroToken(hero_id=obs.hero.card_id,
                         # PowerState.Entity.card 已经扣除 DAMAGE；不可再次扣血。
                         health=obs.hero.health,
                         armor=int(obs.hero.tags.get("ARMOR", "0")), tier=obs.tavern_tier, gold=obs.gold,
                         upgrade_cost=obs.upgrade_cost if obs.upgrade_cost is not None else 0,
                         power_id=power.card_id if power else "", power_cost=power.cost or 0 if power else 0,
                         power_usable=any(a.kind == Kind.HERO_POWER for _, a in kept))
        tokens = {z: tuple(self._token(c, z, i) for i, c in enumerate(cards)) for z, cards in groups.items()}
        extra = {"frozen_tavern": bool(obs.shop) and all(str(c.tags.get("FROZEN", "0")) == "1" for c in obs.shop)}
        if obs.phase == Phase.CHOICE:
            # v3 的第二身份必须来自当前公开日志，不能在适配后悄悄丢掉赠礼语义。
            extra["pending_choice_options"] = [dict(index=i, card_id=c.card_id,
                secondary_id=c.tags.get("DARK_GIFT_CARD_ID"), label=None, name=None,
                secondary_name=None, cost=c.cost if c.cost is not None else -1)
                for i, c in enumerate(obs.choices)]
        # 无日志证据的大厅、对手牌、随机数、牌池、未来商店均不填充。
        observation = Observation(turn=obs.turn, phase=str(obs.phase), hero=hero,
                                  board=tokens["board"], hand=tokens["hand"], tavern=tokens["tavern"],
                                  pending_choice=PendingChoice("client_choice", tokens["choice"]) if obs.phase == Phase.CHOICE else None,
                                  candidates=tuple(c for c, _ in kept),
                                  extra=extra)
        return BoundObservation(observation, obs, tuple(a for _, a in kept), fingerprint(obs), tuple(excluded))

    @staticmethod
    def _candidate(a, obs, slots):
        if a.sub_option_id is not None or a.selected or a.interaction:
            raise BridgeBlocked("分支/多选/磁力交互交给已验证的旧控制器")
        if obs.interaction and obs.interaction.mode != "choice":
            raise BridgeBlocked("当前特殊交互不属于模型动作空间")
        names = {Kind.BUY: "buy", Kind.SELL: "sell", Kind.REFRESH: "reroll", Kind.UPGRADE: "upgrade",
                 Kind.FREEZE: "freeze", Kind.HERO_POWER: "hero_power", Kind.ACTIVATE: "activate",
                 Kind.END: "end_turn", Kind.CHOOSE: "choose"}
        if a.kind == Kind.PLAY:
            card = next((c for c in obs.hand if c.entity_id == a.source), None)
            if card is None or card.card_type not in ("MINION", "SPELL", "BATTLEGROUND_SPELL"):
                raise BridgeBlocked("未知手牌类型")
            kind = "play_minion" if card.card_type == "MINION" else "play_spell"
        elif a.kind in names:
            kind = names[a.kind]
        else:
            raise BridgeBlocked("模型不支持此动作类型")
        source = slots.get(a.source, ("", -1))
        target = slots.get(a.target, ("", -1))
        if a.target is not None and a.target not in slots:
            raise BridgeBlocked("目标不在可见区域")
        if a.target == obs.hero.entity_id:
            raise BridgeBlocked("模型无独立英雄目标槽；禁止与无目标动作混同")
        if kind in ("buy", "sell", "play_minion", "play_spell", "choose") and source[1] < 0:
            raise BridgeBlocked("来源实体缺失")
        # 按钮本身不占实体槽，技能的来源由 hero.power_id 表示。
        if kind not in ("buy", "sell", "play_minion", "play_spell", "activate"):
            source = ("", -1)
        if kind == "activate" and source[0] != "board":
            raise BridgeBlocked("额外控制按钮不在模型实体空间")
        option = slots[a.source][1] if kind == "choose" else -1
        position = a.position - 1 if a.position is not None else -1
        cost = a.resource_cost
        if cost is None and kind == "upgrade":
            cost = obs.upgrade_cost
        if cost is None and kind in ("play_minion", "play_spell", "hero_power", "reroll", "freeze"):
            card = next((c for c in obs.hand + obs.controls if c.entity_id == a.source), None)
            cost = card.cost if card else None
        return CandidateAction(kind=kind, src_zone=source[0], src_slot=source[1],
                               tgt_zone=target[0], tgt_slot=target[1], position=position, option=option,
                               cost=cost if cost is not None else -1)


class Selector(Protocol):
    def select(self, obs: Observation) -> int: ...


class BridgePolicy:
    """可直接注入旧 AgentController；复杂交互可显式委派旧策略。"""
    def __init__(self, selector: Selector, *, adapter=None, special_policy=None):
        self.selector, self.adapter, self.special_policy = selector, adapter or ObservationAdapter(), special_policy
        self.last_bound = None

    def decide(self, observation):
        if observation.phase not in (Phase.RECRUIT, Phase.CHOICE):
            return self._special(observation)
        try:
            bound = self.adapter.convert(observation)
        except UnsupportedInteraction:
            return self._special(observation)
        self.last_bound = bound
        # 部分特殊候选若被排除，不让 AI 抢先 end_turn；沿用旧协议处理。
        if bound.excluded:
            return self._special(observation)
        index = self.selector.select(bound.observation)
        return Decision(bound.action_at(index), observation.revision, "hsbg_local:" + type(self.selector).__name__)

    def _special(self, observation):
        if self.special_policy is None:
            raise BridgeBlocked("当前交互超出模型动作空间，且未配置旧策略")
        legal = self.adapter.resolver.resolve(observation).for_policy(observation)
        decision = self.special_policy.decide(legal)
        if (decision.action.kind != Kind.WAIT and decision.action not in legal.actions) or decision.revision != observation.revision:
            raise BridgeBlocked("旧策略未返回当前已核实的合法动作")
        return decision

    def on_verified(self, *args, **kwargs):
        callback = getattr(self.special_policy, "on_verified", None)
        if callback:
            callback(*args, **kwargs)

    def on_decision_adopted(self, *args, **kwargs):
        callback = getattr(self.special_policy, "on_decision_adopted", None)
        if callback:
            callback(*args, **kwargs)

    def on_abandoned(self, *args, **kwargs):
        callback = getattr(self.special_policy, "on_abandoned", None)
        if callback:
            callback(*args, **kwargs)


@dataclass(frozen=True)
class CycleResult:
    candidate: CandidateAction
    action: Action
    dispatch: Dispatch
    outcome: object | None


def offline_cycle(provider, executor, selector, *, adapter=None):
    """单步离线闭环；真实输入仍应通过旧 AgentController 的完整门槛。"""
    if not getattr(executor, "offline_only", False):
        raise BridgeBlocked("此离线入口拒绝真实输入执行器")
    adapter = adapter or ObservationAdapter()
    before = provider.observe()
    bound = adapter.convert(before)
    if bound.excluded:
        raise BridgeBlocked("候选含模型不支持的交互")
    index = selector.select(bound.observation)
    action = bound.action_at(index)
    fresh = provider.observe()
    # 重检合法性、版本、时间及所有候选身份，禁止在推理后使用旧槽位。
    current = adapter.convert(fresh)
    if bound.stamp != current.stamp:
        raise BridgeBlocked("推理期间游戏状态或合法动作发生变化")
    dispatch = executor.execute(action, before.revision)
    outcome = verify(action, before, provider.observe()) if dispatch.delivery == Delivery.ACCEPTED else None
    return CycleResult(bound.observation.candidates[index], action, dispatch, outcome)
