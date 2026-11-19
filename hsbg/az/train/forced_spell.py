"""强制买法术（forced spell-buy）探索课程：让零频动作真的被采到，并如实记账。

课程在指定回合及配额内引入合法法术购买，分别记录强制购买和策略自主购买。
2026-11-14 的旧实验仅检查第八个商店槽；这些历史计数无法还原各槽的法术购买频率。

本模块只做三件事，全部是纯逻辑、可单测（与 ``forced_tier`` 同构）：

1. **判定**：每次 learner 决策上判断"这次是否强制买法术"；
2. **记账**：每局配额、`blocked`（掩码不合法）、`prob_skips`（概率门控）、
   `policy_spell_actions`（策略**自己**按了多少次 —— 用来判断它到底有没有学会）；
3. **汇总**：逐局 meta → 采集级统计。

设计约束（与 ``forced_tier`` / PPO 口径一致，一条都不能少）：

- **只强制真正合法的动作**：判据是引擎给的合法掩码里商店法术槽那一位
  与原生观测中的酒馆物品类型共同确认。本模块不读金币、不补金币、不改引擎状态；
  不合法就**不强制**，并记一次 ``blocked``。
- **被强制的决策不进策略梯度**：调用方把该条记录标成 ``forced``（与动作上限守卫、
  换位守卫、强制升本同口径：``logprob=0``，PPO 的策略/辅助损失排除，只保留价值链）。
  这样既不伪造 on-policy ratio，也不会把"脚本替我按的键"当成策略的选择。
- **关闭时逐位一致**：``per_game <= 0`` 时不构造对象、不读引擎、不新增 record 字段。
- **概率门控**：``force_prob < 1`` 时按概率强制，用于"逐步解除强制"；
  未强制且合法时调用方可写 ``teacher_action``（DAgger 标签）。

统计口径（每局一条）：

    forced               本次采集里被强制买法术的次数（应该 > 0 才算机制打开）
    blocked              计划要强制但掩码不合法（引擎口径下不该长期为真）
    prob_skips           概率门控挡下的次数
    policy_spell_actions 策略**未被强制**时自己选择买法术的次数（学会没学会看这个）
    legal_decisions      掩码里法术槽合法的决策数（证明"可达"）
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from .forced_tier import parse_prob_schedule, prob_at
from hsbg.env.action_space_std import A_BUY, N_SHOP
from hsbg.env.obs_std import ITEM_F, ITEM_IS_MINION, ITEM_PRESENT, SLICE_SHOP, SPELL_PRESENT


def legal_spell_buys(features, mask):
    """Return current native buy actions for affordable, occupied spell slots."""
    shop = features[SLICE_SHOP].reshape(N_SHOP, ITEM_F)
    return tuple(
        A_BUY + i for i, item in enumerate(shop)
        if mask[A_BUY + i] and item[ITEM_PRESENT] > .5
        and item[ITEM_IS_MINION] < .5 and item[SPELL_PRESENT] > .5
    )

#: 每局默认最多强制买几次
DEFAULT_FORCED_SPELL_PER_GAME = 3
#: 默认从第几回合开始允许强制（之前买法术通常挤掉经营经济）
DEFAULT_FORCED_SPELL_TURN_MIN = 6

#: 回合上限（仅防非法输入；引擎回合远小于此）
_TURN_MAX = 200


def _check_non_negative_int(name: str, value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数: {value!r}") from exc
    if v < 0:
        raise ValueError(f"{name} 必须 >= 0: {v}")
    return v


class ForcedSpellDecision:
    """一次 learner 决策上的强制判定结果（纯数据类，刻意不用 dataclass）。

    与 ``ForcedTierDecision`` 同因：本模块要能在登录节点按文件路径加载执行，
    ``@dataclass`` 在 ``exec_module`` 期间反查注解类型的跨版本行为不一致。
    """

    __slots__ = ("force", "teacher", "blocked", "turn", "remaining", "action")

    def __init__(self) -> None:
        #: 是否强制把动作改成"买法术"
        self.force: bool = False
        #: 是否写教师标签（合法、但本次没有强制）
        self.teacher: bool = False
        #: 计划内但掩码不合法（只用于记账，不改动作）
        self.blocked: bool = False
        #: 本次决策的回合号
        self.turn: int = -1
        #: 本局剩余可强制次数（强制之后的值）
        self.remaining: int = 0
        #: 强制时应该用的动作下标（不需要强制时为 None）
        self.action: Optional[int] = None

    def as_record(self) -> Dict[str, object]:
        return {"force": bool(self.force), "turn": int(self.turn),
                "remaining": int(self.remaining), "action": self.action, "teacher": self.teacher}

    def __repr__(self) -> str:  # 仅调试/日志用
        return (f"ForcedSpellDecision(force={self.force}, teacher={self.teacher}, "
                f"blocked={self.blocked}, turn={self.turn}, "
                f"remaining={self.remaining})")


class _GameState:
    """单局状态：剩余配额 + 本局计数。"""

    __slots__ = ("remaining", "forced", "blocked", "prob_skips",
                 "policy_spell_actions", "legal_decisions")

    def __init__(self, per_game: int) -> None:
        self.remaining = int(per_game)
        self.forced = 0
        self.blocked = 0
        self.prob_skips = 0
        self.policy_spell_actions = 0
        self.legal_decisions = 0


class ForcedSpellTracker:
    """按局维护强制买法术状态与统计（一个实例服务多局，状态用 ``game_id/seat`` 隔离）。"""

    def __init__(self, per_game: object = DEFAULT_FORCED_SPELL_PER_GAME, *,
                 turn_min: object = DEFAULT_FORCED_SPELL_TURN_MIN,
                 prob_spec: object = None) -> None:
        self.per_game = _check_non_negative_int("per_game", per_game)
        self.turn_min = _check_non_negative_int("turn_min", turn_min)
        if self.turn_min > _TURN_MAX:
            raise ValueError(f"turn_min 必须在 0..{_TURN_MAX}: {self.turn_min}")
        #: 分段常数概率表（按累计决策数查表）；空 ⇒ 恒 1.0
        self.prob_schedule = parse_prob_schedule(prob_spec, default=1.0)
        self._state: Dict[Tuple[int, int], _GameState] = {}
        #: ``decide()`` 被调用的次数（= 累计决策数）；概率表默认拿它当"时钟"，
        #: 这样调用方（PPO 热路径）不必再单独维护一个计数器。
        self._decisions_seen = 0
        #: 已结束对局的累计（``reset()`` 时折叠进来，避免逐局状态无限增长）
        self.forced_actions = 0
        self.blocked = 0
        self.prob_skips = 0
        self.policy_spell_actions = 0
        self.legal_decisions = 0
        self.teacher_labels = 0

    # ---- 生命周期 ----
    @property
    def enabled(self) -> bool:
        return bool(self.per_game > 0)

    def reset(self) -> None:
        """一局开始：清空按局状态（整次采集的累计计数保留）。"""
        self._state.clear()

    def _game(self, game_id: Any, seat: Any) -> _GameState:
        key = (int(game_id), int(seat))
        st = self._state.get(key)
        if st is None:
            st = _GameState(self.per_game)
            self._state[key] = st
        return st

    # ---- 主判定 ----
    def decide(self, *, game_id: Any = 0, seat: Any = 0, turn: Any = 0,
               spell_action: Optional[int], decisions: Optional[float] = None,
               force_prob: Optional[float] = None,
               rng: Optional[object] = None,
               teacher_enabled: bool = True) -> ForcedSpellDecision:
        """判断本次决策是否强制"买法术"。

        ``spell_action`` 来自当前可购买的法术槽；没有可购法术时传 None。
        ``decisions``：查 ``prob_schedule`` 用的累计决策数；``None`` ⇒ 用本对象的
        ``decide()`` 调用次数（PPO 热路径走这个默认值，避免多维护一个计数器）。
        ``force_prob`` 给定时优先用它（便于测试与逐段覆盖）。
        """
        out = ForcedSpellDecision()
        if not self.enabled:
            return out
        spell_legal = spell_action is not None
        if spell_legal and not A_BUY <= spell_action < A_BUY + N_SHOP:
            raise ValueError("invalid native spell purchase index")
        self._decisions_seen += 1
        turn = int(turn)
        out.turn = turn
        st = self._game(game_id, seat)
        out.remaining = int(st.remaining)
        if turn < self.turn_min or st.remaining <= 0:
            # 未启用/配额用完：仍然统计"策略自己按了"与"是否可达"
            if spell_legal:
                st.legal_decisions += 1
                self.legal_decisions += 1
            return out
        if not spell_legal:
            st.blocked += 1
            self.blocked += 1
            out.blocked = True
            return out
        st.legal_decisions += 1
        self.legal_decisions += 1
        p = (float(force_prob) if force_prob is not None
             else prob_at(self.prob_schedule,
                          float(self._decisions_seen if decisions is None
                                else decisions)))
        if p < 1.0:
            if self._draw(rng) >= p:
                st.prob_skips += 1
                self.prob_skips += 1
                if teacher_enabled:
                    out.teacher = True
                    out.action = int(spell_action)
                    self.teacher_labels += 1
                return out
        out.force = True
        out.action = int(spell_action)
        st.remaining -= 1
        st.forced += 1
        self.forced_actions += 1
        out.remaining = int(st.remaining)
        return out

    def note_policy_action(self, *, game_id: Any = 0, seat: Any = 0,
                           is_spell_purchase: bool, forced: bool = False) -> None:
        """记账：策略**未被强制**时自己选择了买法术（判断"学会没学会"的唯一口径）。"""
        if not self.enabled or bool(forced):
            return
        if not is_spell_purchase:
            return
        st = self._game(game_id, seat)
        st.policy_spell_actions += 1
        self.policy_spell_actions += 1

    @staticmethod
    def _draw(rng: Optional[object]) -> float:
        if rng is None:
            return 0.0
        try:
            return float(rng.random())  # numpy Generator / random.Random
        except AttributeError:  # pragma: no cover - 防御
            return 0.0

    # ---- 统计 ----
    def game_stats(self, game_id: Any = None, seat: Any = None) -> Dict[str, object]:
        """当前（尚未折叠的）对局统计。未启用时返回 ``{}``（不新增 meta 字段）。"""
        if not self.enabled:
            return {}
        if game_id is None:
            rows = list(self._state.values())
        else:
            key = (int(game_id), 0 if seat is None else int(seat))
            rows = [self._state[key]] if key in self._state else []
        if not rows:
            # 没有这局的数据 ⇒ 返回 {}（调用方用 `if game: meta[...]` 判断，
            # 返回全零 dict 会给每一局都写一条无意义的 meta）
            return {}
        return {
            "forced": int(sum(r.forced for r in rows)),
            "blocked": int(sum(r.blocked for r in rows)),
            "prob_skips": int(sum(r.prob_skips for r in rows)),
            "policy_spell_actions": int(sum(r.policy_spell_actions for r in rows)),
            "legal_decisions": int(sum(r.legal_decisions for r in rows)),
            "per_game": int(self.per_game),
            "turn_min": int(self.turn_min),
        }

    def stats(self) -> Dict[str, object]:
        """整次采集的累计统计。未启用时返回 ``{}``。"""
        if not self.enabled:
            return {}
        return {
            "forced_actions": int(self.forced_actions),
            "blocked": int(self.blocked),
            "prob_skips": int(self.prob_skips),
            "policy_spell_actions": int(self.policy_spell_actions),
            "legal_decisions": int(self.legal_decisions),
            "teacher_labels": int(self.teacher_labels),
            "per_game": int(self.per_game),
            "turn_min": int(self.turn_min),
        }


def summarize_forced_spell(metas: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """把逐局 ``forced_spell_game`` 行汇总成采集级统计（与 tier 同构）。"""
    rows = [m for m in metas if isinstance(m, dict)]
    keys = ("forced", "blocked", "prob_skips", "policy_spell_actions", "legal_decisions")
    out: Dict[str, object] = {k: int(sum(int(r.get(k) or 0) for r in rows)) for k in keys}
    out["games"] = len(rows)
    return out if rows else {}


def aggregate_forced_spell(games_meta: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """从逐局 meta 里的 ``forced_spell_game`` 汇总（默认关闭 -> ``{}``）。"""
    rows = [m["forced_spell_game"] for m in games_meta
            if isinstance(m, dict) and isinstance(m.get("forced_spell_game"), dict)]
    return summarize_forced_spell(rows)


def forced_spell_rates(stats: Dict[str, object]) -> Dict[str, float]:
    """把计数换算成"每局"比率，便于跨规模比较。"""
    games = float(stats.get("games") or 0)
    if games <= 0:
        return {}
    return {k: float(stats.get(k) or 0) / games
            for k in ("forced", "policy_spell_actions", "legal_decisions")}


__all__ = [
    "DEFAULT_FORCED_SPELL_PER_GAME",
    "DEFAULT_FORCED_SPELL_TURN_MIN",
    "ForcedSpellDecision",
    "ForcedSpellTracker",
    "legal_spell_buys",
    "aggregate_forced_spell",
    "forced_spell_rates",
    "summarize_forced_spell",
]
