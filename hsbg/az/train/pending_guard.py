"""待选链去循环守卫。

`Play` / `HeroPower` 打开目标或 Choose 菜单时，引擎允许取消；策略可能在
``Play -> Cancel -> Play`` 之间循环，但其实第一次 Play 之后盘面/手牌没有变化。
本守卫不修改引擎，只在采样前改写 mask：

1. ``commit_play``：Play/HeroPower 刚打开的待选菜单，屏蔽 Cancel，要求选择
   目标或模式；
2. 同一回合里某个 Play 之后已经 Cancel 过，则本回合该 hand 的 Play 全部屏蔽；
3. 同一英雄技能槽 Cancel 后，本回合该槽屏蔽；
4. 单回合 Cancel 达到 ``cancel_cap`` 后屏蔽 Cancel（若没有其他合法动作则回退
   原 mask，避免制造非法动作）。

状态按 ``(game_id, seat, turn)`` 隔离，默认只在显式启用时生效。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple
import json
import numpy as np

CANCEL = 7
A_PLAY = 46
N_POS = 8
A_PLAY_HI = A_PLAY + 10 * N_POS
A_HERO_POWER = 4
N_POWERS = 2
A_HERO_HI = A_HERO_POWER + N_POWERS


def without_cancel(mask: Any) -> np.ndarray:
    """Clear the single Cancel action bit; all other legal actions are preserved."""
    out = np.asarray(mask, dtype=bool).copy()
    out[CANCEL] = False
    return out


def play_hand(action: int) -> int | None:
    a = int(action)
    return (a - A_PLAY) // N_POS if A_PLAY <= a < A_PLAY_HI else None


def hero_slot(action: int) -> int | None:
    a = int(action)
    return a - A_HERO_POWER if A_HERO_POWER <= a < A_HERO_HI else None


@dataclass
class PendingGuardStats:
    decisions: int = 0
    cancel_seen: int = 0
    commit_blocked: int = 0
    cap_blocked: int = 0
    banned_play_masks: int = 0
    banned_power_masks: int = 0
    fallback_masks: int = 0

    def add(self, other: "PendingGuardStats") -> None:
        for k in ("decisions", "cancel_seen", "commit_blocked", "cap_blocked",
                  "banned_play_masks", "banned_power_masks", "fallback_masks"):
            setattr(self, k, getattr(self, k) + int(getattr(other, k)))


@dataclass
class _SeatTurn:
    turn: int
    last_action: int = -1
    prev_action: int = -1
    cancel_count: int = 0
    banned_play: Set[int] = field(default_factory=set)
    banned_power: Set[int] = field(default_factory=set)


class PendingGuard:
    def __init__(self, *, cancel_cap: int = 2, commit_play: bool = True) -> None:
        if cancel_cap < 1:
            raise ValueError("cancel_cap 必须 >= 1")
        self.cancel_cap = int(cancel_cap)
        self.commit_play = bool(commit_play)
        self._turns: Dict[Tuple[int, int], _SeatTurn] = {}
        self.stats = PendingGuardStats()

    def reset(self) -> None:
        self._turns.clear()
        self.stats = PendingGuardStats()

    def clear_state(self) -> None:
        self._turns.clear()

    def _state(self, game_id: Any, seat: Any, turn: Any) -> _SeatTurn:
        key = (int(game_id), int(seat))
        t = int(turn)
        st = self._turns.get(key)
        if st is None or st.turn != t:
            st = _SeatTurn(turn=t)
            self._turns[key] = st
        return st

    @staticmethod
    def _pending(game: Any, seat: int) -> bool:
        try:
            p = json.loads(game.player_json(int(seat)))
            return bool(p.get("pending"))
        except BaseException:
            return False

    def guarded_mask(self, mask: Any, *, game_id: Any, seat: Any, turn: Any,
                     game: Any) -> Tuple[np.ndarray, bool]:
        m = np.asarray(mask, dtype=bool)
        out = m.copy()
        self.stats.decisions += 1
        st = self._state(game_id, seat, turn)
        pending = self._pending(game, seat)
        blocked = 0
        if pending:
            if self.commit_play and (play_hand(st.last_action) is not None
                                     or hero_slot(st.last_action) is not None):
                if bool(out[CANCEL]):
                    out[CANCEL] = False
                    blocked += 1
                    self.stats.commit_blocked += 1
            if st.cancel_count >= self.cancel_cap and bool(out[CANCEL]):
                out[CANCEL] = False
                blocked += 1
                self.stats.cap_blocked += 1
        else:
            for a in range(A_PLAY, A_PLAY_HI):
                if bool(out[a]) and play_hand(a) in st.banned_play:
                    out[a] = False
                    blocked += 1
                    self.stats.banned_play_masks += 1
            for a in range(A_HERO_POWER, A_HERO_HI):
                if bool(out[a]) and hero_slot(a) in st.banned_power:
                    out[a] = False
                    blocked += 1
                    self.stats.banned_power_masks += 1
        if blocked and not bool(out.any()):
            # 没有可选目标等极端情况：不能让守卫制造非法动作。
            self.stats.fallback_masks += 1
            return m.copy(), False
        return out, False

    def note_action(self, action: Any, *, game_id: Any, seat: Any, turn: Any) -> None:
        a = int(action)
        st = self._state(game_id, seat, turn)
        st.prev_action = st.last_action
        st.last_action = a
        if a == CANCEL:
            self.stats.cancel_seen += 1
            st.cancel_count += 1
            h = play_hand(st.prev_action)
            if h is not None:
                st.banned_play.add(int(h))
            p = hero_slot(st.prev_action)
            if p is not None:
                st.banned_power.add(int(p))

    def to_meta(self) -> Dict[str, Any]:
        return {
            "pending_guard": True,
            "cancel_cap": int(self.cancel_cap),
            "commit_play": bool(self.commit_play),
            "pending_guard_decisions": int(self.stats.decisions),
            "pending_guard_cancel_seen": int(self.stats.cancel_seen),
            "pending_guard_commit_blocked": int(self.stats.commit_blocked),
            "pending_guard_cap_blocked": int(self.stats.cap_blocked),
            "pending_guard_banned_play_masks": int(self.stats.banned_play_masks),
            "pending_guard_banned_power_masks": int(self.stats.banned_power_masks),
            "pending_guard_fallback_masks": int(self.stats.fallback_masks),
        }


__all__ = ["PendingGuard", "PendingGuardStats", "CANCEL", "without_cancel",
           "play_hand", "hero_slot"]
