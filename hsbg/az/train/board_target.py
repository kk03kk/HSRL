"""专家相对的过程奖励：把「这一回合该长到多大/该有哪些牌」变成逐回合信号。

背景（`docs/analysis/ENTRY_S1_20261112.md`）：中后期入场课程把一张**真的**真人 T10–T11
战团交到模型手上，但模型几回合内就把 Σ(atk+hp) 从 ≈2,000 做到 ≈1,500，而真人同回合是
5,600（T12）→ 27,000（T14）。只给终局名次时，这个差距要经过 4–6 个回合、8 人混战与
战斗方差之后才变成一个名次差，信噪比很低。

本模块补一条**专家相对**的密集信号：库里的每一帧都带有**同一个真人这一局每一回合**的
板面（`tools/build_human_chain.py` 产出的 `chain_turn` / `chain_strength` /
`chain_cards`）。于是每个回合都能问两件事：

1. **强度**：``L(Σ_learner) − L(Σ_expert,该回合)``，``L(Σ)=log1p(Σ)/log1p(30000)``；
   负值 = 比这位真人同回合更弱（该涨没涨）。
2. **阵容**：``|learner 卡集合 ∩ 专家该回合卡集合| / |专家卡集合|``（召回率），
   0 = 一张核心牌都没留住。

两者线性组合后乘 ``board_target_coef``，挂到**该回合第一条 learner 决策**上（与
:mod:`hsbg.az.train.tier_target` 完全相同的 `queue`/`flush` 机制，避免与势函数塑形
互相覆盖写）。

**边界（必须记住）**：

- 专家序列来自**一个**真人对局，它不是"最优解"，只是"一个强者做到过什么"；
  因此这是**引导**而不是监督标签，不能单独当目标，只作为名次奖励之外的附加项。
- 召回率只看"有没有留住"，不看"有没有多买垃圾"；所以必须同时开强度项，否则
  模型可以靠囤牌刷召回率。
- 一回合只读一次 ``player_json``（与 tier_target 同口径），成本约 6–50 µs/回合/局。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 与训练奖励同一口径的强度归一化（见 rank utility）。
_L_DENOM = math.log1p(30000.0)

#: ``|L(Σ_learner) − L(Σ_expert)|`` 的默认裁剪，避免极端局把奖励拉爆。
DEFAULT_CAP = 1.0


def log_strength(total: Any) -> float:
    """``L(Σ) = log1p(Σ)/log1p(30000)``；非法/负值按 0。"""
    try:
        s = float(total)
    except (TypeError, ValueError):
        return 0.0
    if not (s > 0.0) or not math.isfinite(s):
        return 0.0
    return math.log1p(s) / _L_DENOM


def _cfg_value(cfg: Any, name: str, explicit: Any, default: Any) -> Any:
    if explicit is not None:
        return explicit
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class BoardTargetTracker:
    """逐回合「专家相对」奖励；``coef=0``（默认）时 ``enabled=False``，零开销。"""

    def __init__(self, cfg: Any = None, *, coef: Optional[float] = None,
                 w_card: Optional[float] = None, w_str: Optional[float] = None,
                 cap: Optional[float] = None) -> None:
        self.coef = float(_cfg_value(cfg, "board_target_coef", coef, 0.0) or 0.0)
        self.w_card = float(_cfg_value(cfg, "board_target_w_card", w_card, 1.0))
        self.w_str = float(_cfg_value(cfg, "board_target_w_str", w_str, 0.5))
        self.cap = float(_cfg_value(cfg, "board_target_cap", cap, DEFAULT_CAP))
        self.enabled = bool(self.coef != 0.0)
        #: turn -> 专家该回合 Σ(atk+hp)
        self.strength: Dict[int, int] = {}
        #: turn -> 专家该回合的卡 id 集合
        self.cards: Dict[int, frozenset] = {}
        self.source_turn = 0
        self.evaluated: set = set()
        self.pending: List[Tuple[int, int, float]] = []
        self.dropped = 0
        self.applied = 0
        self.n_turns = 0
        self.str_delta_sum = 0.0
        self.card_recall_sum = 0.0
        self.reward_sum = 0.0

    # ── 目标装载 ────────────────────────────────────────────────────────
    def set_chain(self, turns: Any, strengths: Any, cards: Any,
                  source_turn: Any = 0) -> bool:
        """从库帧的 ``chain_*`` 装目标（缺字段/长度不齐 -> 不装，返回 False）。"""
        if not self.enabled:
            return False
        try:
            ts = [int(t) for t in (turns or [])]
            ss = [int(s) for s in (strengths or [])]
        except (TypeError, ValueError):
            return False
        if not ts or len(ts) != len(ss):
            return False
        cs = cards if isinstance(cards, (list, tuple)) else [None] * len(ts)
        self.strength = {t: s for t, s in zip(ts, ss)}
        self.cards = {}
        for t, lst in zip(ts, cs):
            if isinstance(lst, (list, tuple)):
                try:
                    self.cards[t] = frozenset(int(c) for c in lst)
                except (TypeError, ValueError):
                    pass
        self.source_turn = int(source_turn or 0)
        return bool(self.strength)

    def _target_turn(self, turn: int) -> Optional[int]:
        """``turn`` 上没有观测时退到**最近的已观测回合**（与战团库同一套夹取口径）。"""
        if not self.strength:
            return None
        t = int(turn)
        if t in self.strength:
            return t
        keys = self.strength.keys()
        lo = min(keys)
        hi = max(keys)
        if t < lo:
            return lo
        if t > hi:
            return hi
        return min(keys, key=lambda x: (abs(x - t), x))

    # ── 观测 ────────────────────────────────────────────────────────────
    def observe(self, game: Any, seat: int,
                turn: Optional[int] = None) -> float:
        """读一次 ``player_json`` 并给出该回合的增量；同回合只算一次。"""
        if not self.enabled or not self.strength:
            return 0.0
        try:
            t = int(game.turn if turn is None else turn)
        except (TypeError, ValueError):
            return 0.0
        if t in self.evaluated:
            return 0.0
        self.evaluated.add(t)
        # 入场那一个回合**不打分**：此时 learner 的战团就是库里的那一帧，
        # 阵容召回合天然是 1.0、强度差天然是 0 —— 白送一份奖励只会把尺度搞乱。
        if t <= int(self.source_turn):
            return 0.0
        tt = self._target_turn(t)
        if tt is None:
            return 0.0
        try:
            import json
            pj = json.loads(game.player_json(int(seat)))
            board = pj.get("board") or []
        except BaseException:  # noqa: BLE001 - 奖励绝不能打断采集
            return 0.0
        total = 0
        mine = set()
        for m in board:
            try:
                total += int(m.get("atk") or 0) + int(m.get("hp") or 0)
                mine.add(int(m.get("card") or 0))
            except (TypeError, ValueError):
                continue
        d_str = log_strength(total) - log_strength(self.strength.get(tt, 0))
        d_str = max(-self.cap, min(self.cap, d_str))
        want = self.cards.get(tt) or frozenset()
        recall = (len(mine & want) / float(len(want))) if want else 0.0
        delta = float(self.coef) * (self.w_str * d_str + self.w_card * recall)
        self.n_turns += 1
        self.str_delta_sum += d_str
        self.card_recall_sum += recall
        self.reward_sum += delta
        return delta

    # ── 待应用增量 ──────────────────────────────────────────────────────
    def queue(self, index: int, game_id: int, delta: float) -> bool:
        if not self.enabled or float(delta) == 0.0:
            return False
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return False
        if idx < 0:
            return False
        self.pending.append((idx, int(game_id), float(delta)))
        return True

    def flush(self, records: Sequence[Dict[str, Any]]) -> int:
        if not self.enabled or not self.pending:
            return 0
        applied = 0
        for idx, gid, delta in self.pending:
            if not (0 <= idx < len(records)):
                self.dropped += 1
                continue
            rec = records[idx]
            if int(rec.get("game_id", -1)) != int(gid):
                self.dropped += 1
                continue
            rec["reward"] = float(rec.get("reward", 0.0)) + float(delta)
            applied += 1
        self.applied += applied
        self.pending = []
        return applied

    def stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "board_target_coef": float(self.coef),
            "board_target_w_str": float(self.w_str),
            "board_target_w_card": float(self.w_card),
            "board_target_cap": float(self.cap),
            "board_target_turns": int(self.n_turns),
            "board_target_applied": int(self.applied),
            "board_target_dropped": int(self.dropped),
            "board_target_str_delta_mean": (
                round(self.str_delta_sum / self.n_turns, 4) if self.n_turns else None),
            "board_target_card_recall_mean": (
                round(self.card_recall_sum / self.n_turns, 4) if self.n_turns else None),
            "board_target_reward_sum": round(self.reward_sum, 3),
        }


#: 跨 worker 聚合口径（与 ``aggregate_tier_target`` 一致：均值/求和，不全丢）。
_SUM_KEYS = ("board_target_applied", "board_target_dropped", "board_target_turns",
             "board_target_reward_sum")
_MEAN_KEYS = ("board_target_str_delta_mean", "board_target_card_recall_mean")
_FIRST_KEYS = ("board_target_coef", "board_target_w_str", "board_target_w_card",
               "board_target_cap")


def aggregate_board_target(metas: Iterable[Any]) -> Dict[str, Any]:
    """把多局的 ``stats()`` 聚合成一条 history 记录（禁用时返回 ``{}``）。"""
    sums: Dict[str, float] = {}
    means: Dict[str, List[float]] = {}
    first: Dict[str, Any] = {}
    for m in metas:
        if not isinstance(m, dict):
            continue
        for k in _SUM_KEYS:
            v = m.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] = sums.get(k, 0.0) + float(v)
        for k in _MEAN_KEYS:
            v = m.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                means.setdefault(k, []).append(float(v))
        for k in _FIRST_KEYS:
            if k in m and k not in first and m[k] is not None:
                first[k] = m[k]
    out: Dict[str, Any] = {}
    out.update(first)
    for k, v in sums.items():
        out[k] = int(v) if float(v).is_integer() else round(v, 3)
    for k, vs in means.items():
        out[k] = round(sum(vs) / len(vs), 4) if vs else None
    return out


__all__ = ["BoardTargetTracker", "aggregate_board_target", "log_strength",
           "DEFAULT_CAP"]
