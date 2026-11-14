"""留存随从滚雪球奖励（retained-growth shaping）：直接奖励"把留下来的随从养大"。

动机（`docs/DECISIONS.md` D-052）
================================
名次奖励在我们所处的板面区间里**对棋力不敏感**：控制住"死在第几回合"之后，
战团强度对名次的解释力 R²=0.000–0.004（n=286–639），而死亡回合单独解释 65%。
原因是战斗结果是板面比值的**阶跃函数**——我们 T12 只有真人 p50 的 0.29×，
1434 和 2000 面对 4983 都是"整队被清空"，所以板面的边际作用为零。

可是真人赢家的增长有非常明确的机制：`RUNS_20261113.md` §5.62 —— 中后期增长
**≈90% 来自"养留下来的随从"**（T10→11 每只留存随从 **+690**），换牌只占 ~10%；
而我们只有 **+40**（差 17×）。这个量**不依赖赢**，所以即使在不可赢的场里也可测、可学。

因此本模块把"每只留存随从每回合涨了多少"直接写进 reward。三条不可作弊性：

1. **死了就拿不到**（没有回合 ⇒ 没有增重）⇒ 不能靠早死刷分；
2. **卖掉的随从不算留存**（当回合不在名单里）⇒ 不能靠倒腾刷分；
3. **按每只留存随从的增重算，不是按整块板**⇒ 买一堆新随从不等于长大，
   必须**留着并养大**才算。

与既有口径一致
==============
- 强度 = `atk + hp`，卡 id 去掉金色后缀 `_G`（与 `tools/report_stat_growth.py`、
  `workflows/eval_human_gap.retained_gain_by_turn` 同口径）；
- 每回合得分 `min(1, 每只留存随从增重 / 690)`，690 = 真人 T10→11 的实测值；
- 每局总奖励**有上界** `cap`（默认 = coef）⇒ 与名次项（0..1 量级）同阶，
  不会像 `Φ=log Σ` 那样变成方差机器（§5.74）。

关闭时逐位一致
==============
`coef=0`（默认）⇒ `enabled=False`，`observe` 恒返回 0.0、`queue` 直接拒绝、
`flush` 不碰 records、`stats()` 返回 `{}` —— 采集路径**逐位不变**。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 每只留存随从每回合增重的归一化基准（真人赢家 T10→11 实测 +690）
DEFAULT_NORMALIZER = 690.0
#: 从第几回合开始计分（更早的回合随从更替快，留存增重没有代表性）
DEFAULT_MIN_TURN = 9
#: 每局总奖励上界（相对名次项 0..1 同阶）
DEFAULT_CAP = 0.5

#: player_json 里 minion 的字段名在不同版本可能不同，按顺序取第一个存在的
# 引擎 `minion_json` 的实际字段是 ``"id"``（卡 id 字符串）+ ``"cid"``（数字下标）+ ``"eid"``（实体 id）。
# **2026-11-14 踩过**：只认 ``"card"`` 会让取不到卡 id ⇒ 强度恒 0 ⇒ 奖励**永远付不出去**
# 而日志里只看到 reward_sum=0（与 §5.71 的"配了但量级为 0"同一类错误）。
_CARD_KEYS = ("id", "card", "card_id", "cardId", "cid")
_ATK_KEYS = ("atk", "attack", "Attack")
_HP_KEYS = ("hp", "health", "Health")


def _cfg(cfg: Any, name: str, explicit: Any, default: Any) -> Any:
    if explicit is not None:
        return explicit
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _norm_card(raw: Any) -> str:
    s = str(raw or "")
    return s[:-2] if s.endswith("_G") else s


def _num(d: Dict[str, Any], keys: Sequence[str]) -> float:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def board_strength_by_card(board: Any) -> Dict[str, float]:
    """把 ``player_json`` 的 board 折成 ``{card_id: 该卡所有副本的 (atk+hp) 之和}``。

    按卡归并（而不是按实体 id）是有意的：与 ``report_stat_growth.py`` /
    ``retained_gain_by_turn`` 的口径一致，这样训练里的数字与面板上的数字可直接比较。
    两张同名随从会被视为"这一摞"，增重按摞算——仍然满足"必须留着并养大"。
    """
    out: Dict[str, float] = {}
    if not isinstance(board, list):
        return out
    for m in board:
        if not isinstance(m, dict):
            continue
        cid = ""
        for k in _CARD_KEYS:
            if m.get(k) is not None:
                cid = _norm_card(m.get(k))
                break
        if not cid:
            continue
        out[cid] = out.get(cid, 0.0) + _num(m, _ATK_KEYS) + _num(m, _HP_KEYS)
    return out


def kept_gain(prev: Dict[str, float], cur: Dict[str, float]) -> Tuple[float, int]:
    """→ (每只留存随从的平均增重, 留存只数)。只在两侧都出现的卡算"留存"。"""
    if not prev or not cur:
        return 0.0, 0
    kept = [c for c in prev if c in cur]
    if not kept:
        return 0.0, 0
    total = 0.0
    for c in kept:
        d = cur[c] - prev[c]
        if d > 0.0:          # 只奖励"养大"，掉血/被削不扣分（避免与战斗波动纠缠）
            total += d
    return total / float(len(kept)), len(kept)


class RetainedGrowthTracker:
    """一局一个：每回合读 learner 战团，按"留存随从的增重"发有界奖励。"""

    def __init__(self, cfg: Any = None, *, coef: Optional[float] = None,
                 min_turn: Optional[int] = None,
                 normalizer: Optional[float] = None,
                 cap: Optional[float] = None) -> None:
        self.coef = float(_cfg(cfg, "retained_growth_coef", coef, 0.0) or 0.0)
        self.enabled = self.coef != 0.0
        self.min_turn = int(_cfg(cfg, "retained_growth_min_turn", min_turn,
                                 DEFAULT_MIN_TURN) or DEFAULT_MIN_TURN)
        self.normalizer = float(_cfg(cfg, "retained_growth_normalizer", normalizer,
                                     DEFAULT_NORMALIZER) or DEFAULT_NORMALIZER)
        if self.normalizer <= 0.0:
            raise ValueError(f"normalizer 必须 > 0: {self.normalizer}")
        self.cap = float(_cfg(cfg, "retained_growth_cap", cap, DEFAULT_CAP) or 0.0)
        if self.cap < 0.0:
            raise ValueError(f"cap 必须 >= 0: {self.cap}")

        self.evaluated = set()
        self.pending: List[Tuple[int, int, float]] = []
        self.floor = int(self.min_turn) - 1
        self._prev: Optional[Dict[str, float]] = None

        self.turns = 0
        self.kept_sum = 0.0
        self.gain_sum = 0.0
        self.score_sum = 0.0
        self.reward_sum = 0.0
        self.applied = 0
        self.dropped = 0
        self.errors = 0

    # ── 配置 ────────────────────────────────────────────────────────────
    def set_floor(self, turn: Any) -> None:
        """入场局：把计分起点抬到注入回合之后（注入的战团不是自己养的）。"""
        try:
            t = int(turn)
        except (TypeError, ValueError):
            return
        self.floor = max(int(self.min_turn), t)
        self._prev = None

    # ── 主观测 ──────────────────────────────────────────────────────────
    def observe(self, game: Any, seat: int = 0,
                turn: Optional[int] = None) -> float:
        """读一次 ``player_json``，返回**该回合应加的 reward 增量**；同回合只算一次。"""
        if not self.enabled:
            return 0.0
        try:
            t = int(game.turn if turn is None else turn)
        except (TypeError, ValueError):
            return 0.0
        if t in self.evaluated:
            return 0.0
        self.evaluated.add(t)
        try:
            pj = json.loads(game.player_json(int(seat)))
            cur = board_strength_by_card(pj.get("board") or [])
        except BaseException:  # noqa: BLE001 - 奖励绝不能打断采集
            self.errors += 1
            return 0.0
        prev, self._prev = self._prev, cur
        # 入场那一帧（以及它之前）只用来设置基线，不发奖励
        if t <= int(self.floor) or prev is None:
            return 0.0
        if t < int(self.min_turn):
            return 0.0
        gain, n_kept = kept_gain(prev, cur)
        self.turns += 1
        self.kept_sum += float(n_kept)
        self.gain_sum += float(gain)
        score = min(1.0, max(0.0, gain) / float(self.normalizer))
        self.score_sum += score
        room = float(self.cap) - float(self.reward_sum)
        if room <= 0.0:
            return 0.0
        delta = min(float(self.coef) * score, room)
        if delta <= 0.0:
            return 0.0
        self.reward_sum += delta
        return float(delta)

    # ── 待应用增量（与 comp_target 同构）──────────────────────────────────
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

    # ── 统计 ────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "retained_growth_coef": float(self.coef),
            "retained_growth_min_turn": int(self.min_turn),
            "retained_growth_normalizer": float(self.normalizer),
            "retained_growth_cap": float(self.cap),
            "retained_growth_turns": int(self.turns),
            "retained_growth_kept_mean": (round(self.kept_sum / self.turns, 3)
                                          if self.turns else None),
            "retained_growth_gain_mean": (round(self.gain_sum / self.turns, 1)
                                          if self.turns else None),
            "retained_growth_score_mean": (round(self.score_sum / self.turns, 4)
                                           if self.turns else None),
            "retained_growth_reward_sum": round(self.reward_sum, 4),
            "retained_growth_applied": int(self.applied),
            "retained_growth_dropped": int(self.dropped),
            "retained_growth_errors": int(self.errors),
        }


def aggregate_retained_growth(games_meta: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """把逐局 ``retained_growth_*`` 行汇总成采集级统计（默认关闭 -> ``{}``）。

    与 ``aggregate_comp_target`` 同构：吃**逐局 meta**，键名带 ``retained_growth_`` 前缀。
    """
    rows = [m for m in games_meta
            if isinstance(m, dict) and "retained_growth_coef" in m]
    if not rows:
        return {}
    keys = ("retained_growth_turns", "retained_growth_applied",
            "retained_growth_dropped", "retained_growth_errors")
    out: Dict[str, Any] = {k: int(sum(int(r.get(k) or 0) for r in rows)) for k in keys}
    for k in ("retained_growth_reward_sum",):
        out[k] = round(sum(float(r.get(k) or 0.0) for r in rows), 4)
    first = rows[0]
    for k in ("retained_growth_coef", "retained_growth_min_turn",
              "retained_growth_normalizer", "retained_growth_cap"):
        out[k] = first.get(k)
    out["retained_growth_games"] = len(rows)
    # 每回合平均增重（跨局加权）：直接与真人 +690 对照
    turns = out["retained_growth_turns"]
    if turns:
        gains = [float(r.get("retained_growth_gain_mean") or 0.0) for r in rows
                 if r.get("retained_growth_gain_mean") is not None]
        out["retained_growth_gain_mean"] = (round(sum(gains) / len(gains), 1)
                                            if gains else None)
    return out


__all__ = [
    "DEFAULT_CAP",
    "DEFAULT_MIN_TURN",
    "DEFAULT_NORMALIZER",
    "RetainedGrowthTracker",
    "aggregate_retained_growth",
    "board_strength_by_card",
    "kept_gain",
]
