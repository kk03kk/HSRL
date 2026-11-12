"""阵容覆盖奖励（comp-target reward）：把 **33 个流派的核心卡** 直接写进 reward。

动机
====
2026-11-12 的面板证据（`docs/analysis/TIER_TEMPO_20261112.md`）显示：把升本节奏修好以后
（T12 tier 追平真人 5.83 vs 5.78），**同一等级下我们的板面仍只有真人的 0.22 倍**。
`tools/analyze_composition_coverage.py` 给出同一结论的另一种读法：

| side | ≥1 张 S 核心卡 | ≥2 张核心卡 | ≥3 张核心卡 |
|---|---:|---:|---:|
| 真人吃鸡局 | 0.604 | 0.438 | 0.111 |
| base | 0.229 | 0.125 | 0.000 |

也就是说模型在打**低倍率的 B 级阵容**：`永恒骑士`（B）这种板面会封顶，而 S 级阵容是
乘法放大的。名次奖励管不了这件事——对 7 个引擎 bot 已经 1.18 名，赢不需要 S 阵容；
`board_target` 又只和"同样回合的专家战团"比总量，不区分"是不是同一个流派拼出来的"。

做法
====
读 ``config/strategy_cards.json``（33 个流派，每个带 ``tier`` 与 ``core_card_ids``），
在 learner 每个回合起点读一次战团的卡 id，对每个流派算

    得分_流派 = 等级权重(S=1.0/A=0.6/B=0.3) · min(1, 命中核心卡数 / max(2, 核心卡数))

取所有流派的**最大值**作为"本回合阵容完成度"，奖励 **本局历史最高完成度的增量**：

    delta = coef · (max_{τ≤t} 得分_τ − max_{τ<t} 得分_τ)

这样有三条性质，正好对上用户的要求：

1. **奖励"拼成了什么"，不指定"必须怎么拼"** —— 只认核心卡命中，动作由策略自己选；
2. **转型是被允许的**：用历史最高值而不是当回合值，晚转型拼成更强的阵容照样拿分，
   中途卖掉换方向不会被扣分；
3. **每局奖励有上界** ``coef · 1.0``，与名次奖励（终局 0..1）同量级，不会压过棋力信号。

安全性
======
- ``coef=0``（默认）时对象不启用：:meth:`observe` / :meth:`queue` / :meth:`flush`
  全是空操作，:meth:`stats` 返回 ``{}``，采集路径**逐位不变**；
- 卡 id 映射走 ``hsbg_engine.card_info``（``player_json`` 里 ``Minion.card`` 是 0-based
  CardId，高位是金色标记，所以用 ``(raw & 0x0fff) + 1``，与
  ``workflows/eval_two_models_six_bots.py`` 的 ``CardNames`` 同一口径）；
- 读状态失败一律按 0 处理并累加 ``errors``，**绝不打断采集**；
- 入场回合（``floor``）不打分：那一帧战团是真人库里注入的，白送满分会污染尺度。

实现约定与 ``tier_target`` / ``board_target`` 完全一致：每局一个 tracker，
``queue`` 到待应用列表，采集结束再 ``flush`` 一次性写进 ``record["reward"]``。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 流派等级权重（默认；可用 ``--comp-target-tier-w "S:1.0,A:0.5,B:0.2"`` 覆盖）。
DEFAULT_TIER_W: Dict[str, float] = {"S": 1.0, "A": 0.6, "B": 0.3}

#: 默认流派库（``tools/build_strategy_card_map.py`` 产出，33 个流派）。
DEFAULT_CARDS_PATH = "config/strategy_cards.json"

#: 默认从第 8 回合开始打分：更早的回合阵容还没成形，打分会鼓励"为了凑卡乱买"。
DEFAULT_MIN_TURN = 8

#: 反刷分门槛（默认 2 张核心卡才算"在拼这个流派"）：线性给分会退化成"只买一张"。
DEFAULT_MIN_HITS = 2

#: ``load_comps`` 的结果缓存：``(path, 权重表) -> comps``（每局都会构造 tracker）。
_COMPS_CACHE: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}


def parse_tier_weights(spec: Any) -> Dict[str, float]:
    """``"S:1.0,A:0.6,B:0.3"`` → ``{"S":1.0,...}``；空/None 用默认表。"""
    if spec is None:
        return dict(DEFAULT_TIER_W)
    if isinstance(spec, dict):
        out = {str(k).strip().upper(): float(v) for k, v in spec.items()}
        return out or dict(DEFAULT_TIER_W)
    text = str(spec).strip()
    if not text:
        return dict(DEFAULT_TIER_W)
    out = {}
    for raw in text.replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"非法等级权重 {item!r}（应为 S:1.0）")
        k, v = item.split(":", 1)
        out[k.strip().upper()] = float(v)
    if not out:
        raise ValueError(f"等级权重为空：{spec!r}")
    return out


def format_tier_weights(w: Dict[str, float]) -> str:
    return ",".join(f"{k}:{float(v):g}" for k, v in sorted(w.items()))


def _norm_card(card_id: Any) -> str:
    """卡 id 归一：去掉金色后缀 ``_G``（与 ``analyze_composition_coverage`` 同口径）。"""
    s = str(card_id or "")
    return s[:-2] if s.endswith("_G") else s


def load_comps(path: str, tier_w: Optional[Dict[str, float]] = None,
               min_hits: int = 2) -> List[Dict[str, Any]]:
    """读流派库 → ``[{name, tier, weight, cards, denom, need}, ...]``（跳过无核心卡的流派）。

    结果按 ``(path, 权重表, min_hits)`` 缓存在模块级：每局都会 ``CompTargetTracker(cfg)``，
    不缓存就是每轮 512 次 JSON 读（网络盘上不是零成本）。返回的是**共享对象**，
    调用方只读不改（:meth:`CompTargetTracker.score_board` 只做集合交）。
    """
    weights = dict(tier_w or DEFAULT_TIER_W)
    key = (str(path), format_tier_weights(weights), int(min_hits))
    cached = _COMPS_CACHE.get(key)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    comps: List[Dict[str, Any]] = []
    for c in data.get("comps") or []:
        cards = tuple(sorted({_norm_card(x) for x in (c.get("core_card_ids") or [])
                              if str(x or "").strip()}))
        if not cards:
            continue
        tier = str(c.get("tier") or "").strip().upper()
        comps.append({
            "name": str(c.get("name") or ""),
            "tier": tier,
            "weight": float(weights.get(tier, 0.0)),
            # 只有 1 张核心卡的流派不该靠"买到那一张"拿满分，所以分母至少 2。
            "denom": float(max(2, len(cards))),
            # 反刷分门槛：命中数不足就不给分（核心卡本来就少的流派按自己张数算）。
            "need": float(min(int(max(1, min_hits)), len(cards))),
            "cards": frozenset(cards),
        })
    if not comps:
        raise ValueError(f"{path} 里没有可用的核心卡（core_card_ids 全空）")
    _COMPS_CACHE[key] = comps
    return comps


class CompTargetTracker:
    """一局一个：按回合读 learner 战团卡 id，奖励"历史最高阵容完成度"的增量。"""

    def __init__(self, cfg: Any = None, *, coef: Optional[float] = None,
                 min_turn: Optional[int] = None, cards_path: Optional[str] = None,
                 tier_w: Any = None, cap: Optional[float] = None,
                 min_hits: Optional[int] = None, lookup: Any = None) -> None:
        self.coef = float(_cfg(cfg, "comp_target_coef", coef, 0.0) or 0.0)
        self.enabled = self.coef != 0.0
        self.min_turn = int(_cfg(cfg, "comp_target_min_turn", min_turn,
                                DEFAULT_MIN_TURN) or DEFAULT_MIN_TURN)
        self.cap = float(_cfg(cfg, "comp_target_cap", cap, 1.0) or 1.0)
        self.cards_path = str(_cfg(cfg, "comp_target_cards", cards_path,
                                  DEFAULT_CARDS_PATH) or DEFAULT_CARDS_PATH)
        self.tier_w = parse_tier_weights(
            _cfg(cfg, "comp_target_tier_w", tier_w, None))
        #: 反刷分门槛：命中核心卡数不足就不给分（H1 实测线性部分给分会被"只买一张"刷）。
        self.min_hits = int(_cfg(cfg, "comp_target_min_hits", min_hits,
                                 DEFAULT_MIN_HITS) or DEFAULT_MIN_HITS)
        self.comps: List[Dict[str, Any]] = (
            load_comps(self.cards_path, self.tier_w, self.min_hits)
            if self.enabled else [])
        self._lookup = lookup
        self._lookup_failed = False
        self._cache: Dict[int, str] = {}

        self.evaluated = set()
        self.pending: List[Tuple[int, int, float]] = []
        # 入场局用它把打分起点推到入场回合**之后**；默认 = min_turn-1（即不额外限制）
        self.floor = int(self.min_turn) - 1
        self.best = 0.0
        self.best_name: Optional[str] = None
        self.best_tier: Optional[str] = None

        self.turns = 0
        self.errors = 0
        self.best_sum = 0.0
        self.s_turns = 0
        self.reward_sum = 0.0
        self.applied = 0
        self.dropped = 0

    # ── 配置 ────────────────────────────────────────────────────────────
    def set_floor(self, turn: Any) -> None:
        """把打分起点抬到 ``turn``（入场局用：注入那一帧不算自己拼的）。"""
        try:
            t = int(turn)
        except (TypeError, ValueError):
            return
        self.floor = max(int(self.min_turn), t)

    # ── 卡 id 映射 ──────────────────────────────────────────────────────
    def card_id(self, raw: Any) -> str:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return ""
        if idx in self._cache:
            return self._cache[idx]
        if self._lookup is None and not self._lookup_failed:
            try:
                from .archetype_bias import engine_card_id_lookup
                self._lookup = engine_card_id_lookup()
            except Exception:  # noqa: BLE001 - 引擎/模块缺失时退化为"没有卡 id"
                self._lookup = None
            if self._lookup is None:
                self._lookup_failed = True
        cid = ""
        if self._lookup is not None:
            try:
                # player_json 的 Minion.card 是 0-based CardId（高位是金色标记）
                cid = _norm_card(self._lookup((idx & 0x0FFF) + 1))
            except Exception:  # noqa: BLE001 - 卡表外索引
                cid = ""
        self._cache[idx] = cid
        return cid

    # ── 打分 ────────────────────────────────────────────────────────────
    def score_board(self, have: Iterable[str]) -> Tuple[float, Optional[str],
                                                        Optional[str], int]:
        """→ (最高分, 流派名, 流派等级, 该流派命中数)。

        ``min_hits``（默认 2）是**反刷分门槛**：2026-11-13 的 H1 实验显示，线性部分给分
        会让模型只买"任意一张核心卡"就把水位抬到 0.17，然后在面板上毫无改善
        （训练内 `c_best` 0.13→0.17，面板仍是 0.14）。所以命中数不足 ``min_hits`` 时
        该流派直接记 0；核心卡本来就少于门槛的流派（如只有 1 张核心卡的 S 级）
        按它自己的核心卡数处理。
        """
        have_set = {c for c in have if c}
        best = 0.0
        best_name = None
        best_tier = None
        best_hits = 0
        for c in self.comps:
            if c["weight"] <= 0.0:
                continue
            hits = len(c["cards"] & have_set)
            if hits < c["need"]:
                continue
            s = c["weight"] * min(1.0, hits / c["denom"])
            if s > best + 1e-12:
                best, best_name, best_tier, best_hits = s, c["name"], c["tier"], hits
        return best, best_name, best_tier, best_hits

    def observe(self, game: Any, seat: int, turn: Optional[int] = None) -> float:
        """读一次 ``player_json``，返回该回合应加的 reward 增量；同回合只算一次。"""
        if not self.enabled:
            return 0.0
        try:
            t = int(game.turn if turn is None else turn)
        except (TypeError, ValueError):
            return 0.0
        if t in self.evaluated:
            return 0.0
        self.evaluated.add(t)
        if t < int(self.min_turn):
            return 0.0
        try:
            pj = json.loads(game.player_json(int(seat)))
            board = pj.get("board") or []
        except BaseException:  # noqa: BLE001 - 奖励绝不能打断采集
            self.errors += 1
            return 0.0
        have = []
        for m in board:
            try:
                cid = self.card_id(m.get("card"))
            except Exception:  # noqa: BLE001
                cid = ""
            if cid:
                have.append(cid)
        best, name, tier, _hits = self.score_board(have)
        # 入场那一帧（floor）只用来**设置水位**，不发奖励：战团是真人库里注入的。
        if t <= int(self.floor):
            self.best = max(self.best, min(self.cap, best))
            return 0.0
        self.turns += 1
        self.best_sum += best
        if tier == "S" and best > 0.0:
            self.s_turns += 1
        prev = self.best
        delta = 0.0
        if best > prev:
            self.best = min(self.cap, best)
            self.best_name = name
            self.best_tier = tier
            delta = float(self.coef) * (self.best - prev)
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
            "comp_target_coef": float(self.coef),
            "comp_target_min_turn": int(self.min_turn),
            "comp_target_tier_w": format_tier_weights(self.tier_w),
            "comp_target_cards": self.cards_path,
            "comp_target_turns": int(self.turns),
            "comp_target_best_mean": (round(self.best_sum / self.turns, 4)
                                      if self.turns else None),
            "comp_target_s_turns": int(self.s_turns),
            "comp_target_reward_sum": round(self.reward_sum, 4),
            "comp_target_applied": int(self.applied),
            "comp_target_dropped": int(self.dropped),
            "comp_target_errors": int(self.errors),
        }


def _cfg(cfg: Any, name: str, explicit: Any, default: Any) -> Any:
    if explicit is not None:
        return explicit
    if cfg is not None:
        return getattr(cfg, name, default)
    return default


def aggregate_comp_target(metas: Iterable[Any]) -> Dict[str, Any]:
    """多 worker 汇总；没有任何 ``comp_target_coef`` 时返回 ``{}``（旧路径零键增量）。"""
    rows = [m for m in metas if isinstance(m, dict) and "comp_target_coef" in m]
    if not rows:
        return {}
    first = rows[0]
    turns = sum(int(m.get("comp_target_turns", 0) or 0) for m in rows)
    best_sum = sum(float(m.get("comp_target_best_mean") or 0.0)
                   * int(m.get("comp_target_turns", 0) or 0) for m in rows)
    return {
        "comp_target_coef": float(first.get("comp_target_coef", 0.0)),
        "comp_target_min_turn": int(first.get("comp_target_min_turn", DEFAULT_MIN_TURN)),
        "comp_target_tier_w": str(first.get("comp_target_tier_w", "")),
        "comp_target_cards": str(first.get("comp_target_cards", "")),
        "comp_target_turns": int(turns),
        "comp_target_best_mean": (round(best_sum / turns, 4) if turns else None),
        "comp_target_s_turns": int(sum(int(m.get("comp_target_s_turns", 0) or 0)
                                       for m in rows)),
        "comp_target_reward_sum": round(float(sum(
            float(m.get("comp_target_reward_sum", 0.0) or 0.0) for m in rows)), 4),
        "comp_target_applied": int(sum(int(m.get("comp_target_applied", 0) or 0)
                                       for m in rows)),
        "comp_target_dropped": int(sum(int(m.get("comp_target_dropped", 0) or 0)
                                       for m in rows)),
        "comp_target_errors": int(sum(int(m.get("comp_target_errors", 0) or 0)
                                      for m in rows)),
    }


__all__ = [
    "DEFAULT_TIER_W", "DEFAULT_CARDS_PATH", "DEFAULT_MIN_TURN",
    "parse_tier_weights", "format_tier_weights", "load_comps",
    "CompTargetTracker", "aggregate_comp_target",
]
