"""2 模型 + 6 bot 配对评测的统计口径（纯函数，不依赖引擎/torch，便于单测）。

口径（与 ``AGENTS.md`` 一致，见 ``docs/EVAL_DUEL.md``）：

1. **战团数值强度主参考是 P50 = 每张战团 ``Σatk + Σhp`` 的中位数**。
   不允许把「攻击中位数 + 生命中位数」冒充总攻血 P50；两者作为辅助量单独报告
   （``atk_p50`` / ``hp_p50``），且必须与 ``body_p50`` 并列出现。
2. **一个 (对局, 模型, 回合) 只取一个回合末帧**（模型选择 EndTurn 的时点），因此
   ``frames == games_at_turn`` 是构造事实而不是巧合；报告仍同时给出
   ``frames`` / ``games_at_turn`` / ``survivors``，防止把决策步帧当成回合帧。
3. 名次统计以**对局**为重采样单位（同一局内两个座位相关，不能当独立样本）。
   两模型同局对打时用**配对差分**（同 seed、同 6 个 bot 面板），差分以对局为单位
   bootstrap。
4. 计算过程不使用截断/加权平滑：样本不足时如实返回 ``None``，不给点估计。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

N_PLAYERS = 8

#: 名次 -> 效用 ``u(r)=1-2(r-1)/(N-1)``（与 ``hsbg.az.train.common.RANK_UTILITY`` 同式）。
def utility_of_rank(rank: int, num_players: int = N_PLAYERS) -> float:
    r = int(rank)
    if not 1 <= r <= int(num_players):
        return 0.0
    return float(1.0 - 2.0 * (r - 1) / (int(num_players) - 1))


def board_totals(board: Iterable[Any]) -> Dict[str, int]:
    """战团 -> ``{size, atk, hp, body}``；``body = Σatk + Σhp``（当前血量）。"""
    atk = hp = size = 0
    for minion in board or ():
        atk += int(minion["atk"])
        hp += int(minion["hp"])
        size += 1
    return {"size": int(size), "atk": int(atk), "hp": int(hp), "body": int(atk + hp)}


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """线性插值分位数（与 ``numpy.percentile`` 默认口径一致）；空样本 -> ``None``。"""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, float(q)))


def strength_summary(values: Sequence[float]) -> Dict[str, Any]:
    """一组战团总值（``Σatk + Σhp``）的分位数摘要；``p50`` 是主参考。"""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"frames": 0, "p25": None, "p50": None, "p75": None, "p90": None,
                "mean": None, "min": None, "max": None}
    return {
        "frames": int(arr.size),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _frames_by_turn(frames: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        out[int(frame["turn"])].append(frame)
    return out


def turn_table(frames: Iterable[Dict[str, Any]], *, turns: Optional[Iterable[int]] = None,
               tier_thresholds: Sequence[int] = (4, 5, 6)) -> Dict[str, Dict[str, Any]]:
    """逐回合表：到达该回合的**条件分布**（frames / games / survivors + P50 及辅助量）。

    ``frames``：该回合的回合末帧数；``games_at_turn``：其中来自不同对局的数量；
    ``survivors``：帧内 ``health > 0`` 的数量（到达并存活）。三者并列，禁止只报 P50。
    """
    grouped = _frames_by_turn(frames)
    wanted = sorted(grouped) if turns is None else sorted({int(t) for t in turns})
    table: Dict[str, Dict[str, Any]] = {}
    for turn in wanted:
        rows = grouped.get(turn, [])
        if not rows:
            table[str(turn)] = {
                "frames": 0, "games_at_turn": 0, "survivors": 0,
                "body_p25": None, "body_p50": None, "body_p75": None,
                "body_p90": None, "body_mean": None,
                "atk_p50": None, "hp_p50": None, "size_p50": None,
                "gold_p50": None, "tier_p50": None, "tier_mean": None,
                "board_size_p50": None,
            }
            for threshold in tier_thresholds:
                table[str(turn)][f"tier_ge{int(threshold)}"] = None
            continue
        bodies = [int(r["body"]) for r in rows]
        tiers = [int(r["tier"]) for r in rows]
        entry: Dict[str, Any] = {
            "frames": len(rows),
            "games_at_turn": len({int(r["game"]) for r in rows}),
            "survivors": sum(1 for r in rows if int(r.get("health") or 0) > 0),
            **{f"body_{k}": v for k, v in strength_summary(bodies).items() if k != "frames"},
            "board_size_p50": percentile([int(r["size"]) for r in rows], 50),
            "size_p50": percentile([int(r["size"]) for r in rows], 50),
            # 辅助量：单独的中位数，绝不与 body_p50 混用
            "atk_p50": percentile([int(r["atk"]) for r in rows], 50),
            "hp_p50": percentile([int(r["hp"]) for r in rows], 50),
            "gold_p50": percentile([int(r.get("gold") or 0) for r in rows], 50),
            "tier_p50": percentile(tiers, 50),
            "tier_mean": float(np.mean(tiers)),
        }
        for threshold in tier_thresholds:
            entry[f"tier_ge{int(threshold)}"] = float(np.mean(
                [t >= int(threshold) for t in tiers]))
        table[str(turn)] = entry
    return table


def rank_counts(ranks: Iterable[int], num_players: int = N_PLAYERS) -> Dict[str, int]:
    counts = Counter(int(r) for r in ranks)
    return {str(r): int(counts.get(r, 0)) for r in range(1, int(num_players) + 1)}


def rank_distribution(ranks: Sequence[int], num_players: int = N_PLAYERS) -> Dict[str, float]:
    counts = Counter(int(r) for r in ranks)
    n = len(ranks)
    if n == 0:
        return {str(r): None for r in range(1, int(num_players) + 1)}
    return {str(r): float(counts.get(r, 0) / n) for r in range(1, int(num_players) + 1)}


def _bootstrap_mean(values: np.ndarray, *, n_boot: int, seed: int) -> Optional[List[float]]:
    if values.size == 0:
        return None
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(int(seed))
    n = int(values.size)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def rank_summary(ranks: Sequence[int], *, num_players: int = N_PLAYERS,
                 bootstrap: int = 2000, seed: int = 20261107) -> Dict[str, Any]:
    """单模型名次摘要；重采样单位 = 对局。"""
    known = np.asarray([int(r) for r in ranks if 1 <= int(r) <= int(num_players)],
                       dtype=np.float64)
    out: Dict[str, Any] = {
        "games": len(ranks),
        "games_ranked": int(known.size),
        "mean_rank": None, "mean_rank_ci95": None,
        "top4_rate": None, "win_rate": None, "mean_utility": None,
        "median_rank": None, "rank_counts": rank_counts(ranks, num_players),
        "rank_distribution": rank_distribution(ranks, num_players),
    }
    if known.size == 0:
        return out
    out.update({
        "mean_rank": float(known.mean()),
        "mean_rank_ci95": _bootstrap_mean(known, n_boot=bootstrap, seed=seed),
        "median_rank": float(np.median(known)),
        "top4_rate": float((known <= 4).mean()),
        "win_rate": float((known == 1).mean()),
        "mean_utility": float(np.mean([utility_of_rank(r, num_players) for r in known])),
    })
    return out


def paired_delta(a_ranks: Sequence[float], b_ranks: Sequence[float], *,
                 bootstrap: int = 2000, seed: int = 20261107) -> Dict[str, Any]:
    """配对差分 ``Δ = rank_A - rank_B``（同一对局、同一 seed、同一 bot 面板）。

    ``Δ < 0`` 表示 A 名次更好（名次数字更小 = 更强）。差分以对局为重采样单位。
    """
    pairs = [(float(a), float(b)) for a, b in zip(a_ranks, b_ranks)
             if a is not None and b is not None and a > 0 and b > 0]
    out: Dict[str, Any] = {
        "pairs": len(pairs), "mean_delta": None, "median_delta": None,
        "mean_delta_ci95": None, "a_better": 0, "b_better": 0, "ties": 0,
        "a_better_rate": None, "b_better_rate": None,
        "interpretation": "Δ = rank_A - rank_B；负值 = A 名次更好。两模型同局对打，"
                          "bot 面板与 seed 完全相同，属配对样本。",
    }
    if not pairs:
        return out
    deltas = np.asarray([a - b for a, b in pairs], dtype=np.float64)
    out.update({
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "mean_delta_ci95": _bootstrap_mean(deltas, n_boot=bootstrap, seed=seed),
        "a_better": int((deltas < 0).sum()),
        "b_better": int((deltas > 0).sum()),
        "ties": int((deltas == 0).sum()),
        "a_better_rate": float((deltas < 0).mean()),
        "b_better_rate": float((deltas > 0).mean()),
    })
    return out


def reach_tier_stats(frames: Iterable[Dict[str, Any]], *, thresholds: Sequence[int] = (4, 5, 6),
                     max_turn: int = 20) -> Dict[str, Dict[str, Any]]:
    """达到酒馆等级 ``tier >= t`` 的首个回合（按对局统计，不看帧）。"""
    first: Dict[int, Dict[int, int]] = defaultdict(dict)
    for frame in frames:
        if int(frame["turn"]) > int(max_turn):
            continue
        game, turn, tier = int(frame["game"]), int(frame["turn"]), int(frame["tier"])
        prev = first[game].get(tier)
        if prev is None or turn < prev:
            first[game][tier] = turn
    out: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:
        turns: List[int] = []
        for game in sorted(first):
            best = min((turn for tier, turn in first[game].items()
                        if tier >= int(threshold)), default=None)
            if best is not None:
                turns.append(int(best))
        out[str(int(threshold))] = {
            "games_reached": len(turns),
            "share_reached": (float(len(turns) / len(first)) if first else None),
            "turn_p50": percentile(turns, 50),
            "turn_p25": percentile(turns, 25),
            "turn_p75": percentile(turns, 75),
            "turn_min": (min(turns) if turns else None),
        }
    return out


def final_state_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """逐局最终状态（最后一回合末）的 P50/分布：等级、战团、血量。"""
    rows = list(rows)
    tiers = [int(r["tier"]) for r in rows]
    bodies = [int(r["body"]) for r in rows]
    sizes = [int(r["size"]) for r in rows]
    healths = [int(r["health"]) for r in rows]
    out = {
        "games": len(rows),
        "tier_p50": percentile(tiers, 50),
        "tier_counts": {str(t): int(tiers.count(t)) for t in range(1, 8)},
        "tier_ge4_rate": float(np.mean([t >= 4 for t in tiers])) if tiers else None,
        "tier_ge5_rate": float(np.mean([t >= 5 for t in tiers])) if tiers else None,
        "tier_ge6_rate": float(np.mean([t >= 6 for t in tiers])) if tiers else None,
        "body_p50": percentile(bodies, 50),
        "body_p25": percentile(bodies, 25),
        "body_p75": percentile(bodies, 75),
        "body_p90": percentile(bodies, 90),
        "body_mean": float(np.mean(bodies)) if bodies else None,
        "size_p50": percentile(sizes, 50),
        "health_p50": percentile(healths, 50),
    }
    return out


def select_best_games(rows: Sequence[Dict[str, Any]], *, limit: int,
                      order: str = "rank") -> List[Dict[str, Any]]:
    """挑「表现最好」的局并保持稳定顺序。

    ``order="rank"``：名次升序（1 最好），同名次按最终战团总值降序，再按 seed 升序；
    ``order="body"``：最终战团总值降序，再按名次升序，再按 seed 升序。
    """
    rows = [r for r in rows if r is not None]
    if order == "rank":
        key = lambda r: (int(r.get("rank") or 99), -int(r.get("final_body") or 0),
                         int(r.get("seed") or 0))
    elif order == "body":
        key = lambda r: (-int(r.get("final_body") or 0), int(r.get("rank") or 99),
                         int(r.get("seed") or 0))
    else:
        raise ValueError(f"未知排序 {order!r}（只支持 rank|body）")
    return sorted(rows, key=key)[: int(limit)]


def merge_bot_reference(frames: Sequence[Dict[str, Any]], *, turn: int) -> Optional[float]:
    """bot 面板在 ``turn`` 的战团总值 P50（同帧口径的参照）。"""
    values = [int(f["body"]) for f in frames if int(f["turn"]) == int(turn)]
    return percentile(values, 50)


def as_float(value: Any) -> Optional[float]:
    """格式化用：非有限值一律 ``None``（避免 JSON 里出现 NaN/Inf）。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


__all__ = [
    "N_PLAYERS", "utility_of_rank", "board_totals", "percentile", "strength_summary",
    "turn_table", "rank_counts", "rank_distribution", "rank_summary", "paired_delta",
    "reach_tier_stats", "final_state_summary", "select_best_games",
    "merge_bot_reference", "as_float",
]
