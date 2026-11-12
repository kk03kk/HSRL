#!/usr/bin/env python3
"""真人基线：从 Firestone 数据集提取"人类高手怎么打"的量化曲线。

为什么需要它：要判断模型"中后期差"，必须有一个**同口径的真人标尺**。没有标尺，
所有关于"后期不行"的说法都只是印象。

数据源：``data/real_library/firestone-20261112-1000/``

* ``library.jsonl`` —— 每回合的 self/opponent 状态快照（board_detail、tier、gold、
  strength、archetype、mmr、placement）。
* 原始 PMS ``data/firestone_replays/<refresh>/pms/<id>.json`` —— 每局的经济明细：
  ``tavernTimings``、``rerollsOverTurn``、``coinsWastedOverTurn``、
  ``minionsBoughtOverTurn``、``minionsSoldOverTurn``、``totalStatsOverTurn``、
  ``leaderboardPositionOverTurn``、``compositionsOverTurn``。

**必须先知道的数据偏差**（本脚本会把它打在最前面）：这个 feed 是 Firestone 的
``bgs-perfect-games``，实测 995 局里 816 局是第 1 名。所以：

* 曲线是「赢家怎么打」，**不是**随机天梯分布；
* 不能拿它估计流派胜率；
* 用它当对手面板，等于让模型对抗「人类赢家前期的板面」——这正是我们想要的强度标尺；
* 拿它当"平均真人水平"会系统性偏强。``--require-place 1`` 可以只保留冠军局对比。

用法（服务器上，标准库即可）::

    PYTHONPATH=engine/python:. python tools/analyze_human_baseline.py \
        --dataset data/real_library/firestone-20261112-1000 \
        --pms data/firestone_replays/refresh-20261112T031450Z/pms \
        --out artifacts/runs/human-baseline-<date>/baseline.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: 只看这些 stage 的板面（与 build_human_curve_bank.py 同口径）
BOARD_STAGES = ("combat_start",)

#: 升本里程碑
TIER_MILESTONES = (4, 5, 6)


def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _mean(values: Iterable[float]) -> float:
    v = list(values)
    return statistics.fmean(v) if v else float("nan")


def _r(x: float, nd: int = 2) -> Optional[float]:
    return None if x != x else round(x, nd)  # NaN -> None


def load_games(dataset: Path) -> Dict[str, Dict[str, Any]]:
    """按 game_id 汇总 library.jsonl 里的 self 视角。"""
    games: Dict[str, Dict[str, Any]] = {}
    with (dataset / "library.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gid = r["game_id"]
            g = games.get(gid)
            if g is None:
                g = games[gid] = {
                    "game_id": gid,
                    "placement": r.get("placement"),
                    "mmr": r.get("mmr"),
                    "archetype": r.get("archetype"),
                    "split": r.get("split"),
                    "turns": 0,
                    "tier_by_turn": {},
                    "gold_by_turn": {},
                    "strength_by_turn": {},
                    "board_n_by_turn": {},
                    "stages": set(),
                    "tribes": r.get("tribes"),
                }
            if r.get("perspective") != "self":
                continue
            t = int(r["turn"])
            g["turns"] = max(g["turns"], t)
            g["stages"].add(r.get("stage"))
            if isinstance(r.get("tier"), int):
                g["tier_by_turn"][t] = r["tier"]
            if isinstance(r.get("gold"), int):
                g["gold_by_turn"][t] = r["gold"]
            if isinstance(r.get("strength"), (int, float)) and r.get("stage") in BOARD_STAGES:
                g["strength_by_turn"][t] = float(r["strength"])
                g["board_n_by_turn"][t] = len(r.get("board_detail") or [])
    for g in games.values():
        g["stages"] = sorted(s for s in g["stages"] if s)
    return games


def load_pms_economy(pms_dir: Path, wanted: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """读每局 PMS 的经济明细。文件名是 reviewId，不是 game_id，所以按局数全读。

    只在 ``--with-pms`` 时调用：1000 个文件约 245 MB，读一次约 10-20 秒。
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not pms_dir.is_dir():
        return out
    for path in sorted(pms_dir.glob("*.json")):
        try:
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            d = json.loads(raw)
        except Exception:
            continue
        out[path.stem] = d
    return out


def curve(games: Dict[str, Dict[str, Any]], field: str) -> Dict[int, Dict[str, Any]]:
    by_turn: Dict[int, List[float]] = collections.defaultdict(list)
    for g in games.values():
        for t, v in g[field].items():
            by_turn[int(t)].append(float(v))
    table: Dict[int, Dict[str, Any]] = {}
    for t in sorted(by_turn):
        v = by_turn[t]
        if len(v) < 20:
            continue
        table[t] = {
            "n": len(v),
            "mean": _r(_mean(v)),
            "p25": _r(_pct(v, 0.25)),
            "p50": _r(_pct(v, 0.50)),
            "p75": _r(_pct(v, 0.75)),
            "p90": _r(_pct(v, 0.90)),
        }
    return table


def tier_distribution(games: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_turn: Dict[int, List[int]] = collections.defaultdict(list)
    for g in games.values():
        for t, v in g["tier_by_turn"].items():
            by_turn[int(t)].append(int(v))
    table: Dict[int, Dict[str, Any]] = {}
    for t in sorted(by_turn):
        v = by_turn[t]
        if len(v) < 20:
            continue
        hist = dict(sorted(collections.Counter(v).items()))
        table[t] = {
            "n": len(v),
            "mean": _r(_mean(v)),
            "mode": collections.Counter(v).most_common(1)[0][0],
            "tier6_share": _r(sum(1 for x in v if x >= 6) / len(v), 3),
            "tier5plus_share": _r(sum(1 for x in v if x >= 5) / len(v), 3),
            "hist": hist,
        }
    return table


def tier_up_timing(games: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """第一次到达各 tier 的回合数分布 —— 这就是「升本节奏」。"""
    out: Dict[str, Any] = {}
    for k in TIER_MILESTONES:
        turns = []
        for g in games.values():
            first = next((t for t in sorted(g["tier_by_turn"]) if g["tier_by_turn"][t] >= k), None)
            if first is not None:
                turns.append(first)
        if not turns:
            continue
        c = collections.Counter(turns)
        out[f"tier{k}"] = {
            "n": len(turns),
            "mean": _r(_mean(turns)),
            "mode": c.most_common(1)[0][0],
            "p50": _r(_pct(turns, 0.5), 1),
            "hist": dict(sorted(c.items())),
        }
    return out


def archetype_table(games: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for g in games.values():
        groups[g["archetype"] or "(unlabeled)"].append(g)
    rows = []
    for name, gs in groups.items():
        final_strength = [g["strength_by_turn"][max(g["strength_by_turn"])]
                          for g in gs if g["strength_by_turn"]]
        t6 = [next((t for t in sorted(g["tier_by_turn"]) if g["tier_by_turn"][t] >= 6), None)
              for g in gs]
        t6 = [x for x in t6 if x is not None]
        rows.append({
            "archetype": name,
            "games": len(gs),
            "mean_mmr": _r(_mean(g["mmr"] for g in gs if isinstance(g["mmr"], int)), 0),
            "mean_turns": _r(_mean(g["turns"] for g in gs)),
            "mean_t6_turn": _r(_mean(t6)),
            "mean_final_strength": _r(_mean(final_strength), 0),
            "p50_final_strength": _r(_pct(final_strength, 0.5), 0),
        })
    rows.sort(key=lambda r: -r["games"])
    return rows


def economy_summary(pms: Dict[str, Any]) -> Dict[str, Any]:
    """PMS 经济曲线：rerolls / 浪费金币 / 买卖次数 / 名次曲线。"""
    fields = {
        "rerolls": "rerollsOverTurn",
        "coins_wasted": "coinsWastedOverTurn",
        "bought": "minionsBoughtOverTurn",
        "sold": "minionsSoldOverTurn",
        "total_stats": "totalStatsOverTurn",
        "position": "leaderboardPositionOverTurn",
        "hp": "hpOverTurn",
    }
    acc: Dict[str, Dict[int, List[float]]] = {k: collections.defaultdict(list) for k in fields}
    n = 0
    for d in pms.values():
        if not isinstance(d, dict):
            continue
        n += 1
        for key, src in fields.items():
            seq = d.get(src)
            if isinstance(seq, list):
                for i, v in enumerate(seq):
                    if isinstance(v, (int, float)):
                        acc[key][i + 1].append(float(v))
    out: Dict[str, Any] = {"games": n}
    for key, by_turn in acc.items():
        out[key] = {t: {"n": len(v), "mean": _r(_mean(v))}
                    for t, v in sorted(by_turn.items()) if len(v) >= 20}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Firestone 真人基线曲线")
    ap.add_argument("--dataset", default="data/real_library/firestone-20261112-1000")
    ap.add_argument("--pms", default="", help="原始 PMS 目录；给了才算经济曲线")
    ap.add_argument("--out", default="", help="写出完整 JSON 的路径")
    ap.add_argument("--require-place", type=int, default=0,
                    help="只统计某一名的对局（1=只看冠军局）")
    ap.add_argument("--top", type=int, default=12, help="流派表显示前 N 行")
    args = ap.parse_args(argv)

    dataset = Path(args.dataset)
    if not (dataset / "library.jsonl").is_file():
        print(f"ERROR: missing {dataset/'library.jsonl'}", file=sys.stderr)
        return 2

    games = load_games(dataset)
    all_games = games
    if args.require_place:
        games = {k: g for k, g in games.items() if g["placement"] == args.require_place}

    places = collections.Counter(g["placement"] for g in all_games.values())
    print("=" * 78)
    print("HUMAN BASELINE  (Firestone bgs-perfect-games feed)")
    print("=" * 78)
    print(f"dataset           : {dataset}")
    print(f"games             : {len(all_games)}  (used here: {len(games)})")
    print(f"placement dist    : {dict(sorted((str(k), v) for k, v in places.items()))}")
    print("  !! this feed is WINNER-BIASED: do not read it as a ladder sample,")
    print("     and do not use it to estimate archetype win rates.")
    unlabelled = sum(1 for g in games.values() if not g["archetype"])
    print(f"unlabelled comps  : {unlabelled}/{len(games)}")
    stages = collections.Counter(s for g in games.values() for s in g["stages"])
    print(f"stages present    : {dict(stages)}")
    print()

    tier = tier_distribution(games)
    print("-- tavern tier by turn (self) --")
    print(f"{'turn':>4} {'n':>5} {'mean':>5} {'mode':>5} {'T6sh':>6} {'T5+sh':>6}")
    for t, row in tier.items():
        print(f"{t:>4} {row['n']:>5} {row['mean']:>5} {row['mode']:>5} "
              f"{row['tier6_share']:>6} {row['tier5plus_share']:>6}")
    print()

    timing = tier_up_timing(games)
    print("-- TIER-UP TIMING (first turn reaching tier) --")
    for k, row in timing.items():
        print(f"  {k}: n={row['n']} mean_turn={row['mean']} mode={row['mode']} "
              f"hist={row['hist']}")
    print()

    for field, label in (("strength_by_turn", "board strength (sum atk+hp)"),
                         ("gold_by_turn", "gold"), ("board_n_by_turn", "board size")):
        table = curve(games, field)
        print(f"-- {label} by turn --")
        print(f"{'turn':>4} {'n':>5} {'mean':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8}")
        for t, row in table.items():
            print(f"{t:>4} {row['n']:>5} {row['mean']:>8} {row['p25']:>8} "
                  f"{row['p50']:>8} {row['p75']:>8} {row['p90']:>8}")
        print()

    arch = archetype_table(games)
    print(f"-- archetype table (top {args.top} by games) --")
    print(f"{'archetype':<28} {'games':>6} {'mmr':>7} {'turns':>6} {'T6turn':>7} "
          f"{'final_str':>10} {'p50_str':>8}")
    for r in arch[: args.top]:
        print(f"{r['archetype']:<28} {r['games']:>6} {str(r['mean_mmr']):>7} "
              f"{str(r['mean_turns']):>6} {str(r['mean_t6_turn']):>7} "
              f"{str(r['mean_final_strength']):>10} {str(r['p50_final_strength']):>8}")
    print()

    economy = None
    if args.pms:
        pms = load_pms_economy(Path(args.pms), games)
        economy = economy_summary(pms)
        print(f"-- economy (from PMS, {economy['games']} games) --")
        for key in ("rerolls", "coins_wasted", "bought", "sold", "total_stats"):
            table = economy.get(key) or {}
            if not table:
                continue
            head = " ".join(f"T{t}={row['mean']:.1f}" for t, row in list(table.items())[:10])
            print(f"  {key:<12} {head}")
        print()

    if args.out:
        payload = {
            "schema": "hsai.human_baseline/v1",
            "dataset": str(dataset),
            "pms": args.pms or None,
            "require_place": args.require_place or None,
            "games_total": len(all_games),
            "games_used": len(games),
            "placement_dist": {str(k): v for k, v in places.items()},
            "warning": "bgs-perfect-games feed is winner-biased; not a ladder sample",
            "tier_by_turn": tier,
            "tier_up_timing": timing,
            "strength_by_turn": curve(games, "strength_by_turn"),
            "gold_by_turn": curve(games, "gold_by_turn"),
            "board_n_by_turn": curve(games, "board_n_by_turn"),
            "archetypes": arch,
            "economy": economy,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
