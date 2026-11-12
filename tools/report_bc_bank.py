#!/usr/bin/env python3
"""盘点 ``data/firestone_bc/bc_samples.jsonl.gz``：够不够用来教"中后期选阵容"。

为什么需要它
------------
``bc-coef`` / ``dpo-coef`` / ``board_problems`` 全部吃这个样本池，但从没人看过它的
**质量分布**。整池 282,717 条听起来很多，可是：

- 绝大多数是第 1–4 回合的"买最左边那个"，对"中后期强势阵容"没有任何信息量；
- 池子的 MMR 下界决定它教的是"高手"还是"路人"。第一条记录就是 **6790**。

所以真正要回答的是三个数：**≥8k MMR 有多少条**、其中 **turn>=8 且 tier>=4 有多少条**、
以及这些晚局样本的动作类型构成。只有最后一类才可能教出 S 级阵容。

用法（只读 gzip+json，不碰 torch/引擎）::

    python tools/report_bc_bank.py
    python tools/report_bc_bank.py --path data/firestone_bc/bc_samples.jsonl.gz --late-turn 8
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "data/firestone_bc/bc_samples.jsonl.gz")

MMR_BUCKETS = ((0, 6000), (6000, 7000), (7000, 8000), (8000, 9000),
               (9000, 10000), (10000, 12000), (12000, 99999))


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def bucket(mmr: Optional[float]) -> str:
    if mmr is None:
        return "none"
    for lo, hi in MMR_BUCKETS:
        if lo <= mmr < hi:
            return f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
    return "none"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BC 样本池质量盘点")
    ap.add_argument("--path", default=DEFAULT)
    ap.add_argument("--late-turn", type=int, default=8)
    ap.add_argument("--late-tier", type=int, default=4)
    ap.add_argument("--mmr-min", type=int, default=8000)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.path):
        raise SystemExit(f"没有样本池：{args.path}")

    n = 0
    bad = 0
    mmr_hist: collections.Counter = collections.Counter()
    act_all: collections.Counter = collections.Counter()
    act_late: collections.Counter = collections.Counter()
    turn_hist: collections.Counter = collections.Counter()
    late = 0
    late_hi = 0
    hi = 0
    hi_turns: collections.Counter = collections.Counter()
    games_hi = set()
    games_late_hi = set()
    no_mmr = 0

    with _open(args.path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r: Dict[str, Any] = json.loads(line)
            except ValueError:
                bad += 1
                continue
            n += 1
            m = r.get("mmr")
            try:
                m = None if m is None else float(m)
            except (TypeError, ValueError):
                m = None
            if m is None:
                no_mmr += 1
            mmr_hist[bucket(m)] += 1
            at = str(r.get("action_type") or "?")
            act_all[at] += 1
            t = r.get("turn")
            try:
                t = int(t) if t is not None else None
            except (TypeError, ValueError):
                t = None
            if t is not None:
                turn_hist[min(t, 20)] += 1
            tier = r.get("tier")
            try:
                tier = int(tier) if tier is not None else None
            except (TypeError, ValueError):
                tier = None
            is_late = (t is not None and t >= args.late_turn
                       and tier is not None and tier >= args.late_tier)
            if is_late:
                late += 1
                act_late[at] += 1
            if m is not None and m >= args.mmr_min:
                hi += 1
                if t is not None:
                    hi_turns[min(t, 20)] += 1
                gid = r.get("game_id")
                if gid:
                    games_hi.add(gid)
                if is_late:
                    late_hi += 1
                    if gid:
                        games_late_hi.add(gid)

    print(f"path={args.path}")
    print(f"records={n}  bad_json={bad}  no_mmr={no_mmr}")
    print(f"late = turn>={args.late_turn} & tier>={args.late_tier}: {late} "
          f"({100.0 * late / max(n, 1):.1f}%)")
    print(f"mmr>={args.mmr_min}: {hi} ({100.0 * hi / max(n, 1):.1f}%), "
          f"games={len(games_hi)}, late&mmr>={args.mmr_min}: {late_hi} "
          f"games={len(games_late_hi)}")

    print()
    print("MMR 分布")
    for lo, hi2 in MMR_BUCKETS:
        k = f"{lo}-{hi2}" if hi2 < 99999 else f"{lo}+"
        c = mmr_hist.get(k, 0)
        print(f"  {k:>10} {c:>8} {100.0 * c / max(n, 1):>6.1f}%")
    if mmr_hist.get("none"):
        print(f"  {'none':>10} {mmr_hist['none']:>8}")

    print()
    print("turn 分布（20=20+）")
    for t in sorted(turn_hist):
        c = turn_hist[t]
        print(f"  T{t:<3} {c:>8} {100.0 * c / max(n, 1):>6.1f}%  "
              f"(mmr>={args.mmr_min}: {hi_turns.get(t, 0)})")

    print()
    print(f"动作类型（全体 {args.top} 个）")
    for a, c in act_all.most_common(args.top):
        print(f"  {a:<16} {c:>8} {100.0 * c / max(n, 1):>6.1f}%")
    print()
    print(f"动作类型（晚局 turn>={args.late_turn} & tier>={args.late_tier}）")
    if not act_late:
        print("  （没有晚局样本 —— BC 教不了中后期）")
    for a, c in act_late.most_common(args.top):
        print(f"  {a:<16} {c:>8} {100.0 * c / max(late, 1):>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
