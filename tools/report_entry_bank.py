"""统计入场库的「回合 × 观测等级」分布（纯标准库，可在登录节点跑）。

用途：M1 需要「T8 且观测等级 ≥5」的真人帧。开训前先确认池子里真的有足够多的这种帧，
否则 `--entry-min-tier 5` 会在作业启动时 fail-fast（那是刻意的，但白占一次排队）。

例：
    python3 tools/report_entry_bank.py \
        --bank data/real_library/firestone-bank-6k/train-m6k.jsonl --turns 6-10
"""

from __future__ import annotations

import argparse
import collections
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--turns", default="7-10", help="回合区间 a-b（闭区间）")
    ap.add_argument("--perspective", default="",
                    help="只统计某个 perspective（self/opponent；空=全部）")
    args = ap.parse_args()

    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)
    want = args.perspective.strip()

    rows = 0
    by_turn: collections.Counter = collections.Counter()
    by_turn_tier: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    games: collections.defaultdict[int, set] = collections.defaultdict(set)
    by_gold: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    by_hand: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    no_tier = 0
    no_detail = 0
    for line in open(args.bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            r = json.loads(line)
        except BaseException:  # noqa: BLE001
            continue
        if want and str(r.get("perspective") or "") != want:
            continue
        try:
            t = int(r.get("turn", 0))
        except BaseException:  # noqa: BLE001
            continue
        if not (lo <= t <= hi):
            continue
        if not isinstance(r.get("board_detail"), list):
            no_detail += 1
        tier = r.get("tier")
        if tier is None:
            no_tier += 1
            continue
        by_turn[t] += 1
        by_turn_tier[t][int(tier)] += 1
        gold = r.get("gold")
        hand_n = r.get("hand_n")
        if hand_n is None and isinstance(r.get("hand"), list):
            hand_n = len(r["hand"])
        if isinstance(gold, (int, float)):
            gb = ">=9" if gold >= 9 else "6-8" if gold >= 6 else "3-5" if gold >= 3 else "<3"
            by_gold[t][gb] += 1
        if isinstance(hand_n, (int, float)):
            hb = ">=6" if hand_n >= 6 else "4-5" if hand_n >= 4 else "2-3" if hand_n >= 2 else "<2"
            by_hand[t][hb] += 1
        if r.get("game_id"):
            games[t].add(str(r["game_id"]))

    print(f"bank={args.bank} rows={rows} perspective={want or 'all'} turns={lo}..{hi}")
    print(f"{'turn':>5} {'rows':>6} {'games':>6} " + " ".join(f"{'t' + str(k):>6}"
                                                             for k in range(1, 8)))
    for t in range(lo, hi + 1):
        c = by_turn_tier.get(t) or collections.Counter()
        print(f"{t:>5} {by_turn.get(t, 0):>6} {len(games.get(t, ())):>6} "
              + " ".join(f"{c.get(k, 0):>6}" for k in range(1, 8)))
    print()
    print(f"{'turn':>5}  gold(<3/3-5/6-8/>=9)          hand(<2/2-3/4-5/>=6)")
    for t in range(lo, hi + 1):
        g = by_gold.get(t) or collections.Counter()
        h = by_hand.get(t) or collections.Counter()
        print(f"{t:>5}  {g.get('<3',0):>5}/{g.get('3-5',0):<6}/{g.get('6-8',0):<6}/{g.get('>=9',0):<7} "
              f"{h.get('<2',0):>5}/{h.get('2-3',0):<6}/{h.get('4-5',0):<6}/{h.get('>=6',0):<7}")
    tot = sum(by_turn.values())
    ge5 = sum(v for t in by_turn_tier for k, v in by_turn_tier[t].items() if k >= 5)
    print(f"合计 {tot} 行；其中 tier>=5 有 {ge5} 行 "
          f"（{100.0 * ge5 / tot if tot else 0:.1f}%）；缺 tier 字段 {no_tier} 行，"
          f"缺 board_detail {no_detail} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
