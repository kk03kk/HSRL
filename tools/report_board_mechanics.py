"""真人库的"机制密度"：每回合金卡数 / 关键词率 / 每格攻血（纯标准库，登录节点可跑）。

动机（专家先验 C1）：后期每格强度 700–3900，靠的是乘法机制而不只是"买更多随从"。
其中最容易测的一项是**金卡（三连）**：金卡=属性翻倍+三连奖励（发现更高等级随从），
它同时是战力与经济的乘法器。模型侧同名指标见面板的 `golden_by_turn`。

例：
    python3 tools/report_board_mechanics.py \
        --bank data/real_library/firestone-bank-6k/train-m6k.jsonl --turns 8-14
"""

from __future__ import annotations

import argparse
import collections
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--turns", default="8-14")
    ap.add_argument("--min-tier", type=int, default=0,
                    help="只用观测等级 >= 本值的帧（0=全部）")
    args = ap.parse_args()

    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)

    per_turn = collections.defaultdict(lambda: {
        "boards": 0, "golden": 0, "minions": 0, "boards_with_golden": 0, "top1": 0.0,
        "atk": 0, "hp": 0, "kw": 0,
    })
    for line in open(args.bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except BaseException:  # noqa: BLE001
            continue
        try:
            t = int(r.get("turn", 0))
        except BaseException:  # noqa: BLE001
            continue
        if not (lo <= t <= hi):
            continue
        tier = r.get("tier")
        if args.min_tier and (tier is None or int(tier) < args.min_tier):
            continue
        detail = r.get("board_detail")
        if not isinstance(detail, list) or not detail:
            continue
        d = per_turn[t]
        d["boards"] += 1
        g = 0
        for m in detail:
            if not isinstance(m, dict):
                continue
            d["minions"] += 1
            d["atk"] += int(m.get("atk") or 0)
            d["hp"] += int(m.get("hp") or 0)
            d["kw"] += int(m.get("kw") or 0) + int(m.get("bonus_kw") or 0)
            if m.get("golden"):
                g += 1
        sums = [int(m.get("atk") or 0) + int(m.get("hp") or 0)
                for m in detail if isinstance(m, dict)]
        if sums and sum(sums) > 0:
            d["top1"] += max(sums) / sum(sums)
        d["golden"] += g
        if g:
            d["boards_with_golden"] += 1

    print(f"bank={args.bank} turns={lo}..{hi} min_tier={args.min_tier}")
    print(f"{'turn':>5} {'boards':>7} {'minion':>7} {'golden':>7} {'gold/game':>10} "
          f"{'有金卡局':>9} {'atk':>7} {'hp':>7} {'kw/grid':>8}")
    for t in range(lo, hi + 1):
        d = per_turn.get(t)
        if not d or not d["boards"]:
            continue
        n = d["boards"]
        print(f"{t:>5} {n:>7} {d['minions']:>7} {d['golden']:>7} "
              f"{d['golden'] / n:>10.2f} {d['boards_with_golden'] / n:>9.2f} "
              f"{d['atk'] / d['minions']:>7.1f} {d['hp'] / d['minions']:>7.1f} "
              f"{d['kw'] / d['minions']:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
