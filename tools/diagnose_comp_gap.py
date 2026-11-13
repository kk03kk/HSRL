#!/usr/bin/env python3
"""阵容差距的细粒度诊断：模型「拼到哪一张就停了」，以及**缺的是几级的卡**。

为什么需要
----------
`report_comp_timeline` 只能给"完成度"这一个数。H2 把完成度从 0.16 抬到 0.256
（真人 0.426），但**卡在哪一步**不知道：

- 若缺的都是 **5/6 级**核心卡 → 说明是"能不能上到高阶并买到它"（经济/升本）问题；
- 若缺的分布和卡池一致 → 说明是"识别不出该拼哪个"（策略）问题。

两种诊断的处方完全不同，所以这里按**核心卡酒馆等级**分开统计命中率。

输入：`panel.json` 的 `boards_by_turn`（2026-11-13 01:30 起 `eval_human_gap.py` 会写）
+ `config/strategy_cards.json`（`core[*].tavern_tier`）。

用法::

    python tools/diagnose_comp_gap.py \
        --panel h2=artifacts/runs/gap-h2-20261113/panel.json --turns 12,14
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def norm(cid: Any) -> str:
    s = str(cid or "")
    return s[:-2] if s.endswith("_G") else s


def load_comps(path: str) -> List[Dict[str, Any]]:
    """→ ``[{name, tier, weight, cards: [(card_id, tavern_tier)], denom}]``。"""
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    out: List[Dict[str, Any]] = []
    for c in data.get("comps") or []:
        cards = []
        for item in c.get("core") or []:
            cid = norm(item.get("card_id"))
            if not cid:
                continue
            try:
                tt = int(item.get("tavern_tier") or 0)
            except (TypeError, ValueError):
                tt = 0
            cards.append((cid, tt))
        if not cards:
            # 退回 core_card_ids（没有 tavern_tier 信息时记 0）
            ids = [norm(x) for x in (c.get("core_card_ids") or []) if str(x or "").strip()]
            cards = [(cid, 0) for cid in ids]
        if not cards:
            continue
        tier = str(c.get("tier") or "").strip().upper()
        w = {"S": 1.0, "A": 0.6, "B": 0.3}.get(tier, 0.0)
        out.append({"name": str(c.get("name") or ""), "tier": tier,
                    "weight": w, "cards": cards, "denom": max(2, len(cards))})
    return out


def best_comp(comps: List[Dict[str, Any]], have: set) -> Tuple[Optional[Dict], int]:
    best, best_hits = None, -1
    for c in comps:
        hits = sum(1 for cid, _t in c["cards"] if cid in have)
        if hits > best_hits:
            best, best_hits = c, hits
    return best, max(0, best_hits)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="核心卡命中率按酒馆等级拆解")
    ap.add_argument("--panel", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--strategy", default="config/strategy_cards.json")
    ap.add_argument("--turns", default="12,14")
    ap.add_argument("--min-comp-games", type=int, default=10,
                    help="某流派被选中至少这么多局才进榜（默认 10）")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    comps = load_comps(args.strategy)
    turns = [int(x) for x in str(args.turns).replace(" ", "").split(",") if x]

    for spec in args.panel:
        tag, _, path = spec.partition("=")
        if not path or not os.path.isfile(path):
            raise SystemExit(f"--panel 需要 TAG=PATH，收到 {spec!r}")
        panel = json.loads(open(path, encoding="utf-8-sig").read())
        print(f"\n===== {tag}  ({path})")
        for m in panel.get("models") or []:
            bt = m.get("boards_by_turn") or {}
            if not bt:
                print(f"  [{m.get('name')}] 没有 boards_by_turn（旧面板），跳过")
                continue
            for t in turns:
                by_tier: Dict[int, List[int]] = collections.defaultdict(lambda: [0, 0])
                comp_games: collections.Counter = collections.Counter()
                comp_hits: collections.Counter = collections.Counter()
                comp_cards: collections.Counter = collections.Counter()
                n_games = 0
                for _seed, per_turn in bt.items():
                    have_ids = per_turn.get(str(t))
                    if have_ids is None:
                        continue
                    n_games += 1
                    have = set(have_ids)
                    c, hits = best_comp(comps, have)
                    if c is None:
                        continue
                    comp_games[c["name"]] += 1
                    comp_hits[c["name"]] += hits
                    comp_cards[c["name"]] += len(c["cards"])
                    for cid, tt in c["cards"]:
                        by_tier[tt][1] += 1
                        if cid in have:
                            by_tier[tt][0] += 1
                if not n_games:
                    continue
                print(f"  [{m.get('name')}] T{t}  n={n_games}")
                line = "    核心卡命中率按酒馆等级: "
                for tt in sorted(by_tier):
                    hit, tot = by_tier[tt]
                    if not tot:
                        continue
                    line += f"T{tt}:{hit}/{tot}={hit / tot:.0%}  "
                print(line)
                top = [(name, comp_games[name],
                        comp_hits[name] / max(1, comp_games[name]))
                       for name in comp_games
                       if comp_games[name] >= args.min_comp_games]
                top.sort(key=lambda kv: -kv[1])
                if top:
                    print("    最常被当作'目标流派'的（命中数/核心卡数）:")
                    for name, g, mean_hits in top[: args.top]:
                        print(f"      {name:<24} 局数={g:<4} 平均命中={mean_hits:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
