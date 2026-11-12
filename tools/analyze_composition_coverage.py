#!/usr/bin/env python3
"""阵容掌握度：模型 vs 真人赢家，各自的终局板面能拼出哪些流派。

为什么需要它：`docs/analysis/HUMAN_GAP_20261112.md` 只证明了"战力差距"，
不能说明差距来自"不会选阵容"还是"会选但执行不出来"。这个脚本把攻略（33 个流派，
5 个 S 级）变成可判定的标准：

    一局"拼出某流派" = 该局终局板面含有该流派的 ≥2 张核心卡。

只用 **核心卡**（攻略的 ``### 核心卡牌``），不用附加卡——核心卡才是流派的启动条件。

输入
----
* ``config/strategy_cards.json`` —— ``tools/build_strategy_card_map.py`` 产出。
* 真人：``data/real_library/firestone-20261112-1000/library.jsonl``
  （``perspective=self``、``stage=combat_start``、每局取最大回合）。
* 模型：``eval_human_gap`` 面板产出的 ``panel.json``（``models[].final_boards``）。

口径声明
--------
* 真人侧**胜者偏差**（816/995 第 1 名），所以"真人拼出率"是**冠军水平**，不是平均。
* 模型侧对手是 7 个引擎 bot，真人侧是 8 个真人。只比"板面构成"，不比名次。
* 两边都只看**终局板面**；中途拼过又卖掉的不计入。这会低估双方，但方向一致。

用法（服务器上，标准库即可）::

    PYTHONPATH=engine/python:. python tools/analyze_composition_coverage.py \
        --panel artifacts/runs/human-gap-all5-20261112/panel.json \
        --human data/real_library/firestone-20261112-1000/library.jsonl \
        --min-turn 10 --top 8
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def load_strategy(path: Path) -> List[Dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["comps"]


def human_final_boards(path: Path, min_turn: int) -> List[Dict[str, Any]]:
    last: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("perspective") != "self":
                continue
            if r.get("stage") != "combat_start":
                continue
            gid = r["game_id"]
            t = int(r["turn"])
            if gid not in last or t >= int(last[gid]["turn"]):
                ids = []
                for m in r.get("board_detail") or []:
                    cid = m.get("card_id")
                    if isinstance(cid, str):
                        ids.append(cid[:-2] if cid.endswith("_G") else cid)
                last[gid] = {"turn": t, "card_ids": ids,
                             "placement": r.get("placement"),
                             "archetype": r.get("archetype")}
    return [v for v in last.values() if v["turn"] >= min_turn and v["card_ids"]]


def score_board(ids: Iterable[str], comps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一局板面对每个流派的命中数，以及命中最多的那个。"""
    have: Set[str] = set(ids)
    per_comp = {}
    for c in comps:
        core = set(c.get("core_card_ids") or [])
        if not core:
            continue
        hit = len(core & have)
        if hit:
            per_comp[c["name"]] = hit
    best = max(per_comp.items(), key=lambda kv: kv[1]) if per_comp else (None, 0)
    tiers = {c["name"]: c["tier"] for c in comps}
    return {
        "per_comp": per_comp,
        "best_comp": best[0],
        "best_hits": best[1],
        "best_tier": tiers.get(best[0]) if best[0] else None,
    }


def summarise(boards: List[Dict[str, Any]], comps: List[Dict[str, Any]],
              label: str) -> Dict[str, Any]:
    tiers = collections.Counter(c["tier"] for c in comps)
    rows = [score_board(b["card_ids"], comps) for b in boards]
    n = max(len(rows), 1)

    def share(pred) -> float:
        return round(sum(1 for r in rows if pred(r)) / n, 3)

    tier_of = {c["name"]: c["tier"] for c in comps}
    any_tier = {
        t: share(lambda r, t=t: any(tier_of.get(k) == t for k in r["per_comp"]))
        for t in ("S", "A", "B")
    }
    assembled = {k: share(lambda r, k=k: r["best_hits"] >= k) for k in (2, 3, 4)}

    # 命中最多的流派排行（只统计 >=2 的，即"真的拼起来"）
    counts = collections.Counter()
    for r in rows:
        for name, hit in r["per_comp"].items():
            if hit >= 2:
                counts[name] += 1
    top = [{"comp": name, "tier": tier_of.get(name), "games": c,
            "share": round(c / n, 3)} for name, c in counts.most_common()]
    return {
        "label": label,
        "games": len(rows),
        "any_core_by_tier": any_tier,
        "assembled_at_least": assembled,
        "top_comps": top,
        "comp_tier_counts": dict(sorted(tiers.items())),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阵容掌握度：模型 vs 真人")
    ap.add_argument("--strategy", default="config/strategy_cards.json")
    ap.add_argument("--panel", default="")
    ap.add_argument("--human", default="data/real_library/firestone-20261112-1000/library.jsonl")
    ap.add_argument("--min-turn", type=int, default=10)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    comps = load_strategy(Path(args.strategy))
    sides: List[Dict[str, Any]] = []

    hp = Path(args.human)
    if hp.is_file():
        sides.append(summarise(human_final_boards(hp, args.min_turn), comps,
                               "human-winners"))
    else:
        print(f"WARN: human data missing at {hp}", file=sys.stderr)

    if args.panel:
        panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
        for m in panel["models"]:
            boards = [b for b in m.get("final_boards") or [] if b["turn"] >= args.min_turn]
            if boards:
                sides.append(summarise(boards, comps, m["name"]))

    if not sides:
        print("ERROR: nothing to compare", file=sys.stderr)
        return 2

    print(f"compositions: {len(comps)}  "
          f"tiers={dict(sorted(collections.Counter(c['tier'] for c in comps).items()))}  "
          f"min_turn={args.min_turn}")
    print()
    hdr = f"{'side':<16} {'games':>6} {'≥1 S':>7} {'≥1 A':>7} {'≥1 B':>7} " \
          f"{'2+core':>7} {'3+core':>7} {'4+core':>7}"
    print(hdr)
    for s in sides:
        a, asm = s["any_core_by_tier"], s["assembled_at_least"]
        print(f"{s['label']:<16} {s['games']:>6} {a['S']:>7} {a['A']:>7} {a['B']:>7} "
              f"{asm[2]:>7} {asm[3]:>7} {asm[4]:>7}")

    print()
    for s in sides:
        top = s["top_comps"][: args.top]
        if not top:
            print(f"-- {s['label']}: no composition assembled (>=2 core cards) --")
            continue
        line = "  ".join(f"{t['comp'].split(' - ')[-1]}({t['tier']}) {t['share']:.0%}"
                         for t in top)
        print(f"-- {s['label']:<14} {line}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"schema": "hsai.composition_coverage/v1",
                                   "min_turn": args.min_turn,
                                   "sides": sides}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nwritten: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
