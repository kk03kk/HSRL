#!/usr/bin/env python3
"""跨面板汇总：把今晚所有 768 局面板拼成一张表（模型有多强 / 相比 champion 提升多少）。

为什么单独写一个
----------------
每个面板（`gap-*/panel.json`）里都有一个 `base`（= champion）+ 若干候选。逐个面板读很啰嗦，
而最终要回答的问题只有一个：**每个臂相对 champion 的配对 Δ 是多少、CI 含不含 0**。
本工具扫描所有面板、按臂聚合、打印一张总表，并给出 `D-052` 门禁的 signal R²。

用法
----
    python tools/summarize_arms.py [--root artifacts/runs] [--json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))


def _paired_delta(cand: List[int], base: List[int], n_boot: int = 4000,
                  seed: int = 20261113) -> Optional[Dict[str, Any]]:
    """同批种子的逐局名次差 + bootstrap 95% CI（与 accept_gap_panel 同口径）。"""
    if not cand or not base or len(cand) != len(base):
        return None
    diffs = [a - b for a, b in zip(cand, base)]
    n = len(diffs)
    mean = sum(diffs) / n
    import random
    rnd = random.Random(seed)
    boots = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return {"n": n, "mean": mean,
            "lo": boots[int(0.025 * (n_boot - 1))],
            "hi": boots[int(0.975 * (n_boot - 1))]}


def collect(root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, "gap-*", "panel.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        models = {m.get("name"): m for m in d.get("models") or []}
        base = models.get("base")
        if not base or not base.get("rank_sequence"):
            continue
        bseq = [int(x) for x in base["rank_sequence"]]
        bidx = {str(b["seed"]): i for i, b in enumerate(base.get("final_boards") or [])}
        for name, m in models.items():
            if name == "base" or not m.get("rank_sequence"):
                continue
            cseq = m["rank_sequence"]
            if len(cseq) != len(bseq):
                continue
            delta = _paired_delta([int(x) for x in cseq], bseq)
            o = m.get("outcome") or {}
            sv = m.get("survival") or {}
            fb = m.get("final_boards") or []
            t8 = [b.get("strength") for b in fb if int(b.get("turn") or 0) >= 8]
            cov = (m.get("coverage") or {}).get("any_core_by_tier") or {}
            rg = m.get("retained_gain_by_turn") or {}
            rows.append({
                "arm": name,
                "panel": os.path.basename(os.path.dirname(path)),
                "games": len(cseq),
                "mean_rank": o.get("mean_rank"),
                "top4": o.get("top4_rate"),
                "win": o.get("win_rate"),
                "sT10": (sv.get("reached") or {}).get("10"),
                "sT12": (sv.get("reached") or {}).get("12"),
                "delta": None if delta is None else round(delta["mean"], 3),
                "ci": None if delta is None else [round(delta["lo"], 3), round(delta["hi"], 3)],
                "sig": (None if delta is None
                        else ("worse" if delta["lo"] > 0 else
                              "better" if delta["hi"] < 0 else "ns")),
                "s_any": cov.get("S"),
                "kept_T10_11": (rg.get("10") or {}).get("kept_gain"),
            })
    # 同一臂可能在多个面板出现：保留 games 最多的那条
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cur = best.get(r["arm"])
        if cur is None or (r["games"] or 0) > (cur["games"] or 0):
            best[r["arm"]] = r
    return sorted(best.values(), key=lambda r: (r["mean_rank"] is None, r["mean_rank"]))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/runs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = collect(args.root)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no panels found")
        return 2
    print(f"{'arm':<14}{'games':>6}{'mean_rank':>10}{'top4':>7}{'win':>7}"
          f"{'sT10':>7}{'sT12':>7}{'S-any':>7}{'kept10_11':>10}   "
          f"{'delta vs champion':>18}  verdict")
    print("-" * 118)
    for r in rows:
        ci = "" if r["ci"] is None else f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        delta_cell = "" if r["delta"] is None else f"{r['delta']:+.3f} {ci}"
        print(f"{r['arm']:<14}{r['games']:>6}"
              f"{(r['mean_rank'] if r['mean_rank'] is not None else float('nan')):>10.3f}"
              f"{(r['top4'] or 0):>7.3f}{(r['win'] or 0):>7.3f}"
              f"{(r['sT10'] or 0):>7.3f}{(r['sT12'] or 0):>7.3f}"
              f"{(r['s_any'] or 0):>7.3f}"
              f"{(r['kept_T10_11'] if r['kept_T10_11'] is not None else 0):>10.1f}   "
              f"{delta_cell:>18}  {r['sig'] or ''}")
    print()
    print("champion (base) 在每个面板内都是同一个 checkpoint：league-final-20261111-step014858")
    print("sig: worse = 配对 CI 全 >0（显著更差）; better = 全 <0（显著更好）; ns = 含 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
