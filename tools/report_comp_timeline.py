#!/usr/bin/env python3
"""阵容完成度时间线：用**与训练奖励同一口径**评估"什么时候拼齐了阵容"。

为什么需要
----------
四件套里的"核心卡覆盖"只看**终局板面**，于是有两个盲点：

1. 看不到"什么时候拼齐"——第 8 回合就成型的阵容和第 15 回合才凑齐的阵容，
   在终局覆盖率里一模一样；
2. 看不到"转型"——中途拼过 S 级又拆掉的，终局覆盖率会记成 0。

``workflows/eval_human_gap.py`` 现在会写 ``boards_by_turn``（逐回合卡 id），
本脚本按 ``hsbg/az/train/comp_target.py`` 的同一套口径
（等级权重 S=1.0/A=0.6/B=0.3，``min(1, 命中 / max(2, 核心卡数))``）逐回合算完成度。

用法::

    python tools/report_comp_timeline.py \
        --panel artifacts/runs/gap-g24-20261113/panel.json \
        --turns 8-16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hsbg.az.train.comp_target import (  # noqa: E402
    DEFAULT_CARDS_PATH, load_comps, parse_tier_weights,
)


def score(comps: List[Dict[str, Any]], have: List[str]) -> Tuple[float, Optional[str],
                                                                 Optional[str], int]:
    have_set = set(have)
    best, name, tier, hits_best = 0.0, None, None, 0
    for c in comps:
        hits = len(c["cards"] & have_set)
        if not hits:
            continue
        s = c["weight"] * min(1.0, hits / c["denom"])
        if s > best + 1e-12:
            best, name, tier, hits_best = s, c["name"], c["tier"], hits
    return best, name, tier, hits_best


def _pct(nums: List[float], q: float) -> Optional[float]:
    if not nums:
        return None
    s = sorted(nums)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def human_boards_by_turn(path: str, mmr_min: int) -> Dict[str, Dict[str, List[str]]]:
    """真人库 → ``game_id -> turn -> [card_id]``（``perspective==self``、combat_start）。

    与模型侧同口径：只取 ``stage == combat_start``（= 招募结束时的战团），
    ```card_id`` 去 ``_G`` 后缀。
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("perspective") != "self" or r.get("stage") != "combat_start":
                continue
            mmr = r.get("mmr")
            try:
                if mmr is not None and float(mmr) < float(mmr_min):
                    continue
            except (TypeError, ValueError):
                pass
            ids = []
            for m in r.get("board_detail") or []:
                cid = m.get("card_id")
                if isinstance(cid, str):
                    ids.append(cid[:-2] if cid.endswith("_G") else cid)
            out.setdefault(str(r.get("game_id")), {})[str(int(r["turn"]))] = ids
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="阵容完成度时间线")
    ap.add_argument("--panel", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--strategy", default=DEFAULT_CARDS_PATH)
    ap.add_argument("--tier-w", default=None)
    ap.add_argument("--turns", default="8-16")
    ap.add_argument("--min-turn", type=int, default=6,
                    help="面板里 boards_by_turn 记录的最小回合（默认 6）")
    ap.add_argument("--human", default="",
                    help="真人库 library.jsonl；给了就把真人同口径曲线一起打印")
    ap.add_argument("--human-mmr-min", type=int, default=8000)
    args = ap.parse_args(argv)

    comps = load_comps(args.strategy, parse_tier_weights(args.tier_w))
    lo, _, hi = args.turns.partition("-")
    t_lo, t_hi = int(lo), int(hi or lo)

    def cells_for(bt: Dict[str, Dict[str, List[str]]], t: int) -> str:
        vals, s_tier = [], 0
        for _seed, turns in bt.items():
            have = turns.get(str(t))
            if have is None:
                continue
            s, _name, tier, _h = score(comps, have)
            vals.append(s)
            if tier == "S" and s > 0:
                s_tier += 1
        n = len(vals)
        if not n:
            return f"{'—':>26}"
        mean = sum(vals) / n
        p50 = _pct(vals, 0.5) or 0.0
        half = sum(1 for v in vals if v >= 0.5) / n
        return f"{mean:>7.3f}{p50:>6.3f}{half:>7.1%}{s_tier / n:>6.0%}"

    human: Optional[Dict[str, Dict[str, List[str]]]] = None
    if args.human:
        human = human_boards_by_turn(args.human, args.human_mmr_min)
        print(f"\n真人库 {args.human}（perspective=self, mmr>={args.human_mmr_min}）："
              f"{len(human)} 局")
        print(f"{'turn':>4} {'n':>4}{'human':>26}")
        for t in range(t_lo, t_hi + 1):
            n = sum(1 for turns in human.values() if str(t) in turns)
            print(f"{t:>4} {n:>4}{cells_for(human, t)}")

    for spec in args.panel:
        tag, _, path = spec.partition("=")
        if not path or not os.path.isfile(path):
            raise SystemExit(f"--panel 需要 TAG=PATH 且文件存在，收到 {spec!r}")
        panel = json.loads(open(path, encoding="utf-8-sig").read())
        print(f"\n== {tag}  ({path})")
        print("   阵容完成度（与训练奖励同口径）：mean / p50 / ≥0.5 占比 / 最优是 S 级占比")
        print(f"{'turn':>4} {'n':>4}" + "".join(
            f"{m.get('name', '?'):>26}" for m in panel.get("models") or []))
        for t in range(t_lo, t_hi + 1):
            row = f"{t:>4}"
            cells = []
            for m in panel.get("models") or []:
                cells.append(cells_for(m.get("boards_by_turn") or {}, t))
            n_any = 0
            for m in panel.get("models") or []:
                bt = m.get("boards_by_turn") or {}
                n_any = max(n_any, sum(1 for turns in bt.values() if str(t) in turns))
            print(f"{row}{n_any:>4}" + "".join(cells))
        if not any((m.get("boards_by_turn") or {}) for m in panel.get("models") or []):
            print("   （该面板没有 boards_by_turn —— 生成于 2026-11-13 01:30 之前，"
                  "需要重跑面板才能看到这个口径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
