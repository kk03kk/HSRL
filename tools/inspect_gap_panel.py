#!/usr/bin/env python3
"""Compact per-turn comparison across ``hsai.human_gap_panel/v1`` panels.

Why this exists
---------------
``workflows/eval_human_gap.py`` prints only turns 6/8/10/12/14 and only the
p50 board strength.  When a candidate checkpoint makes the *aggregate* ratio go
down we need to know **which turns regressed**: a drop concentrated in T2-T8
means the early game was forgotten (the entry curriculum replaced it), while a
drop in T12-T16 means the late game itself got worse.  Those two diagnoses call
for opposite fixes, and the aggregate number cannot tell them apart.

Usage
-----
    python tools/inspect_gap_panel.py \
        --panel e3=artifacts/runs/human-gap-20261112-211828/panel.json \
        --panel f1=artifacts/runs/human-gap-f1-20261112/panel.json \
        --model base --model e3 --model f1 \
        --delta e3:base --delta f1:base

Reads only the panel JSON; runs on the login node (no torch, no engine).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_panel(path: Path) -> Dict[str, Any]:
    # utf-8-sig: 允许手工用 PowerShell 另存的 JSON（带 BOM）
    return json.loads(path.read_text(encoding="utf-8-sig"))


def index_models(panel: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {m["name"]: m for m in panel.get("models") or []}


def curve(model: Dict[str, Any], turn: str) -> Optional[Dict[str, Any]]:
    return (model.get("curves") or {}).get(turn)


def human_p50(panel: Dict[str, Any], human: Dict[str, Any], turn: str) -> Optional[float]:
    return ((human.get("strength_by_turn") or {}).get(turn) or {}).get("p50")


def human_tier(human: Dict[str, Any], turn: str) -> Optional[float]:
    return ((human.get("tier_by_turn") or {}).get(turn) or {}).get("mean")


def fmt(x: Optional[float], nd: int = 0, width: int = 8) -> str:
    if x is None:
        return " " * (width - 2) + "- "
    return f"{x:{width}.{nd}f}"


def turns_of(models: Dict[str, Dict[str, Any]], want: Optional[str]) -> List[str]:
    seen = set()
    for m in models.values():
        seen.update((m.get("curves") or {}).keys())
    ts = sorted(seen, key=int)
    if want:
        lo, _, hi = want.partition("-")
        lo_i = int(lo)
        hi_i = int(hi) if hi else lo_i
        ts = [t for t in ts if lo_i <= int(t) <= hi_i]
    return ts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="per-turn panel comparison")
    ap.add_argument("--panel", action="append", default=[], required=True,
                    metavar="TAG=PATH")
    ap.add_argument("--model", action="append", default=[],
                    help="模型名（默认：所有面板里出现过的名字，按字母序）")
    ap.add_argument("--delta", action="append", default=[],
                    metavar="AFTER:BEFORE", help="配对差值 after-before（同面板）")
    ap.add_argument("--turns", default="2-20")
    ap.add_argument("--human", default="artifacts/runs/human-baseline-20261112/baseline.json")
    ap.add_argument("--human-tier", action="store_true",
                    help="额外打印真人 tier_by_turn 原始字段（反推 tier-target 目标表用）")
    args = ap.parse_args(argv)

    panels: Dict[str, Dict[str, Any]] = {}
    tagged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in args.panel:
        tag, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--panel 需要 TAG=PATH，收到 {spec!r}")
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"panel not found: {p}")
        panels[tag] = load_panel(p)
        tagged[tag] = index_models(panels[tag])

    human: Dict[str, Any] = {}
    hp = Path(args.human)
    if hp.is_file():
        human = json.loads(hp.read_text(encoding="utf-8"))

    names = args.model or sorted({n for ms in tagged.values() for n in ms})
    rows: List[tuple] = []
    for tag, ms in tagged.items():
        for name in names:
            if name in ms:
                rows.append((tag, name, ms[name]))

    turns = turns_of({f"{t}:{n}": m for t, n, m in rows}, args.turns)

    # ---- absolute board strength p50 + ratio vs human ----
    print("BOARD STRENGTH p50  (sum(atk+hp) at recruitment end)")
    head = f"{'turn':>4} {'human':>8}" + "".join(
        f" {t + ':' + n:>18}" for t, n, _ in rows)
    print(head)
    for t in turns:
        line = f"{t:>4} {fmt(human_p50(panels[list(tagged)[0]], human, t), 0):>8}"
        for tag, name, model in rows:
            c = curve(model, t)
            s = c.get("strength_p50") if c else None
            h = human_p50(panels[tag], human, t)
            ratio = (s / h) if (s and h) else None
            line += f" {fmt(s, 0, 9)}{fmt(ratio, 3, 9)}"
        print(line)

    # ---- tier tempo (does the model reach T5/T6 at the human turn?) ----
    print()
    print("TIER  (mean tavern tier)")
    print(f"{'turn':>4} {'human':>8}" + "".join(f" {t + ':' + n:>18}" for t, n, _ in rows))
    for t in turns:
        line = f"{t:>4} {fmt(human_tier(human, t), 2):>8}"
        for tag, name, model in rows:
            c = curve(model, t)
            line += f" {fmt(c.get('tier_mean') if c else None, 2, 9)}" + " " * 9
        print(line)

    # ---- paired deltas inside one panel ----
    for spec in args.delta:
        after, _, before = spec.partition(":")
        tag = next((tg for tg, ms in tagged.items() if after in ms and before in ms), None)
        if tag is None:
            print(f"\n[delta {spec}] 跳过了：没有哪个面板同时含 {after} 与 {before}")
            continue
        a, b = tagged[tag][after], tagged[tag][before]
        print()
        print(f"DELTA {after} - {before}  (panel {tag})")
        print(f"{'turn':>4} {'n_after':>8} {'n_before':>8} "
              f"{'str_after':>10} {'str_before':>10} {'str%':>8} "
              f"{'tier_d':>7} {'gold_d':>7}")
        for t in turns:
            ca, cb = curve(a, t), curve(b, t)
            sa = ca.get("strength_p50") if ca else None
            sb = cb.get("strength_p50") if cb else None
            rel = (100.0 * (sa - sb) / sb) if (sa and sb) else None
            ta = ca.get("tier_mean") if ca else None
            tb = cb.get("tier_mean") if cb else None
            ga = ca.get("gold_p50") if ca else None
            gb = cb.get("gold_p50") if cb else None
            print(f"{t:>4} {ca.get('n') if ca else 0:>8} {cb.get('n') if cb else 0:>8} "
                  f"{fmt(sa, 0, 10)} {fmt(sb, 0, 10)} {fmt(rel, 1, 8)} "
                  f"{fmt((ta - tb) if (ta is not None and tb is not None) else None, 2, 7)} "
                  f"{fmt((ga - gb) if (ga is not None and gb is not None) else None, 2, 7)}")

    # ---- 真人 tier 曲线原始字段（用来反推 tier-target 目标表，避免拍脑袋）----
    if args.human_tier:
        tb = human.get("tier_by_turn") or {}
        print()
        print("HUMAN tier_by_turn 原始字段")
        for t in turns:
            d = tb.get(t)
            if d:
                print(f"  T{t}: {json.dumps(d, ensure_ascii=False, sort_keys=True)}")

    # ---- 结局 / 存活 ----
    print()
    print("OUTCOME")
    print(f"{'panel:model':>26} {'mean_rank':>10} {'top4':>7} {'win':>7} "
          f"{'>t8':>6} {'>t10':>6} {'>t12':>6} {'>t14':>6} {'t_p50':>6} {'illegal':>8}")
    for tag, name, model in rows:
        o = model.get("outcome") or {}
        sv = model.get("survival") or {}
        r = sv.get("reached") or {}
        print(f"{tag + ':' + name:>26} {fmt(o.get('mean_rank'), 3, 10)} "
              f"{fmt(o.get('top4_rate'), 3, 7)} {fmt(o.get('win_rate'), 3, 7)} "
              f"{fmt(r.get('8'), 3, 6)} {fmt(r.get('10'), 3, 6)} "
              f"{fmt(r.get('12'), 3, 6)} {fmt(r.get('14'), 3, 6)} "
              f"{fmt(sv.get('max_turn_p50'), 1, 6)} {fmt(o.get('illegal_actions'), 0, 8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
