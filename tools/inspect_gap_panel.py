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
    ap.add_argument("--action-mix", action="store_true",
                    help="打印逐回合的动作构成均值（买/卖/刷新/出牌…）——用来判断"
                         "禁卖起始战团（K1/K2）这种规则会命中哪些回合")
    ap.add_argument("--mechanics", action="store_true",
                    help="打印逐回合金卡/每格攻血（专家 C1；需新面板的 mechanics_by_turn）")
    ap.add_argument("--combat", action="store_true",
                    help="打印逐回合实际战斗胜负（需要面板开 collect_exposure）")
    ap.add_argument("--economy", action="store_true",
                    help="打印逐回合金币 p50（区分『没钱升本』与『有钱不升』）")
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

    # ---- 金币存量：判断"没钱升本"还是"有钱不升" ----
    if args.economy:
        print()
        print("GOLD p50（回合结束时的金币余额；升本 5 级要 9、6 级要 10）")
        print(f"{'turn':>4}" + "".join(f" {t + ':' + n:>16}" for t, n, _ in rows))
        for t in turns:
            line = f"{t:>4}"
            for tag, name, model in rows:
                c = curve(model, t) or {}
                line += f" {fmt(c.get('gold_p50'), 1, 16)}"
            print(line)

    # ---- 实际战斗胜率（专家先验 G1：Σ 不代表战力）----
    if args.combat:
        print()
        print("COMBAT（逐回合实际对阵；win/tie/dmg/ghost/mirror 均为有战斗记录的局）")
        keys = ("combats", "win", "tie", "damage_taken_mean", "ghost", "mirror_applied")
        for tag, name, model in rows:
            cb = model.get("combat_by_turn") or {}
            if not cb:
                print(f"  [{tag}:{name}] 没有 combat_by_turn（旧面板或未开 collect_exposure）")
                continue
            print(f"  [{tag}:{name}]  " + " ".join(f"{k[:10]:>11}" for k in keys))
            for t in turns:
                d = cb.get(t)
                if not d:
                    continue
                print(f"    T{t:<3} " + " ".join(f"{float(d.get(k, 0) or 0):>11.2f}"
                                                for k in keys))

    # ---- 机制密度：金卡 / 每格攻血（专家 C1；真人对照见 tools/report_board_mechanics.py）----
    if args.mechanics:
        print()
        print("MECHANICS（每局金卡数 / 有金卡的局占比 / 每格攻血）")
        keys = ("golden_per_game", "with_golden", "atk_per_minion", "hp_per_minion")
        for tag, name, model in rows:
            mb = model.get("mechanics_by_turn") or {}
            if not mb:
                print(f"  [{tag}:{name}] 没有 mechanics_by_turn（旧面板）")
                continue
            print(f"  [{tag}:{name}]  " + " ".join(f"{k[:14]:>15}" for k in keys))
            for t in turns:
                d = mb.get(t)
                if not d:
                    continue
                print(f"    T{t:<3} " + " ".join(f"{float(d.get(k, 0) or 0):>15.2f}"
                                                for k in keys))

    # ---- 训练内 tier 曲线原始字段（反推 tier-target 目标表，避免拍脑袋）----
    if args.human_tier:
        tb = human.get("tier_by_turn") or {}
        print()
        print("HUMAN tier_by_turn 原始字段")
        for t in turns:
            d = tb.get(t)
            if d:
                print(f"  T{t}: {json.dumps(d, ensure_ascii=False, sort_keys=True)}")

    # ---- 阵容规模：板面格数 vs 每格强度（区分"没铺满"与"每格太弱"）----
    print()
    print("BOARD SIZE  board_n_p50 / (strength_p50 ÷ board_n_p50)  —— 区分「没铺满」与「每格太弱」")
    print(f"{'turn':>4}" + "".join(f" {t + ':' + n:>20}" for t, n, _ in rows))
    for t in turns:
        line = f"{t:>4}"
        for tag, name, model in rows:
            c = curve(model, t) or {}
            n_min = c.get("board_n_p50")
            s = c.get("strength_p50")
            per = (s / n_min) if (s and n_min) else None
            line += f" {fmt(n_min, 1, 7)}{fmt(per, 0, 13)}"
        print(line)

    # ---- 逐回合动作构成（判断"禁卖起始战团"会命中哪些回合）----
    if args.action_mix:
        print()
        print("ACTION MIX（逐回合、每局均值）")
        groups = ("buy", "sell", "refresh", "play", "upgrade", "dark_gift",
                  "hero_power", "choose", "activate", "magnetize")
        for tag, name, model in rows:
            am = model.get("action_mix_by_turn") or {}
            if not am:
                print(f"  [{tag}:{name}] 没有 action_mix_by_turn")
                continue
            print(f"  [{tag}:{name}]  "
                  + " ".join(f"{g[:6]:>7}" for g in groups) + "   (games)")
            for t in turns:
                d = am.get(t) or {}
                mean = d.get("mean") or {}
                if not mean:
                    continue
                print(f"    T{t:<3} "
                      + " ".join(f"{float(mean.get(g, 0) or 0):>7.2f}" for g in groups)
                      + f"   ({d.get('games', '?')})")
            # used_rate：当回合"做过这件事"的局占比（旧面板没有这个字段，跳过）
            rate = {t: (am.get(t) or {}).get("used_rate") or {} for t in turns}
            if any(rate.values()):
                keys = ("sell", "upgrade", "dark_gift")
                print(f"  [{tag}:{name}] USED RATE  "
                      + " ".join(f"{g[:8]:>8}" for g in keys))
                for t in turns:
                    r = rate.get(t) or {}
                    if not r:
                        continue
                    print(f"    T{t:<3} "
                          + " ".join(f"{float(r.get(g, 0) or 0):>8.2f}" for g in keys))

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
