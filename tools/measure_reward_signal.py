#!/usr/bin/env python3
"""奖励信号灵敏度测量（D-052 门禁）：这个场里的名次奖励，到底能不能看见"棋力"？

为什么需要它
------------
2026-11-14 的根因结论：在我们所处的板面比值区间（T12 ≈ 真人 p50 的 0.29×），
战斗结果已经饱和（整队被清空），于是**名次对板面强度几乎不敏感**：
控制住"死在第几回合"之后 R² = 0.000–0.004，而死亡回合单独解释 65%。
结果是 PPO 只能学"活久一点"，继续训练反而把板子练薄（自建 T8 板面 270→206→168）。
一晚上四种奖励形状因此全部无效——它们都长在同一个看不见棋力的指标上。

用法
----
    python tools/measure_reward_signal.py artifacts/runs/gap-xxx/panel.json [--json]

判据（写死在这里，避免事后调门槛）
    signal R²(板面→名次 | 死亡回合) >= 0.05   -> PASS，这个场可以用来训练
    < 0.05                                    -> FAIL，训练等于在拟合噪声

口径
----
- 板面强度 = 该回合 board 上所有随从 `atk + hp` 之和（卡 id 去金色 `_G`）；
- **必须在共同回合、且只取活得比它更久的局**（否则"晚死 ⇒ 板大 ⇒ 名次好"会把
  死亡回合的效应算进板面，得到虚假的高相关）；
- 同时报逐回合**战斗胜率**与受伤均值：比值远离 1 时胜率会贴在 0 或 1 上，
  那就是"阶跃函数的平坦段"，此时再谈奖励系数没有意义。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

#: 通过门禁的最小（偏）相关平方
SIGNAL_R2_MIN = 0.05
#: 在这些回合上测板面（太早随从未成形，太晚样本骤减）
MEASURE_TURNS = ("8", "9", "10", "11")
#: 只取"活得比测量回合至少多这么多回合"的局，保证板面是死前测到的
SURVIVE_MARGIN = 2


def _stats(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float, int]]:
    n = len(xs)
    if n < 30:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    r = sxy / math.sqrt(sxx * syy)
    return r, r * r, n


def _resid(xs: List[float], ys: List[float]) -> List[float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    b = (sxy / sxx) if sxx > 0 else 0.0
    return [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]


def _board_strength(ids: Any, table: Dict[str, float]) -> Optional[float]:
    if not isinstance(ids, list) or not ids:
        return None
    total = 0.0
    for cid in ids:
        key = str(cid)
        if key.endswith("_G"):
            key = key[:-2]
        v = table.get(key)
        if v is None:
            return None
        total += v
    return total


def load_card_strength(path: str) -> Dict[str, float]:
    """卡 id -> atk+hp。只读引擎的 cards.json，不需要 hsbg_engine。"""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    cards = raw if isinstance(raw, list) else (raw.get("cards") or [])
    out: Dict[str, float] = {}
    for c in cards:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None:
            continue
        atk = next((c.get(k) for k in ("atk", "attack", "Attack")
                    if isinstance(c.get(k), (int, float))), 0)
        hp = next((c.get(k) for k in ("hp", "health", "Health")
                   if isinstance(c.get(k), (int, float))), 0)
        out[str(cid)] = float(atk) + float(hp)
    return out


def measure_panel(panel_path: str, table: Dict[str, float]) -> List[Dict[str, Any]]:
    with open(panel_path, encoding="utf-8") as fh:
        d = json.load(fh)
    results: List[Dict[str, Any]] = []
    for m in d.get("models") or []:
        fb = m.get("final_boards") or []
        rs = m.get("rank_sequence") or []
        if not fb or not rs or len(fb) != len(rs):
            continue
        seed2 = {str(b.get("seed")): (float(r), int(b.get("turn") or 0))
                 for b, r in zip(fb, rs)}
        row: Dict[str, Any] = {"model": m.get("name"), "games": len(rs)}

        # 1) 名次 ~ 死亡回合（基线解释力）
        st = _stats([float(t) for _, t in seed2.values()],
                    [float(r) for r, _ in seed2.values()])
        if st:
            row["death_turn_r2"] = round(st[1], 3)

        # 2) 名次 ~ 板面（控制死亡回合的偏相关）—— 这是门禁看的量
        bbt = m.get("boards_by_turn") or {}
        per_turn: Dict[str, Any] = {}
        for t in MEASURE_TURNS:
            xs: List[float] = []
            ys: List[float] = []
            ts: List[float] = []
            for seed, per in bbt.items():
                if not isinstance(per, dict):
                    continue
                rec = seed2.get(str(seed))
                if rec is None:
                    continue
                rank, death = rec
                if death < int(t) + SURVIVE_MARGIN:
                    continue
                s = _board_strength(per.get(t), table)
                if s is None:
                    continue
                xs.append(s)
                ys.append(rank)
                ts.append(death)
            raw = _stats(xs, ys)
            partial = _stats(_resid(ts, xs), _resid(ts, ys)) if len(xs) >= 30 else None
            if raw is None or partial is None:
                continue
            per_turn[t] = {"n": raw[2], "r": round(raw[0], 3), "r2": round(raw[1], 3),
                           "partial_r": round(partial[0], 3),
                           "partial_r2": round(partial[1], 3)}
        row["board_by_turn"] = per_turn
        vals = [v["partial_r2"] for v in per_turn.values()]
        row["signal_r2_max"] = max(vals) if vals else None

        # 3) 战斗胜率（判断是否处在阶跃的平坦段）
        cb = m.get("combat_by_turn") or {}
        row["combat"] = {t: {"win": cb[t].get("win"),
                             "dmg_taken": cb[t].get("damage_taken_mean")}
                         for t in cb if isinstance(cb[t], dict) and int(t) >= 7}
        results.append(row)
    return results


def verdict(rows: List[Dict[str, Any]]) -> Tuple[bool, str]:
    best = None
    for r in rows:
        v = r.get("signal_r2_max")
        if v is not None and (best is None or v > best):
            best = v
    if best is None:
        return False, "no measurable turn (样本不足，无法判定)"
    if best >= SIGNAL_R2_MIN:
        return True, f"PASS: signal R2={best:.3f} >= {SIGNAL_R2_MIN}"
    return False, (f"FAIL: signal R2={best:.3f} < {SIGNAL_R2_MIN} —— "
                   f"该场里的名次对板面几乎不敏感，训练等于拟合噪声（D-052）")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel", help="hsai.human_gap_panel/v1 panel.json")
    ap.add_argument("--cards", default="engine/crates/hsbg-engine/data/cards.json")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = ap.parse_args(argv)
    if not os.path.exists(args.panel):
        print(f"panel not found: {args.panel}", file=sys.stderr)
        return 2
    if not os.path.exists(args.cards):
        print(f"cards not found: {args.cards}", file=sys.stderr)
        return 2
    table = load_card_strength(args.cards)
    rows = measure_panel(args.panel, table)
    ok, msg = verdict(rows)
    if args.json:
        print(json.dumps({"panel": args.panel, "models": rows,
                          "signal_r2_min": SIGNAL_R2_MIN,
                          "pass": ok, "verdict": msg}, ensure_ascii=False, indent=2))
    else:
        print(f"panel: {args.panel}")
        print(f"{'model':<12}{'games':>6}{'R2(death)':>11}{'signal R2':>11}"
              f"{'  per-turn partial r (T8/T9/T10/T11)':>40}")
        for r in rows:
            pt = r.get("board_by_turn") or {}
            cells = " ".join(f"T{t}={pt[t]['partial_r']:+.3f}" for t in MEASURE_TURNS
                             if t in pt)
            print(f"{str(r['model']):<12}{r['games']:>6}{r.get('death_turn_r2', 0):>11.3f}"
                  f"{(r.get('signal_r2_max') or 0):>11.3f}   {cells}")
        print()
        print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
