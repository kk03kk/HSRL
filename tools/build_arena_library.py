#!/usr/bin/env python
"""构建"校准战团库"：让每一回合的对手强度 = 当前模型该回合强度 × 系数。

设计（用户口径）
----------------
- **前期（模型还没上 5–6 本）略弱于模型**：×0.85。目的是鼓励升本、打经济，
  不要在低本混日子（低本混日子换不来战力，却会白丢血）。
- **第 9 回合回正**：×1.00（此时模型通常刚上 5 本，给一个势均力敌的检查点）。
- **后期只比模型高一点、且缓慢上升**：×1.05 → ×1.15，每回合只加 0.02~0.03。
  绝不出现"一个回合跳 3–4 倍"那种模型永远赢不了的曲线。

实现要点
--------
1. 对手**不缩放身材**：每条都是真人库里的**真实战团**，只是在强度带里挑出来的。
2. 强度带 = ``[target×(1-jitter), target×(1+jitter)]``，默认 jitter=0.10。
   同一回合内随机抽，得到"同一强度目标 + 形状抖动"，避免模型过拟合单一形状。
3. **高回合允许借用更早回合的真实战团**（弱玩家也能活到 15 回合），
   并把 ``turn`` 改写成目标回合、原回合存进 ``_src_turn`` —— 采集侧按 turn 分桶，
   不改写就查不到该回合的对手。
4. 输出可直接当 ``--human-boards`` 用（保留 ``board_detail``，full 保真度不丢）。

用法::

    python tools/build_arena_library.py \
        --pool data/real_library/firestone-bank-6k/train-m6k.jsonl \
        --panel artifacts/runs/human-gap-all5-20261112/panel.json --panel-model final14858 \
        --out-dir artifacts/libraries/arena-cal-v1 --per-turn 96
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

#: 每回合的目标强度系数（相对"当前模型该回合战斗开始板面中位数"）。
#: 前段 <1（略弱，鼓励升本），第 9 回合回正，后段缓慢 >1（只高一点点）。
DEFAULT_MULT: Dict[int, float] = {
    1: 0.85, 2: 0.85, 3: 0.85, 4: 0.85, 5: 0.85, 6: 0.85, 7: 0.85, 8: 0.90,
    9: 1.00, 10: 1.05, 11: 1.08, 12: 1.10, 13: 1.12, 14: 1.12, 15: 1.15, 16: 1.15,
}


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_model_curve(panel_path: str, model_name: str) -> Dict[int, float]:
    """从面板 JSON 里取某模型"战斗开始时板面中位数"的逐回合曲线。"""
    d = json.load(open(panel_path, encoding="utf-8"))
    for m in d.get("models") or []:
        if m.get("name") != model_name:
            continue
        rows = (m.get("vs_human") or {}).get("per_turn") or []
        out = {}
        for r in rows:
            t, v = _num(r.get("turn")), _num(r.get("model_strength_p50"))
            if t is not None and v is not None:
                out[int(t)] = float(v)
        return out
    raise SystemExit(f"model {model_name!r} not found in {panel_path}")


def load_pool(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def pick_boards(rows: List[Dict[str, Any]], turn: int, target: float,
                jitter: float, per_turn: int, rng: random.Random,
                min_window: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """为目标回合挑真实战团；不够就逐步放宽强度带、再放宽回合窗口。"""
    lo, hi = target * (1.0 - jitter), target * (1.0 + jitter)
    band, window = (lo, hi), min_window
    for _ in range(8):
        cands = []
        for r in rows:
            st, src = _num(r.get("strength")), _num(r.get("turn"))
            if st is None or src is None:
                continue
            if not (turn - window <= int(src) <= turn):
                continue
            if not (band[0] <= st <= band[1]):
                continue
            cands.append(r)
        if len(cands) >= max(8, per_turn // 4):
            break
        if window < turn + 8:
            window += 2            # 先放宽窗口（借更早回合的真实战团）
        else:
            band = (band[0] * 0.9, band[1] * 1.1)   # 再放宽强度带
    if not cands:
        return [], {"turn": turn, "target": round(target, 1), "n": 0,
                    "band": [round(band[0], 1), round(band[1], 1)], "window": window}
    rng.shuffle(cands)
    chosen = cands[:per_turn]
    out = []
    for r in chosen:
        row = dict(r)
        row["_src_turn"] = int(r.get("turn") or 0)
        row["_target_turn"] = int(turn)
        row["_target_strength"] = round(float(target), 1)
        row["turn"] = int(turn)      # 采集侧按 turn 分桶，必须改写
        out.append(row)
    vals = sorted(_num(r.get("strength")) for r in out)
    q = lambda p: vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]
    return out, {"turn": turn, "target": round(target, 1), "n": len(out),
                 "band": [round(band[0], 1), round(band[1], 1)], "window": window,
                 "realized_p10": q(0.10), "realized_p50": q(0.50),
                 "realized_p90": q(0.90),
                 "src_turn_min": min(r["_src_turn"] for r in out),
                 "src_turn_max": max(r["_src_turn"] for r in out)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="build calibrated warband library")
    ap.add_argument("--pool", default="data/real_library/firestone-bank-6k/train-m6k.jsonl")
    ap.add_argument("--panel", default="artifacts/runs/human-gap-all5-20261112/panel.json")
    ap.add_argument("--panel-model", default="final14858")
    ap.add_argument("--out-dir", default="artifacts/libraries/arena-cal-v1")
    ap.add_argument("--per-turn", type=int, default=96)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--mult", default="", help="override e.g. 2:0.85,9:1.0,12:1.1")
    a = ap.parse_args(argv)

    mult = dict(DEFAULT_MULT)
    if str(a.mult).strip():
        for part in str(a.mult).split(","):
            k, _, v = part.partition(":")
            if k.strip().isdigit() and v.strip():
                mult[int(k)] = float(v)

    curve = load_model_curve(a.panel, a.panel_model)
    rows = load_pool(a.pool)
    print(f"pool rows={len(rows)}  baseline model={a.panel_model}  turns in curve={len(curve)}")

    rng = random.Random(int(a.seed))
    os.makedirs(a.out_dir, exist_ok=True)
    out_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for t in sorted(mult):
        base = curve.get(t)
        if base is None:
            summary.append({"turn": t, "target": None, "n": 0, "note": "no model baseline"})
            continue
        target = base * float(mult[t])
        got, info = pick_boards(rows, t, target, float(a.jitter), int(a.per_turn), rng)
        out_rows.extend(got)
        summary.append(info)

    path = os.path.join(a.out_dir, "boards.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta = {"schema": "hsai.arena_calibrated_library/v1",
            "source_pool": a.pool, "baseline_panel": a.panel,
            "baseline_model": a.panel_model, "jitter": float(a.jitter),
            "per_turn_cap": int(a.per_turn), "mult": {str(k): v for k, v in sorted(mult.items())},
            "boards": len(out_rows), "per_turn": {str(s["turn"]): s for s in summary}}
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")

    print()
    print("=== calibrated library: per-turn strength (real boards, no stat scaling) ===")
    print(f"{'turn':>4} {'model_p50':>10} {'mult':>6} {'target':>8} {'n':>4} "
          f"{'realized_p10':>13} {'realized_p50':>13} {'realized_p90':>13} {'src turns':>12}")
    for s in summary:
        t = s["turn"]
        base = curve.get(t)
        if base is None:
            print(f"{t:>4} {'-':>10} {mult.get(t, 0):>6} {'-':>8} {0:>4}")
            continue
        src = f"{s.get('src_turn_min')}..{s.get('src_turn_max')}" if s.get("n") else "-"
        f = lambda v: ("-" if v is None else f"{v:.0f}")
        print(f"{t:>4} {base:>10.0f} {mult.get(t, 0):>6.2f} {s['target']:>8} {s['n']:>4} "
              f"{f(s.get('realized_p10')):>13} {f(s.get('realized_p50')):>13} "
              f"{f(s.get('realized_p90')):>13} {src:>12}")
    print()
    print(f"wrote {path}  ({len(out_rows)} boards)  + summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
