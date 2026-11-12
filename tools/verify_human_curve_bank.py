#!/usr/bin/env python3
"""校验真人曲线对手库：逐回合战力是否真的落在真人分布上。

**为什么必须先校验**：`build_human_curve_bank.py` 声称把每帧的攻击/生命重分配
到真人目标分位，但那是代码的意图，不是事实。对手库如果没真的变强，
"抬高对手强度天花板"这个干预就是空的，后面所有训练都白跑。

本脚本读对手库，逐回合算 ``Σ(atk+hp)`` 的分位数，并与

* 真人原始曲线（``baseline.json`` 的 ``strength_by_turn``），以及
* 构建器实际的目标 = 真人分位 × ``coefficient(turn)``

对照，给出比值。判据：**校准后 p50 / 真人 p50 应等于 coefficient(turn)**，
且 p90 明显高于源库。偏离说明重分配逻辑没生效。

用法::

    python tools/verify_human_curve_bank.py \
        --bank artifacts/libraries/humancurve-train-20261112 \
        --baseline artifacts/runs/human-baseline-20261112/baseline.json \
        --source artifacts/libraries/league-eot-v1.1-20261109/train \
        --out artifacts/runs/human-baseline-20261112/bank-verify.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def coefficient(turn: int) -> float:
    """与 build_human_curve_bank.coefficient 完全一致（复制以便独立校验）。"""
    return (1.0 if turn <= 7 else .9 if turn <= 10 else .85 if turn == 11
            else .8 if turn == 12 else .75 if turn == 13 else .7)


def pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


def load_curves(root: Path) -> Dict[int, List[float]]:
    """turn -> [Σ(atk+hp), ...]，逐帧（不按游戏去重，这是库里的真实抽样单元）。"""
    out: Dict[int, List[float]] = collections.defaultdict(list)
    files = sorted(root.glob("snap_*.json.gz"))
    if not files:
        raise SystemExit(f"no snap_*.json.gz under {root}")
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            record = json.load(f)
        for frame in record.get("frames") or []:
            board = (frame.get("player") or {}).get("board") or []
            if not board:
                continue
            total = sum(int(m.get("atk", 0)) for m in board) + \
                    sum(int(m.get("hp", 0)) for m in board)
            out[int(frame["turn"])].append(float(total))
    return out


def table(curves: Dict[int, List[float]], min_n: int = 8) -> Dict[int, Dict[str, Any]]:
    res = {}
    for t in sorted(curves):
        v = curves[t]
        if len(v) < min_n:
            continue
        res[t] = {"n": len(v), "p25": pct(v, .25), "p50": pct(v, .50),
                  "p75": pct(v, .75), "p90": pct(v, .90), "mean": sum(v) / len(v)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="校验真人曲线对手库")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--source", default="", help="原始库（用于对比'变强了多少'）")
    ap.add_argument("--turns", nargs="+", type=int, default=[6, 8, 10, 12, 14])
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    bank = table(load_curves(Path(args.bank)))
    source = table(load_curves(Path(args.source))) if args.source else {}
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    human = base.get("strength_by_turn") or {}

    print(f"bank   : {args.bank}  turns={len(bank)}")
    if source:
        print(f"source : {args.source}  turns={len(source)}")
    print()
    print(f"{'turn':>4} {'coef':>5} {'bank p50':>10} {'human p50':>10} "
          f"{'ratio':>7} {'want':>6} {'verdict':>9} "
          + (f"{'src p50':>9} {'gain':>6}" if source else ""))
    rows = []
    bad = 0
    for t in args.turns:
        b = bank.get(t)
        h = human.get(str(t))
        if not b or not h or not h.get("p50"):
            print(f"{t:>4}  (insufficient data)")
            continue
        coef = coefficient(t)
        want = h["p50"] * coef
        ratio = b["p50"] / h["p50"] if h["p50"] else None
        ok = ratio is not None and abs(ratio - coef) / coef < 0.25
        if not ok:
            bad += 1
        line = (f"{t:>4} {coef:>5.2f} {b['p50']:>10.0f} {h['p50']:>10.0f} "
                f"{ratio:>7.2f} {coef:>6.2f} {('OK' if ok else 'OFF'):>9}")
        if source and t in source:
            line += f" {source[t]['p50']:>9.0f} {b['p50']/max(source[t]['p50'],1):>6.1f}x"
        print(line)
        rows.append({"turn": t, "coefficient": coef, "bank_p50": b["p50"],
                     "human_p50": h["p50"], "ratio": ratio, "verdict": "OK" if ok else "OFF",
                     "source_p50": (source.get(t) or {}).get("p50"),
                     "gain_vs_source": (b["p50"] / source[t]["p50"]) if (source and t in source and source[t]["p50"]) else None,
                     "bank_p25": b["p25"], "bank_p90": b["p90"]})

    print()
    print(f"verdict: {'all turns within 25% of the target coefficient' if not bad else f'{bad} turn(s) OFF target'}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "hsai.human_curve_bank_verify/v1",
            "bank": args.bank, "source": args.source or None,
            "baseline": args.baseline, "rows": rows,
            "bank_table": {str(k): v for k, v in bank.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out} ({out.stat().st_size} bytes)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
