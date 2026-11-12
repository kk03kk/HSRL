#!/usr/bin/env python3
"""换牌/搜牌诊断：逐回合把模型的卖、买、刷新次数与真人 PMS 对齐。

**为什么需要逐回合**：`analyze_action_economy.py` 给的是每局总量，而
"模型不换牌"这个假设的关键是*什么时候*换。真人在第 10 回合以后卖牌量会抬起来
（腾位置给流派核心卡），如果模型在同样的回合段卖牌量塌下去，那就是信用分配问题
的指纹。逐回合也能把"活得久所以总量高"这个混淆去掉。

数据来源：
* 模型：``eval_human_gap`` 面板的 ``models[].action_mix_by_turn[t].mean``
  （键来自 ``hsbg/az/train/common.ACTION_SEGMENTS``：buy/sell/refresh/...）
* 真人：Firestone PMS 的 ``minionsSoldOverTurn`` / ``minionsBoughtOverTurn`` /
  ``rerollsOverTurn``，形如 ``[{"turn": n, "value": v}, ...]``

用法::

    PYTHONPATH=engine/python:. python tools/analyze_churn.py \
        --panel artifacts/runs/human-gap-strength-20261112/panel.json \
        --pms data/firestone_replays/refresh-20261112T031450Z/pms \
        --out artifacts/runs/human-baseline-20261112/churn.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HUMAN_FIELDS = {"sell": "minionsSoldOverTurn",
                "buy": "minionsBoughtOverTurn",
                "refresh": "rerollsOverTurn"}


def _per_turn(seq: Any) -> Dict[int, float]:
    """``[{"turn": n, "value": v}]`` -> {n: v}。兼容 dict 形式。"""
    out: Dict[int, float] = {}
    if isinstance(seq, list):
        for item in seq:
            if isinstance(item, dict):
                t = item.get("turn")
                v = item.get("value")
                if isinstance(t, int) and isinstance(v, (int, float)):
                    out[t] = float(v)
    elif isinstance(seq, dict):
        for k, v in seq.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def human_curves(pms_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """field -> turn -> {mean, n}。逐局等权（同一局同一回合只算一次）。"""
    acc: Dict[str, Dict[int, List[float]]] = {k: collections.defaultdict(list)
                                              for k in HUMAN_FIELDS}
    if not pms_dir.is_dir():
        return {}
    for path in sorted(pms_dir.glob("*.json")):
        try:
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            d = json.loads(raw)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for key, field in HUMAN_FIELDS.items():
            for t, v in _per_turn(d.get(field)).items():
                acc[key][t].append(v)
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for key, by_turn in acc.items():
        out[key] = {t: {"mean": statistics.fmean(v), "n": len(v)}
                    for t, v in sorted(by_turn.items()) if len(v) >= 20}
    return out


def model_curves(panel: Dict[str, Any], key: str) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for m in panel.get("models") or []:
        mix = m.get("action_mix_by_turn") or {}
        curve = {}
        for t, row in mix.items():
            v = (row.get("mean") or {}).get(key)
            if isinstance(v, (int, float)):
                curve[int(t)] = float(v)
        if curve:
            out[m["name"]] = dict(sorted(curve.items()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="逐回合换牌/搜牌诊断")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--pms", required=True)
    ap.add_argument("--turns", nargs="+", type=int, default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    human = human_curves(Path(args.pms))
    if not human:
        print("ERROR: no human PMS series found", file=sys.stderr)
        return 2

    payload: Dict[str, Any] = {"schema": "hsai.churn/v1", "turns": args.turns,
                               "sections": {}}

    for key in ("sell", "refresh", "buy"):
        hcurve = human.get(key) or {}
        mcurves = model_curves(panel, key)
        print(f"===== {key} per turn =====")
        head = f"{'turn':>4} {'human':>7}"
        for name in mcurves:
            head += f" {name[:14]:>15}"
        print(head)
        rows = {}
        for t in args.turns:
            h = (hcurve.get(t) or {}).get("mean")
            line = f"{t:>4} {('%.2f' % h) if h is not None else '-':>7}"
            cell = {"human": h}
            for name, curve in mcurves.items():
                v = curve.get(t)
                cell[name] = v
                ratio = (v / h) if (v is not None and h) else None
                cell[f"{name}_ratio"] = ratio
                line += f" {(('%.2f (%.2fx)' % (v, ratio)) if ratio else
                            (('%.2f' % v) if v is not None else '-')):>15}"
            print(line)
            rows[str(t)] = cell
        payload["sections"][key] = rows
        print()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
