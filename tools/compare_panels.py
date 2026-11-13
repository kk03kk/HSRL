"""跨面板配对比较：两个 panel.json 里**同名模型**逐局配对（同 seed 才能配对）。

用途：有些评测变量是**全局选项**（例如 `--forced-tier-plan` 会作用到面板里的所有模型），
所以"开/关它"无法放进同一个面板。只要两次运行用同一 `--seed`（默认 930000），
就可以按 seed 把每一局对上，做配对 bootstrap。

例：
    python tools/compare_panels.py \
        --a off=artifacts/runs/gap-base-combat-20261113/panel.json \
        --b forced=artifacts/runs/gap-forced-expert-20261113/panel.json \
        --model base
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, List, Optional, Tuple


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def models_of(panel: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tag, models in (panel.get("panels") or {}).items():
        for name, m in (models or {}).items():
            out[name] = m
    for name, m in (panel.get("models") or {}).items():
        out[name] = m
    return out


def ranks(model: dict) -> Dict[str, float]:
    seq = model.get("rank_sequence") or []
    out: Dict[str, float] = {}
    for item in seq:
        if isinstance(item, dict):
            seed, rank = item.get("seed"), item.get("rank")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            seed, rank = item[0], item[1]
        else:
            continue
        if seed is None or rank is None:
            continue
        out[str(seed)] = float(rank)
    return out


def boot_ci(deltas: List[float], n_boot: int = 5000, seed: int = 20261113
            ) -> Tuple[float, float, float]:
    mean = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return mean, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="TAG=PATH（参照侧）")
    ap.add_argument("--b", required=True, help="TAG=PATH（对照侧）")
    ap.add_argument("--model", default="", help="模型名（默认取两边共有的第一个）")
    args = ap.parse_args()

    ta, _, pa = args.a.partition("=")
    tb, _, pb = args.b.partition("=")
    ma, mb = models_of(load(pa)), models_of(load(pb))
    shared = [n for n in ma if n in mb]
    if not shared:
        raise SystemExit(f"两个面板没有同名模型；A={sorted(ma)} B={sorted(mb)}")
    name = args.model or shared[0]
    if name not in ma or name not in mb:
        raise SystemExit(f"模型 {name!r} 不在两个面板里；A={sorted(ma)} B={sorted(mb)}")

    ra, rb = ranks(ma[name]), ranks(mb[name])
    paired = [k for k in ra if k in rb]
    if not paired:
        raise SystemExit("没有可配对的对局（seed 不同？）")
    deltas = [rb[k] - ra[k] for k in paired]
    mean, lo, hi = boot_ci(deltas)
    better = sum(1 for d in deltas if d < 0)
    worse = sum(1 for d in deltas if d > 0)

    print(f"model={name}  配对局数={len(paired)}")
    print(f"  {ta}: mean_rank={sum(ra[k] for k in paired) / len(paired):.4f}")
    print(f"  {tb}: mean_rank={sum(rb[k] for k in paired) / len(paired):.4f}")
    print(f"  Δ({tb} - {ta}) = {mean:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   ({tb} 更好 {better} 局 / 更差 {worse} 局)")
    verdict = ("显著更好" if hi < 0 else "显著更差" if lo > 0 else "不能区分（CI 跨 0）")
    print(f"  判定：{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
