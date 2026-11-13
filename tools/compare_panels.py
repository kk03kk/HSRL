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
    """面板里 ``models`` 是**列表**（每项带 ``name``），与 accept_gap_panel 同一口径。"""
    return {m.get("name"): m for m in (panel.get("models") or []) if isinstance(m, dict)}


def ranks(model: dict) -> Dict[str, float]:
    """逐局名次。

    面板里 ``rank_sequence`` 实际是**纯整数列表**（第 i 项 = seed ``base+i`` 那一局，
    见 ``evaluate.py`` 的 ``[int(r) for r in ranks]``），所以这里用**下标**当配对键；
    两次运行只要 `--seed` 与 `--games` 相同，下标就一一对应。
    （如果将来改成分数对/带 seed 的字典，这个函数也兼容。）
    """
    seq = model.get("rank_sequence") or []
    out: Dict[str, float] = {}
    for i, item in enumerate(seq):
        if isinstance(item, dict):
            seed, rank = item.get("seed"), item.get("rank")
            if seed is not None and rank is not None:
                out[str(seed)] = float(rank)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out[str(item[0])] = float(item[1])
        else:
            try:
                out[str(i)] = float(item)
            except (TypeError, ValueError):
                continue
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
