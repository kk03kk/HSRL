#!/usr/bin/env python
"""真人战团库的**强度曲线**（按回合的分位表）：动态难度课程的绝对刻度。

为什么要这个工具
----------------
名次在本项目的对手场里已经饱和（D-052：固定死亡回合下 R²(board→rank)=0.000–0.004），
所以「模型有没有变强」必须换一把尺子。最干净的尺子是**真人战团本身的强度分布**：

- 每回合把真人库的战团按 Σ(atk+hp) 排序，取 p10/p25/p50/p75/p90；
- 模型「能在该回合打赢 p40 的真人战团」= 一个跨运行可比、与训练过程无关的绝对刻度；
- 难度控制器就工作在这张表上：把目标分位 q 换算成本回合的目标强度，
  再由采集侧从真人库里挑最接近该强度的**真实**战团注入（不缩放身材、不造数据）。

输出既打印（人看），也写 JSON（控制器/面板消费）。ASCII 只为了在任何 locale 下都能打。

用法::

    python tools/pool_strength_curve.py \
        --pool data/real_library/firestone-bank-6k/train-m6k.jsonl \
        --out artifacts/state/human_curve.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

#: 输出用的分位（与 human_board_pool.SAMPLE_PERCENTILES 对齐，另加 p75/p90 便于比值）。
QUANTILES = (("p10", 0.10), ("p25", 0.25), ("p33", 0.33), ("p50", 0.50),
             ("p67", 0.67), ("p75", 0.75), ("p90", 0.90))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="human board strength curve by turn")
    ap.add_argument("--pool", default="data/real_library/firestone-bank-6k/train-m6k.jsonl")
    ap.add_argument("--out", default="artifacts/state/human_curve.json")
    ap.add_argument("--turns", default="", help="comma list, default = all turns in pool")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.pool):
        print(f"[error] pool not found: {a.pool}", file=sys.stderr)
        return 2
    from hsbg.az.train.human_board_pool import HumanBoardPool

    pool = HumanBoardPool(a.pool, sample="auto")
    if not pool.kept:
        print(f"[error] pool empty after filters: {a.pool}", file=sys.stderr)
        return 2

    want = ([int(x) for x in a.turns.split(",") if x.strip()]
            if str(a.turns).strip() else list(pool.turns()))
    curve = {}
    print(f"pool={a.pool} boards={pool.boards if hasattr(pool, 'boards') else pool.kept} "
          f"turns={pool.turns()[0]}..{pool.turns()[-1]}")
    head = "turn      n" + "".join(f"{q:>9}" for q, _ in QUANTILES)
    print(head)
    print("-" * len(head))
    for t in want:
        if not pool.has(t):
            continue
        row = {"n": int(len(pool.index.get(t) or []))}
        for name, q in QUANTILES:
            v = pool.strength_quantile(t, q)
            row[name] = int(v) if v is not None else None
        curve[str(int(t))] = row
        print(f"{int(t):>4}{row['n']:>7}" + "".join(
            f"{(row[name] if row[name] is not None else -1):>9}" for name, _ in QUANTILES))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    payload = {"schema": "hsrl.human_curve/v1", "pool": a.pool,
               "boards": int(pool.kept), "quantiles": {k: q for k, q in QUANTILES},
               "curve": curve}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
