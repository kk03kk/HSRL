"""真人库内部检验：**动作吞吐 vs 战团强度**是否正相关。

动机（§5.39）：真人 T12 每回合 play 22 / sell 10.4，模型只有 8 / 1.2。
但在把它当靶子之前，必须先问：**在真人的对局里，"循环多"是否真的伴随"战团更强"？**
如果同一个回合里动作多的玩家并没有更强，那"多循环"就只是个风格，不是因果。

做法：按 `game_id` 取 `perspective=self` 的帧，在指定回合用
「该回合 `play` + `sell` 次数」与「同回合战团 Σ(atk+hp)」做 Spearman 秩相关，
并按动作量分箱看强度中位数（分箱表比单一相关系数更能看出形状）。
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from typing import Dict, List, Tuple


def spearman(xs: List[float], ys: List[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(xs) < 8:
        return float("nan")
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--turns", default="10-12", help="在这些回合上做相关（含端点）")
    args = ap.parse_args()
    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)

    rows: Dict[int, List[Tuple[float, float]]] = collections.defaultdict(list)
    for line in open(args.bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except BaseException:  # noqa: BLE001
            continue
        if str(r.get("perspective") or "") != "self":
            continue
        t = int(r.get("turn") or 0)
        if not (lo <= t <= hi):
            continue
        counts = r.get("action_counts") or {}
        if not isinstance(counts, dict):
            continue
        acts = float(counts.get("play_minion", 0) + counts.get("play_spell", 0)
                     + counts.get("sell", 0))
        detail = r.get("board_detail")
        if not isinstance(detail, list) or not detail:
            continue
        strength = sum(float(m.get("atk") or 0) + float(m.get("hp") or 0) for m in detail)
        rows[t].append((acts, strength))

    print(f"bank={args.bank} turns={lo}..{hi}（perspective=self）")
    print(f"{'turn':>5} {'n':>6} {'spearman(动作, 强度)':>22}  分箱中位强度（动作量四分位）")
    for t in range(lo, hi + 1):
        pairs = rows.get(t) or []
        if len(pairs) < 8:
            continue
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        pairs.sort(key=lambda p: p[0])
        n = len(pairs)
        q = [pairs[: n // 4], pairs[n // 4: n // 2], pairs[n // 2: 3 * n // 4],
             pairs[3 * n // 4:]]
        med = [statistics.median([p[1] for p in grp]) if grp else 0 for grp in q]
        print(f"{t:>5} {n:>6} {rho:>22.3f}  "
              + " / ".join(f"{m:>7.0f}" for m in med))
    print("\n读法：若 rho 明显为正、且分箱中位强度单调上升，则'循环多'与'更强'同向；"
          "若 rho≈0，则吞吐只是风格，不是强度的前提。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
