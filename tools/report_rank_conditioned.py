"""条件名次：把"死得早"与"活到后期打不过"分开。

问题（§5.46）：所有课程臂名次都在噪声内、等级更低、T10 战斗胜率更高但 T12 更差。
这既可能是"死得太早"（根本没机会运营后期），也可能是"活到后期也打不过"。
两者的处方完全不同，所以必须分开量。

做法：面板里 `boards_by_turn[seed]` 的**最大回合**就是该局活到的回合；
把每局按"是否活到 T"分组，算各组的 `mean_rank`（名次越低越好）。
若两组的差距主要来自"能不能活到"，说明瓶颈是存活；若"活到 T12 的子集"里模型仍明显更差，
说明瓶颈是中后期运营本身。

例：
    python3 tools/report_rank_conditioned.py \
      --panel base=artifacts/runs/gap-n1m2b-mid-20261113/panel.json --turn 12
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def per_game(model: dict) -> Dict[str, Dict[str, float]]:
    """seed -> {rank, max_turn}。"""
    bbt = model.get("boards_by_turn") or {}
    seq = model.get("rank_sequence") or []
    ranks = [float(x) for x in seq] if seq and not isinstance(seq[0], dict) else []
    out: Dict[str, Dict[str, float]] = {}
    for i, (seed, turns) in enumerate(sorted(bbt.items())):
        mt = 0
        if isinstance(turns, dict) and turns:
            try:
                mt = max(int(t) for t in turns)
            except (TypeError, ValueError):
                mt = 0
        rank = ranks[i] if i < len(ranks) else float("nan")
        out[str(seed)] = {"rank": rank, "max_turn": float(mt)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--turn", type=int, default=12)
    args = ap.parse_args()
    T = int(args.turn)

    print(f"条件名次（活到 T{T} 与否）  名次越低越好")
    print(f"{'model':>16} {'games':>6} {'mean_rank':>10} "
          f"{'活到 T%d 的局' % T:>12} {'其中 mean_rank':>14} "
          f"{'没活到的局':>10} {'其中 mean_rank':>14}")
    for spec in args.panel:
        tag, _, path = spec.partition("=")
        doc = load(path)
        for m in doc.get("models") or []:
            pg = per_game(m)
            if not pg:
                continue
            alive = [v for v in pg.values() if v["max_turn"] >= T]
            dead = [v for v in pg.values() if v["max_turn"] < T]
            allr = sum(v["rank"] for v in pg.values()) / max(len(pg), 1)
            ar = (sum(v["rank"] for v in alive) / len(alive)) if alive else float("nan")
            dr = (sum(v["rank"] for v in dead) / len(dead)) if dead else float("nan")
            print(f"{tag + ':' + str(m.get('name')):>16} {len(pg):>6} {allr:>10.3f} "
                  f"{len(alive):>12} {ar:>14.3f} {len(dead):>10} {dr:>14.3f}")
    print("\n读法：若'活到 T'子集里名次差不多、差别全在能不能活到 → 瓶颈是存活；"
          "若活到 T 的子集里仍明显更差 → 瓶颈是中后期运营。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
