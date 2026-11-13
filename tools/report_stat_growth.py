"""战团的增长来自哪里：**养大老随从** 还是 **换成新随从**。

问题（§5.47）：真人 T10→T11 战团 Σ 从 ~1958 涨到 ~4400（2.25×），模型只有 1.3×/回合。
两条可能的路径，处方完全不同：
- 「养」：留在场上的随从属性自己长大（磁力/宝石/法术/成长引擎）→ 要教"投资"；
- 「换」：把小的换成天生更大的（升本买高星）→ 要教"升本/找质量"。

做法：对同一局（`perspective=self`）的相邻两回合，按 `card_id`（剥 `_G`）配对：
- `kept`：两回合都在的随从 → 看它们的 Σ 平均涨了多少；
- `new`：只在后一回合出现的 → 看它们的平均 Σ（对比被换掉的旧的）；
并给出"总增长中有多少来自 kept 的增重 vs 新随从的净贡献"。

例：
    python3 tools/report_stat_growth.py \
      --bank data/real_library/firestone-bank-6k/train-m6k.jsonl --turns 9-12
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Dict, List, Tuple


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--turns", default="9-12", help="统计 t→t+1（t 取该区间）")
    args = ap.parse_args()
    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)

    games: Dict[str, Dict[int, Dict[str, float]]] = collections.defaultdict(dict)
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
        gid = str(r.get("game_id") or "")
        detail = r.get("board_detail")
        if not gid or not isinstance(detail, list) or not detail:
            continue
        t = int(r.get("turn") or 0)
        board: Dict[str, float] = {}
        for m in detail:
            cid = m.get("card_id")
            if not isinstance(cid, str):
                continue
            cid = cid[:-2] if cid.endswith("_G") else cid
            board[cid] = float(m.get("atk") or 0) + float(m.get("hp") or 0)
        games[gid][t] = board

    acc: Dict[int, Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for turns in games.values():
        for t in range(lo, hi + 1):
            a, b = turns.get(t), turns.get(t + 1)
            if not a or not b:
                continue
            kept = [k for k in a if k in b]
            gone = [k for k in a if k not in b]
            new = [k for k in b if k not in a]
            d = acc[t]
            d["total_before"].append(sum(a.values()))
            d["total_after"].append(sum(b.values()))
            if kept:
                d["kept_gain"].append(sum(b[k] - a[k] for k in kept) / len(kept))
            if new:
                d["new_mean"].append(sum(b[k] for k in new) / len(new))
            if gone:
                d["gone_mean"].append(sum(a[k] for k in gone) / len(gone))
            d["n_kept"].append(float(len(kept)))
            d["n_new"].append(float(len(new)))

    def mean(v: List[float]) -> float:
        return sum(v) / len(v) if v else 0.0

    print(f"bank={args.bank} turns={lo}..{hi}（perspective=self）")
    print(f"{'t->t+1':>7} {'n':>5} {'Σ前':>8} {'Σ后':>8} {'倍数':>6} "
          f"{'留存数':>7} {'新进数':>7} {'留存的平均增重':>15} {'新进平均Σ':>10} {'换掉平均Σ':>10}")
    for t in sorted(acc):
        d = acc[t]
        n = len(d["total_before"])
        before, after = mean(d["total_before"]), mean(d["total_after"])
        print(f"{t:>3}->{t + 1:<3} {n:>5} {before:>8.0f} {after:>8.0f} "
              f"{(after / before if before else 0):>6.2f} "
              f"{mean(d['n_kept']):>7.2f} {mean(d['n_new']):>7.2f} "
              f"{mean(d['kept_gain']):>15.1f} {mean(d['new_mean']):>10.1f} "
              f"{mean(d['gone_mean']):>10.1f}")
    print("\n读法：『留存的平均增重』大 ⇒ 靠养；『新进平均Σ』远高于『换掉平均Σ』⇒ 靠换质量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
