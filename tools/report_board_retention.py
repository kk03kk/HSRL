"""战团留存率：相邻回合之间有多少张卡留在场上（复利 vs 换血的直接度量）。

动机：§5.32 说差距是"成长速率"（真人每格 ×2.3/回合，模型 ×1.3/回合）；
§5.35 又证明"引擎在场"当回合只值 1.12–1.23× —— 所以引擎不是当回合的强度，
大概率是**复利的前提**：养过的随从要留在场上，属性才会滚起来。
如果模型每回合把战团换掉一半，它就永远在"重新从基础身材开始"。

口径：同一局里取相邻回合 `t` 与 `t+1` 的战团 card_id 集合（金卡剥 `_G`），
留存 = |交集| / max(|上一回合战团|, 1)；分母用上一回合格数，避免"多买一张"虚高。

- 真人库：按 `game_id` 分组的 `board_detail`；
- 面板：`boards_by_turn[seed][turn] = [card_id]`。

例：
    python3 tools/report_board_retention.py \
      --bank data/real_library/firestone-bank-6k/train-m6k.jsonl \
      --panel base=artifacts/runs/gap-mech-base-20261113/panel.json --turns 8-12
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Dict, List, Set


def _retention(pairs: collections.Counter, per_turn: collections.Counter) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for t in sorted(per_turn):
        n = per_turn[t]
        out[t] = (pairs[t] / n) if n else 0.0
    return out


def bank_retention(bank: str, lo: int, hi: int) -> Dict[int, float]:
    games: Dict[str, Dict[int, Set[str]]] = collections.defaultdict(dict)
    for line in open(bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except BaseException:  # noqa: BLE001
            continue
        gid = str(r.get("game_id") or "")
        if not gid:
            continue
        t = int(r.get("turn") or 0)
        if not (lo <= t <= hi + 1):
            continue
        detail = r.get("board_detail")
        if not isinstance(detail, list) or not detail:
            continue
        cids = set()
        for m in detail:
            cid = m.get("card_id")
            if isinstance(cid, str):
                cids.add(cid[:-2] if cid.endswith("_G") else cid)
        games[gid][t] = cids
    sums: collections.Counter = collections.Counter()
    cnt: collections.Counter = collections.Counter()
    for turns in games.values():
        for t in range(lo, hi + 1):
            a, b = turns.get(t), turns.get(t + 1)
            if not a or not b:
                continue
            sums[t] += len(a & b) / max(len(a), 1)
            cnt[t] += 1
    return _retention(sums, cnt)


def panel_retention(panel: str, lo: int, hi: int) -> Dict[str, Dict[int, float]]:
    with open(panel, "r", encoding="utf-8-sig") as fh:
        doc = json.load(fh)
    out: Dict[str, Dict[int, float]] = {}
    for m in doc.get("models") or []:
        name = m.get("name")
        bbt = m.get("boards_by_turn") or {}
        sums: collections.Counter = collections.Counter()
        cnt: collections.Counter = collections.Counter()
        for _seed, turns in bbt.items():
            if not isinstance(turns, dict):
                continue
            for t in range(lo, hi + 1):
                a = turns.get(str(t))
                b = turns.get(str(t + 1))
                if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
                    continue
                sums[t] += len(set(a) & set(b)) / max(len(set(a)), 1)
                cnt[t] += 1
        out[str(name)] = _retention(sums, cnt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="")
    ap.add_argument("--panel", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--turns", default="8-12", help="统计 t→t+1 的留存，t 取该区间")
    args = ap.parse_args()

    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)
    cols: List[tuple] = []
    if args.bank:
        cols.append(("human-bank", bank_retention(args.bank, lo, hi)))
    for spec in args.panel:
        tag, _, path = spec.partition("=")
        for name, per in panel_retention(path, lo, hi).items():
            cols.append((f"{tag}:{name}", per))

    print(f"{'retention':>10} " + "".join(f"{c[0]:>20}" for c in cols))
    for t in range(lo, hi + 1):
        line = f"  T{t}→{t + 1:<3} "[:11]
        for _name, per in cols:
            v = per.get(t)
            line += f"{('%.3f' % v) if v is not None else '—':>20}"
        print(line)
    print("\n（值 = 上一回合战团里有几张留在下一回合，分母=上一回合格数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
