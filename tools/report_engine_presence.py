"""引擎卡在场率：真人库 vs 模型面板，逐回合"战团里有没有那张引擎"。

动机（专家先验 C1 + `COMP_SHARED_CORES_20261113.md`）：33 个流派实际上围绕少数几张
T5–T6 引擎（铜须 12 个流派、巴琳达 7、罗杰斯 4、瑞文 4 …）。如果模型从来没把引擎放上场，
那"经济→成长"的回路根本没开始，讨论数值高低没有意义。

两个数据源，同一口径：
- 真人库 JSONL（`board_detail[].card_id`，去掉 `_G`）；
- 面板 panel.json 的 `boards_by_turn[seed][turn] = [card_id]`（已去 `_G`）。

例：
    python3 tools/report_engine_presence.py \
      --bank data/real_library/firestone-bank-6k/train-m6k.jsonl \
      --panel base=artifacts/runs/gap-mech-base-20261113/panel.json \
      --turns 8-13
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 被 ≥3 个流派列为核心的引擎卡（中文名 → 用途标签）。卡 id 从版本卡表里查。
DEFAULT_ENGINES = [
    "布莱恩·铜须",
    "巴琳达·斯通赫尔斯",
    "空军上将罗杰斯",
    "提图斯·瑞文戴尔",
    "骄傲的私掠者",
    "达卡莱附魔师",
    "热气球测绘员",
    "香蕉猛猿",
    "执念诗心龙",
]


def load_name_ids() -> Dict[str, str]:
    """卡名 → card_id（版本卡表，去掉 `_G` 变体）。"""
    out: Dict[str, str] = {}
    path = os.path.join(ROOT, "data", "p2", "cards_37.json")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = c.get("id")
            zh = c.get("name_zh")
            if isinstance(cid, str) and isinstance(zh, str) and not cid.endswith("_G"):
                out.setdefault(zh, cid)
    return out


def bank_presence(bank: str, ids: Set[str], lo: int, hi: int
                  ) -> Dict[int, Dict[str, float]]:
    per = collections.defaultdict(lambda: {
        "boards": 0, "with_engine": 0,
        "sum_with": 0.0, "sum_without": 0.0, "n_with": 0, "n_without": 0})
    for line in open(bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except BaseException:  # noqa: BLE001
            continue
        t = int(r.get("turn") or 0)
        if not (lo <= t <= hi):
            continue
        detail = r.get("board_detail")
        if not isinstance(detail, list) or not detail:
            continue
        d = per[t]
        d["boards"] += 1
        cids = set()
        strength = 0.0
        for m in detail:
            cid = m.get("card_id")
            if isinstance(cid, str):
                cids.add(cid[:-2] if cid.endswith("_G") else cid)
            strength += float(m.get("atk") or 0) + float(m.get("hp") or 0)
        if cids & ids:
            d["with_engine"] += 1
            d["sum_with"] += strength
            d["n_with"] += 1
        else:
            d["sum_without"] += strength
            d["n_without"] += 1
    out: Dict[int, Dict[str, float]] = {}
    for t, d in per.items():
        out[t] = {
            "boards": d["boards"],
            "with_engine": d["with_engine"] / max(d["boards"], 1),
            # 「一台引擎值多少战力」：有引擎 vs 无引擎的战团 Σ 均值
            "str_with": (d["sum_with"] / d["n_with"]) if d["n_with"] else 0.0,
            "str_without": (d["sum_without"] / d["n_without"]) if d["n_without"] else 0.0,
            "lift": ((d["sum_with"] / d["n_with"]) / (d["sum_without"] / d["n_without"])
                     if d["n_with"] and d["n_without"] and d["sum_without"] else 0.0),
        }
    return out


def panel_presence(panel: str, ids: Set[str], lo: int, hi: int
                   ) -> Dict[str, Dict[int, float]]:
    with open(panel, "r", encoding="utf-8-sig") as fh:
        doc = json.load(fh)
    out: Dict[str, Dict[int, float]] = {}
    for m in doc.get("models") or []:
        name = m.get("name")
        bbt = m.get("boards_by_turn") or {}
        per = collections.defaultdict(lambda: {"boards": 0, "with_engine": 0})
        for _seed, turns in bbt.items():
            if not isinstance(turns, dict):
                continue
            for t, cards in turns.items():
                try:
                    ti = int(t)
                except (TypeError, ValueError):
                    continue
                if not (lo <= ti <= hi):
                    continue
                d = per[ti]
                d["boards"] += 1
                if isinstance(cards, list) and (set(cards) & ids):
                    d["with_engine"] += 1
        out[str(name)] = {t: d["with_engine"] / max(d["boards"], 1) for t, d in per.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="")
    ap.add_argument("--panel", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--turns", default="8-13")
    ap.add_argument("--names", default="", help="逗号分隔的中文卡名（默认内置 9 张）")
    args = ap.parse_args()

    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)
    name_ids = load_name_ids()
    names = [n for n in (args.names.split(",") if args.names else DEFAULT_ENGINES) if n.strip()]
    ids = {name_ids[n.strip()] for n in names if n.strip() in name_ids}
    missing = [n for n in names if n.strip() not in name_ids]
    print(f"引擎卡 {len(ids)}/{len(names)} 命中卡表；未找到：{missing or '无'}")

    cols: List[tuple] = []
    bank_lift: Dict[int, Dict[str, float]] = {}
    if args.bank:
        b = bank_presence(args.bank, ids, lo, hi)
        bank_lift = b
        cols.append(("human-bank", {t: b[t]["with_engine"] for t in b}))
    for spec in args.panel:
        tag, _, path = spec.partition("=")
        for name, per in panel_presence(path, ids, lo, hi).items():
            cols.append((f"{tag}:{name}", per))

    print(f"{'turn':>5}" + "".join(f" {c[0]:>22}" for c in cols))
    for t in range(lo, hi + 1):
        line = f"{t:>5}"
        for _name, per in cols:
            v = per.get(t)
            if isinstance(v, dict):
                v = v.get("with_engine")
            line += f" {('%.3f' % v) if v is not None else '—':>22}"
        print(line)
    if args.bank and bank_lift:
        print()
        print("引擎的「战力溢价」（真人库）：有引擎 vs 无引擎的战团 Σ 均值")
        print(f"{'turn':>5} {'有引擎':>13} {'无引擎':>13} {'倍数':>7}")
        for t in sorted(bank_lift):
            d = bank_lift[t]
            print(f"{t:>5} {d['str_with']:>13.0f} {d['str_without']:>13.0f} "
                  f"{d['lift']:>7.2f}")
    print("\n（值 = 该回合战团里至少含 1 张引擎卡的局占比）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
