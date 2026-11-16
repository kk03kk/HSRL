#!/usr/bin/env python3
"""把 N 份「分片评测」的结果合并成一份与串行完全等价的面板。

设计原则：**不重写任何聚合逻辑**。每个分片用 ``workflows/eval_human_gap.py`` 的
``--dump-rows 1`` 落盘了 ``rows.jsonl.gz``（喂给聚合的原始 rows）和 ``extra.json``
（outcome / rank_sequence / combat_exposure）。合并时把这些原始输入拼起来，再调用
**同一个文件里的同一批函数**重算 —— 等价性由构造保证，不靠人工对齐口径。

用法：
    python tools/merge_panels.py --out <dir>/panel.json <shard0_dir> <shard1_dir> ...
（每个参数是包含 rows.jsonl.gz 的分片目录；也接受直接给 panel.json，此时只做逐局部分拼接）
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.eval_human_gap import (  # noqa: E402
    _counts_all, _counts_by_turn, _gold_by_turn, _opponents_by_turn, _strength_by_seed,
    _trace, action_mix_by_turn, boards_by_turn, combat_by_turn, compare,
    economy_curves, final_boards, mechanics_by_turn, retained_gain_by_turn, survival,
)


def load_rows(shard: Path) -> List[Dict[str, Any]]:
    p = shard / "rows.jsonl.gz"
    if not p.is_file():
        return []
    out = []
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_extra(shard: Path) -> Dict[str, Any]:
    p = shard / "extra.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def merge_outcome(extras: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按逐局名次重算名次类指标（与 evaluate.py 同口径）；其余计数直接累加。"""
    ranks: List[int] = []
    for e in extras:
        for r in e.get("rank_sequence") or []:
            try:
                ranks.append(int(r))
            except (TypeError, ValueError):
                pass
    games = sum(int(e.get("games") or 0) for e in extras)
    decisions = sum(int((e.get("outcome") or {}).get("decisions") or 0) for e in extras)
    out = {
        "games": games,
        "illegal_actions": sum(int((e.get("outcome") or {}).get("illegal_actions") or 0) for e in extras),
        "truncated_games": sum(int((e.get("outcome") or {}).get("truncated_games") or 0) for e in extras),
        "decisions": decisions,
        "decisions_per_game": round(decisions / games, 1) if games else None,
    }
    ok = [r for r in ranks if 1 <= r <= 8]
    if ok:
        out["mean_rank"] = round(sum(ok) / len(ok), 4)
        out["top4_rate"] = round(sum(1 for r in ok if r <= 4) / len(ok), 4)
        out["win_rate"] = round(sum(1 for r in ok if r == 1) / len(ok), 4)
    return out


def merge(shard_dirs: List[Path], out_path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    extras: List[Dict[str, Any]] = []
    exposure: List[Dict[str, Any]] = []
    hist: collections.Counter = collections.Counter()
    ranks: List[int] = []
    names = set()
    for d in shard_dirs:
        rows.extend(load_rows(d))
        e = load_extra(d)
        if e:
            extras.append(e)
            names.add(str(e.get("name")))
            exposure.extend(e.get("combat_exposure") or [])
            for k, v in (e.get("action_hist") or {}).items():
                hist[str(k)] += int(v)
            for r in e.get("rank_sequence") or []:
                try:
                    ranks.append(int(r))
                except (TypeError, ValueError):
                    pass
    if not rows:
        raise SystemExit("no rows.jsonl.gz found in any shard dir")

    name = next(iter(names)) if len(names) == 1 else "+".join(sorted(names))
    entry: Dict[str, Any] = {
        "name": name,
        "games": len(_strength_by_seed(rows)) or None,
        "shards": [str(d) for d in shard_dirs],
        "outcome": merge_outcome(extras),
        "rank_sequence": ranks or None,
        "action_hist": dict(hist.most_common()) or None,
        "survival": survival(rows),
        "curves": economy_curves(rows),
        "action_mix_by_turn": action_mix_by_turn(rows),
        "combat_by_turn": combat_by_turn(exposure),
        "mechanics_by_turn": mechanics_by_turn(rows),
        "retained_gain_by_turn": retained_gain_by_turn(rows),
        "final_boards": final_boards(rows),
        "boards_by_turn": boards_by_turn(rows),
        "strength_by_seed": _strength_by_seed(rows),
        "trace": _trace(rows),
        "trinkets_by_turn": _counts_by_turn(rows, "trinkets"),
        "opponents_by_turn": _opponents_by_turn(exposure),
        "gold_by_turn": _gold_by_turn(rows),
        "heroes": _counts_all(rows, "hero"),
        "rows": len(rows),
    }
    doc = {"models": [entry], "merged_from": [str(d) for d in shard_dirs],
           "merge": {"schema": "hsai.panel.merge/v1", "shards": len(shard_dirs)}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    cv = entry["curves"]
    print("merged %-16s games=%-5s rows=%-6d  T12=%-7s T16=%-7s rank=%s"
          % (name, entry["games"], len(rows),
             (cv.get("12") or {}).get("strength_p50"),
             (cv.get("16") or {}).get("strength_p50"),
             (entry["outcome"] or {}).get("mean_rank")))
    print("wrote", out_path)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+", help="分片目录（含 rows.jsonl.gz）或 panel.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dirs = []
    for s in a.shards:
        p = Path(s)
        dirs.append(p.parent if p.is_file() else p)
    merge(dirs, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
