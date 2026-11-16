#!/usr/bin/env python3
"""离线等价性测试：分片合并出来的面板必须与「不分片直接算」逐位相同。

不需要 GPU、不需要引擎：直接构造合成 rows，写成分片目录（rows.jsonl.gz + extra.json），
跑 tools/merge_panels.py，再拿同一批 rows 直接调用 eval_human_gap 的聚合函数比对。
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.merge_panels import merge  # noqa: E402
from workflows.eval_human_gap import (  # noqa: E402
    _counts_all, _counts_by_turn, _gold_by_turn, _strength_by_seed,
    action_mix_by_turn, boards_by_turn, combat_by_turn, economy_curves,
    final_boards, mechanics_by_turn, retained_gain_by_turn, survival,
)

rng = random.Random(7)
CARDS = ["BG26_152", "BG36_366", "BG36_367", "BG29_611", "BG32_111", "BG36_366_G"]
RACES = ["MECHANICAL", "BEAST", "DEMON"]


def make_rows(seeds):
    rows = []
    for seed in seeds:
        for turn in range(1, 19):
            n = rng.randint(1, 7)
            board = []
            for pos in range(n):
                cid = rng.choice(CARDS)
                board.append({"position": pos, "card_id": cid, "name": cid,
                              "atk": rng.randint(1, 300), "hp": rng.randint(1, 300),
                              "golden": cid.endswith("_G"),
                              "keywords": rng.sample(["TAUNT", "DIVINE_SHIELD", "REBORN"], rng.randint(0, 2)),
                              "tier": rng.randint(1, 6)})
            rows.append({
                "seed": seed, "turn": turn, "tier": min(6, 1 + turn // 3),
                "gold": rng.randint(0, 12), "health": rng.randint(1, 40),
                "hand_n": rng.randint(0, 10), "actions": rng.randint(1, 40),
                "eot_attack": sum(m["atk"] for m in board),
                "eot_board_hp": sum(m["hp"] for m in board),
                "eot_board": board, "board": board,
                "board_stats": {m["card_id"]: [m["atk"], m["hp"]] for m in board},
                "hero": rng.randint(1, 143), "trinkets": rng.choice([None, [], [3037], [2707, 3043]]),
                "rewards": None, "consecutive_repeats": rng.randint(0, 5),
                "gold_events": [{"action": 8, "kind": "buy", "before": 5, "after": 2, "net_change": -3}],
            })
    return rows


def exposure(seeds):
    return [{"seed": s, "turn": t, "opponent": o, "ghost": False, "mirror_applied": True,
             "source_strength": 100.0 + 10 * t, "source_turn": t}
            for s in seeds for t in range(6, 19) for o in range(1, 8)]


def write_shard(d: Path, rows, extra):
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "rows.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (d / "extra.json").write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")


def main():
    seeds = list(range(930000, 930032))
    rows = make_rows(seeds)
    exp = exposure(seeds)
    ranks = [rng.randint(1, 8) for _ in seeds]
    half = len(seeds) // 2
    shards = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, (lo, hi) in enumerate(((0, half), (half, len(seeds)))):
            src = set(seeds[lo:hi])            # 严格不相交
            sr = [r for r in rows if r["seed"] in src]
            se = [e for e in exp if e["seed"] in src]
            extra = {"name": "t", "outcome": {"decisions": 1000, "illegal_actions": 0,
                                              "truncated_games": 0, "decisions_per_game": 31.2},
                     "rank_sequence": ranks[lo:hi], "action_hist": {"buy": 10, "roll": 3},
                     "combat_exposure": se, "seconds": 1.0, "games": hi - lo, "seed": 930000}
            write_shard(td / ("shard%d" % i), sr, extra)
            shards.append(td / ("shard%d" % i))
        doc = merge(shards, td / "panel.json")
    got = doc["models"][0]

    # 直接对全量 rows 算一遍
    want = {
        "curves": economy_curves(rows),
        "survival": survival(rows),
        "action_mix_by_turn": action_mix_by_turn(rows),
        "mechanics_by_turn": mechanics_by_turn(rows),
        "retained_gain_by_turn": retained_gain_by_turn(rows),
        "final_boards": final_boards(rows),
        "boards_by_turn": boards_by_turn(rows),
        "strength_by_seed": _strength_by_seed(rows),
        "trinkets_by_turn": _counts_by_turn(rows, "trinkets"),
        "gold_by_turn": _gold_by_turn(rows),
        "heroes": _counts_all(rows, "hero"),
        "combat_by_turn": combat_by_turn(exp),
    }
    bad = []
    for k, v in want.items():
        if json.dumps(got.get(k), sort_keys=True, ensure_ascii=False) != \
           json.dumps(v, sort_keys=True, ensure_ascii=False):
            bad.append(k)
    o = got.get("outcome") or {}
    if o.get("mean_rank") != round(sum(ranks) / len(ranks), 4):
        bad.append("outcome.mean_rank")
    print("merged keys ok:", sorted(k for k in want if k not in bad))
    if bad:
        print("MISMATCH:", bad)
        return 1
    print("EQUIV_OK  所有聚合块与「不分片直接算」逐位一致；mean_rank=%s" % o.get("mean_rank"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
