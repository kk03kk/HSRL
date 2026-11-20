"""Count native cancellation opportunities and shop actions between archived freeze presses."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.env.action_space_std import A_BUY, A_CANCEL, A_FREEZE, A_REFRESH, N_SHOP
from hsbg.runtime.artifacts import atomic_json, file_identity


def sample(path):
    report = json.loads(path.read_bytes())
    trace = path.with_suffix(".decisions.jsonl.gz")
    if not report["trace"]["complete"] or trace.stat().st_size != report["trace"]["bytes"]:
        raise ValueError(f"incomplete source trace {path.name}")
    result = Counter()
    seeds = []
    with gzip.open(trace, "rt", encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "game_start":
                expected = report["rows"][len(seeds)]
                if expected["seed"] != row["seed"]:
                    raise ValueError("unexpected game order")
                current_turn, decisions = None, 0
            elif row["event"] == "action":
                if row["seed"] != expected["seed"] or row["action_index"] != decisions:
                    raise ValueError("noncontiguous trace")
                if row["turn"] != current_turn:
                    current_turn = row["turn"]
                    presses, last_press, shop_changed = 0, None, False
                    result["turns"] += 1
                action, legal = row["action"], row["legal_mask"]
                if legal[A_CANCEL]:
                    result["native_cancel_available"] += 1
                    result["cancel_only_native_option"] += int(sum(bool(x) for x in legal) == 1)
                if action == A_FREEZE:
                    presses += 1
                    result["freeze_presses"] += 1
                    if last_press is not None:
                        result["second_or_later_press"] += 1
                        result["after_refresh_or_buy"] += int(shop_changed)
                        result["consecutive_presses"] += int(last_press == decisions - 1)
                    if presses == 3:
                        result["turns_with_three_presses"] += 1
                    shop_changed, last_press = False, decisions
                elif action == A_REFRESH or A_BUY <= action < A_BUY + N_SHOP:
                    shop_changed = True
                decisions += 1
            elif row["event"] == "game_end":
                if (row["rank"], decisions) != (expected["rank"], expected["decisions"]) or not row["finished"]:
                    raise ValueError("archived game outcome mismatch")
                result["decisions"] += decisions
                seeds.append(row["seed"])
                if len(seeds) == 2:
                    break
    if len(seeds) != 2:
        raise ValueError("missing archived games")
    return dict(report=file_identity(path), trace=report["trace"]["sha256"], seeds=seeds, counts=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    parts = sorted(args.source.glob("shard-[0-9][0-9].json"))
    args.out.mkdir(exist_ok=False)
    if not parts:
        raise ValueError("missing source shards")
    with ProcessPoolExecutor(min(8, len(parts))) as pool:
        samples = list(pool.map(sample, parts))
    total = Counter()
    for item in samples:
        total.update(item["counts"])
    seeds = [seed for item in samples for seed in item["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate game")
    atomic_json(args.out / "complete.json", dict(games=len(seeds), counts=total, samples=samples))
