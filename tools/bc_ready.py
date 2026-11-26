"""Check the frozen input bundle before the first HSRL BC update."""
import argparse
import csv
import json
from pathlib import Path

SPLITS = Path(__file__).resolve().parents[1] / "deploy" / "slurm" / "bc_splits.tsv"


def check_evaluator_parity(run, previous_pairs):
    with previous_pairs.open(newline="", encoding="utf-8") as stream:
        expected = {
            int(row["seed"]): (int(row["seat"]), int(row["strategy_v2_placed_rank"]))
            for row in csv.DictReader(stream)
        }
    count = 0
    for path in (run / "data" / "pilot-original").glob("games-*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            game = json.loads(line)["game"]
            if expected.get(game["seed"]) != (game["seat"], game["rank"]):
                raise ValueError(f"BC arena diverged from strategy evaluator at seed {game['seed']}")
            count += 1
    if count != 128:
        raise ValueError("Incomplete evaluator parity probe")
    return count


def examine(run):
    data = run / "data"
    splits = {}
    seeds = set()
    for line in SPLITS.read_text(encoding="utf-8").splitlines():
        name, start, count, field, capture = line.split()
        first, games = int(start), int(count)
        manifest = json.loads((data / name / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((data / name / "audit.json").read_text(encoding="utf-8"))
        if (
            (manifest["first_seed"], manifest["games"], manifest["field"]) != (first, games, field)
            or audit["games"] != games
            or bool(manifest["rows"]) != (capture == "rows")
        ):
            raise ValueError(f"{name}: frozen split differs from its plan")
        span = set(range(first, first + games))
        if not span.isdisjoint(seeds):
            raise ValueError(f"{name}: seeds overlap a different split")
        seeds.update(span)
        splits[name] = {"manifest": manifest, "audit": audit}
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parity", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    splits = examine(run)
    cpu = json.loads((run / "preflight-cpu.json").read_text(encoding="utf-8"))
    gpu = json.loads((run / "preflight-cuda.json").read_text(encoding="utf-8"))
    fingerprint = gpu["engine"]["source_hash"]
    for probe in (cpu, gpu):
        if (
            probe["engine"]["source_hash"] != fingerprint or not probe["backward_ok"]
            or probe["optimizer_updates"] != 0 or probe["parameters"] != 18_147_729
            or probe["unchanged_games_when_labelling"] != 8
        ):
            raise ValueError("Incomplete model or labelling preflight")
    if any(x["manifest"]["source_hash"] != fingerprint for x in splits.values()):
        raise ValueError("The prepared shards use a different engine")
    result = {
        "ready_for_bc": True, "optimizer_updates": 0,
        "strategy_evaluator_parity_games": check_evaluator_parity(run, args.parity),
        "engine_fingerprint": fingerprint, "parameters": gpu["parameters"], "gpu": gpu["gpu"],
        "train": ["train-original", "train-mixed"], "development": ["dev-original", "dev-mixed"],
        "independent_holdout": ["test-original", "test-mixed"],
        "placement_utility": gpu["placement_utility"],
        "splits": {name: x["audit"] for name, x in splits.items()},
    }
    output = run / "preparation.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    (run / "READY_FOR_BC").write_text(fingerprint + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "splits"}), flush=True)


if __name__ == "__main__":
    main()
