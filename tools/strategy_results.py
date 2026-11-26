"""Summarize matched strategy games and paired 95% intervals."""
import argparse
import csv
import json
import statistics
from pathlib import Path

REWARDS = (1.0, 0.3, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25)


def panel_metrics(placements):
    return {
        "rank": placements,
        "top4": [int(place <= 4) for place in placements],
        "top2": [int(place <= 2) for place in placements],
        "first": [int(place == 1) for place in placements],
        "utility": [REWARDS[place - 1] for place in placements],
    }


def differences(candidate, baseline):
    values = [a - b for a, b in zip(candidate, baseline, strict=True)]
    mean = statistics.fmean(values)
    error = 1.96 * statistics.stdev(values) / len(values) ** 0.5
    return {"mean": mean, "paired_95_interval": [mean - error, mean + error]}


def summarize(directory: Path):
    with (directory / "pairs.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        policies = [key.removesuffix("_rank") for key in fields if key.endswith("_rank")]
        rows = list(reader)
    if len(rows) < 2 or len(policies) < 2:
        raise ValueError("expected multiple complete matched games and policies")
    seeds = [int(row["seed"]) for row in rows]
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("incomplete or non-contiguous seed list")
    if any(int(row["seat"]) != seed % 8 for seed, row in zip(seeds, rows, strict=True)):
        raise ValueError("inconsistent subject seat")
    ranks = {}
    for policy in policies:
        values = [int(row[f"{policy}_rank"]) for row in rows]
        if any(row[f"{policy}_problem"] or not 1 <= rank <= 8
               for rank, row in zip(values, rows, strict=True)):
            raise ValueError(f"{policy}: invalid game")
        ranks[policy] = panel_metrics(values)

    with (directory / "v2-progress.csv").open(encoding="utf-8", newline="") as stream:
        progress = list(csv.DictReader(stream))
    if len(progress) != len(rows) * len(policies):
        raise ValueError("incomplete formation telemetry")
    for policy in policies:
        policy_rows = sorted(
            (row for row in progress if row["policy"] == policy),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in policy_rows] != seeds:
            raise ValueError(f"{policy}: missing formation rows")
        ranks[policy]["formed"] = [int(bool(row["formed_routes"])) for row in policy_rows]
        ranks[policy]["formed_and_first"] = [
            int(bool(row["formed_routes"]) and row["rank"] == "1") for row in policy_rows
        ]

    metrics = {policy: {key: statistics.fmean(values) for key, values in scans.items()}
               for policy, scans in ranks.items()}
    comparisons = {}
    for index, candidate in enumerate(policies):
        for reference in policies[:index]:
            comparisons[f"{candidate} vs {reference}"] = {
                key: differences(values, ranks[reference][key])
                for key, values in ranks[candidate].items()
            }


    return {
        "seed_start": seeds[0],
        "seed_end_exclusive": seeds[-1] + 1,
        "paired_seeds": len(rows),
        "unique_lobbies": len({row["lobby_types"] for row in rows}),
        "metrics": metrics,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.directory), ensure_ascii=False, indent=2)
    (args.directory / "paired-metrics.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
