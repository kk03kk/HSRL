"""Summarize identical-budget PPO hardware runs and their shared development panel."""
import argparse
import json
from pathlib import Path
from statistics import median

from taverngym.bc_train import json_file, paired

ARMS = ("full-host", "full-resident", "mig-single", "mig-dual", "a100")
FIELDS = ("original", "mixed")
UTILITY = (1.0, 0.3, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(path):
    config = read(path / "summary.json")["config"]
    rounds = [read(p) for p in sorted(path.glob("iteration-*.json"))]
    if [r["iteration"] for r in rounds] != list(range(1, 7)):
        raise ValueError(f"{path}: six complete target-scale PPO iterations required")
    samples = [row["sampling"] for row in rounds]
    updates = [row["optimization"] for row in rounds]
    for index, (sample, update) in enumerate(zip(samples, updates, strict=True)):
        start = config["first_seed"] + index * 1024
        if (
            sample["terminal_events"] != 1024
            or sample["seed_range"] != [start, start + 1024]
            or set(sample["fields"]) != set(FIELDS)
            or any(sample["fields"][field]["games"] != 512 for field in FIELDS)
            or update["after"]["kl"] > 0.05
            or update["stop"] != "all_minibatches"
            or update["updated_decisions"] != sample["decisions"]
        ):
            raise ValueError(f"{path}: round {index + 1} failed the PPO experiment contract")
    elapsed = [
        r["rollout_seconds"] + r["update_seconds"] + r["checkpoint_seconds"]
        for r in samples
    ]
    devices = config["actors"]
    result = {
        "devices": config["gpu"], "concurrency": config["concurrent"],
        "resident": config["resident_rollout"],
        "rollout_s": median([r["rollout_seconds"] for r in samples]),
        "update_s": median([r["update_seconds"] for r in samples]),
        "checkpoint_s": median([r["checkpoint_seconds"] for r in samples]),
        "round_s": median(elapsed), "round_s_all": elapsed,
        "throughput_games_per_s": 1024 * len(elapsed) / sum(elapsed),
        "billed_gpu_slot_hours_per_10000_games": (10000 / 1024) * devices * sum(elapsed) / len(elapsed) / 3600,
        "peak_learner_gib": max(row["sampling"]["gpu_peak_allocated_bytes"] for row in rounds) / 2**30,
        "max_policy_kl": max(u["after"]["kl"] for u in updates),
        "max_replay_logratio_rms": max(u["before"]["log_ratio_rms"] for u in updates),
        "decisions": sum(s["decisions"] for s in samples),
        "optimizer_steps": sum(u["optimizer_steps"] for u in updates),
    }
    if devices == 2:
        result["actor_s_by_round"] = [s["actor_seconds"] for s in samples]
        result["merge_s"] = median([s["merge_seconds"] for s in samples])
        result["broadcast_s"] = median([s["broadcast_seconds"] for s in samples])
    return result


def read_arms(root):
    library = {}
    base = {}
    for arm in ARMS:
        path = root / f"eval-{arm}"
        if not (path / "report.json").is_file():
            continue
        report = read(path / "report.json")
        games = {}
        for field in FIELDS:
            with (path / f"pairs-{field}.jsonl").open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            games[field] = [r["ppo"] for r in rows]
            reference = [r["bc18"] for r in rows]
            if base.setdefault(field, reference) != reference or len(reference) != 2048:
                raise ValueError(f"{arm}: comparison used a different frozen BC baseline")
        library[arm] = games, report
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    timings = {arm: stats(run / arm) for arm in ARMS if (run / arm / "summary.json").exists()}
    results = read_arms(run)
    comparators = results["full-resident"][0] if "full-resident" in results else None
    comparison = {}
    for arm, (games, report) in results.items():
        fields = {}
        for field in FIELDS:
            fields[field] = {"versus_bc": report["fields"][field]["versus_bc18"]}
            if comparators is not None and arm != "full-resident":
                baseline = {r["seed"]: (field, r) for r in comparators[field]}
                fields[field]["versus_full_resident"] = paired(games[field], baseline, UTILITY)["paired"]
        comparison[arm] = fields
    json_file(run / "hardware-report.json", {"timings": timings, "quality": comparison})


if __name__ == "__main__":
    main()
