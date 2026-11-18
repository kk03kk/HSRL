"""Reproduce the frozen D-128 decision from the validated three-model panel."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import file_identity
from tools.compare_research_panels import load_sealed_panel


def seed_vectors(model, seeds):
    order = model["seed_sequence"]
    ranks = model["rank_sequence"]
    if len(order) != len(seeds) or len(ranks) != len(seeds) or sorted(order) != seeds:
        raise ValueError("incomplete or different game seed cohort")
    by_seed = dict(zip(order, ranks))
    rank = np.array([by_seed[s] for s in seeds])
    if not np.isin(rank, np.arange(1, 9)).all():
        raise ValueError("invalid placement")
    boards = model["strength_by_seed"]
    if set(boards) != {str(s) for s in seeds}:
        raise ValueError("incomplete board histories")
    turns = [boards[str(s)] for s in seeds]
    reached = np.array(["12" in t for t in turns], dtype=int)
    growth = np.array([np.log1p(t["12"]) - np.log1p(t["9"])
                       if "12" in t and "9" in t else np.nan for t in turns])
    if reached.sum() != model["curves"]["12"]["n"]:
        raise ValueError("inconsistent T12 survival count")
    return rank, reached, growth


def contrast(models, seeds, utility, gate):
    parent_rank, parent_alive, parent_growth = seed_vectors(models["parent"], seeds)
    rng = np.random.default_rng(118128)
    draws = rng.integers(len(seeds), size=(10000, len(seeds)))
    arms = {}
    joint_rank = []
    joint_growth = []
    for name in ("original_a", "original_b"):
        rank, alive, growth = seed_vectors(models[name], seeds)
        rank_diff = rank - parent_rank
        growth_diff = growth - parent_growth
        common = np.isfinite(growth_diff)
        denominators = common[draws].sum(axis=1)
        growth_sums = np.where(common, growth_diff, 0)[draws].sum(axis=1)
        sampled_growth = np.divide(growth_sums, denominators, where=denominators > 0,
                                   out=np.full(len(draws), np.nan))
        joint_growth.append(sampled_growth)
        sampled_rank = rank_diff[draws].mean(axis=1)
        joint_rank.append(sampled_rank)
        arms[name] = {
            "common_games": int(common.sum()),
            "growth_difference": float(growth_diff[common].mean()) if common.any() else None,
            "growth_ci95": interval(sampled_growth),
            "T12_survival_difference": float((alive - parent_alive).mean()),
            "rank_difference": float(rank_diff.mean()),
            "rank_ci95": interval(sampled_rank),
            "original_utility_difference": float((utility[rank - 1] - utility[parent_rank - 1]).mean()),
        }
    growth_ci = interval(np.mean(joint_growth, axis=0))
    rank_ci = interval(np.mean(joint_rank, axis=0))
    empty_common = growth_ci is None
    checks = {
        "positive_growth_each": all(a["growth_difference"] is not None and a["growth_difference"] >
                                    gate["common_T9_T12_growth_difference_each_arm_gt"] for a in arms.values()),
        "joint_growth_lower": None if empty_common else
            growth_ci[0] > gate["mean_of_two_growth_differences_paired_bootstrap_lower95_gt"],
        "T12_survival_each": all(a["T12_survival_difference"] >=
                                 gate["T12_survival_rate_difference_each_arm_gte"] for a in arms.values()),
        "joint_rank_upper": rank_ci[1] < gate["mean_of_two_rank_differences_paired_bootstrap_upper95_lt"],
        "original_utility_each": all(a["original_utility_difference"] >=
                                     gate["original_rank_utility_difference_each_arm_gte"] for a in arms.values()),
    }
    return {"arms": arms, "equal_weight_growth_ci95": growth_ci, "equal_weight_rank_ci95": rank_ci,
            "checks": checks, "decision": "inconclusive" if empty_common else
            ("passes" if all(checks.values()) else "does_not_pass")}


def interval(samples):
    return np.percentile(samples, [2.5, 97.5]).tolist() if np.isfinite(samples).all() else None


def assess(directory, verification):
    plan = json.loads((ROOT / "config/research/natural12m-20261118.json").read_text())
    evaluation = json.loads((ROOT / plan["assessment"]["plan"]).read_text())
    run = json.loads((directory / "run.json").read_text())
    panel = load_sealed_panel(directory / "panel.json")
    complete = json.loads((directory / "complete.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if (complete["validation"] != file_identity(directory / "validation.json") or
            validation["valid"] is not True or validation["panel"] != complete["panel"] or
            run["plan"] != evaluation or run["source"]["tracked_dirty"] or
            panel["protocol"]["protocol_sha256"] != evaluation["expected_protocol_sha256"]):
        raise ValueError("evaluation identity or validation mismatch")
    training = {r["seed"]: r for r in map(json.loads, verification.read_text().splitlines())}
    expected = {evaluation["models"][0]["checkpoint"]: evaluation["models"][0]["sha256"]}
    for arm in plan["arm_environment"]:
        endpoint = training[int(arm["HSAI_ARENA_SEED"])]
        decisions = endpoint["totals"]["decisions"]
        limit = int(plan["shared_environment"]["HSAI_ARENA_STOP_DECISIONS"])
        if not decisions - endpoint["last_rollout"] < limit <= decisions:
            raise ValueError("endpoint differs from frozen budget")
        expected[arm["HSAI_ARENA_OUT"] + "/train/model.pt"] = endpoint["files"]["model.pt"]["sha256"]
    models = {m["name"]: m for m in panel["models"]}
    if len(panel["models"]) != 3 or set(models) != {m["name"] for m in evaluation["models"]}:
        raise ValueError("unexpected models")
    for item in evaluation["models"]:
        path = item["checkpoint"]
        m = models[item["name"]]
        if (m["checkpoint"] != path or m["checkpoint_sha256"] != expected[path] or
                run["inputs"][path]["sha256"] != expected[path] or
                m["outcome"]["illegal_actions"] or m["outcome"]["truncated_games"]):
            raise ValueError("wrong endpoint or invalid games")
    for path, sha in plan["asset_sha256"].items():
        if "v14hold-snap/" in path and run["inputs"][path]["sha256"] != sha:
            raise ValueError("holdout library mismatch")
    seeds = list(range(evaluation["seed"], evaluation["seed"] + evaluation["games"]))
    utility = np.array([float(v) for v in plan["shared_environment"]["HSAI_ARENA_RANK_UTILITY"].split(",")])
    return {"panel": complete["panel"], "protocol_sha256": validation["protocol_sha256"],
            **contrast(models, seeds, utility, plan["assessment"]["scale_gate"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_directory", type=Path)
    parser.add_argument("endpoint_check_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(args.evaluation_directory, args.endpoint_check_jsonl), indent=2, allow_nan=False))
