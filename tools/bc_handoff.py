"""Verify PPO-compatible sampling from a frozen, independently evaluated BC checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from taverngym import N_ACTIONS, OBS_CARDS, OBS_FLOATS, build_info
from taverngym._core import BcArena
from taverngym.bc_train import configure, json_file, play, tensors, value_target
from taverngym.entity_policy import EntityPolicy

UTILITY = [1.0, 0.3, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25]


def parameters(arena):
    indices, *payload = arena.observe()
    size = len(indices) // 4
    views = (
        np.frombuffer(data, dtype=dtype).reshape(size, width)
        for data, dtype, width in zip(
            payload, ("<f4", "<i4", "u1"), (OBS_FLOATS, OBS_CARDS, N_ACTIONS), strict=True
        )
    )
    return tensors(*views)


def check_policy(model):
    arena = BcArena(list(range(9480000, 9480016)), True, 2)
    samples = 0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(12):
            batch = parameters(arena)
            result = model(*batch)
            distribution = torch.distributions.Categorical(logits=result.logits)
            actions = distribution.sample()
            if (
                not batch[-1][torch.arange(len(actions), device="cuda"), actions].all()
                or not torch.isfinite(result.value).all()
                or not torch.isfinite(distribution.log_prob(actions)).all()
            ):
                raise RuntimeError("Checkpoint gave an invalid policy sample or value")
            samples += len(actions)
            arena.step(actions.tolist())
        batch = parameters(arena)
        old_distribution = torch.distributions.Categorical(logits=model(*batch).logits)
        actions = old_distribution.sample()
        old_log_prob = old_distribution.log_prob(actions)
    with torch.autocast("cuda", torch.bfloat16):
        result = model(*batch)
        new_log_prob = torch.distributions.Categorical(logits=result.logits).log_prob(actions)
        if not torch.allclose((new_log_prob - old_log_prob).exp(), torch.ones_like(old_log_prob)):
            raise RuntimeError("An unchanged policy produced a probability ratio unequal to one")
    (-new_log_prob.mean() + 0.05 * result.value.square().mean()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Policy handoff gradients are invalid")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--games", type=int, default=1024, help="Stochastic complete games per field")
    parser.add_argument("--arena-batch", type=int, default=384)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    run = args.run
    report = json.loads((run / "evaluation/report.json").read_text())
    checkpoint = run / "selected.pt"
    with checkpoint.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if not report["ppo_gate"] or sha != report["checkpoint_sha256"]:
        raise ValueError("Handoff requires a frozen checkpoint that passed both independent fields")
    configure(19377)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = saved["meta"]
    if (
        meta["source"] != report["source"] or meta["source"] != build_info()["source_hash"]
        or meta["placement_utility"] != UTILITY
        or not torch.equal(value_target(torch.arange(8), torch.tensor(meta["placement_utility"])),
                           torch.tensor(UTILITY))
        or args.games < 1
    ):
        raise ValueError("The handoff uses a different engine, reward, or invalid game count")

    model = EntityPolicy(**meta["model"]).cuda()
    model.load_state_dict(saved["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params != 18_147_729:
        raise ValueError("Unexpected number of model parameters")
    samples = check_policy(model)
    fields = {}
    start = perf_counter()
    for field, first in (("original", 9550000), ("mixed", 9560000)):
        seeds = list(range(first, first + args.games))
        games = play(
            model, seeds, field, args.arena_batch, args.workers,
            run / f"handoff-failure-{field}.json", sample=True,
        )
        if [g["seed"] for g in games] != seeds or any(g["rank"] not in range(1, 9) for g in games):
            raise ValueError(f"Incomplete stochastic handoff for {field}")
        fields[field] = {
            "games": len(games), "actions": sum(g["actions"] for g in games),
            "first_seed": first, "mean_rank": sum(g["rank"] for g in games) / len(games),
            "budget_turns": sum(g["budget_turns"] for g in games),
            "recruit_turns": sum(len(g["rounds"]) for g in games),
        }
    check = {
        "checkpoint_sha256": sha, "parameters": n_params, "model": meta["model"],
        "valid_samples": samples, "same_weights_ratio": True, "optimizer_updates": 0,
        "finite_log_probabilities_value_and_gradients": True, "stochastic": fields,
        "stochastic_seconds": perf_counter() - start, "gpu": torch.cuda.get_device_name(),
    }
    json_file(run / "selected-check.json", check)
    print(check, flush=True)


if __name__ == "__main__":
    main()
