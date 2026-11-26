"""Alternate synchronization settings while replaying one frozen, complete PPO rollout."""
import argparse
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from taverngym.bc_train import configure, emit, json_file
from taverngym.ppo_train import load_reference, make_model, sample, update


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("init", type=Path)
    parser.add_argument("admission", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    configure(2601126)
    torch.use_deterministic_algorithms(True)
    saved, digest = load_reference(args.init, args.admission)
    reference = make_model(saved["model"])
    reference.requires_grad_(False)
    batch, sampled = sample(
        reference, 11030000, 1024, 1024, 10, "bf16", args.out.with_name("replay-failure.json"),
    )
    control = None
    rounds = []
    for defer in (False, True, True, False):
        model = make_model(saved["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=9e-6, fused=True, weight_decay=0)
        recipe = SimpleNamespace(
            seed=2601126, resident_rollout=True, defer_metrics=defer, microbatch=1024, precision="bf16",
            kl_check_every=4, lr_scale=1.5, bc_kl_scale=1.0, minibatch=2048,
        )
        torch.cuda.synchronize()
        start = perf_counter()
        info = update(model, reference, optimizer, batch, recipe, 0)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        state = (
            {key: tensor.cpu() for key, tensor in model.state_dict().items()},
            {
                index: {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in group.items()
                }
                for index, group in optimizer.state_dict()["state"].items()
            },
        )
        if control is None:
            control = state
        if not equal(state, control):
            raise RuntimeError("Timing treatment changed weights or optimizer state")
        row = {"defer_metrics": defer, "seconds": elapsed, "diagnostics": info}
        rounds.append(row)
        emit({"defer_metrics": defer, "seconds": elapsed, "exact": True})
        del model, optimizer, state
    json_file(args.out, {
        "exact": True, "checkpoint_sha256": digest,
        "rollout_sha256": sampled["rollout_sha256"], "rounds": rounds,
    })


if __name__ == "__main__":
    main()
