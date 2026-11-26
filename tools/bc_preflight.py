"""Exercise the HSRL observation, loss, and arena contracts without updating any model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import taverngym
from taverngym._core import BcArena
from taverngym.bc_train import Demonstrations, ROW, configure, score
from taverngym.entity_policy import EntityPolicy

REWARD = [1.0, 0.3, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25]


def check_teacher_targets():
    actions = dict(taverngym.layout())
    games = 0
    for mixed in (False, True):
        seeds = [9001000, 9001003, 9001006, 9001007]
        plain = BcArena(seeds, mixed, 1)
        labelled = BcArena(seeds, mixed, 4)
        for step in range(6000):
            view = plain.observe()
            assert view == labelled.observe(), (mixed, step)
            n = len(view[0]) // 4
            if n == 0:
                break
            mask = np.frombuffer(view[-1], dtype="u1").reshape(n, -1)
            targets = np.frombuffer(labelled.teacher_targets(), dtype="<u2").reshape(n, 3)
            assert mask[np.arange(n), targets[:, 0]].all()
            assert not mask[:, actions["A_CANCEL"]].any()
            assert not mask[:, actions["A_MOVE"] :].any()
            move = mask.argmax(axis=1).tolist()
            plain.step(move)
            labelled.step(move)
        else:
            raise RuntimeError("Native labelling failed to reach the end of a complete game")
        assert json.loads(plain.results()) == json.loads(labelled.results())
        games += len(seeds)
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    assert taverngym.build_is_current(Path.cwd()) is True
    torch.set_num_threads(2)
    torch.manual_seed(2601126)
    if args.device == "cuda":
        configure(2601126)
    data = Demonstrations(args.data, load_rows=True)
    sample = np.concatenate([np.array(t[:16]) for t in data.tables])
    rng = np.random.default_rng(473)
    selected = rng.choice(len(sample), size=min(64, len(sample)), replace=False)
    rows = sample[selected]
    assert rows.dtype == ROW and len(rows) > 0

    device = torch.device(args.device)
    model = EntityPolicy().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 18_147_729
    representative = next(model.parameters())
    before = representative.detach().clone()
    inputs = [
        torch.from_numpy(np.array(rows[key], dtype=dtype, order="C", copy=True)).to(device)
        for key, dtype in (("features", "f4"), ("cards", "i4"), ("mask", "?"))
    ]
    target = torch.from_numpy(rows["action"].astype("i8")).to(device)
    rank = torch.from_numpy(rows["rank"].astype("i8") - 1).to(device)
    utility = torch.tensor(REWARD, device=device)
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        out = model(*inputs)
    legal = inputs[-1]
    assert bool(torch.isfinite(out.logits[legal]).all())
    assert bool(torch.isneginf(out.logits[~legal]).all())
    nll, placement, value = score(out, target, rank, utility)
    multi = legal.sum(dim=1) > 1
    loss = nll[multi].mean() + 0.05 * placement + 0.05 * value
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert torch.equal(before, representative)
    if args.device == "cuda":
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        assert not optimizer.state

    result = {
        "parameters": params, "input_rows": len(rows), "backward_ok": True, "optimizer_updates": 0,
        "placement_utility": REWARD, "device": args.device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "dtype": str(out.logits.dtype), "engine": taverngym.build_info(),
        "unchanged_games_when_labelling": check_teacher_targets(),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
