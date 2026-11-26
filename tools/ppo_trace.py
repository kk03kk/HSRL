"""Collect complete masked trajectories in the fixed-shape production evaluator."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import taverngym.bc_train as bc
import taverngym.ppo_train as ppo

class Recorder:
    def __init__(self, file, field, agent, mode):
        self.file, self.field, self.agent, self.mode = file, field, agent, mode

    def start(self, seeds):
        self.seeds = seeds
        self.actions = [[] for _ in seeds]

    def step(self, arena, slots, f, c, mask, actions, choice):
        indices = np.frombuffer(slots, dtype="<u4")
        assert mask[np.arange(len(indices)), actions].all()
        for slot, action in zip(indices.tolist(), actions.tolist(), strict=True):
            self.actions[slot].append(action)
        return actions

    def finish(self, finished):
        for seed, game, actions in zip(self.seeds, finished, self.actions, strict=True):
            assert seed == game["seed"] and len(actions) == game["actions"]
            print(json.dumps(dict(field=self.field, agent=self.agent, mode=self.mode,
                                  game=game, actions=actions), separators=(",", ":")),
                  file=self.file)


def model(path):
    return ppo.make_model(torch.load(path, weights_only=False, map_location="cpu")["model"], 512)


def collect(args):
    if min(args.greedy, args.sample) < 0 or args.greedy + args.sample == 0:
        raise ValueError("Use a positive number of games in at least one mode")
    bc.configure(args.seed)
    weights = {"ppo": model(args.checkpoint), "bc": model(args.reference)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    windows = (("greedy", args.greedy, args.first),
               ("sample", args.sample, args.first + args.greedy))
    with tmp.open("w") as stream:
        for mode, count, first in windows:
            if not count:
                continue
            for agent, net in weights.items():
                torch.manual_seed(args.seed)
                bc.play(net, list(range(first, first + count)), args.field, 384, args.workers,
                        args.out.with_suffix(".failure.json"),
                        Recorder(stream, args.field, agent, mode), sample=mode == "sample")
                print(args.field, mode, agent, count, flush=True)
    tmp.replace(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "reference", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--field", choices=("original", "mixed"), required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--greedy", type=int, default=256)
    parser.add_argument("--sample", type=int, default=128)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--seed", type=int, default=132879)
    collect(parser.parse_args())
