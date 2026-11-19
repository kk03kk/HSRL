"""Paired test of retaining material on a short board when near elimination."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare, estimate
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol, read
from workflows.peer_health_switch import field
from workflows.peer_ppo_panel import trial


class RetainMaterial:
    def __init__(self, controller):
        self.controller = controller
        self.evaluate = controller.evaluator
        self.model = self.evaluate.model
        controller.evaluator = self

    def reset(self):
        self.controller.reset()
        self.overrides = []

    def decide(self, game, seat):
        self.game = game
        return self.controller.decide(game, seat)

    def __call__(self, features, cards, mask, seat):
        prior, value = self.evaluate(features, cards, mask, seat)
        choice = int(prior.argmax())
        if self.game.turn < 7 or not 16 <= choice < 23:
            return prior, value
        player = json.loads(self.game.player_json(seat))
        board = player["board"]
        if player["health"] + player["armor"] > 15 or len(board) > 4:
            return prior, value
        stats = [unit["atk"] + unit["hp"] for unit in board]
        total = sum(stats)
        if 100 * stats[choice - 16] < 15 * max(1, total):
            return prior, value
        eligible = mask.copy()
        for i, stat in enumerate(stats):
            if 100 * stat >= 15 * max(1, total):
                eligible[16+i] = False
        if not eligible.any():
            return prior, value
        new, value = self.evaluate(features, cards, eligible, seat)
        replacement = int(new.argmax())
        if not eligible[replacement]:
            raise ValueError("invalid alternate choice")
        self.overrides.append([self.game.turn, choice, replacement])
        return new, value


def run(args):
    out = args.out
    if args.split == "train":
        proto, digest = field()
    else:
        proto, _ = holdout_protocol()
        if args.split == "transfer":
            path = Path("artifacts/libraries/human-turn-transfer-20261119-01")
            digest = library_digest(path)
            if digest != "8a769ef0f43fc8edf2a718514898b1ae4e60831aa8bf37355e11c4d2627c4be0":
                raise ValueError("transfer field changed")
            proto = replace(proto, mirror_snaps=path.as_posix())
        else:
            digest = read(ROOT/"config/research/human-turn-3702-ppo.json")["library_digests"]["hold"]
    source = source_identity(ROOT)
    parent = file_identity(INITIAL)
    if source["tracked_dirty"] or parent["sha256"] != INITIAL_SHA:
        raise ValueError("wrong source or parent")
    if args.worker is not None:
        import torch
        torch.set_num_threads(1)
        plan = read(out/"plan.json")
        if (proto.sha256(), digest, parent) != (plan["protocol"], plan["library"], plan["parent"]):
            raise ValueError("panel assets differ")
        pool = MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)
        def policy():
            return make_eval_policy(str(INITIAL), protocol=proto, device=args.device)["policy"]
        raw, keep = policy(), RetainMaterial(policy())
        rows, edits = [], []
        for j in range(args.worker, args.games, 8):
            seed = args.seed + j
            a, b = [trial(p, pool, seed, event_turns=True) for p in (raw, keep)]
            if not keep.overrides and a != b:
                raise ValueError(f"untouched game differs at {seed}")
            rows.append(dict(parent=a, candidate=b))
            edits.append(dict(seed=seed, overrides=keep.overrides.copy()))
        atomic_json(out/f"panel-{args.worker}.json", rows)
        atomic_json(out/f"changes-{args.worker}.json", edits)
        return
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/"plan.json", dict(source=source, seed=args.seed, games=args.games, split=args.split, device=args.device,
        parent=parent, protocol=proto.sha256(), field=proto.as_dict(), library=digest,
        rule="T7+, HP+armor<=15, board<=4: protect each 15%+ unit on a proposed material sale",
        gate="positive lower 95% CI on all-game utility and T12 paired log strength lower CI >= -0.12",
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"), job=os.environ["SLURM_JOB_ID"]))
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--worker", str(i)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(8) as workers:
        list(workers.map(launch, range(8)))
    report = compare(out, args.seed, args.games)
    effects = []
    for i in range(8):
        for row, edits in zip(read(out/f"panel-{i}.json"), read(out/f"changes-{i}.json"), strict=True):
            if row["parent"]["seed"] != edits["seed"]:
                raise ValueError("override seed mismatch")
            if edits["overrides"]:
                effects.append(row["candidate"]["metrics"]["utility"] - row["parent"]["metrics"]["utility"])
    report["affected"] = estimate(effects)
    report["passed"] = report["delta"]["utility"]["ci95"][0] > 0 and report["delta"]["log12"]["ci95"][0] >= -.12
    atomic_json(out/"complete.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "hold", "transfer"))
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--games", required=True, type=int)
    parser.add_argument("--worker", type=int)
    run(parser.parse_args())
