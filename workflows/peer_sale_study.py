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
    def __init__(self, controller, *, extended=False):
        self.controller = controller
        self.evaluate = controller.evaluator
        self.model = self.evaluate.model
        controller.evaluator = self
        self.extended = extended

    def reset(self):
        self.controller.reset()
        self.overrides = []

    def decide(self, game, seat):
        self.game = game
        self.last_override = None
        choice = self.controller.decide(game, seat)
        if self.last_override is not None and choice.info.get("forced"):
            self.overrides.pop()
            self.last_override = None
        return choice

    def __call__(self, features, cards, mask, seat):
        prior, value = self.evaluate(features, cards, mask, seat)
        choice = int(prior.argmax())
        if self.game.turn < 7 or not 16 <= choice < 23:
            return prior, value
        player = json.loads(self.game.player_json(seat))
        board = player["board"]
        low = player["health"] + player["armor"] <= 15
        short = len(board) <= 4
        extra = self.extended and player["gold"] < 3 and (short or (low and len(board) <= 6))
        if not ((low and short) or extra):
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
        self.last_override = (choice, replacement, "base" if low and short else "extra")
        return new, value


class ActionTrace:
    def __init__(self, controller):
        self.controller = controller

    def reset(self):
        self.controller.reset()
        self.events = []

    def decide(self, game, seat):
        choice = self.controller.decide(game, seat)
        self.events.append((int(game.turn), seat, int(choice.action), getattr(self.controller, "last_override", None)))
        return choice


def first_change(left, right, extension):
    for index,(a,b) in enumerate(zip(left,right)):
        if a[:3] == b[:3]:
            continue
        reason = b[3]
        if (a[:2] != b[:2] or a[3] is not None or reason is None
                or reason != (a[2], b[2], "extra" if extension else "base")):
            raise ValueError(f"unexplained first action change at decision {index}: {a} / {b}")
        return dict(index=index, turn=a[0], original=a[2], replacement=b[2], reason=reason[2])
    if len(left) != len(right):
        raise ValueError("identical action prefix produced an unexplained episode length")
    return None


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
        raw = RetainMaterial(policy()) if args.extension else policy()
        keep = RetainMaterial(policy(), extended=args.extension)
        left, right = ActionTrace(raw), ActionTrace(keep)
        rows, edits = [], []
        for j in range(args.worker, args.games, 8):
            seed = args.seed + j
            a, b = [trial(p, pool, seed, event_turns=True) for p in (left, right)]
            first = first_change(left.events, right.events, args.extension)
            if first is None and a != b:
                raise ValueError(f"untouched game differs at {seed}")
            rows.append(dict(parent=a, candidate=b))
            edits.append(dict(seed=seed, overrides=keep.overrides.copy(), affected=first is not None, first=first))
        atomic_json(out/f"panel-{args.worker}.json", rows)
        atomic_json(out/f"changes-{args.worker}.json", edits)
        return
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/"plan.json", dict(source=source, seed=args.seed, games=args.games, split=args.split, device=args.device,
        parent=parent, protocol=proto.sha256(), field=proto.as_dict(), library=digest,
        rule="T7+, HP+armor<=15, board<=4: protect each 15%+ unit on a proposed material sale",
        comparison=("extension vs original retention: gold<3 and (board<=4 or low HP and board<=6)"
                    if args.extension else "original retention vs raw"),
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
            if edits["affected"]:
                effects.append(row["candidate"]["metrics"]["utility"] - row["parent"]["metrics"]["utility"])
    report["affected"] = estimate(effects)
    report["passed"] = report["delta"]["utility"]["ci95"][0] > 0 and report["delta"]["log12"]["ci95"][0] >= -.12
    atomic_json(out/"complete.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "hold", "transfer"))
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    parser.add_argument("--extension", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--games", required=True, type=int)
    parser.add_argument("--worker", type=int)
    run(parser.parse_args())
