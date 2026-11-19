"""Learn when to close recruitment from paired, single-decision full-game outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import estimate
from workflows.peer_finetune_holdout import read
from workflows.peer_health_switch import field
from workflows.peer_ppo_panel import trial

BASE = Path("artifacts/runs/nus-peer-sale-package-20261119-1")
BASE_SHA = "9ecf26967e850d96e73ef8bd05b93388c30c2086d2bce1e04b4ce53403e102d9"
START = 329410000
FIT, CHECK = 768, 384
# Only public, predecision variables. Subgroups and capacity fixed before collecting labels.
CUTS = {"hp":[10,15,25], "size":[4,5,6], "gold":[0,1],"share":[.15,.25,.4],
        "turn":[9,11,13], "hand":[0,2], "strength":[300,1500,5000]}


class CloseRecruitment:
    def __init__(self, controller, intervene):
        self.controller = controller
        self.evaluate = controller.evaluator
        self.model = self.evaluate.model
        controller.evaluator = self
        self.intervene = intervene

    def reset(self):
        self.controller.reset()
        self.case = None

    def decide(self, game, seat):
        self.game = game
        return self.controller.decide(game, seat)

    def __call__(self, features, cards, mask, seat):
        prior, value = self.evaluate(features, cards, mask, seat)
        choice = int(prior.argmax())
        if self.case is not None or self.game.turn < self.first_turn or not mask[0] or not 16 <= choice < 23:
            return prior, value
        p = json.loads(self.game.player_json(seat))
        board = p["board"]
        stats = [u["atk"]+u["hp"] for u in board]
        total = max(1,sum(stats))
        if p["gold"] > 2 or stats[choice-16]*100 < 8*total:
            return prior, value
        self.case = dict(turn=self.game.turn, hp=p["health"]+p["armor"], size=len(board), gold=p["gold"],
            share=stats[choice-16]/total, strength=total, hand=len(p["hand"]), sale=choice)
        if not self.intervene:
            return prior, value
        action = np.zeros_like(prior)
        action[0] = 1.
        return action, value


def worker(out, index):
    import torch
    torch.set_num_threads(1)
    proto, _ = field()
    pool = MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)
    def arm(stop):
        return CloseRecruitment(make_eval_policy(str(BASE/"model.pt"), protocol=proto, device="cpu")["policy"], stop)
    control, stop = arm(False), arm(True)
    rows = []
    for j in range(index, FIT+CHECK, 16):
        seed = START+j
        control.first_turn = stop.first_turn = 7+j%7
        a = trial(control, pool, seed, event_turns=True)
        if control.case is None:
            b = a
        else:
            b = trial(stop, pool, seed, event_turns=True)
            if stop.case != control.case or a["lobby"] != b["lobby"]:
                raise ValueError("first eligible state differs")
        rows.append(dict(seed=seed, case=control.case, parent=a, candidate=b))
    atomic_json(out/f"labels-{index}.json", rows)


def outcome(row):
    return row["candidate"]["metrics"]["utility"] - row["parent"]["metrics"]["utility"]


def fit_tree(rows, depth):
    d = np.array([outcome(r) for r in rows])
    mean = float(d.mean())
    # Shrink decisions towards the baseline when the benefit has a noisy sign.
    stop = mean > max(.025, float(d.std()/np.sqrt(len(d))))
    leaf = dict(n=len(rows), mean=mean, stop=bool(stop))
    if not depth:
        return leaf
    best = None
    for name, cuts in CUTS.items():
        for cut in cuts:
            left = [r for r in rows if r["case"][name] <= cut]
            right = [r for r in rows if r["case"][name] > cut]
            if min(len(left),len(right)) < 40:
                continue
            gains = [sum(outcome(r) for r in part) for part in (left,right)]
            score = sum(max(0,g) for g in gains) - max(0,sum(gains))
            if score > 2 and (best is None or score > best[0]):
                best = score,name,cut,left,right
    if best is None:
        return leaf
    _,name,cut,left,right = best
    return dict(n=len(rows), feature=name, cut=cut, left=fit_tree(left,depth-1), right=fit_tree(right,depth-1))


def choose(tree, row):
    if row["case"] is None:
        return False
    while "feature" in tree:
        tree = tree["left" if row["case"][tree["feature"]] <= tree["cut"] else "right"]
    return tree["stop"]


def run(out):
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"] != BASE_SHA or read(BASE/"verified.json")["games"] != 64:
        raise ValueError("unverified parent")
    proto, lib = field()
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source, parent=BASE_SHA, protocol=proto.sha256(), library=lib,
        seed=START, fit=FIT, check=CHECK, max_depth=2, cutpoints=CUTS, job=os.environ["SLURM_JOB_ID"],
        treatment="at most once per game on first 8%+ sale at <=2 gold after assigned T7..13",
        gate="chronological validation utility lower 95% CI >0; check T12 survivor growth before expansion"))
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(out),"--worker",str(i)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(16) as workers:
        list(workers.map(launch,range(16)))
    rows = sorted(sum((read(out/f"labels-{i}.json") for i in range(16)),[]),key=lambda r:r["seed"])
    if [r["seed"] for r in rows] != list(range(START,START+FIT+CHECK)):
        raise ValueError("missing source labels")
    train = [r for r in rows[:FIT] if r["case"]]
    tree = fit_tree(train,2)
    val = rows[FIT:]
    gain = estimate([outcome(r) if choose(tree,r) else 0 for r in val])
    atomic_json(out/"complete.json", dict(tree=tree, fit_eligible=len(train), check_eligible=sum(bool(r["case"]) for r in val),
        selected=sum(choose(tree,r) for r in val), utility=gain,
        always_close=estimate([outcome(r) for r in val]), passed=gain["ci95"][0]>0))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--worker",type=int)
    args=parser.parse_args()
    run(args.out) if args.worker is None else worker(args.out,args.worker)
