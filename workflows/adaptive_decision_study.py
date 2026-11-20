"""Trace focal public decisions in sealed, fully adaptive eight-policy games."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows import peer_adaptive_lobby as lobby
from workflows.peer_finetune_holdout import read
from workflows.peer_intent_scope import REGISTRY, POLICY

RUNS = Path("artifacts/runs")
OLD = RUNS / "nus-peer-intent-adaptive-20261120-1"
RANGES = {"end":(0,1), "roll":(1,2), "freeze":(2,3), "level":(3,4), "power":(4,6),
          "gift":(6,7), "buy":(8,16), "sell":(16,23), "activate":(23,30),
          "choose":(30,46), "play":(46,126), "magnetize":(126,196), "move":(196,245)}


def family(action):
    return next((k for k,(lo,hi) in RANGES.items() if lo <= action < hi),"other")


def weights():
    import torch
    torch.set_num_threads(1)
    proto,_ = lobby.holdout_protocol()
    if (file_identity(REGISTRY/"policy.json")["sha256"] != POLICY
            or file_identity(lobby.INITIAL)["sha256"] != lobby.INITIAL_SHA
            or file_identity(lobby.native.BASE)["sha256"] != lobby.native.PLAN["base_sha"]):
        raise ValueError("registered focal or opponents changed")
    specs = {"intent":"research-policy:"+str(REGISTRY),"peer":str(lobby.INITIAL),"native":str(lobby.native.BASE)}
    return {k:make_eval_policy(path,protocol=proto,device="cpu")["policy"] for k,path in specs.items()}


class PublicTrace:
    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        self.actor.reset()
        self.rows = []
        self.index = 0

    @property
    def overrides(self):
        return self.actor.overrides

    def decide(self, game, seat):
        decision = self.actor.decide(game,seat)
        action = decision.action
        if 7 <= game.turn <= 14 and action < 196:
            pl = json.loads(game.player_json(seat))
            best = {}
            for name,(lo,hi) in RANGES.items():
                a = lo + int(np.argmax(decision.pi[lo:hi]))
                if decision.pi[a] > 0:
                    best[name] = [a,round(float(decision.pi[a]),7)]
            board = [[m["card"],m["atk"]+m["hp"]] for m in pl["board"]]
            row = dict(index=self.index,turn=game.turn,action=action,family=family(action),
                gold=pl["gold"],health=pl["health"]+pl["armor"],tier=pl["tier"],upgrade=pl["upgrade_cost"],
                board=board,hand=len(pl["hand"]),shop=[s["item"] for s in pl["shop"]],
                forced=decision.info["forced"],best=best)
            self.rows.append(row)
        self.index += 1
        return decision


def trace_one(i):
    actor = PublicTrace(ACTORS["intent"])
    result = lobby.play({**ACTORS,"intent":actor},i,True,380410000)
    archive = OLD/f"pair-{i}.json"
    matches = json.loads(json.dumps(result)) == read(archive)["candidate"]
    atomic_json(OUT/f"game-{i}.json",dict(result=result,events=actor.rows,matches=matches,prior=file_identity(archive)))
    return matches


def start(out, n, workers):
    global ACTORS, OUT
    OUT = out
    source = source_identity(ROOT)
    if source["tracked_dirty"] or not 0 < n <= 128 or n%8 or not 0 < workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        raise ValueError("expected clean release, balanced natural seats, allocated workers")
    ACTORS = weights()
    out.mkdir(exist_ok=False)
    plan = read(OLD/"plan.json")
    if plan["candidate"] != POLICY:
        raise ValueError("wrong archived focal actor")
    atomic_json(out/"plan.json",dict(source=source,n=n,old_protocol=plan["protocol_sha256"],policy=POLICY,
        original_plan=file_identity(OLD/"plan.json"),engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),job=os.environ["SLURM_JOB_ID"],
        selection="First n sealed adaptive seeds in natural order; observe only; all eight seats act."))
    with ProcessPoolExecutor(workers,mp_context=mp.get_context("fork")) as pool:
        matches = list(pool.map(trace_one,range(n)))
    atomic_json(out/"complete.json",dict(n=n,exact=sum(matches),mismatched=[i for i,v in enumerate(matches) if not v]))
    if not all(matches):
        raise ValueError("archived trajectories differed; preserve reports for diagnosis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--games",type=int,default=48)
    parser.add_argument("--workers",type=int,default=24)
    args = parser.parse_args()
    start(args.out,args.games,args.workers)
