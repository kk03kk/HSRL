"""Collect donors, seal source pools, and evaluate independent warband fields."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import taverngym
from taverngym import bc_train as bc, ppo_train as ppo
from taverngym._core import BcArena
from taverngym.warband import Fields

WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def policy(checkpoint):
    return ppo.make_model(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],512)


def collect(root, split, checkpoint):
    first, n = {"train":(14300000,3072),"dev":(14310000,1536),"test":(14320000,1536)}[split]
    bc.configure(424)
    model = policy(checkpoint)
    target = root/f"source-data/{split}-ppo.jsonl"
    with target.with_suffix(".tmp").open("w") as stream:
        for offset in range(0,n,128):
            captured = []
            def constructor(seeds, field, workers):
                arena = BcArena.capture(seeds, field=="mixed", workers)
                captured.append(arena)
                return arena
            bc.play(model,list(range(first+offset,first+offset+128)),"mixed",128,WORKERS,
                    root/f"capture-failure-{split}.json",arena_factory=constructor)
            assert len(captured) == 1
            for track in json.loads(captured[0].take_tracks()):
                if len(track["frames"]) >= 5:
                    print(json.dumps(track,separators=(",",":")),file=stream)
    target.with_suffix(".tmp").replace(target)


def consolidate(root):
    identities = {}
    for split in ("train","dev","test"):
        identity = set()
        coverage = Counter()
        turns = defaultdict(list)
        counts = Counter()
        target = root/f"{split}.jsonl"
        first = root/f"source-data/{split}-script-0.jsonl"
        header = json.loads(first.open().readline())
        chunks = sorted((root/"source-data").glob(f"{split}-*.jsonl"))
        assert len(chunks) == 9
        with target.with_suffix(".tmp").open("w") as dst:
            print(json.dumps(header,separators=(",",":")),file=dst)
            for path in chunks:
                with path.open() as src:
                    if "-script-" in path.name:
                        assert json.loads(src.readline()) == header
                    for line in src:
                        row = json.loads(line)
                        counts[row["source"]] += 1
                        identity.add(row["seed"])
                        coverage[row["lobby"]] += 1
                        for frame in row["frames"]:
                            t,p = frame["turn"], frame["player"]
                            power = sum(m["atk"]+m["hp"] for m in p["board"])
                            turns[t].append((power,row["source"]))
                        dst.write(line)
        assert len(coverage) == 252
        for seen in identities.values():
            assert not seen.intersection(identity)
        identities[split] = identity
        target.with_suffix(".tmp").replace(target)
        receipt = dict(source_lobbies=len(identity), combinations=len(coverage),
                       min_lobby_tracks=min(coverage.values()), counts=counts,
                       sha256=ppo.sha256(target), rounds={
                       t:{s:[len(a),*np.quantile(a,[.1,.5,.9,.99]).tolist()] for s in ("all","script","ppo")
                       if (a:= [p for p,origin in rows if s=="all" or origin==s])}
                       for t,rows in sorted(turns.items())})
        (root/f"{split}-receipt.json").write_text(json.dumps(receipt,indent=2))


class Tally:
    def __init__(self):
        self.formations = []
        self.reversals = None
        self.prior = None
        self.reversals_out = []

    def start(self, seeds):
        self.reversals = np.zeros(len(seeds),dtype=np.int64)
        self.prior = np.zeros(len(seeds),dtype=np.bool_)

    def step(self, arena, slots, f, c, mask, actions, choice):
        ids = np.frombuffer(slots,dtype="<u4")
        freeze = actions == 2
        self.reversals[ids] += (self.prior[ids] & freeze)
        self.prior[ids] = freeze
        self.arena = arena
        return actions

    def finish(self, finished):
        self.formations.extend(self.arena.formations())
        self.reversals_out.extend(self.reversals.tolist())


def evaluate(root, panel, field, spec, checkpoint, n, first, destination):
    bc.configure(421)
    fields=Fields(spec)
    tally=Tally()
    games=bc.play(policy(checkpoint),list(range(first,first+n)),field,384,WORKERS,
                  destination.with_suffix(".failure.json"),tally,arena_factory=fields)
    for g,forms,freeze in zip(games,tally.formations,tally.reversals_out,strict=True):
        g["formations"]=forms
        g["freeze_adjacency"]=freeze
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.with_suffix(".tmp").open("w") as out:
        for game in games:
            print(json.dumps(game,separators=(",",":")),file=out)
    destination.with_suffix(".tmp").replace(destination)
    stats=ppo.game_metrics(games)
    stats["formation_rate"]=sum(bool(g["formations"]) for g in games)/n
    stats["freeze_adjacencies_per_game"]=sum(g["freeze_adjacency"] for g in games)/n
    stats["turns"]={t:bc.round_summary(games,t) for t in (5,8,10,12,14,16)}
    stats["formations"]=Counter(f for g in games for f in g["formations"])
    receipt={"panel":panel,"field":field,"checkpoint_sha256":ppo.sha256(checkpoint),
             "first_seed":first,"engine":taverngym.build_info()["source_hash"],
             "environment":fields.identity[field],"stats":stats}
    destination.with_suffix(".json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    s=p.add_subparsers(dest="mode",required=True)
    c=s.add_parser("collect")
    c.add_argument("split")
    c.add_argument("checkpoint",type=Path)
    s.add_parser("consolidate")
    e=s.add_parser("evaluate")
    for key in ("panel","field"):
        e.add_argument(key)
    for key in ("spec","checkpoint"):
        e.add_argument(key,type=Path)
    e.add_argument("n",type=int)
    e.add_argument("first",type=int)
    e.add_argument("destination",type=Path)
    a=p.parse_args()
    if a.mode=="collect":
        collect(a.root,a.split,a.checkpoint)
    elif a.mode=="consolidate":
        consolidate(a.root)
    else:
        a=vars(a); del a["mode"]; evaluate(**a)

if __name__ == "__main__":
    main()
