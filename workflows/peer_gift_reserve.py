"""Test reserving the third paid Dark Gift for turn ten in otherwise natural games."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare
from workflows.peer_intent_scope import REGISTRY,POLICY
from workflows.peer_ppo_panel import trial
from workflows.peer_sale_intent import panel,read,RUNS

START = 389410000
OUT = RUNS/"nus-peer-gift-reserve-20261120-1"


class ReserveGift:
    def __init__(self,actor):
        self.actor = actor
        self.model = actor.model
        actor.parent.evaluator = self

    def reset(self):
        self.actor.reset()
        self.blocks = []
        self.presses = []

    def decide(self,game,seat):
        self.game = game
        choice = self.actor.decide(game,seat)
        if choice.action == 6:
            self.presses.append(game.turn)
        return choice

    def __call__(self,f,c,mask,seat):
        p,v = self.actor(f,c,mask,seat)
        if self.game.turn >= 10 or int(p.argmax()) != 6:
            return p,v
        uses = json.loads(self.game.player_json(seat))["dark_gift_uses"]
        if uses != 2:
            return p,v
        alt = mask.copy()
        alt[6] = False
        if not alt.any():
            return p,v
        self.blocks.append(self.game.turn)
        self.actor.proposal = None
        return self.actor(f,c,alt,seat)


def shard(i):
    import torch
    torch.set_num_threads(1)
    proto,library = panel("development")
    plan = read(OUT/"plan.json")
    if (proto.sha256(),library) != (plan["protocol"],plan["library"]):
        raise ValueError("wrong development field")
    pool = MirrorSnapshotPool(proto.mirror_snaps,sample=proto.mirror_sample)
    def actor():
        return make_eval_policy("research-policy:"+str(REGISTRY),protocol=proto,device="cuda:0")["policy"]
    parent = actor()
    candidate = ReserveGift(actor())
    pairs, actions = [], []
    for seed in range(START+i,START+plan["n"],8):
        a = trial(parent,pool,seed,event_turns=True)
        b = trial(candidate,pool,seed,event_turns=True)
        prior = [t for t,row in sorted(a["turns"].items()) for _ in range(row["actions"].get("dark_gift",0))]
        if (a["lobby"] != b["lobby"] or (not candidate.blocks and a!=b)
                or (len(candidate.presses)>=3 and candidate.presses[2]<10)
                or bool(candidate.blocks) != (len(prior)>=3 and prior[2]<10)
                or (candidate.blocks and candidate.blocks[0]!=prior[2])):
            raise ValueError("gift reserve changed an unrelated game or native press history differs")
        pairs.append(dict(parent=a,candidate=b))
        actions.append(dict(seed=seed,baseline=prior,reserved=candidate.presses,blocked=candidate.blocks))
    atomic_json(OUT/f"panel-{i}.json",pairs)
    atomic_json(OUT/f"actions-{i}.json",actions)


def main():
    proto,lib = panel("development")
    parent = file_identity(REGISTRY/"policy.json")
    source = source_identity(ROOT)
    if source["tracked_dirty"] or parent["sha256"] != POLICY:
        raise ValueError("unexpected incumbent or release")
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source,model=parent,base=read(REGISTRY/"policy.json")["parent"],
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),live_pin=file_identity(ROOT/"config/live/model.json"),
        protocol=proto.sha256(),library=lib,n=512,seed=START,job=os.environ["SLURM_JOB_ID"],
        rule="Suppress only the third voluntary paid Dark Gift before T10 when another legal choice exists."))
    def launch(i):
        with (OUT/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(launch, range(8)))
    result = compare(OUT, START,512)
    records = [r for i in range(8) for r in read(OUT/f"actions-{i}.json")]
    result.update(affected=sum(bool(r["blocked"]) for r in records),
        candidate_used_third=sum(len(r["reserved"])>=3 for r in records),
        passed=result["delta"]["utility"]["ci95"][0]>0 and result["delta"]["growth9to12"]["ci95"][0]>=0)
    atomic_json(OUT/"complete.json",result)


if __name__ == "__main__":
    if len(sys.argv)==2:
        shard(int(sys.argv[1]))
    else:
        main()
