"""Single paired intervention on the first late, affordable spell passed at low gold."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.env.action_space_std import A_BUY, A_END, A_REFRESH, N_SHOP
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare
from workflows.peer_finetune_holdout import read
from workflows.peer_health_switch import field
from workflows.peer_ppo_panel import trial
from workflows.peer_stop_learning import BASE,BASE_SHA

START,N = 351410000,512


class SpendOnSpell:
    def __init__(self,policy,intervene):
        self.policy = policy
        self.evaluate = policy.evaluator
        self.model = self.evaluate.model
        self.intervene = intervene
        policy.evaluator = self

    def reset(self):
        self.case = None
        self.policy.reset()

    def decide(self,game,seat):
        self.game = game
        self.proposed = None
        decision = self.policy.decide(game,seat)
        if self.proposed and not decision.info.get("forced"):
            expected = self.proposed["alternate"] if self.intervene else self.proposed["original"]
            if decision.action != expected:
                raise ValueError("controller did not execute the recorded decision")
            self.case = self.proposed
        return decision

    def __call__(self,features,cards,mask,seat):
        prior,value = self.evaluate(features,cards,mask,seat)
        original = int(prior.argmax())
        if self.case or self.game.turn < 7 or original not in (A_END,A_REFRESH):
            return prior,value
        player = json.loads(self.game.player_json(seat))
        if player["gold"] not in (1,2):
            return prior,value
        slots = [i for i,item in enumerate(player["shop"][:N_SHOP]) if mask[A_BUY+i] and "Spell" in item["item"]]
        if not slots:
            return prior,value
        slot = max(slots,key=lambda i:float(prior[A_BUY+i]))
        item = player["shop"][slot]["item"]["Spell"]
        self.proposed = dict(turn=self.game.turn,gold=player["gold"],hp=player["health"]+player["armor"],
            board=len(player["board"]),hand=len(player["hand"]),original=original,alternate=A_BUY+slot,
            card=item["card"],original_prior=float(prior[original]),alternate_prior=float(prior[A_BUY+slot]))
        if self.intervene:
            altered = np.zeros_like(prior)
            altered[A_BUY+slot] = 1
            return altered,value
        return prior,value


def shard(out,i,seed_start,games):
    import torch
    torch.set_num_threads(1)
    proto,digest = field()
    plan = read(out/"plan.json")
    if (proto.sha256(),digest) != (plan["protocol"],plan["library"]):
        raise ValueError("worker field differs")
    pool = MirrorSnapshotPool(proto.mirror_snaps,sample=proto.mirror_sample)
    def policy(change):
        return SpendOnSpell(make_eval_policy(str(BASE/"model.pt"),protocol=proto,device="cuda:0")["policy"],change)
    base,alternative = policy(False),policy(True)
    rows,cases = [],[]
    for seed in range(seed_start+i,seed_start+games,8):
        a = trial(base,pool,seed,event_turns=True)
        b = trial(alternative,pool,seed,event_turns=True)
        if a["lobby"] != b["lobby"] or base.case != alternative.case or (base.case is None and a != b):
            raise ValueError(f"paired intervention mismatch at {seed}")
        rows.append(dict(parent=a,candidate=b))
        cases.append(dict(seed=seed,case=base.case))
    atomic_json(out/f"panel-{i}.json",rows)
    atomic_json(out/f"cases-{i}.json",cases)


def run(out,seed_start=START,games=N):
    source = source_identity(ROOT)
    if (source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"] != BASE_SHA
            or games <= 0 or games % 8):
        raise ValueError("wrong source or reference policy")
    proto,lib = field()
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,parent=BASE_SHA,protocol=proto.sha256(),library=lib,
        seed=seed_start,games=games,job=os.environ["SLURM_JOB_ID"],engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),rule="First T7+ legal shop spell with 1-2 gold when greedy would pass or refresh. Buy highest policy-probability spell once, then ordinary policy.",
        gate="All-game utility lower paired 95% CI > 0 and common T12 log strength lower CI >= -0.12."))
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(out),"--worker",str(i),"--seed",str(seed_start),
                            "--games",str(games)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(8) as workers:
        list(workers.map(launch,range(8)))
    report = compare(out,seed_start,games)
    cases = [row["case"] for i in range(8) for row in read(out/f"cases-{i}.json") if row["case"]]
    report.update(affected=len(cases),refresh=sum(c["original"]==A_REFRESH for c in cases),
                  passed=report["delta"]["utility"]["ci95"][0]>0 and report["delta"]["log12"]["ci95"][0]>=-.12)
    atomic_json(out/"complete.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--seed",type=int,default=START)
    parser.add_argument("--games",type=int,default=N)
    args=parser.parse_args()
    if args.worker is None:
        run(args.out,args.seed,args.games)
    else:
        shard(args.out,args.worker,args.seed,args.games)
