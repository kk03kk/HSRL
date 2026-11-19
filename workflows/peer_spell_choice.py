"""Learn the conditional terminal value of one affordable spell purchase."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare,estimate
from workflows.peer_finetune_holdout import holdout_protocol,read
from workflows.peer_ppo_panel import trial
from workflows.peer_spell_purchase import SpendOnSpell
from workflows.peer_stop_learning import BASE,BASE_SHA

RUNS = Path("artifacts/runs")
FIT = RUNS/"nus-peer-spell-fit-20261120-1"
INPUTS = (("nus-peer-small-spell-20261120-1",351410000,512),("nus-peer-spell-labels-20261120-1",354410000,1024))
FIELDS = {"hold": (357410000,1024),"transfer": (360410000,512)}
DEV_PROTOCOL = "ee216399c6892c6ce2935368899fbea767f90dbb55bdbaf932051858620390c1"


def context(c):
    health = min(c["hp"]/40,1.5)
    board = c["board"]/7
    hand = c["hand"]/10
    gold = c["gold"]-1
    gap = math.log(max(1e-12,c["original_prior"])/max(1e-12,c["alternate_prior"]))
    return [1,min(c["turn"]/15,1.5),gold,health,board,hand,float(c["original"]==0),
            min(max(gap/5,0),2),hand*gold,health*board]


def fit_half(rows,cards):
    n_context = len(context(rows[0][1]))
    x = np.zeros((len(rows),n_context+len(cards)),np.float64)
    for i,(_,c,_) in enumerate(rows):
        x[i,:n_context] = context(c)
        x[i,n_context+cards.index(c["card"])] = 1
    y = np.asarray([d for _,_,d in rows])
    # Fixed pseudo-observation penalties for the intercept, context and card effects.
    penalty = np.array([4]+[32]*(n_context-1)+[24]*len(cards))
    w = np.linalg.solve(x.T@x+np.diag(penalty),x.T@y)
    if not np.isfinite(w).all():
        raise ValueError("nonfinite choice model")
    return dict(n=len(rows),context=w[:n_context].tolist(),card={str(k):float(v) for k,v in zip(cards,w[n_context:])})


def select(model,c):
    card = str(c["card"])
    if model["support"].get(card,0)<16:
        return False
    f = context(c)
    predictions = [sum(a*b for a,b in zip(f,arm["context"],strict=True))+arm["card"][card] for arm in model["halves"]]
    return min(predictions)>0 and sum(predictions)/2>=.05


def fit():
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"] != BASE_SHA:
        raise ValueError("immutable source and base required")
    rows,ids = [],{}
    for name,start,n in INPUTS:
        run = RUNS/name
        plan,report = read(run/"plan.json"),read(run/"complete.json")
        if (plan["seed"],plan["games"],plan["parent"],plan["protocol"],report["n"]) != (start,n,BASE_SHA,DEV_PROTOCOL,n):
            raise ValueError("wrong training inputs")
        hashes={p.name:file_identity(p) for p in (run/"plan.json",run/"complete.json")}
        coverage=[]
        for i in range(8):
            cases,panel = run/f"cases-{i}.json",run/f"panel-{i}.json"
            hashes.update({p.name:file_identity(p) for p in (cases,panel)})
            for r,pair in zip(read(cases),read(panel),strict=True):
                a,b = pair["parent"],pair["candidate"]
                if a["seed"] != r["seed"] or b["seed"] != r["seed"] or a["lobby"] != b["lobby"]:
                    raise ValueError("unpaired training target")
                coverage.append(r["seed"])
                if r["case"]:
                    rows.append((r["seed"],r["case"],b["metrics"]["utility"]-a["metrics"]["utility"]))
        if sorted(coverage)!=list(range(start,start+n)):
            raise ValueError("incomplete training labels")
        ids[name]=hashes
    rows.sort(key=lambda r:r[0])
    counts = Counter(c["card"] for _,c,_ in rows)
    halves = [fit_half([r for r in rows if r[0]%2==i],sorted(counts)) for i in range(2)]
    if len({r[0] for r in rows})!=len(rows):
        raise ValueError("duplicate training case")
    FIT.mkdir(exist_ok=False)
    atomic_json(FIT/"choice.json",dict(schema="hsai.spell-choice/v1",source=source,parent=BASE_SHA,
        support={str(k):v for k,v in counts.items()},halves=halves,inputs=ids,job=os.environ["SLURM_JOB_ID"]))


def field(split):
    proto,pool = holdout_protocol()
    if split=="hold":
        return proto,pool
    root=Path("artifacts/libraries/human-turn-transfer-20261119-01")
    if library_digest(root)!="8a769ef0f43fc8edf2a718514898b1ae4e60831aa8bf37355e11c4d2627c4be0":
        raise ValueError("transfer field changed")
    return replace(proto,mirror_snaps=root.as_posix()),MirrorSnapshotPool(root,sample=proto.mirror_sample)


def shard(out,split,i):
    import torch
    torch.set_num_threads(1)
    proto,pool = field(split)
    plan = read(out/"plan.json")
    if proto.sha256()!=plan["protocol"] or file_identity(FIT/"choice.json")!=plan["choice"]:
        raise ValueError("evaluation identity differs")
    model = read(FIT/"choice.json")
    selector = lambda c:select(model,c)
    def actor(intervene,choose):
        native=make_eval_policy(str(BASE/"model.pt"),protocol=proto,device="cuda:0")["policy"]
        return SpendOnSpell(native,intervene,choose)
    observe,learned,always = actor(False,selector),actor(True,selector),actor(True,None)
    rows,cases = [],[]
    start,n = FIELDS[split]
    for seed in range(start+i,start+n,8):
        a,b = [trial(p,pool,seed,event_turns=True) for p in (observe,learned)]
        case=observe.case
        if case!=learned.case or a["lobby"]!=b["lobby"] or ((not case or not case["selected"]) and a!=b):
            raise ValueError(f"learned policy intervention differs at {seed}")
        blind=b
        if case and not case["selected"]:
            blind=trial(always,pool,seed,event_turns=True)
            if always.case!=(case|{"selected":True}) or blind["lobby"]!=a["lobby"]:
                raise ValueError(f"unconditional comparator differs at {seed}")
        rows.append(dict(parent=a,candidate=b))
        cases.append(dict(seed=seed,case=case,blind_utility=blind["metrics"]["utility"]))
    atomic_json(out/f"panel-{i}.json",rows)
    atomic_json(out/f"choices-{i}.json",cases)


def assess(out,split):
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"]!=BASE_SHA or read(FIT/"choice.json")["parent"]!=BASE_SHA:
        raise ValueError("wrong release or policy")
    if split=="transfer" and not read(RUNS/"nus-peer-spell-choice-hold-20261120-1"/"complete.json")["passed"]:
        raise ValueError("hold gate failed")
    proto,_=field(split)
    start,n=FIELDS[split]
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,parent=BASE_SHA,choice=file_identity(FIT/"choice.json"),
        job=os.environ["SLURM_JOB_ID"],protocol=proto.sha256(),library=library_digest(proto.mirror_snaps),
        seed=start,games=n,split=split,engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),gate="positive utility lower CI; log12 lower >= -0.12; learned >= always-buy mean utility"))
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,"assess",str(out),"--split",split,"--worker",str(i)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(launch,range(8)))
    report=compare(out,start,n)
    chosen,offered=0,0
    vs_blind,blind_gain=[],[]
    for i in range(8):
        for row,metadata in zip(read(out/f"panel-{i}.json"),read(out/f"choices-{i}.json"),strict=True):
            assert row["parent"]["seed"]==metadata["seed"]
            c=metadata["case"]
            offered+=int(bool(c))
            chosen+=int(bool(c and c["selected"]))
            vs_blind.append(row["candidate"]["metrics"]["utility"]-metadata["blind_utility"])
            blind_gain.append(metadata["blind_utility"]-row["parent"]["metrics"]["utility"])
    report.update(chosen=chosen,offered=offered,vs_blind=estimate(vs_blind),blind_gain=estimate(blind_gain))
    report["passed"] = bool(report["delta"]["utility"]["ci95"][0]>0 and report["delta"]["log12"]["ci95"][0]>=-.12 and report["vs_blind"]["mean"]>=0)
    atomic_json(out/"complete.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("fit","assess"))
    parser.add_argument("out",type=Path,nargs="?")
    parser.add_argument("--split",choices=FIELDS,default="hold")
    parser.add_argument("--worker",type=int)
    args=parser.parse_args()
    if args.mode=="fit":
        fit()
    elif args.worker is None:
        assess(args.out,args.split)
    else:
        shard(args.out,args.split,args.worker)
