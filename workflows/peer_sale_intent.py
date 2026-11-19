"""Learn whether an ordinary late sale is immediately followed by Refresh from public pre-sale data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare
from workflows.peer_health_switch import field
from workflows.peer_finetune_holdout import holdout_protocol,read
from workflows.peer_ppo_panel import trial
from workflows.peer_stop_learning import BASE,BASE_SHA

RUNS=Path("artifacts/runs")
FIT=RUNS/"nus-peer-sale-intent-fit-20261120-1"
SCHEMA=("turn","effective_health","board_count","sold_share","hand_count","playable_hand_count",
        "tier","one_gold_offers","sale_log_margin")
START=365410000
CARDS=Path("engine/crates/hsbg-engine/data/cards.json")
DEVELOPMENT=RUNS/"nus-peer-sale-intent-effect-20261120-1"
PANEL_SEEDS={"development":(368410000,512),"hold":(371410000,768),"market":(374410000,512)}


def panel(split):
    if split=="development":
        return field()
    from hsbg.az.train.personal import library_digest
    if split=="hold":
        proto,_=holdout_protocol()
    else:
        from workflows.peer_minion_transfer import protocol
        proto=protocol(RUNS/"nus-peer-market-build-20261120-1")
        if proto.sha256()!="e1ecbed8489a0a2185b1eb4d9318c1065169fb486fd536c927a617a76e7689dd":
            raise ValueError("expected the sensitivity field with verified market prices")
    return proto,library_digest(proto.mirror_snaps)


def probability(fit,features):
    x=(np.array(features)-fit["center"])/fit["scale"]
    return 1/(1+math.exp(-float(np.clip(fit["bias"]+x@fit["coef"],-30,30))))


class SaleIntent:
    def __init__(self,policy,fit=None,intervene=False):
        self.policy=policy
        self.evaluate=policy.evaluator
        self.model=self.evaluate.model
        policy.evaluator=self
        self.fit=fit
        self.intervene=intervene
        self.costs=[v["cost"] for v in read(CARDS)["cards"]]

    def reset(self):
        self.policy.reset()
        self.events=[]
        self.pending=None
        self.last_turn=None
        self.index=0

    def decide(self,game,seat):
        self.game=game
        self.proposed=None
        choice=self.policy.decide(game,seat)
        if self.pending is not None and choice.action<196:
            if self.pending["turn"]!=game.turn:
                raise ValueError("missing actual economic action before next turn")
            self.pending["next"]=int(choice.action)
            self.pending=None
        if self.proposed and not choice.info.get("forced"):
            row=self.proposed
            if choice.action!=row["executed"]:
                raise ValueError("pre-sale proposal disagrees with the actual action")
            self.events.append(row)
            self.last_turn=game.turn
            if not row["changed"]:
                self.pending=row
        self.index+=1
        return choice

    def __call__(self,f,c,mask,seat):
        p,v=self.evaluate(f,c,mask,seat)
        a=int(p.argmax())
        if self.last_turn==self.game.turn or self.game.turn<7 or not 16<=a<23:
            return p,v
        pl=json.loads(self.game.player_json(seat))
        board=pl["board"]
        hp=pl["health"]+pl["armor"]
        if pl["gold"]!=0 or ((hp<=15)==(len(board)<=4)):
            return p,v
        sizes=[m["atk"]+m["hp"] for m in board]
        total=max(1,sum(sizes))
        if sizes[a-16]*100<15*total:
            return p,v
        rem=mask.copy()
        for j,size in enumerate(sizes):
            if size*100>=15*total:
                rem[16+j]=False
        if not rem.any():
            return p,v
        shop=sum("Spell" in s["item"] and 0<self.costs[s["item"]["Spell"]["card"]]<=1 for s in pl["shop"])
        playable=int(mask[46:126].reshape(10,8).any(1).sum())
        gap=math.log(max(1e-12,float(p[a]))/max(1e-12,float(p[rem].max())))
        features=[self.game.turn,hp,len(board),sizes[a-16]/total,len(pl["hand"]),playable,pl["tier"],shop,min(12,gap)]
        select=self.fit is not None and probability(self.fit,features)>=.8
        changed=bool(self.intervene and select)
        if changed:
            p,v=self.evaluate(f,c,rem,seat)
        self.proposed=dict(index=self.index,turn=self.game.turn,features=features,original=a,executed=int(p.argmax()),
                           selected=select,changed=changed)
        return p,v


def fit_classifier(train):
    x=np.array([e["features"] for e in train],float)
    center=x.mean(0);scale=x.std(0);scale[scale<1e-7]=1.
    design=np.column_stack((np.ones(len(x)),(x-center)/scale))
    labels=np.array([e["next"]==1 for e in train],float)
    weights=np.zeros(design.shape[1]);penalty=np.diag([.001]+[8.]*x.shape[1])
    for _ in range(32):
        pred=1/(1+np.exp(-np.clip(design@weights,-30,30)))
        grad=design.T@(pred-labels)+penalty@weights
        hessian=(design.T*(pred*(1-pred)))@design+penalty
        step=np.linalg.solve(hessian,grad);weights-=step
        if np.linalg.norm(step)<1e-8:break
    return dict(features=SCHEMA,center=center.tolist(),scale=scale.tolist(),bias=float(weights[0]),coef=weights[1:].tolist())


def fit_model(rows):
    missing=sum("next" not in e for r in rows for e in r["events"])
    if missing:
        raise ValueError(f"{missing} sales lack a subsequent observed economic decision")
    train=[e for r in rows if (r["seed"]-START)%3!=2 for e in r["events"]]
    check=[(r["seed"],e) for r in rows if (r["seed"]-START)%3==2 for e in r["events"]]
    if len(train)<40 or not check:
        return dict(passed=False,training=len(train),check=len(check),reason="insufficient natural sales")
    spec=fit_classifier(train)
    chosen=[(s,e) for s,e in check if probability(spec,e["features"])>=.8]
    n=len(chosen);correct=sum(e["next"]==1 for _,e in chosen)
    acc=correct/n if n else 0.
    z=1.96
    lower=((acc+z*z/(2*n))-z*math.sqrt((acc*(1-acc)+z*z/(4*n))/n))/(1+z*z/n) if n else 0.
    return dict(passed=n>=16 and len({s for s,_ in chosen})>=12 and acc>=.8 and lower>=.65,model=spec,
        training=len(train),check=len(check),checked_actions=dict(Counter(e["next"] for _,e in check)),
        selected=n,correct=correct,selected_games=len({s for s,_ in chosen}),precision=acc,wilson_lower=lower)


def shard(out,mode,i):
    import torch
    torch.set_num_threads(1)
    plan=read(out/"plan.json")
    proto,lib=panel(plan.get("split","development"))
    if (proto.sha256(),lib)!=(plan["protocol"],plan["library"]):
        raise ValueError("development field differs")
    pool=MirrorSnapshotPool(proto.mirror_snaps,sample=proto.mirror_sample)
    spec=read(FIT/"complete.json")["model"] if mode=="effect" else None
    def actor(change):
        native=make_eval_policy(str(BASE/"model.pt"),protocol=proto,device="cuda:0")["policy"]
        return SaleIntent(native,spec,change)
    parent=actor(False)
    candidate=actor(True) if mode=="effect" else None
    rows=[];details=[]
    for seed in range(plan["seed"]+i,plan["seed"]+plan["n"],8):
        a=trial(parent,pool,seed,event_turns=True)
        events=parent.events
        if mode=="effect":
            b=trial(candidate,pool,seed,event_turns=True)
            changes=[e for e in candidate.events if e["changed"]]
            eligible=[e for e in events if e["selected"]]
            if bool(changes)!=bool(eligible) or a["lobby"]!=b["lobby"] or (not changes and a!=b):
                raise ValueError("unexplained paired divergence")
            if changes and any(changes[0][k]!=eligible[0][k] for k in ("index","turn","original","features")):
                raise ValueError("first predicted sale differs between arms")
            rows.append(dict(parent=a,candidate=b))
        detail=dict(seed=seed,events=events)
        if candidate is not None:
            detail["interventions"]=changes
        details.append(detail)
    if candidate is not None:atomic_json(out/f"panel-{i}.json",rows)
    atomic_json(out/f"events-{i}.json",details)


def run(out,mode,split):
    if mode=="fit" and split!="development":
        raise ValueError("intent labels can be fitted only on development games")
    source=source_identity(ROOT);proto,lib=panel(split)
    if source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"]!=BASE_SHA:
        raise ValueError("wrong source or model")
    if mode=="effect" and not read(FIT/"complete.json")["passed"]:
        raise ValueError("fresh-seed classifier gate failed")
    if split!="development" and (not read(DEVELOPMENT/"complete.json")["passed"]
            or read(DEVELOPMENT/"plan.json")["classifier"]!=file_identity(FIT/"complete.json")):
        raise ValueError("independent efficacy must pass before confirming the same frozen classifier")
    start,n=(START,768) if mode=="fit" else PANEL_SEEDS[split]
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,model=BASE_SHA,seed=start,n=n,job=os.environ["SLURM_JOB_ID"],
        library=lib,protocol=proto.sha256(),split=split,canonical_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),classifier=file_identity(FIT/"complete.json") if mode=="effect" else None))
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,mode,str(out),"--worker",str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(8) as executor:list(executor.map(launch,range(8)))
    details=[r for i in range(8) for r in read(out/f"events-{i}.json")]
    if sorted(r["seed"] for r in details)!=list(range(start,start+n)):
        raise ValueError("missing trajectories")
    if mode=="fit":
        result=fit_model(details)
    else:
        result=compare(out,start,n)
        result.update(affected=sum(any(e["selected"] for e in r["events"]) for r in details),
            passed=result["delta"]["utility"]["ci95"][0]>0 and result["delta"]["growth9to12"]["ci95"][0]>=0)
    atomic_json(out/"complete.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("fit","effect"))
    parser.add_argument("out",type=Path)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--split",choices=PANEL_SEEDS,default="development")
    args=parser.parse_args()
    if args.worker is None:run(args.out,args.mode,args.split)
    else:shard(args.out,args.mode,args.worker)
