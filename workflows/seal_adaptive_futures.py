"""Analyze repeated decision futures with the original lobby state as the sampling unit."""
from collections import Counter
import math
import os
from pathlib import Path
from statistics import mean
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.summarize_peer_finetune import UTILITY
from workflows.adaptive_trinket_futures import OUT,PREVIOUS,RULE,read


def values(a,b):
    def score(g):
        return math.log1p(g["boards"].get("12",0))
    common=all("12" in g["boards"] for g in (a,b))
    log12=score(b)-score(a)
    growth=log12-math.log1p(b["boards"]["9"])+math.log1p(a["boards"]["9"]) if common else 0.
    return (a["rank"]-b["rank"],UTILITY[b["rank"]-1]-UTILITY[a["rank"]-1],log12,
        int("12" in b["boards"])-int("12" in a["boards"]),growth,float(common))


def grouped(rows,draws):
    v=np.asarray([np.sum(state,axis=0) for state in rows],float)
    n=sum(map(len,rows));k=len(rows[0])
    def measure(total,den):
        return np.column_stack((total[:,:4]/den,total[:,4]/np.maximum(1,total[:,5])))
    actual=measure(v.sum(0)[None,:],n)[0]
    sims=measure(v[draws].sum(1),len(rows)*k)
    out={}
    for j,name in enumerate(("rank_gain","utility","log12_including_elimination","reach12","common_growth")):
        keep=sims[:,j] if j!=4 else sims[v[draws,5].sum(1)>0,j]
        out[name]=dict(mean=float(actual[j]),ci95=np.quantile(keep,[.025,.975]).tolist()) if len(keep) else None
    return dict(states=len(rows),pairs=n,common=int(v[:,5].sum()),effects=out)


def main():
    plan=read(OUT/"plan.json");seal=read(OUT/"complete.json")
    ids=plan["indices"];future={i:[] for i in ids}
    natural=[];source_states=[]
    for i in ids:
        old=read(PREVIOUS/f"game-{i}.json")
        natural.append(values(old["baseline"],old["alternatives"][RULE]["result"])[0])
        source_states.append(old["alternatives"][RULE]["trigger"]["state"])
        for rep in range(4):
            path=OUT/f"pair-{i}-{rep}.json"
            if file_identity(path)!=seal["interventions"][path.name]:
                raise ValueError("replicated game changed")
            r=read(path)
            if r["event"]!=old["alternatives"][RULE]["trigger"] or r["rep"]!=rep:
                raise ValueError("replicate belongs to another action")
            for g in (r["parent"],r["candidate"]):
                if g["seed"]!=old["seed"] or sorted(g["placements"])!=list(range(1,9)) or g["rank"]!=g["placements"][i%8]:
                    raise ValueError("invalid terminal lobby")
            future[i].append(r)
    if len(set(source_states))!=32 or Counter(i%8 for i in ids)!=Counter({s:4 for s in range(8)}):
        raise ValueError("invalid state sampling unit")
    rng=np.random.default_rng(2026112002)
    draws=np.concatenate([rng.choice(np.flatnonzero(np.array(ids)%8==s),(10000,4),replace=True) for s in range(8)],axis=1)
    groups=[];oracle=[];chosen=[];state_rows=[];control_abs=[];flip_abs=[]
    for i,initial in zip(ids,natural):
        raw=future[i]
        arm=[values(r["parent"],r["candidate"]) for r in raw]
        groups.append(arm)
        take=mean(v[0] for v in arm[:2])>0 and mean(v[2] for v in arm[:2])>=0
        picked=[values(r["parent"],r["candidate"] if take else r["parent"]) for r in raw[2:]]
        oracle.append(picked);chosen.append(take)
        state_rows.append(dict(index=i,natural_gain=initial,rep_rank=[v[0] for v in arm],
            fit_mean_rank=mean(v[0] for v in arm[:2]),held_mean_rank=mean(v[0] for v in arm[2:]),take=take))
        control_abs.extend(abs(raw[j]["parent"]["rank"]-raw[k]["parent"]["rank"]) for j,k in ((0,1),(2,3)))
        flip_abs.extend(abs(v[0]) for v in arm)
    report=grouped(oracle,draws)
    signs=Counter()
    for s in state_rows:
        for tag,x,y in (("natural_vs_new",s["natural_gain"],mean(s["rep_rank"])),
                        ("fit_vs_held",s["fit_mean_rank"],s["held_mean_rank"])):
            signs[tag+("_tie" if x*y==0 else "_agree" if x*y>0 else "_disagree")]+=1
    out=OUT.parent/"nus-adaptive-trinket-future-seal-20261120-1"
    out.mkdir(exist_ok=False)
    atomic_json(out/"states.json",state_rows)
    passed=sum(chosen)>=8 and all(report["effects"][k]["ci95"][0]>0 for k in ("rank_gain","common_growth"))
    atomic_json(out/"complete.json",dict(job=os.environ["SLURM_JOB_ID"],plan=file_identity(OUT/"plan.json"),
        source_seal=file_identity(OUT/"complete.json"),states=file_identity(out/"states.json"),choice_pairs=grouped(groups,draws),
        conditional_held=report,states_selected=sum(chosen),signs=signs,mean_abs_rank_same_policy_new_rng=mean(control_abs),
        mean_abs_rank_change_choice_paired_rng=mean(flip_abs),learn_public_selector=passed))


if __name__=="__main__":
    main()
