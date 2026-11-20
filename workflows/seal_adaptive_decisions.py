"""Independently reconstruct adaptive decision effects, restore safeguards, and retain all seeds."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import mean,median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.summarize_peer_finetune import UTILITY,estimate
from workflows.adaptive_decision_study import read


def statistics(data):
    a = [r[0] for r in data]
    b = [r[1] for r in data]
    rank = [x["rank"]-y["rank"] for x,y in data]
    both = [(x,y) for x,y in data if all("12" in g["boards"] for g in (x,y))]
    def change(x,y,t):
        return math.log1p(y["boards"][t])-math.log1p(x["boards"][t])
    growth = [change(x,y,"12")-change(x,y,"9") for x,y in both]
    curves = {}
    for name,rows in (("base",a),("candidate",b)):
        curves[name] = {}
        for t in ("8","9","10","11","12","13","14"):
            vals = [r["boards"][t] for r in rows if t in r["boards"]]
            curves[name][t] = dict(n=len(vals),p50=median(vals) if vals else None)
    return dict(n=len(data),changed=sum(x!=0 for x in rank),rank_gain=estimate(rank),
        mean_abs_rank=mean(map(abs,rank)) if rank else None,utility=estimate([UTILITY[y["rank"]-1]-UTILITY[x["rank"]-1] for x,y in data]),
        growth=estimate(growth),log12=estimate([change(x,y,"12") for x,y in both]),
        ranks={k:dict(Counter(r["rank"] for r in v)) for k,v in (("base",a),("candidate",b))},
        mean_ranks={k:mean(r["rank"] for r in v) if v else None for k,v in (("base",a),("candidate",b))},curves=curves)


def main(run,bank,out):
    plan = read(run/"plan.json")
    prior = read(bank/"plan.json")
    report = read(run/"complete.json")
    if plan["bank_plan"]!=file_identity(bank/"plan.json") or plan["bank_seal"]!=file_identity(bank/"complete.json"):
        raise ValueError("bank evidence changed")
    cards = read(ROOT/"engine/crates/hsbg-engine/data/cards.json")["cards"]
    fields,hashes = {},{}
    for name in plan["rules"]:
        data,triggered = [],[]
        tags = Counter()
        for i in range(prior["n"]):
            path=run/f"game-{i}.json"
            digest=file_identity(path)
            if i in hashes and hashes[i]!=digest:
                raise ValueError("shard changed on reread")
            hashes[i]=digest
            r=read(path)
            a=r["baseline"];arm=r["alternatives"][name];b=arm["result"];e=arm["trigger"]
            expected=read(bank/f"base-{i}.json")["natural"]
            if (any(expected[k]!=v for k,v in a.items() if k!="overrides")
                    or a["seed"]!=b["seed"] or a["seed"]!=prior["seed"]+i or a["focus"]!=b["focus"] or a["focus"]!=i%8
                    or any(sorted(g["placements"])!=list(range(1,9)) or g["placements"][i%8]!=g["rank"] for g in (a,b))):
                raise ValueError("unmatched or invalid terminal game")
            if e is None and a!=b:
                raise ValueError("untriggered arm differs")
            if e is not None:
                if len(e["prefix"])!=64 or len(e["state"])!=64 or e["alternative"]==e["original"]:
                    raise ValueError("missing matched intervention")
                triggered.append((a,b))
                if "cards" in e:
                    tags["all_greater"]+=all(cards[k]["kind"]=="trinket_greater" for k in e["cards"])
                    d=e["costs"][e["original"]-30]-e["costs"][e["alternative"]-30]
                    tags["cheaper_by_two"]+=d>=2
                if "kind" in e:
                    tags[e["kind"]]+=1
            data.append((a,b))
        s=statistics(data);q=report[name]
        for k,v in (("rank_gain","rank_gain"),("utility","utility"),("growth","common_growth"),("log12","common_log12")):
            if abs(s[k]["mean"]-q[v]["mean"])>1e-10:
                raise ValueError("paired statistics differ")
        reach=s["curves"]["candidate"]["12"]["n"]-s["curves"]["base"]["12"]["n"]
        gate=len(triggered)>=12 and s["rank_gain"]["mean"]>=.08 and s["growth"]["mean"]>0 and s["log12"]["mean"]>0 and reach>=-2
        fields[name]=dict(all=s,affected=statistics(triggered),tags=tags,passed_screen=gate)
    passing=[k for k in fields if fields[k]["passed_screen"]]
    out.mkdir(exist_ok=False)
    atomic_json(out/"seal.json",dict(job=os.environ["SLURM_JOB_ID"],plan=file_identity(run/"plan.json"),
        report=file_identity(run/"complete.json"),shards=hashes,fields=fields,
        selected=max(passing,key=lambda k:fields[k]["all"]["rank_gain"]["mean"]) if passing else None))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("run","bank","out"):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    main(a.run,a.bank,a.out)
