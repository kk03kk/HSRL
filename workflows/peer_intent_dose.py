"""Check the value of additional independent sale-intent labels before spending on a larger trainer."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows.peer_sale_intent import FIT,DEVELOPMENT,RUNS,START,fit_classifier,probability,read

DOSE=RUNS/"nus-peer-intent-dose-20261120-1"


def observations(root):
    plan=read(root/"plan.json")
    files=[root/f"events-{i}.json" for i in range(8)]
    rows=[r for file in files for r in read(file)]
    if sorted(r["seed"] for r in rows)!=list(range(plan["seed"],plan["seed"]+plan["n"])):
        raise ValueError("incomplete input")
    return rows,[file_identity(f) for f in files]


def fit():
    source=source_identity(ROOT)
    if source["tracked_dirty"]:raise ValueError("dirty sources")
    first,first_sha=observations(FIT)
    second,second_sha=observations(DEVELOPMENT)
    originals=[r for r in first if (r["seed"]-START)%3!=2]
    assert len(originals)==512 and len({r["seed"] for r in first+second})==1280
    ordered=sorted(originals,key=lambda r:sha256(f"hsai-sale-dose-v1/{r['seed']}".encode()).digest())
    sets={"128_games":ordered[:128],"256_games":ordered[:256],"original":originals,"expanded":first+second}
    models={}
    for key,rows in sets.items():
        events=[e for r in rows for e in r["events"]]
        models[key]=dict(games=len(rows),events=len(events),fit=fit_classifier(events))
    original=read(FIT/"complete.json")["model"]
    for k in ("center","scale","bias","coef"):
        np.testing.assert_allclose(models["original"]["fit"][k],original[k],rtol=0,atol=1e-12)
    DOSE.mkdir(exist_ok=False)
    atomic_json(DOSE/"fit.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],models=models,
        model=read(FIT/"plan.json")["model"],fit_inputs=first_sha+second_sha,original=file_identity(FIT/"complete.json")))


def score():
    models=read(DOSE/"fit.json")["models"]
    fields={};checks=[]
    for split in ("hold","market"):
        folder=RUNS/f"nus-peer-sale-intent-{split}-20261120-1"
        rows,inputs=observations(folder)
        original_return_pass=read(folder/"complete.json")["passed"]
        metrics={}
        for name,model in models.items():
            predictions=[(probability(model["fit"],e["features"]),e["next"]==1) for r in rows for e in r["events"]]
            p,y=np.array(predictions).T
            chosen=p>=.8
            tp=float(y[chosen].sum())
            metrics[name]=dict(evaluated=len(y),selected=int(chosen.sum()),precision=tp/max(1,int(chosen.sum())),
                            recall=tp/max(1,int(y.sum())),brier=float(np.mean((p-y)**2)))
        a,b=metrics["original"],metrics["expanded"]
        check=bool(original_return_pass and b["precision"]>=.90 and b["recall"]>a["recall"]
                   and a["brier"]-b["brier"]>=.01)
        fields[split]=dict(inputs=inputs,metrics=metrics,passed=check)
        checks.append(check)
    target=RUNS/"nus-peer-intent-dose-check-20261120-1"
    target.mkdir(exist_ok=False)
    atomic_json(target/"complete.json",dict(job=os.environ["SLURM_JOB_ID"],fit=file_identity(DOSE/"fit.json"),
        fields=fields,confirm_new_effect=all(checks)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("fit","score"))
    if parser.parse_args().mode=="fit":fit()
    else:score()
