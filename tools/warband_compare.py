"""Quantify paired uncertainty on sealed final panels."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

root=Path(sys.argv[1])
def load(path):
    return json.loads(path.read_text())

def games(path):
    with path.open() as f:
        return [json.loads(line) for line in f]

values = np.array([1,.3,0,-.05,-.1,-.15,-.2,-.25])
def scores(game):
    ranks = np.asarray([g["rank"] for g in game])
    return {
        "rank":ranks,
        "top4":(ranks<=4).astype(float),
        "top2":(ranks<=2).astype(float),
        "first":(ranks==1).astype(float),
        "utility":values[ranks-1],
        "formation":np.asarray([bool(g["formations"]) for g in game],dtype=float),
        "freeze":np.asarray([g["freeze_adjacency"] for g in game],dtype=float),
        "reach12":np.asarray([any(r["turn"]==12 for r in g["rounds"]) for g in game],dtype=float),
    }

def compare(a,b):
    def identity(row):
        return [(g["seed"],g["seat"],g["lobby"]) for g in row]
    assert identity(a)==identity(b)
    rng=np.random.default_rng(81191)
    a,b=scores(a),scores(b)
    result={}
    for k,x in a.items():
        delta=x-b[k]
        d=np.concatenate([delta[rng.integers(len(delta),size=(128,len(delta)))].mean(1) for _ in range(64)])
        result[k]={"delta":float(delta.mean()),"ci95":np.quantile(d,[.025,.975]).tolist(),
                   "ci9875":np.quantile(d,[.00625,.99375]).tolist()}
    return result

def net_metrics(path):
    rows=[load(p) for p in sorted(path.glob("iteration-*.json"))]
    assert rows
    opt=[r["optimization"] for r in rows]
    samples=[r["sampling"] for r in rows]
    return {"iterations":len(rows),"decisions":sum(r["decisions"] for r in samples),
            "max_replay":max(r["before"]["max_abs_log_ratio"] for r in opt),
            "min_kl":min(r["after"]["kl"] for r in opt),
            "max_kl":max(r["after"]["kl"] for r in opt),
            "max_norm":max(r["max_grad_norm"] for r in opt),
            "entropy":float(np.mean([r["entropy"] for r in opt])),
            "updates":sum(r["optimizer_steps"] for r in opt),
            "early_stops":sum(r["stop"]=="target_kl" for r in opt),
            "seconds":sum(r["rollout_seconds"]+r["update_seconds"] for r in samples),
            "frames":sum(g.get("warband",{}).get("frames",0) for r in samples for g in r["fields"].values()),
            "tail_repeats":sum(g.get("warband",{}).get("repeats",0) for r in samples for g in r["fields"].values())}

r={"development":{"environment":load(root/"env-choice.json"),"hyperparameters":load(root/"hyper-choice.json")},
   "training":{p.name:net_metrics(p) for p in sorted((root/"branches").glob("*")) if (p/"summary.json").exists()},
   "final":{},"kl_weight_equality":{}}
for a,b in (("base","nokl"),("lr","lr-nokl")):
    aweights,bweights=(torch.load(root/f"branches/hyper-{key}/latest.pt",weights_only=False,map_location="cpu") for key in (a,b))
    r["kl_weight_equality"][a+"-"+b] = {
        "all_tensors":all(torch.equal(x,bweights["model"][key]) for key,x in aweights["model"].items()),
        "rollouts_equal":aweights["rollout_history"]==bweights["rollout_history"],
    }

for name in ("parent","control","environment","optimizer","human","continuous"):
    receipt={}
    for f in ("bank","human","mixed"):
        them=root/f"test/{name}-{f}.jsonl"
        base=root/f"test/parent-{f}.jsonl"
        control=root/f"test/control-{f}.jsonl"
        receipt[f]={
            "absolute": load(them.with_suffix(".json"))["stats"],
            "paired_parent": compare(games(them),games(base)),
            "paired_control": compare(games(them),games(control)),
        }
    score=receipt["mixed"]["paired_parent"]
    human=receipt["human"]["paired_parent"]
    bank=receipt["bank"]["paired_parent"]
    receipt["gates"]={
        "script_preservation":score["rank"]["ci95"][1]<=.15 and score["top4"]["ci95"][0]>=-.03,
        "human_win_utility":human["utility"]["ci95" if name=="optimizer" else "ci9875"][0]>0,
        "bank_utility":bank["utility"]["ci95"][1]>=0,
        "human_top4":human["top4"]["ci95"][0]>=-.03,
        "freeze":human["freeze"]["ci95"][1]<=.5,
    }
    receipt["qualified"]=all(receipt["gates"].values())
    r["final"][name]=receipt
(root/"final-results.json").write_text(json.dumps(r,indent=2,ensure_ascii=False))
