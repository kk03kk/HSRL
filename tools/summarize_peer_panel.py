"""Summarize the paired peer-model panel, preserving reach and matched outcomes."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


def quant(values, q=.5):
    return sorted(values)[round(q*(len(values)-1))] if values else None


def interval(values):
    a=np.array(values)
    if not len(a):
        return None
    bootstrap=np.random.default_rng(34152).choice(a,(10000,len(a))).mean(1)
    return [float(a.mean()),*np.quantile(bootstrap,[.025,.975]).tolist()]


def analyze(root):
    rows=[json.loads(p.read_bytes()) for p in sorted(root.glob("shard-*/8*.json"))]
    keys=("base","peer","peer_free")
    assert len(rows)==128 and all(set(r)==set(keys) for r in rows)
    for key in keys:
        assert sorted(r[key]["seed"] for r in rows)==list(range(82370000,82370128))
    summary={}
    for key in keys:
        arms=[r[key] for r in rows]
        turns={}
        for t in range(1,19):
            boards=[a["turns"][str(t)] for a in arms if str(t) in a["turns"]]
            total=[b["atk"]+b["hp"] for b in boards]
            turns[t]=dict(n=len(boards),p50=quant(total),p90=quant(total,.9),
                attack=quant([v["atk"] for v in boards]),health=quant([v["hp"] for v in boards]),
                ge2796=sum(v>=2796 for v in total), ge5796=sum(v>=5796 for v in total))
        def metric(name):
            return sum(r["metrics"][name] for r in arms)
        pig=[a for a in arms if a["lobby"]&128]
        cohorts={}
        for label, subset in (("pig",pig),("no_pig",[a for a in arms if not a["lobby"]&128])):
            t12=[a["turns"]["12"] for a in subset if "12" in a["turns"]]
            cohorts[label]=dict(n=len(subset),rank=np.mean([a["metrics"]["rank"] for a in subset]),
                sustained=sum(a["metrics"]["repeat_pig"] for a in subset),reach12=len(t12),
                body12=quant([v["atk"]+v["hp"] for v in t12]))
        summary[key]=dict(n=len(arms),rank=metric("rank")/len(arms),
            counts={k:metric(k) for k in ("top4","top2","win","repeat_pig","pig_bought","pig_offered",
                                          "repeat_element","mech","dragon")},
            turns=turns,cohorts=cohorts,
            max_t12=max(((a["turns"]["12"]["atk"]+a["turns"]["12"]["hp"],a["seed"])
                for a in arms if "12" in a["turns"]),default=None),
            actions=dict(sum((Counter(b["actions"]) for a in arms for b in a["turns"].values()),Counter())))
    pairs={}
    for a,b in (("peer","base"),("peer_free","base"),("peer_free","peer")):
        d={k:interval([r[a]["metrics"][k]-r[b]["metrics"][k] for r in rows])
           for k in ("rank","top4","top2","win")}
        common=[r for r in rows if all("12" in r[k]["turns"] for k in (a,b))]
        body=lambda r,k:r[k]["turns"]["12"]["atk"]+r[k]["turns"]["12"]["hp"]
        d["common_t12"]=dict(n=len(common),a=quant([body(r,a) for r in common]),
            b=quant([body(r,b) for r in common]),median_delta=quant([body(r,a)-body(r,b) for r in common]))
        pairs[a+"-"+b]=d
    return dict(models=summary,paired=pairs)


if __name__=="__main__":
    print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
