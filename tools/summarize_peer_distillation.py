"""Summarize paired natural outcomes and identically restored safety cohorts."""
import json
from pathlib import Path
import sys

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def p50(values):
    return sorted(values)[round(.5*(len(values)-1))] if values else None


def estimate(values):
    a=np.array(values,dtype=float)
    if not len(a):
        return {"n":0}
    rng=np.random.default_rng(859501)
    bootstrap=np.concatenate([a[rng.integers(len(a),size=(500,len(a)))].mean(1) for _ in range(20)])
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist())


def body(row,t):
    board=row["turns"].get(str(t))
    return board["atk"]+board["hp"] if board is not None else None


def prepare(row):
    turns=row["turns"]
    cards={c.removesuffix("_G") for t in ("6","7","8") if t in turns for c in turns[t]["cards"]}
    pig=bool(cards & {"BG31_326","BG36_331"})
    elem=bool(cards & {"BG28_707","BG35_881"})
    late=lambda key:len({t for t,g in row[key] if t>=12 and g>0})>=2
    return dict(pig_prep=int(pig),elem_prep=int(elem),pig_chain=int(pig and late("pig_returns")),
                elem_chain=int(elem and late("element_returns")),pig_late=int(late("pig_returns")),
                elem_late=int(late("element_returns")))


def report(root):
    plan=load(root/"plan.json")
    assert load(root/"evaluated.json")["n"]==plan["test"][1]
    rows=[load(p) for p in root.glob("eval-*/*.json")]
    keys=("base","ordinary","filtered","peer")
    start,n=plan["test"]
    assert sorted(r["base"]["seed"] for r in rows)==list(range(start,start+n))
    for r in rows:
        assert set(r)==set(keys) and len({(a["seed"],a["lobby"]) for a in r.values()})==1
        for a in r.values():
            a["metrics"].update(prepare(a))
            a["metrics"]["reach12"]=int("12" in a["turns"])
    totals={}
    for k in keys:
        arms=[r[k] for r in rows]
        turns={}
        for t in range(6,19):
            vals=[v for a in arms if (v:=body(a,t)) is not None]
            turns[t]=dict(n=len(vals),p50=p50(vals))
        cohorts={}
        for label,bit in (("pig",128),("element",8),("no_pig",-128),("mech",16),("dragon",4)):
            games=[a for a in arms if bool(a["lobby"] & abs(bit))==(bit>0)]
            cohorts[label]=dict(n=len(games),means={m:np.mean([v["metrics"][m] for v in games])
                for m in ("utility","top2","pig_prep","pig_late","pig_chain","elem_prep","elem_late","elem_chain","mech","dragon")})
        totals[k]=dict(turns=turns,means={m:np.mean([a["metrics"][m] for a in arms]) for m in arms[0]["metrics"]},cohorts=cohorts)
    pairs={}
    for left,right in (("base","ordinary"),("base","filtered"),("ordinary","filtered"),("base","peer")):
        delta={m:estimate([r[right]["metrics"][m]-r[left]["metrics"][m] for r in rows]) for m in
            ("rank","utility","win","top2","top4","reach12","pig_chain","elem_chain","mech","dragon")}
        for t in (12,13,14,15):
            both=[(r[left],r[right]) for r in rows if all(body(r[k],t) is not None for k in (left,right))]
            delta[f"log{t}"]=estimate([np.log1p(body(b,t))-np.log1p(body(a,t)) for a,b in both])
            delta[f"growth9to{t}"]=estimate([np.log1p(body(b,t))-np.log1p(body(b,9))-np.log1p(body(a,t))+np.log1p(body(a,9)) for a,b in both])
        pairs[right+"-"+left]=delta
    matched=[r for p in root.glob("matched-*.json") for r in load(p)]
    assert len(list(root.glob("matched-*.json")))==8 and len({r["j"] for r in matched})==len(matched)
    safety={}
    u=[1,.35,-1,-1.15,-1.3,-1.45,-1.6,-1.75]
    for group in ("low","mech","dragon"):
        games=[r["ranks"] for r in matched if group in r["groups"]]
        safety[group]=dict(n=len(games),ranks={k:np.mean([r[k] for r in games]) for k in keys},
            utility_delta={k:estimate([u[r[k]-1]-u[r["base"]-1] for r in games]) for k in keys if k!="base"},
            top4={k:sum(r[k]<=4 for r in games) for k in keys})
    decisions=[r for p in root.glob("returns-*.json") for r in load(p) if not r.get("missed")]
    return dict(n=n,models=totals,paired=pairs,same_state=safety,training_sources=len(decisions),
                peer_selected=sum(r["chosen"] for r in decisions),ties=sum(r["gain"]==0 for r in decisions))


if __name__=="__main__":
    print(json.dumps(report(Path(sys.argv[1])),indent=2))
