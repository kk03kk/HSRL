"""Out-of-source variance of an unbiased two-fidelity estimate of paired rank utility."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.assess_investment_timing import UTILITY

OPEN=Path('artifacts/runs/nus-opening-plans-20261118-2')
OLD=Path('artifacts/runs/nus-d132-plans-20261118-1')


def board_features(row,t):
    fields=[]
    for turn in (t,t+1):
        e=row['eot'].get(str(turn))
        strength,hp,tier=e if e is not None else (0,0,0)
        log=np.log1p(max(0,strength))
        fields.extend([float(e is not None),hp/25,np.clip(hp,0,15)/15,
                       log/7,tier/6,log*hp/175,log*log/49])
    return np.array(fields+[(t-9)/3*v for v in fields])


def pair(base,option,t):
    return (board_features(option,t)-board_features(base,t),
            UTILITY[option['rank']-1]-UTILITY[base['rank']-1])


def opening():
    sources=json.loads((OPEN/'sources.json').read_text())
    groups=[]
    for s in sources:
        t=s['turn']
        runs=[[json.loads(l) for l in (OPEN/f"search-{s['i']}-{rep}.jsonl").read_text().splitlines()] for rep in range(8)]
        assert all([r['route'] for r in run]==list(range(4)) for run in runs)
        groups.append([[pair(run[0],run[j],t) for run in runs] for j in (1,2,3)])
    return groups


def previous():
    sources=json.loads((OLD/'sources.json').read_text())['sources']
    groups,cost=[],[]
    for s in sources[1::2]:
        t=s['event']['turn']
        rows=json.loads((OLD/f"{s['i']:03}-outcomes.json").read_text())['futures']
        groups.append([[pair(r[0],r[2],t) for r in rows]])
        prefix,full=0,0
        for arms in rows:
            for r in (arms[0],arms[2]):
                ends=[i+1 for i,a in enumerate(r['actions']) if a==0]
                prefix+=ends[1] if len(ends)>=2 else len(r['actions'])
                full+=len(r['actions'])
        cost.append([prefix,full])
    return groups,np.asarray(cost)


def centered(group):
    x=np.array([xy[0] for xy in group])
    y=np.array([xy[1] for xy in group])
    return x-x.mean(0),y-y.mean()


def fit(groups):
    data=[centered(arm) for source in groups for arm in source]
    x=np.vstack([r[0] for r in data])
    y=np.concatenate([r[1] for r in data])
    scale=x.std(0)
    scale[scale<1e-6]=1.
    z=x/scale
    return np.linalg.solve(z.T@z+40*np.eye(z.shape[1]),z.T@y)/scale


def measure(groups,coef,cost=None):
    # Four independent extra short futures per full future. The 4/5 shrinkage
    # accounts for uncertainty in the independent cheap-sample estimate.
    stats=[]
    for source in groups:
        components=[]
        for arm in source:
            x,y=centered(arm)
            f=.8*(x@coef)
            components.append([y.var(ddof=1),(y-f).var(ddof=1)+f.var(ddof=1)/4])
        stats.append(np.mean(components,axis=0))
    stats=np.array(stats)
    def metrics(indices):
        before,after=stats[indices].mean(axis=-2).T
        ratio=after/before
        result=dict(variance_ratio=ratio,break_even_prefix_cost=(1/ratio-1)/4)
        if cost is not None:
            short,full=cost[indices].sum(axis=-2).T
            result.update(prefix_cost=short/full,equal_decisions_ratio=ratio*(1+4*short/full))
        return result
    point=metrics(np.arange(len(stats)))
    draws=metrics(np.random.default_rng(13431).integers(len(stats),size=(10000,len(stats))))
    return {k:dict(mean=float(v),ci95=np.quantile(draws[k],[.025,.975]).tolist()) for k,v in point.items()}


if __name__=='__main__':
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    groups=opening()
    coef=fit(groups[::2])
    previous_groups,cost=previous()
    first=measure(groups[1::2],coef)
    second=measure(previous_groups,coef,cost)
    atomic_json(out/'complete.json',dict(opening_transfer_sources=32,trade_transfer_sources=128,
                model=coef.tolist(),opening=first,trade=second,
                qualify=bool(first['variance_ratio']['ci95'][1]<1 and second['equal_decisions_ratio']['ci95'][1]<1),
                opening_manifest=file_identity(OPEN/'sources.json'),trade_manifest=file_identity(OLD/'sources.json')))
