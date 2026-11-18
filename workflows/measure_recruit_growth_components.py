"""Decompose natural T9-to-T12 growth at decision boundaries in the archived parent panel."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.compare_research_panels import load_sealed_panel
from workflows.assess_purchase_consequences import interval


def reduce(ref):
    src=Path(ref['directory'])/ref['file']
    with src.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=ref['sha256']:
            raise ValueError('parent trajectory fingerprint mismatch')
    rows=[];acts=0;seeds=[]
    with gzip.open(src,'rt') as stream:
        if json.loads(next(stream))['identity']!=ref['identity']:
            raise ValueError('unexpected policy identity')
        for line in stream:
            r=json.loads(line)
            if r['event']=='game_start':
                first={};last={};families={};seeds.append(r['seed'])
            elif r['event']=='action':
                acts+=1
                t=r['turn']
                if not 9<=t<=12:
                    continue
                p=r['state_before']['players'][r['seat']]
                board=p['board']
                power=np.log1p(sum(m['atk']+m['hp'] for m in board))
                first.setdefault(t,power)
                if r['action']==0:
                    last[t]=power
                    if t==9:
                        families[t]=[m['card'] for m in board]
            elif r['event']=='game_end':
                if set(last)!={9,10,11,12}:
                    continue
                recruit=sum(last[t]-first[t] for t in (10,11,12))
                between=sum(first[t]-last[t-1] for t in (10,11,12))
                total=last[12]-last[9]
                if abs(recruit+between-total)>1e-7:
                    raise ValueError('growth decomposition failed')
                rows.append(dict(seed=r['seed'],rank=r['rank'],active=recruit,between=between,total=total,
                                 cards=families[9]))
    if acts!=ref['actions'] or seeds!=ref['seeds']:
        raise ValueError('incomplete trace')
    return rows


def summarize(rows):
    stats={}
    for k in ('active','between','total'):
        v=np.array([r[k] for r in rows])
        stats[k]=dict(**interval(v),p50=float(np.median(v)))
    return stats


if __name__=='__main__':
    from workflows.plan_recruit_openings import native
    panel=Path('artifacts/runs/nus-natural12m-eval-20261118-1/panel.json')
    report=load_sealed_panel(panel)
    parent=next(m for m in report['models'] if m['name']=='parent')
    refs=parent['decision_traces']
    with ProcessPoolExecutor(8) as pool:
        rows=[r for part in pool.map(reduce,refs) for r in part]
    info=native('development')[2]
    for r in rows:
        counts=Counter(race for c in r.pop('cards') for race in info[c]['races'] if race!='ALL')
        top=counts.most_common(2)
        r['family']=top[0][0] if top and top[0][1]>=3 and (len(top)==1 or top[0][1]>top[1][1]) else 'MIXED'
    families=Counter(r['family'] for r in rows)
    out=Path('artifacts/runs/nus-growth-components-20261118-1')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'complete.json',dict(panel=file_identity(panel),games=512,common=len(rows),overall=summarize(rows),
                by_family={k:summarize([r for r in rows if r['family']==k]) for k,n in families.items() if n>=15},
                rows=rows))
