"""Learn calibration of a public arithmetic afterstate estimate, shared across effect texts."""
from functools import lru_cache
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json
from workflows.distill_target_effects import cases_for,quality
from workflows.plan_recruit_openings import native

ROOT=Path('artifacts/runs/nus-public-effect-model-20261118-1')


@lru_cache(maxsize=512)
def effect(src):
    card=native('development')[2].get(src-1,{})
    text=re.sub('<[^>]+>','',str(card.get('text',''))).strip().lower()
    who=r'(?:a|a friendly) minion'
    for regex,kind in [(rf'give {who} \+(\d+)/\+(\d+)\.', 'add'),
                       (rf'give {who} \+(\d+) (attack|health)\.', 'add_one'),
                       (rf"set a minion's stats to (\d+)/(\d+)\.", 'set')]:
        m=re.fullmatch(regex,text)
        if m:
            a,b=m.groups()
            if kind=='add_one':
                return 'add',int(a) if b=='attack' else 0,int(a) if b=='health' else 0
            return kind,int(a),int(b)
    return None


def physical(x,b,rule):
    typ,atk,hp=rule
    retention=x[:,73]
    amount=np.full(len(x),float(atk+hp))
    if typ=='set':
        amount-=np.expm1(x[:,74])+np.expm1(x[:,75])
    score=(.35+retention)*amount
    denom=np.maximum(10.,np.maximum(abs(amount),abs(amount[b])))
    return (score-score[b])/denom


def calibration(rows):
    data={'add':[],'set':[]}
    for _,x,y,b,_,src in rows:
        rule=effect(src)
        if rule is None or y[b]<-.99:
            continue
        z=physical(x,b,rule)
        valid=(y>-.99)&(abs(z)>.005)
        if valid.any():
            data[rule[0]].extend(((y[valid]+.20)/z[valid]).tolist())
    return {k:float(np.clip(np.median(v),0.,3.)) if v else 1. for k,v in data.items()}


def predict(rows,scales):
    scores=[]
    for _,x,_,b,_,src in rows:
        rule=effect(src)
        s=physical(x,b,rule)*scales[rule[0]]-.20 if rule else np.full(len(x),-2.)
        scores.append(s)
    return scores


def summarize(rows,scale):
    scores=predict(rows,scale)
    result=quality(rows,scores,0.)
    result['supported_contexts']=sum(effect(r[-1]) is not None for r in rows)
    result['supported_changes']=sum(r[3]!=r[4] and effect(r[-1]) is not None for r in rows)
    result['sources_with_benefit']=sorted({r[-1] for r,s in zip(rows,scores)
            if s.max()>0 and s.argmax()!=r[3] and r[2][s.argmax()]>0})
    return result


def fit(arm):
    rows=cases_for(arm)
    tr=[r for r in rows if (r[0]//2)%5!=4]; cal=[r for r in rows if (r[0]//2)%5==4]
    scale=calibration(tr)
    result=summarize(cal,scale)
    # No case of the same source is allowed into the calibration of its numerical effect model.
    unseen_scales={s:calibration([r for r in tr if r[-1]!=s]) for s in set(r[-1] for r in cal) if effect(s)}
    loo=[predict([r],unseen_scales.get(r[-1],scale))[0] for r in cal]
    unseen=quality(cal,loo,0.)
    return dict(scale=scale,calibration=result,leave_source_out=unseen)


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=False)
    stats=[fit(a) for a in (0,1)]
    atomic_json(ROOT/'assessment.json',dict(arms=stats,passes=all(
        v[k]['precision']>=.70 and v[k]['beneficial']>=8 for v in stats for k in ('calibration','leave_source_out'))))
