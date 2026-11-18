"""Decompose zero-search failure into admissibility, ranking, and detection of a useful switch."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json
from workflows.distill_target_effects import cases_for,student_head,predictions,quality


def stats(rows,scores):
    counts=defaultdict(int)
    by_source=defaultdict(lambda:defaultdict(int))
    for (_,x,y,b,best,src),s in zip(rows,scores):
        pick=int(s.argmax())
        comparable=y[b]>-.99
        positive=best!=b
        for d in (counts,by_source[src]):
            d['contexts']+=1
            d['comparable']+=int(comparable)
            d['positive']+=int(positive)
            d['ranked_positive']+=int(positive and y[pick]>0)
            d['ranked_exact']+=int(positive and pick==best)
            d['retention_only']+=int(positive and y[int(x[:,73].argmax())]>0)
            d['ranked_invalid']+=int(y[pick]<-.99)
    ranked=sorted(((s[j],y[j]>0) for (_,_,y,b,_,_),s in zip(rows,scores)
                   if (j:=int(s.argmax()))!=b),reverse=True)
    top={n:sum(good for _,good in ranked[:n])/min(n,len(ranked)) for n in (8,16,32,64) if ranked}
    cuts=[quality(rows,scores,t) for t in np.arange(-.10,.301,.025)]
    return dict(counts=dict(counts),top_precision=top,grid=cuts,by_source=dict(by_source))


if __name__=='__main__':
    out=Path('artifacts/runs/nus-target-label-diagnosis-20261118-1')
    out.mkdir(parents=True,exist_ok=False)
    arms=[]
    for arm in (0,1):
        rows=[r for r in cases_for(arm) if (r[0]//2)%5==4]
        net,mu,sd,_=student_head(arm)
        arms.append(stats(rows,predictions(net,mu,sd,rows)))
    atomic_json(out/'assessment.json',arms)
