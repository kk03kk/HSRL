"""Report D-132 held-out selection value, opportunity costs and paired temporal attribution."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.assess_investment_timing import UTILITY
from tools.compare_research_panels import bootstrap

SEEDS=(13211,13229)


def fit_selectors(x,y):
    fits=[]
    for seed in SEEDS:
        ids=np.random.default_rng(seed).choice(len(x),112,replace=False)
        xx,yy=x[ids],y[ids]
        center,scale=xx.mean(0),xx.std(0)
        scale[scale<1e-6]=1
        z=(xx-center)/scale
        coef=np.linalg.solve(z.T@z+30*np.eye(z.shape[1]),z.T@(yy-yy.mean()))
        fits.append(dict(seed=seed,center=center.tolist(),scale=scale.tolist(),coef=coef.tolist(),
                         intercept=float(yy.mean())))
    return fits


def predict(fit,x):
    return (x-np.array(fit['center']))/fit['scale']@fit['coef']+fit['intercept']


def evaluate(root,out):
    out.mkdir(parents=True,exist_ok=False)
    sources=json.loads((root/'sources.json').read_text())['sources']
    assert len(sources)==256 and len({s['seed'] for s in sources})==256
    features,utility,ranks,more,later,hashes=[],[],[],[],[],{}
    for i,s in enumerate(sources):
        path=root/f'{i:03}-outcomes.json'
        row=json.loads(path.read_text())
        assert i==s['i']==row['i'] and s['field']==('transfer' if i%2 else 'development')
        futures=row['futures']
        if len(futures)!=8 or any(len(f)!=3 for f in futures):
            raise ValueError('wrong suffix budget')
        for f in futures:
            if len(f[1]['changes'])!=1 or f[0]['changes'] or not f[2]['changes']:
                raise ValueError('intervention audit failed')
            if f[1]['changes'][0]!=f[2]['changes'][0] or f[0]['actions'][0]!=s['event']['choices'][0]:
                raise ValueError('first action mismatch')
            if any(c[0] not in (s['event']['turn'],s['event']['turn']+1) for c in f[2]['changes']):
                raise ValueError('intervention outlived its option')
        hashes[path.name]=file_identity(path)['sha256']
        r=np.array([[a['rank'] for a in f] for f in futures])
        ranks.append(r.mean(0))
        utility.append(np.array(UTILITY)[r-1].mean(0))
        more.append(np.mean([len(f[2]['changes'])>1 for f in futures]))
        later.append(np.mean([any(c[0]>s['event']['turn'] for c in f[2]['changes']) for f in futures]))
        features.append(s['event']['features'])
    train=np.arange(0,256,2)
    test=np.arange(1,256,2)
    x=np.array(features)
    u=np.array(utility)
    r=np.array(ranks)[test]
    delta=u[:,2]-u[:,0]
    fits=fit_selectors(x[train],delta[train])
    preds=np.array([predict(f,x[test]) for f in fits])
    take=preds>0
    rate=take.mean(0)
    u=u[test]
    direct=u[:,1]-u[:,0]
    follow=u[:,2]-u[:,1]
    total=u[:,2]-u[:,0]
    ev=[sources[j]['event'] for j in test]
    def summarize(mask):
        mask=np.asarray(mask,dtype=bool)
        n=int(sum(mask))
        if n==0:
            return dict(n=0)
        weighted=lambda v:bootstrap(v[mask])
        def conditional(weight, vector):
            denom=weight[mask].sum()
            return float((weight*vector)[mask].sum()/denom) if denom else None
        return dict(n=n,adopted=[int(g[mask].sum()) for g in take],first=weighted(direct),
                    follow=weighted(follow),full=weighted(total),gated=weighted(rate*total),
                    versus_always=weighted((rate-1)*total),gated_first=weighted(rate*direct),
                    gated_follow=weighted(rate*follow),gated_rank=weighted(rate*(r[:,2]-r[:,0])),
                    accepted_full=conditional(rate,total),refused_full=conditional(1-rate,total),
                    extra_change=float(np.array(more)[test][mask].mean()),
                    next_turn_change=float(np.array(later)[test][mask].mean()))
    alltest=summarize(np.ones(128,dtype=bool))
    groups={}
    strata=dict(family=[e['board_family'] for e in ev],trade=[e['kind'] for e in ev],
                health=['at_most_15' if e['health']<=15 else 'above_15' for e in ev],
                support=['at_least_two' if e['support']>=2 else 'under_two' for e in ev],
                window=[str((e['turn']-7)//2) for e in ev])
    for name,values in strata.items():
        groups[name]={k:summarize(np.array(values)==k) for k in sorted(set(values))}
    byfit=[dict(seed=f['seed'],taken=int(t.sum()),utility=bootstrap(t*total),
                versus_always=bootstrap((t-1)*total)) for f,t in zip(fits,take)]
    gate=alltest['gated']['ci95'][0]>0 and alltest['versus_always']['mean']>0
    selected=[dict(seed=sources[i]['seed'],family=e['board_family'],turn=e['turn'],health=e['health'],
                   cards=e['cards'],kind=e['kind'],pred=preds[:,j].tolist(),first=direct[j],
                   follow=follow[j],total=total[j]) for j,(i,e) in enumerate(zip(test,ev))]
    atomic_json(out/'heldout.json',selected)
    atomic_json(out/'evidence.json',hashes)
    atomic_json(out/'fits.json',fits)
    atomic_json(out/'complete.json',dict(advance=bool(gate),sources=256,suffixes=6144,heldout=alltest,
               screened={f:sum(s['screened'] for s in sources if s['field']==f) for f in ('development','transfer')},
               families_train=dict(Counter(sources[i]['event']['board_family'] for i in train)),
               fit_results=byfit,groups=groups,source_manifest=file_identity(root/'sources.json'),
               fits=file_identity(out/'fits.json'),evidence=file_identity(out/'evidence.json')))


if __name__=='__main__':
    evaluate(Path(sys.argv[1]),Path(sys.argv[2]))
