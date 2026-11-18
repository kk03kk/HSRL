"""Supplementary outcomes for complete, paired conditional suffix studies.

These descriptors do not change any formal evaluation or promotion rule.
Call only after admission of the full policy arms and evidence identities.
"""
import hashlib
from collections import Counter, defaultdict

import numpy as np


def summarize(rows, references=('parent',), rank_utility=None):
    indexed=defaultdict(dict)
    names={r['name'] for r in rows}
    if not rows or not set(references)<=names:
        raise ValueError('missing outcome reference or rows')
    for row in rows:
        if type(row['rank']) is not int or not 1<=row['rank']<=8:
            raise ValueError('invalid final rank')
        key=(row['case'],row['variant'],row['replica'])
        if row['name'] in indexed[key]:raise ValueError('duplicate outcome')
        indexed[key][row['name']]=row
    for arms in indexed.values():
        if set(arms)!=names:raise ValueError('unpaired outcome')
        first=next(iter(arms.values()))
        for row in arms.values():
            if any(row[k]!=first[k] for k in ('source_seed','future_seed','initial_sha256')):
                raise ValueError('outcome identity differs')
    summary={};paired={}
    metrics={'win_rate':lambda rank:float(rank==1), 'top4_rate':lambda rank:float(rank<=4)}
    if rank_utility is not None:
        utility=np.asarray(rank_utility,dtype=float)
        if utility.shape!=(8,) or not np.isfinite(utility).all():raise ValueError('explicit finite eight-rank utility required')
        metrics['frozen_rank_utility']=lambda rank:float(utility[rank-1])
    for variant in sorted({r['variant'] for r in rows}):
        for name in sorted(names):
            rr=[r for r in rows if r['name']==name and r['variant']==variant]
            summary[name+'/'+variant]={'games':len(rr),'sources':len({r['case'] for r in rr}),
                'rank_counts':dict(sorted(Counter(r['rank'] for r in rr).items())),
                **{key:float(np.mean([fn(r['rank']) for r in rr])) for key,fn in metrics.items()}}
        for reference in references:
            for name in sorted(names-{reference}):
                estimates={}
                for metric,fn in metrics.items():
                    groups=defaultdict(list)
                    for (case,v,replica),arms in indexed.items():
                        if v==variant:groups[case].append(fn(arms[name]['rank'])-fn(arms[reference]['rank']))
                    x=np.array([np.mean(groups[k]) for k in sorted(groups)])
                    # Stable per contrast: adding other policies cannot change these intervals.
                    token='/'.join((variant,name,reference,metric)).encode('ascii')
                    seed=int.from_bytes(hashlib.sha256(token).digest()[:8],'little')
                    rng=np.random.default_rng(seed)
                    boot=rng.choice(x,size=(5000,len(x)),replace=True).mean(1)
                    estimates[metric]={'mean':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                        'sources':len(x),'paired_replicates':sum(map(len,groups.values()))}
                paired[name+'-'+reference+'/'+variant]=estimates
    result={'summary':summary,'paired':paired,
        'scope':'Supplementary exploratory win/top4 outcomes, not new promotion rules',
        'method':'Average paired future replicas within source, then 5000 source-cluster bootstrap samples',
        'limitations':['Artificial endowments','Repeated exploratory development sources',
            'Multiple exploratory contrasts; intervals are not multiplicity-adjusted']}
    if rank_utility is not None:
        result['rank_utility_table']=utility.tolist()
        result['scope']='Supplementary outcomes and explicitly supplied rank utility; no changed evaluation or training rule'
    return result
