"""Paired research comparison with explicit survivor conditioning and uncertainty."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.az.train.eval_policy import assert_same_protocol
from hsbg.runtime.artifacts import atomic_json,file_identity


def bootstrap(values,seed=11818):
    values=np.asarray(values,dtype=float)
    if not len(values):return {'n':0}
    rng=np.random.default_rng(seed)
    estimates=np.mean(values[rng.integers(len(values),size=(10000,len(values)))],axis=1)
    return {'n':len(values),'mean':float(np.mean(values)),
            'ci95':[float(x) for x in np.quantile(estimates,[.025,.975])]}


def compare(panels,reference):
    protocol=assert_same_protocol(*panels)
    models={}
    for panel in panels:
        for m in panel['models']:
            if m['name'] in models:raise ValueError('duplicate model name')
            models[m['name']]=m
    base=models[reference]
    br=dict(zip(base['seed_sequence'],base['rank_sequence']))
    bs=base['strength_by_seed']
    report={'protocol':protocol,'reference':reference,'models':{},
            'scope':'Development evidence; paired bootstrap conditional on tested seeds, not promotion',
            'survivor_warning':'Strength and growth contrasts condition on common surviving games'}
    for name,m in models.items():
        rank=dict(zip(m['seed_sequence'],m['rank_sequence']))
        if rank.keys()!=br.keys():raise ValueError('nonidentical seed cohort')
        summary={'checkpoint_sha256':m['checkpoint_sha256'],'outcome':m['outcome'],
                 'rank_delta_candidate_minus_reference':bootstrap([rank[k]-br[k] for k in sorted(br)]),
                 'turns':{}}
        ss=m['strength_by_seed']
        for turn in (7,8,9,10,11,12,13,14):
            key=str(turn);curve=m['curves'].get(key,{})
            common=sorted(s for s in ss if key in ss[s] and key in bs.get(s,{}))
            rows=[np.log1p(ss[s][key])-np.log1p(bs[s][key]) for s in common]
            summary['turns'][key]={'all_survivor_n':curve.get('n',0),'all_survivor_p50':curve.get('strength_p50'),
                'common_n':len(common),'common_candidate_p50':float(np.median([ss[s][key] for s in common])) if common else None,
                'common_reference_p50':float(np.median([bs[s][key] for s in common])) if common else None,
                'common_log_strength_difference':bootstrap(rows)}
        common=sorted(s for s in ss if all(t in ss[s] and t in bs.get(s,{}) for t in ('9','12')))
        growth=[np.log1p(ss[s]['12'])-np.log1p(ss[s]['9'])-np.log1p(bs[s]['12'])+np.log1p(bs[s]['9']) for s in common]
        summary['common_T9_T12_log_growth_difference']=bootstrap(growth)
        report['models'][name]=summary
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('panels',nargs='+')
    p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=compare([json.loads(Path(x).read_text(encoding='utf-8')) for x in a.panels],a.reference)
    r['inputs']={x:file_identity(x) for x in a.panels};atomic_json(a.out,r)
    for name,m in r['models'].items():
        print(json.dumps({'name':name,'rank_delta':m['rank_delta_candidate_minus_reference'],
                          'T12':m['turns']['12'],'growth':m['common_T9_T12_log_growth_difference']}))
