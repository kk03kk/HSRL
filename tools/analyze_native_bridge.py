"""Strict full-cohort native purchase effects under two fixed continuations."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.analyze_economy_suffix import analyze


def rank_interaction(rows):
    arms={'parent','parent_buy','growth','growth_buy'};pairs=defaultdict(dict)
    for r in rows:
        key=(r['case'],r['replica'])
        if r['name'] in pairs[key]:raise ValueError('duplicate bridge outcome')
        pairs[key][r['name']]=r
    sources=defaultdict(list)
    for (case,replica),group in pairs.items():
        if set(group)!=arms:raise ValueError('incomplete bridge arms')
        signatures={(r['source_seed'],r['future_seed'],r['initial_sha256'],r['variant']) for r in group.values()}
        if len(signatures)!=1:raise ValueError('bridge paired identity differs')
        if any(type(r['rank']) is not int or not 1<=r['rank']<=8 for r in group.values()):raise ValueError('invalid bridge rank')
        for name in ('parent_buy','growth_buy'):
            if group[name].get('bridge_first_actions')!=1:raise ValueError('bridge first-action coverage differs')
        sources[case].append((group['growth_buy']['rank']-group['growth']['rank'])
            -(group['parent_buy']['rank']-group['parent']['rank']))
    x=np.asarray([np.mean(v) for _,v in sorted(sources.items())])
    if not len(x):raise ValueError('empty bridge cohort')
    rng=np.random.default_rng(12653801);boot=rng.choice(x,size=(5000,len(x)),replace=True).mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
        'sources':len(x),'paired_replicates':sum(map(len,sources.values())),
        'definition':'(growth_buy - growth) - (parent_buy - parent); negative means first purchase helps more under growth continuation'}


def main(args):
    root=Path(args.root);run=json.loads((root/'run.json').read_text());complete=json.loads((root/'complete.json').read_text())
    if run['plan']!=file_identity(args.plan):raise ValueError('bridge plan changed')
    plan=json.loads(Path(args.plan).read_text())
    if set(complete['models'])!=set(plan['models']):raise ValueError('bridge model coverage differs')
    for n,item in complete['models'].items():
        if item['sha256']!=plan['models'][n]['sha256']:raise ValueError('bridge model changed')
    expected=run['source_filter']['included_sources']*plan['replicas']*len(plan['models'])
    if complete['rollouts']!=expected:raise ValueError('bridge full source coverage differs')
    result=analyze([root],list(plan['models']),['privateer_bridge'],['growth'])
    result['purchase_rank_interaction']=rank_interaction(complete['rows'])
    result['source_selection']=run['source_filter']
    result['scope']='One guarded native first purchase and unchanged continuation; explicit precursor, not natural acquisition or an optimal Q estimate'
    atomic_json(Path(args.out)/'complete.json',result)
    print(json.dumps({'rollouts':expected,'interaction':result['purchase_rank_interaction']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
