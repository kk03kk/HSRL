"""Evaluate paired curriculum pilots by initial natural lobby, with common-seed intervals."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.lobby_filter import is_excluded
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.assess_natural12m import seed_vectors
from tools.compare_research_panels import load_sealed_panel, bootstrap


def assess(directory, verification):
    import hsbg_engine
    plan = json.loads(Path('config/evaluation/lobby-practice-20261118.json').read_text())
    parent_plan = json.loads(Path('config/research/natural12m-20261118.json').read_text())
    endpoints = json.loads(verification.read_text())
    panel = load_sealed_panel(directory/'panel.json')
    run = json.loads((directory/'run.json').read_text())
    complete = json.loads((directory/'complete.json').read_text())
    valid = json.loads((directory/'validation.json').read_text())
    if (not valid['valid'] or valid['panel'] != complete['panel'] or
            complete['validation'] != file_identity(directory/'validation.json') or
            run['plan'] != plan or run['source']['tracked_dirty'] or
            panel['protocol']['protocol_sha256'] != plan['expected_protocol_sha256']):
        raise ValueError('invalid or different assessment')
    seeds = list(range(plan['seed'],plan['seed']+plan['games']))
    weak = np.array([not is_excluded(hsbg_engine.Game(seed=s).observe(0)[0],('DRAGON','MECHANICAL'))
                     for s in seeds])
    stats, vectors = {}, {}
    utility = np.array([float(v) for v in parent_plan['shared_environment']['HSAI_ARENA_RANK_UTILITY'].split(',')])
    models = {m['name']:m for m in panel['models']}
    for spec in plan['models']:
        name = spec['name']
        model = models[name]
        sha = spec['sha256'] if name=='parent' else endpoints[name]['files']['model.pt']['sha256']
        if (model['checkpoint_sha256']!=sha or run['inputs'][spec['checkpoint']]['sha256']!=sha or
                model['outcome']['illegal_actions'] or model['outcome']['truncated_games']):
            raise ValueError('wrong or invalid endpoint')
        rank,reach,growth = seed_vectors(model,seeds)
        vectors[name] = dict(rank=rank,reach=reach,growth=growth,utility=utility[rank-1])
    for label,keep in [('all',np.ones_like(weak)),('neither',weak),('either',~weak)]:
        group = dict(n=int(keep.sum()),models={}, comparisons={})
        for name,vec in vectors.items():
            xs = [models[name]['strength_by_seed'][str(s)]['12'] for s,use,alive in zip(seeds,keep,vec['reach']) if use and alive]
            group['models'][name] = dict(rank=float(vec['rank'][keep].mean()),
                    wins=int((vec['rank'][keep]==1).sum()),reach=int(vec['reach'][keep].sum()),
                    t12_p50=float(np.median(xs)), utility=float(vec['utility'][keep].mean()))
        joint = []
        for letter in 'ab':
            mix = vectors['mixed_'+letter]
            for ref in ('natural_'+letter,'parent'):
                deltas = {k:(v-vectors[ref][k])[keep] for k,v in mix.items()}
                group['comparisons']['mixed_'+letter+'-'+ref] = {
                    k:bootstrap(v[np.isfinite(v)]) for k,v in deltas.items()}
                if ref!='parent':
                    joint.append(deltas['utility'])
        group['equal_weight_utility'] = bootstrap(np.mean(joint,axis=0))
        stats[label] = group
    def both(group,metric,threshold,strict):
        means=[stats[group]['comparisons'][f'mixed_{x}-natural_{x}'][metric].get('mean') for x in 'ab']
        return all(v is not None and (v>threshold if strict else v>=threshold) for v in means)
    checks = dict(weak_utility=both('neither','utility',0,True),
                  weak_growth=both('neither','growth',0,True),
                  natural_reach=both('all','reach',-.03,False),
                  natural_utility=stats['all']['equal_weight_utility']['mean']>=0,
                  either_utility_bound=stats['either']['equal_weight_utility']['ci95'][0]>-.15)
    return dict(panel=complete['panel'], protocol=valid['protocol_sha256'], checks=checks,
                passed=all(checks.values()),groups=stats)


if __name__=='__main__':
    atomic_json(Path(sys.argv[3]),assess(Path(sys.argv[1]),Path(sys.argv[2])))
