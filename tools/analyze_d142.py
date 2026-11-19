"""Paired learning-curve and seed-clustered factorial analysis for D-142."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.az.train.eval_policy import assert_same_protocol
from tools.compare_research_panels import load_sealed_panel

UTILITY = [1, .35, -1, -1.15, -1.30, -1.45, -1.60, -1.75]
CHECKPOINTS = {'P':'5058768bf90e989a943c6e21800a0767095a81a2034f4d4bf8d3d603028896a1',
               'M':'a1df8427c0a7efbe9e460d132226df8f4a0329e09a1ef3ff64055982153b0efc',
               'E':'e1117934c76c37090212ec02e75d8ef52b21280c8d47f7dd5eb6089a228f74ad'}
# Absolute benchmark fixed from the already published parent panel (D-142).
METRICS = ('rank','utility','first','top2','top4','bottom2','reach12','reach12_at_least_1414')


def series(row):
    rank = row['rank']
    return dict(zip(METRICS, (rank, UTILITY[rank-1], rank == 1, rank <= 2, rank <= 4,
                              rank >= 7, '12' in row['strength'], row['strength'].get('12', -1) >= 1414)))


def paired(a, b, *, eligible=True):
    if not eligible:
        return {}
    av, bv = series(a), series(b)
    values = {key: bv[key]-av[key] for key in METRICS}
    left, right = a['strength'], b['strength']
    if all(t in left and t in right for t in ('9','12')):
        d9, d12 = (np.log1p(right[t])-np.log1p(left[t]) for t in ('9','12'))
        values.update(growth=d12-d9, common_log12=d12, common_log9=d9)
    return values


def summarize(rows):
    metric = [series(x) for x in rows]
    result = {k: np.mean([r[k] for r in metric]) for k in METRICS}
    result['ranks'] = [sum(x['rank'] == k for x in rows) for k in range(1,9)]
    result['turns'] = {}
    for turn in ('7','8','9','10','11','12','14'):
        values = sorted(x['strength'][turn] for x in rows if turn in x['strength'])
        if values:
            result['turns'][turn] = dict(n=len(values), p50=values[round(.5*(len(values)-1))])
    return result


def estimate(groups):
    # Each outer row is ONE original seed. Weight each eligible matched
    # opening equally; resample the original seeds including their missingness.
    keys = sorted({key for group in groups for record in group for key in record})
    n = len(groups)
    draws = np.random.default_rng(142).integers(n, size=(10000,n))
    out = {}
    for k in keys:
        xs = [[record[k] for record in group if k in record] for group in groups]
        sums = np.array([sum(x) for x in xs])
        counts = np.array([len(x) for x in xs])
        numerator = sums[draws].sum(axis=1)
        denominator = counts[draws].sum(axis=1)
        samples = numerator[denominator>0] / denominator[denominator>0]
        out[k] = dict(n=int(counts.sum()), seeds=int(np.count_nonzero(counts)),
                      mean=float(sums.sum()/counts.sum()), ci95=np.quantile(samples,[.025,.975]).tolist())
    return out


def midpoint(paths):
    panels = [load_sealed_panel(path) for path in paths]
    protocol = assert_same_protocol(*panels)
    by_model = {}
    for key, panel in zip('PME', panels):
        m, = panel['models']
        if m['checkpoint_sha256'] != CHECKPOINTS[key]:
            raise ValueError('panel checkpoint mismatch for '+key)
        seeds = m['seed_sequence']
        if sorted(seeds) != list(range(82370000,82370512)) or len(m['rank_sequence']) != len(seeds):
            raise ValueError('not the declared 512 seeds')
        by_model[key] = {seed:dict(rank=r, strength=m['strength_by_seed'][str(seed)])
                         for seed,r in zip(seeds,m['rank_sequence'])}
    diff = {a+b:estimate([[paired(by_model[a][s],by_model[b][s])] for s in sorted(by_model[a])])
            for a,b in ('PM','ME','PE')}
    return dict(protocol=protocol, panels={k:file_identity(p) for k,p in zip('PME',paths)},
                summary={k:summarize(list(v.values())) for k,v in by_model.items()}, contrasts=diff)


def crossover(directory):
    complete = json.loads((directory/'complete.json').read_text())
    if not complete['valid'] or complete['logical_trajectories'] != 1024:
        raise ValueError('invalid run')
    for p, identity in complete['files'].items():
        if file_identity(directory/p) != identity:
            raise ValueError('changed evidence '+p)
    plan = json.loads((directory/'plan.json').read_text())
    if any(plan['models'][k][1] != CHECKPOINTS[k] for k in 'PE'):
        raise ValueError('crossover checkpoint mismatch')
    with gzip.open(directory/'trajectories.json.gz','rt') as stream:
        rows = json.load(stream)
    cells, eligible = {}, {}
    for r in rows:
        k, early = r['seed'],r['early']
        eligible[k,early] = bool(r['opening'])
        for late, data in r['legs'].items():
            cells[k,early+late] = data
    seeds = list(range(87419000,87419256))
    if set(cells) != {(s,c) for s in seeds for c in ('PP','PE','EP','EE')} or len(rows) != 512:
        raise ValueError('incomplete factorial')
    def change(s, left, right, live=False):
        return paired(cells[s,left], cells[s,right], eligible=not live or eligible[s,left[0]])
    diff = {}
    for a,b in [('PP','PE'),('EP','EE'),('PP','EP'),('PE','EE'),('PP','EE')]:
        diff[a+'-'+b] = estimate([[change(s,a,b)] for s in seeds])
        if a[0] == b[0]:
            diff[a+'-'+b+'|T9'] = estimate([[change(s,a,b,True)] for s in seeds])
    for name,pairs in {'late':[('PP','PE'),('EP','EE')],'early':[('PP','EP'),('PE','EE')]}.items():
        diff[name] = estimate([[change(s,a,b) for a,b in pairs] for s in seeds])
    diff['late|T9'] = estimate([[change(s,a,b,True) for a,b in [('PP','PE'),('EP','EE')]] for s in seeds])
    interactions = []
    for s in seeds:
        e,p = change(s,'EP','EE'),change(s,'PP','PE')
        interactions.append([{k:e[k]-p[k] for k in e.keys() & p.keys()}])
    diff['interaction'] = estimate(interactions)
    outcomes = {}
    for early in 'PE':
        matched = [(cells[s,early+'P'], cells[s,early+'E']) for s in seeds if eligible[s,early]]
        outcomes[early] = dict(better=sum(e['rank'] < p['rank'] for p,e in matched),
                               equal=sum(e['rank'] == p['rank'] for p,e in matched),
                               worse=sum(e['rank'] > p['rank'] for p,e in matched),
                               identical_actions=sum(e['actions'] == p['actions'] for p,e in matched))
    return dict(complete=complete, eligible={k:sum(eligible[s,k] for s in seeds) for k in 'PE'},
                outcomes_at_T9=outcomes,
                summary={c:summarize([cells[s,c] for s in seeds]) for c in ('PP','PE','EP','EE')}, contrasts=diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--panels',nargs=3,type=Path)
    group.add_argument('--cross',type=Path)
    a = parser.parse_args()
    atomic_json(a.out,midpoint(a.panels) if a.panels else crossover(a.cross))
