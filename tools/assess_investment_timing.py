"""Fit a fixed public ridge selector on development, then measure paired whole-game outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.compare_research_panels import bootstrap


UTILITY = [1., .35, -1., -1.15, -1.30, -1.45, -1.60, -1.75]
OPTIONS = {'tier': ('upgrade', 'delay'), 'sell': ('retain', 'parent')}


def read(root, name, plan):
    run = json.loads((root/'run.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    n, first = plan[name]['games_per_arm'], plan[name]['first_seed']
    if (complete['run'] != file_identity(root/'run.json') or complete['full_games'] != n*4 or
            (run['split'],run['games'],run['seed']) != (name,n,first)):
        raise ValueError('study run differs from preregistration')
    games = [json.loads((root/f'{i}.json').read_text()) for i in range(first,first+n)]
    if any(set(g) != set(plan['arms']) for g in games):
        raise ValueError('incomplete arms')
    return games,run


def features(event):
    return [event['turn'], event['health']+event['armor'], event['gold'], event['hand'],
            event['size'], np.log1p(event['strength'])]


def utility(row):
    return UTILITY[row['rank']-1]


def fit(games, kinds):
    x, y = [], []
    for game in games:
        for kind in kinds:
            event = game['parent']['events'].get(kind)
            if event:
                invest, alternative = OPTIONS[kind]
                x.append(features(event))
                y.append(utility(game[invest])-utility(game[alternative]))
    x = np.asarray(x)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1
    z = (x-center)/scale
    intercept = np.mean(y)
    weights = np.linalg.solve(z.T@z+20*np.eye(z.shape[1]), z.T@(y-intercept))
    return dict(n=len(y), center=center.tolist(), scale=scale.tolist(), intercept=float(intercept),
                weights=weights.tolist())


def choose(event, model):
    z = (np.asarray(features(event))-model['center'])/model['scale']
    return bool(model['intercept']+z@model['weights'] > 0)


def compare(a,b):
    def growth(row):
        return np.log1p(row['eot']['12'][0])-np.log1p(row['eot']['9'][0])
    common = [(x,y) for x,y in zip(a,b) if all(k in r['eot'] for k in ('9','12') for r in (x,y))]
    return dict(utility=bootstrap([utility(x)-utility(y) for x,y in zip(a,b)]),
                rank=bootstrap([x['rank']-y['rank'] for x,y in zip(a,b)]),
                reach12=bootstrap([int('12' in x['eot'])-int('12' in y['eot']) for x,y in zip(a,b)]),
                growth=bootstrap([growth(x)-growth(y) for x,y in common]))


def assess(games, fits):
    report = {}
    for kind,(invest,alt) in OPTIONS.items():
        rows = []
        for g in games:
            event = g['parent']['events'].get(kind)
            row = {k:g[k] for k in ('parent', invest, alt)}
            for name,model in fits.items():
                row['fit_'+name] = g[invest if event and choose(event,model) else alt] if event else g['parent']
            rows.append(row)
        evs = [g['parent']['events'].get(kind) for g in games]
        comparisons = {}
        for name in [invest,alt]+['fit_'+s for s in fits]:
            comparisons[name] = {ref:compare([r[name] for r in rows],[r[ref] for r in rows])
                                 for ref in dict.fromkeys(('parent',invest,alt)) if ref != name}
        stages = {}
        if kind == 'tier':
            for label,interval in [('milestone',(7,8)), ('optional_six',(9,10))]:
                subset = [g for e,g in zip(evs,games) if e and e['turn'] in interval]
                stages[label] = compare([g[invest] for g in subset],[g[alt] for g in subset])
        report[kind] = dict(triggered=sum(bool(e) for e in evs), game_seeds=len(games),
                            invest_choices={s:sum(bool(e) and choose(e,m) for e in evs) for s,m in fits.items()},
                            comparisons=comparisons, timing_strata=stages,
                            source_cards=len({e['card'] for e in evs if e and 'card' in e}),
                            source_heroes=len({str(e['hero']) for e in evs if e}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('development',type=Path)
    p.add_argument('transfer',type=Path)
    p.add_argument('out',type=Path)
    args = p.parse_args()
    plan = json.loads(Path('config/research/investment-timing-20261118.json').read_text())
    dev, run_d = read(args.development,'development',plan)
    held, run_h = read(args.transfer,'transfer',plan)
    if run_d['revision'] != run_h['revision']:
        raise ValueError('different rollout source')
    fits = {k:fit(dev,[k]) for k in OPTIONS}
    fits['pooled'] = fit(dev,list(OPTIONS))
    atomic_json(args.out,dict(fits=fits, development=assess(dev,fits), transfer=assess(held,fits),
                              runs=[file_identity(p/'run.json') for p in [args.development,args.transfer]]))
