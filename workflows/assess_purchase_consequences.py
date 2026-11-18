"""Fit purchase-effect heads on development games, then open the disjoint transfer panel."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.learn_purchase_consequences import LABELS, ROOT


def read(start, stop):
    rows = []
    for shard in range(start, stop):
        directory = ROOT / f'shard-{shard:02}'
        done = json.loads((directory / 'complete.json').read_text())
        batch = [json.loads(line) for line in (directory / 'games.jsonl').read_text().splitlines()]
        if done['games'] != 64 or [r['i'] for r in batch] != list(range(64*shard, 64*(shard+1))):
            raise ValueError('incomplete or repeated source set')
        rows.extend(batch)
    return rows


def arrays(rows):
    offers = np.array([row['sample']['x'] for row in rows])
    baseline = offers[:, 0, :4]
    # State-only effects are allowed for both models; offer differences encode the intervention.
    rich = np.column_stack((offers[:, 0], offers[:, 1] - offers[:, 0]))
    outcomes = np.array([[a['y'] for a in row['outcomes']] for row in rows])
    return {'state': baseline, 'rich': rich}, outcomes[:, 1] - outcomes[:, 0]


def fit(x, y):
    mean, scale, center = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5), y.mean(axis=0)
    z = (x - mean) / scale
    w = np.linalg.solve(z.T @ z + 32 * np.eye(z.shape[1]), z.T @ (y - center))
    return {k: v.tolist() for k, v in dict(mean=mean, scale=scale, center=center, weight=w).items()}


def predict(head, x):
    return ((x - head['mean']) / head['scale']) @ np.asarray(head['weight']) + head['center']


def interval(values):
    rng = np.random.default_rng(135391)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(len(v), size=(5000, len(v)))].mean(axis=1)
    return dict(n=len(v), mean=float(v.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def prepare():
    rows = [r for r in read(0, 16) if r['sample']]
    heads = {}
    for key, group in [('all', rows), ('unseen_mechanical', [r for r in rows if r['sample']['family'] != 'MECHANICAL'])]:
        x, y = arrays(group)
        heads[key] = dict(models={k: fit(v, y) for k, v in x.items()}, train_n=len(y),
                          label_scale=np.maximum(y.std(axis=0), .001).tolist())
    if (ROOT / 'heads.json').exists():
        raise FileExistsError('heads already sealed')
    atomic_json(ROOT / 'heads.json', heads)


def assess():
    heads = json.loads((ROOT / 'heads.json').read_text())
    all_rows = read(16, 24)
    report = {}
    for key, spec in heads.items():
        rows = [r for r in all_rows if r['sample'] and (key == 'all' or r['sample']['family'] == 'MECHANICAL')]
        x, y = arrays(rows)
        pred = {k: predict(spec['models'][k], v) for k, v in x.items()}
        errors = {k: ((v-y) / spec['label_scale'])**2 for k, v in pred.items()}
        improvement = errors['state'] - errors['rich']
        pick = pred['rich'][:, -1] > 0
        utility = np.where(pick, y[:, -1], 0)
        if key == 'all':
            utility = np.pad(utility, (0, len(all_rows)-len(rows)))
        report[key] = dict(opportunities=len(rows), choices=int(pick.sum()), utility=interval(utility),
                           improvement=interval(improvement[:, :-1].mean(axis=1)),
                           by_label={k: interval(improvement[:, j]) for j, k in enumerate(LABELS)},
                           outcome_mean=y.mean(axis=0).tolist(), train_n=spec['train_n'])
    overall = report['all']
    atomic_json(ROOT / 'assessment.json', dict(results=report, heads=file_identity(ROOT / 'heads.json'),
                shortlist=overall['utility']['ci95'][0] > 0 and overall['improvement']['mean'] > 0))


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        prepare()
    elif sys.argv[1] == 'assess':
        assess()
    else:
        raise ValueError('unknown stage')
