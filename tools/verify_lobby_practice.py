"""Bind the four small pilot endpoints, observed training mix, and paired configuration."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity


def read(path):
    return json.loads(path.read_text())


def finite(node):
    if isinstance(node, dict):
        return all(finite(v) for v in node.values())
    if isinstance(node, list):
        return all(finite(v) for v in node)
    return not isinstance(node, float) or math.isfinite(node)


def main(out):
    pilot = read(Path('config/research/lobby-practice-20261118.json'))
    base = read(Path(pilot['reference']))
    for path,sha in base['asset_sha256'].items():
        if file_identity(path)['sha256'] != sha:
            raise ValueError('source asset changed: '+path)
    endpoints = {}
    for letter,seed in zip('ab',pilot['training_seeds']):
        configs = []
        for arm,p in [('natural',0.),('mixed',.25)]:
            name = f'{arm}_{letter}'
            root = Path(f'artifacts/runs/nus-practice-{arm}-{letter}-20261118-1/train')
            last,summary = read(root/'latest.json'),read(root/'summary.json')
            cfg = summary['config']
            states = [json.loads(l) for l in (root/'history.jsonl').read_text().splitlines()]
            rows = [r for r in states if 'collect' in r]
            if not (cfg['seed']==last['seed']==seed and cfg['lobby_filter_probability']==p and
                    last['final'] and len(rows)==summary['iterations']==last['history_len']==pilot['rollouts_per_arm']
                    and all(finite(r['ppo']) for r in rows)):
                raise ValueError('invalid endpoint: '+name)
            configs.append({k:v for k,v in cfg.items() if k not in ('out','lobby_filter_probability')})
            keys = ('games','decisions','practice_games_observed','practice_games_without_races',
                    'mirror_missing_learner_alive','mirror_missing_learner_dead','mirror_missing_learner_unknown')
            errors = ('invalid_games','illegal_actions','truncated_games','truncated_combats',
                      'truncated_recruit_contexts','dropped_decisions','mirror_rejected','mirror_errors')
            counts = {k:sum(r['collect'][k] for r in rows) for k in keys+errors}
            if (any(counts[k] for k in errors) or counts['games']!=counts['practice_games_observed'] or
                    counts['decisions']!=last['cum_decisions'] or counts['games']!=len(rows)*pilot['games_per_rollout']):
                raise ValueError('invalid pilot counts: '+name)
            for fn in ('model.pt','trainer.pt'):
                if file_identity(root/fn)!=last['files'][fn]:
                    raise ValueError('endpoint identity differs: '+name)
            endpoints[name] = dict(seed=seed, counts=counts,files=last['files'])
        if configs[0]!=configs[1]:
            raise ValueError('unmatched settings: '+str([k for k in configs[0] if configs[0][k]!=configs[1][k]]))
    atomic_json(out, endpoints)


if __name__=='__main__':
    main(Path(sys.argv[1]))
