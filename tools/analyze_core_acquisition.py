"""Case-clustered mechanism analysis of the predeclared acquisition study."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json, file_identity


def analyze(complete, run, cases, utility):
    if not complete['valid'] or complete['rollouts'] != len(complete['rows']):
        raise ValueError('incomplete acquisition evidence')
    pairs = defaultdict(dict)
    for row in complete['rows']:
        if not row['validation']['valid'] or row['validation']['ranks'] != [row['rank']]:
            raise ValueError('invalid conditional trace')
        key = (row['model'], row['core'], row['case'], row['replica'])
        if row['arm'] in pairs[key]:
            raise ValueError('duplicate branch outcome')
        pairs[key][row['arm']] = row
    clustered = defaultdict(lambda: defaultdict(list))
    arm_rows = defaultdict(list)
    def turn(row, offset):
        return next((t for t in row['turns'] if t['turn'] == row['initial_turn'] + offset), None)
    for (model, core, case, replica), arms in pairs.items():
        if set(arms) != {'natural', 'buy'}:
            raise ValueError('missing paired arm')
        a, b = arms['buy'], arms['natural']
        if a['seed'] != b['seed']:
            raise ValueError('future seeds differ')
        ta, tb = turn(a, 2), turn(b, 2)
        metrics = {
            'rank_buy_minus_natural': a['rank'] - b['rank'],
            'training_utility_buy_minus_natural': utility[a['rank']-1] - utility[b['rank']-1],
            'top4_probability_delta': int(a['rank'] <= 4) - int(b['rank'] <= 4),
            'win_probability_delta': int(a['rank'] == 1) - int(b['rank'] == 1),
            'survive_two_turns_delta': int(ta is not None) - int(tb is not None),
            'two_turn_strength_dead_as_zero_delta': (ta['strength'] if ta else 0) - (tb['strength'] if tb else 0),
        }
        if ta and tb:
            metrics['two_turn_log_strength_common_survivors_delta'] = float(np.log1p(ta['strength']) - np.log1p(tb['strength']))
        for name, value in metrics.items():
            clustered[(model, core, name)][case].append(value)
        for arm, row in arms.items():
            arm_rows[(model, core, arm)].append(row)
    rng = np.random.default_rng(11901999)
    groups = {}
    for (model, core, metric), case_values in clustered.items():
        x = np.array([np.mean(v) for _, v in sorted(case_values.items())])
        boot = x[rng.integers(len(x), size=(10000, len(x)))].mean(-1)
        group = groups.setdefault(model + '/' + core, {'metrics': {}, 'arms': {}})
        group['metrics'][metric] = {'mean': float(x.mean()), 'source_games': len(x),
                                   'paired_rollouts': sum(len(v) for v in case_values.values()),
                                   'case_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()}
    for (model, core, arm), rows in arm_rows.items():
        first = [turn(r, 0) for r in rows]
        groups[model + '/' + core]['arms'][arm] = {
            'rollouts': len(rows), 'mean_rank': float(np.mean([r['rank'] for r in rows])),
            'core_played_games': sum(r['core_events'].get('played', 0) > 0 for r in rows),
            'core_sold_games': sum(r['core_events'].get('sold', 0) > 0 for r in rows),
            'core_present_first_eot_games': sum(t is not None and t['core_present'] for t in first),
            'reached_two_turns_later': sum(turn(r, 2) is not None for r in rows)}
    return {'groups': groups, 'training_utility_table': utility,
            'utility_source': 'Declared diagnostic metric, matching stage_arena.sbatch default; distinct from evaluator default linear utility',
            'cases': [{k: c[k] for k in ('case', 'core', 'source_seed', 'turn', 'health', 'gold', 'board_size', 'prior_turn_actions')} for c in cases],
            'scope': 'Conditional T9-T12 training-source study. Legal availability is not a correct-purchase label. No model promotion.',
            'survival_caution': 'Common-survivor strength is conditional; dead-as-zero strength is an explicit composite, not standard M1.',
            'controller_caution': 'Both arms reset policy history; first intervention consumes the same action budget. Not exact original-game controller continuation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--complete', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--utility', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    complete = json.loads(Path(args.complete).read_text())
    run = json.loads(Path(args.run).read_text())
    if file_identity(args.run) != complete['run'] or file_identity(args.cases) != run['cases']:
        raise ValueError('study/run/case identity mismatch')
    utility = [float(x) for x in args.utility.split(',')]
    if len(utility) != 8 or not np.isfinite(utility).all():
        raise ValueError('eight finite placement utility values required')
    with gzip.open(args.cases, 'rt', encoding='utf-8') as stream:
        cases = [json.loads(line) for line in stream]
    result = analyze(complete, run, cases, utility)
    result['inputs'] = {name: file_identity(getattr(args, name)) for name in ('complete', 'run', 'cases')}
    atomic_json(args.out, result)
    for name, group in result['groups'].items():
        print(json.dumps({'group': name, 'rank': group['metrics']['rank_buy_minus_natural'],
                          'utility': group['metrics']['training_utility_buy_minus_natural'],
                          'buy_arm': group['arms']['buy']}))


if __name__ == '__main__':
    main()
