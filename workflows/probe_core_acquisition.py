"""Paired natural-state suffix rollouts of buying a core versus the base policy.

This is a conditional training-source diagnostic, not a formal match or an
expert-label generator. Only the first action is intervened on. The current
opponents are preserved, future RNG is paired, and every learner step is sealed.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity, environment_identity
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.policies import Decision
from hsbg.evaluation.trace import DecisionRecorder, validate_trace

FAMILIES = ('BG_LOE_077', 'BG36_352')
_CACHE = {}


def select_cases(bank, maximum):
    import hsbg_engine as engine
    root = Path(bank)
    manifest = json.loads((root / 'manifest.json').read_text())
    if file_identity(root / 'states.jsonl.gz') != manifest['data']:
        raise ValueError('natural source bank changed')
    normals = {c['id']: c.get('normal') or c['id'] for c in json.loads(
        Path('engine/crates/hsbg-engine/data/cards.json').read_text())['cards']}
    def normal(index):
        return normals[engine.card_info(index + 1)[0]]
    candidates = []
    with gzip.open(root / 'states.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['seed'] % 5 == 0 or row['seat'] != 0 or not 9 <= row['turn'] <= 12:
                continue
            state = row['state']
            player = state['players'][0]
            if player['pending'] or player['ended_turn'] or player['health'] <= 0:
                continue
            owned = {normal(item_card(x)) for x in player['board'] + player['hand']}
            game = engine.Game.from_json(json.dumps(state))
            mask = bytes(game.legal_mask(0))
            for i, slot in enumerate(player['shop'][:8]):
                core = normal(item_card(slot))
                if core in FAMILIES and core not in owned and mask[8+i]:
                    candidates.append({'source_seed': row['seed'], 'turn': row['turn'],
                        'core': core, 'buy_action': 8+i, 'state': state,
                        'gold': player['gold'], 'health': player['health'],
                        'board_size': len(player['board']), 'prior_turn_actions': player['actions_this_turn']})
    # Selection uses only source identifiers and pre-action state, never outcomes.
    selected, used_games, counts = [], set(), Counter()
    for case in sorted(candidates, key=lambda c: (c['source_seed'], c['turn'], c['core'], c['buy_action'])):
        if case['source_seed'] in used_games or counts[case['core']] >= maximum:
            continue
        case['case'] = len(selected)
        selected.append(case)
        used_games.add(case['source_seed'])
        counts[case['core']] += 1
    if not selected:
        raise ValueError('no natural component-acquisition cases')
    return selected, {'source_manifest': file_identity(root / 'manifest.json'),
                      'candidates': len(candidates), 'selected_per_core': dict(counts),
                      'selection': 'one case per source game, train split only, sorted identifiers before outcomes'}


class FirstBuy:
    name = 'research-first-buy'
    def __init__(self, base, action, enabled):
        self.base, self.action, self.enabled = base, action, enabled
        self.first = True
    def reset(self):
        self.base.reset()
        self.first = True
    def decide(self, game, seat):
        if self.first and self.enabled:
            self.first = False
            if not bytes(game.legal_mask(seat))[self.action]:
                raise RuntimeError('predeclared core purchase became illegal')
            self.base.note_external_purchase(game, seat, self.action)
            pi = np.zeros(245, np.float32)
            pi[self.action] = 1
            return Decision(self.action, pi, {'research_intervention': 'buy_core_first', 'forced': True})
        self.first = False
        return self.base.decide(game, seat)


class AcquisitionRecorder(DecisionRecorder):
    def __init__(self, *args, core, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = core
        self.events = Counter()
        self.first_action = None
        self.normal_by_index = {}
    def _normal(self, item):
        import hsbg_engine as engine
        index = item_card(item)
        if 'normal_card_ids' not in _CACHE:
            cards = json.loads(Path('engine/crates/hsbg-engine/data/cards.json').read_text())['cards']
            _CACHE['normal_card_ids'] = {c['id']: c.get('normal') or c['id'] for c in cards}
        if index not in self.normal_by_index:
            self.normal_by_index[index] = _CACHE['normal_card_ids'][engine.card_info(index + 1)[0]]
        return self.normal_by_index[index]
    def after(self, game, before, decision, executed_action, forced_tier=False):
        a = int(executed_action)
        p = before['state_before']['players'][before['seat']]
        if self.first_action is None:
            self.first_action = a
        item, event = None, None
        if 8 <= a < 16:
            item, event = p['shop'][a-8], 'bought'
        elif 16 <= a < 23:
            item, event = p['board'][a-16], 'sold'
        elif 46 <= a < 126:
            item, event = p['hand'][(a-46)//8], 'played'
        if item is not None and self._normal(item) == self.core:
            self.events[event] += 1
        super().after(game, before, decision, executed_action, forced_tier)


def trial(task):
    import torch
    import hsbg_engine as engine
    from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    torch.set_num_threads(1)
    case, name, checkpoint, arm, replica, root, device = task
    library = 'artifacts/libraries/arena-cal-v14-snap'
    key = (checkpoint, device)
    if key not in _CACHE:
        _CACHE[key] = make_eval_policy(checkpoint, protocol=EvalProtocol(mirror_snaps=library), device=device)
        _CACHE[key]['research_checkpoint_identity'] = file_identity(checkpoint)
    if 'pool' not in _CACHE:
        _CACHE['pool'] = MirrorSnapshotPool(library, sample='p50band')
    built = _CACHE[key]
    seed = 11901000 + case['case'] * 1000 + replica
    state = json.loads(json.dumps(case['state']))
    fresh = json.loads(engine.Game(seed=seed).state_json())
    state['seed'], state['rng'] = fresh['seed'], fresh['rng']
    path = Path(root) / 'trials' / name / f"case-{case['case']:03d}-r{replica:02d}-{arm}"
    path.mkdir(parents=True, exist_ok=False)
    identity = {'name': name, 'checkpoint_sha256': built['research_checkpoint_identity']['sha256'],
                'scope': 'research conditional suffix, not formal evaluation',
                'source_seed': case['source_seed'], 'source_turn': case['turn'],
                'core': case['core'], 'intervention': arm, 'future_seed': seed,
                'reset_controller': True, 'initial_opponents_preserved': True}
    recorder = AcquisitionRecorder(path / 'decisions.jsonl.gz', identity=identity, core=case['core'])
    policy = FirstBuy(built['policy'], case['buy_action'], arm == 'buy')
    try:
        result = evaluate_policy(policy, EvalConfig(games=1, seed=seed,
            research_initial_state=state, collect_economy=True, decision_recorder=recorder,
            mirror_pool=_CACHE['pool'], mirror_strict=False, mirror_seats=7, mirror_p=1.0))
        if result['illegal_actions'] or result['truncated_games']:
            raise RuntimeError('conditional rollout failed quality gates')
        trace = recorder.finish(result)
        validation = validate_trace(path / 'decisions.jsonl.gz', replay=16)
    except BaseException:
        recorder.abort()
        raise
    turns = []
    for row in result['end_turn_economy']:
        board = row['eot_board']
        if board is None:
            raise RuntimeError('missing EOT evidence')
        turns.append({'turn': row['turn'], 'strength': row['eot_attack'] + row['eot_board_hp'],
                      'gold': row['gold'], 'health': row['health'], 'actions': row['actions'],
                      'core_present': any(recorder._normal(m) == case['core'] for m in board)})
    summary = {'case': case['case'], 'source_seed': case['source_seed'], 'initial_turn': case['turn'],
               'core': case['core'], 'model': name, 'arm': arm, 'replica': replica, 'seed': seed,
               'rank': result['rank_sequence'][0], 'first_action': recorder.first_action,
               'core_events': dict(recorder.events), 'turns': turns,
               'trace': trace, 'validation': validation}
    atomic_json(path / 'result.json', {'summary': summary, 'evaluation': result})
    return summary


def compare(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row['model'], row['case'], row['replica'])][row['arm']] = row
    differences = defaultdict(list)
    for (model, case, replica), arms in grouped.items():
        if set(arms) != {'buy', 'natural'}:
            raise ValueError('unpaired conditional outcome')
        a, b = arms['buy'], arms['natural']
        if a['seed'] != b['seed']:
            raise ValueError('paired future seed differs')
        differences[(model, a['core'], case)].append(a['rank'] - b['rank'])
    by_family = defaultdict(list)
    for (model, core, case), values in differences.items():
        by_family[(model, core)].append({'case': case, 'rank_difference': float(np.mean(values)),
                                        'replicas': len(values)})
    rng = np.random.default_rng(11901000)
    report = {}
    for (model, core), cases in by_family.items():
        x = np.array([c['rank_difference'] for c in cases])
        boot = x[rng.integers(len(x), size=(10000, len(x)))].mean(-1)
        report[model + '/' + core] = {'cases': len(cases), 'case_means': cases,
            'buy_minus_natural_rank': float(x.mean()), 'case_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--max-per-core', type=int, default=12)
    parser.add_argument('--replicas', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.max_per_core, args.replicas, args.workers) < 1:
        raise ValueError('positive case, replicate and worker budgets required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    models = dict(x.split('=', 1) for x in args.checkpoint)
    if len(models) != len(args.checkpoint):
        raise ValueError('duplicate model names')
    identities = {name: file_identity(path) for name, path in models.items()}
    plan_path = Path('config/research/core-acquisition-20261118.json')
    plan = json.loads(plan_path.read_text())
    for name, identity in identities.items():
        if name not in plan['models'] or identity['sha256'] != plan['models'][name]['sha256']:
            raise ValueError('model differs from predeclared acquisition study: ' + name)
    cases, selection = select_cases(args.bank, args.max_per_core)
    with gzip.open(out / 'cases.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for case in cases:
            stream.write(json.dumps(case, separators=(',', ':')) + '\n')
    from hsbg.az.train.common import engine_info
    library = Path('artifacts/libraries/arena-cal-v14-snap')
    library_identity = {p.name: file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}
    if not library_identity:
        raise ValueError('missing training snapshot field')
    run = {'args': vars(args), 'selection': selection, 'models': identities, 'engine': engine_info(),
           'plan': file_identity(plan_path),
           'snapshot_files': library_identity,
           'cases': file_identity(out / 'cases.jsonl.gz'),
           'source': source_identity(Path(__file__).resolve().parents[1]), 'environment': environment_identity(),
           'scope': 'Conditional training-source counterfactual; no promotion or automatic training labels'}
    if run['source']['tracked_dirty']:
        raise RuntimeError('committed source required')
    atomic_json(out / 'run.json', run)
    tasks = [(case, name, checkpoint, arm, replica, str(out), args.device)
             for case in cases for replica in range(args.replicas)
             for name, checkpoint in models.items() for arm in ('natural', 'buy')]
    print(json.dumps({'selection': selection, 'rollouts': len(tasks)}), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(trial, tasks):
            rows.append(row)
            if len(rows) % 32 == 0:
                print(json.dumps({'finished': len(rows), 'total': len(tasks)}), flush=True)
    for name, path in models.items():
        if file_identity(path) != identities[name]:
            raise RuntimeError('model changed during counterfactual study')
    if {p.name: file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))} != library_identity:
        raise RuntimeError('snapshot field changed during counterfactual study')
    result = {'valid': True, 'rollouts': len(rows), 'selection': selection, 'models': identities,
              'comparison': compare(rows), 'rows': rows, 'run': file_identity(out / 'run.json'),
              'scope': 'Case-clustered paired research suffix outcomes; not natural-game performance'}
    atomic_json(out / 'complete.json', result)
    print(json.dumps({'comparison': result['comparison']}), flush=True)


if __name__ == '__main__':
    main()
