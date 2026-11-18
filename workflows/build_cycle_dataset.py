"""Build native verified cycle tasks and independent whole-game policy anchors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import hsbg_engine as engine
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.cycle_tasks import KINDS, make_task, solve, achieved, own
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.az.train.common import engine_info
from workflows.build_economic_dataset import observation


def build_task(args):
    source, kind, variant = args
    seed = int(source['seed'])
    game, goal, edits = make_task(source, kind, variant)
    initial = json.loads(game.state_json())
    rows, certificate = [], []
    try:
        while not achieved(game, goal):
            remaining = goal['max_depth'] - len(rows)
            plan = solve(game, goal, max_depth=remaining)
            accepted = set(plan['actions'])
            for salt in (11829, 71829):
                alternate_state = json.loads(game.state_json())
                alternate_state['rng'] = json.loads(engine.Game(seed=seed + salt).state_json())['rng']
                alternate = engine.Game.from_json(json.dumps(alternate_state))
                if M.observe(alternate, 0, M.SCHEMA) != M.observe(game, 0, M.SCHEMA):
                    raise ValueError('RNG replacement changed observation')
                accepted &= set(solve(alternate, goal, max_depth=remaining)['actions'])
            if not accepted:
                raise ValueError('no RNG-robust common label')
            f, c, mask = observation(game)
            target = np.zeros(245, bool)
            target[list(accepted)] = True
            if not (target <= mask).all():
                raise ValueError('teacher label outside native mask')
            rows.append((f, c, mask, target, seed, kind))
            action = min(accepted)
            before = json.loads(game.state_json())
            game.step(0, action)
            certificate.append({'action': action, 'acceptable_actions': sorted(accepted),
                                'remaining_distance': plan['distance'], 'before': before,
                                'after': json.loads(game.state_json())})
            if any(game.diagnostic_counts()):
                raise RuntimeError('native diagnostic in demonstration')
        if not rows:
            raise ValueError('task was already complete')
    except ValueError as error:
        return {'accepted': False, 'seed': seed, 'kind': kind, 'variant': variant,
                'reason': str(error)}
    return {'accepted': True, 'seed': seed, 'kind': kind, 'variant': variant,
            'goal': goal, 'initial': initial, 'interventions': edits,
            'certificate': certificate, 'rows': rows,
            'split': 'dev' if seed % 5 == 0 else 'train'}


class AnchorRecorder:
    """Reservoir sample eight states per turn, including mid-turn decisions."""
    def __init__(self):
        self.rng = np.random.default_rng(1182801)
        self.rows = {}
        self.counts = {}
        self.seed = None
    def begin_game(self, game, game_index, seed, seat):
        self.seed = int(seed)
    def before(self, game, seat):
        key = (self.seed, int(game.turn))
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        rows = self.rows.setdefault(key, [])
        index = len(rows) if count <= 8 else int(self.rng.integers(count))
        if index < 8:
            row = observation(game, seat)
            if index == len(rows):
                rows.append(row)
            else:
                rows[index] = row
        return None
    def after(self, *args, **kwargs):
        pass
    def end_game(self, *args, **kwargs):
        pass


def collect_anchors(checkpoint, games):
    import torch
    from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    torch.set_num_threads(1)
    library = 'artifacts/libraries/arena-cal-v14-snap'
    built = make_eval_policy(checkpoint, protocol=EvalProtocol(mirror_snaps=library), device='cpu')
    recorder = AnchorRecorder()
    result = evaluate_policy(built['policy'], EvalConfig(
        games=games, seed=11808000, collect_economy=True, decision_recorder=recorder,
        mirror_pool=MirrorSnapshotPool(library, sample='p50band'), mirror_strict=False,
        mirror_seats=7, mirror_p=1.0))
    if result['illegal_actions'] or result['truncated_games']:
        raise RuntimeError('anchor source game failed quality gates')
    rows = [r for group in recorder.rows.values() for r in group]
    return rows, {'seeds': result['seed_sequence'], 'ranks': result['rank_sequence'],
                  'rows': len(rows), 'observed_decisions': sum(recorder.counts.values()),
                  'per_turn': {str(t): sum(len(v) for (_, turn), v in recorder.rows.items() if turn == t)
                               for t in sorted({turn for _, turn in recorder.rows})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-sources', type=int, default=192)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--anchor-games', type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    bank = Path(args.bank)
    manifest = json.loads((bank / 'manifest.json').read_text())
    if manifest.get('assessment_only'):raise ValueError('assessment-only bank cannot produce training data')
    if file_identity(bank / 'states.jsonl.gz') != manifest['data']:
        raise ValueError('source bank identity differs')
    groups = {}
    with gzip.open(bank / 'states.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['seat'] != 0 or not int(row['state']['lobby_races']) & (1 << 3):
                continue
            seed = row['seed']
            if seed not in groups or abs(row['turn'] - 9) < abs(groups[seed]['turn'] - 9):
                groups[seed] = row
    chosen = [groups[k] for k in sorted(groups)][:args.max_sources]
    tasks = [(s, kind, variant) for s in chosen for kind in KINDS for variant in (0, 1)
             if kind != 'activate_draw' or int(s['state']['lobby_races']) & (1 << 5)]
    allrows, rejected, counts = [], [], {}
    with gzip.open(out / 'tasks.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3) as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for task in pool.map(build_task, tasks):
                if not task['accepted']:
                    rejected.append(task)
                    continue
                allrows.extend(task.pop('rows'))
                key = task['split'] + '/' + task['kind']
                counts[key] = counts.get(key, 0) + 1
                stream.write(json.dumps(task, separators=(',', ':')) + '\n')
    for kind in KINDS:
        if min(counts.get('train/' + kind, 0), counts.get('dev/' + kind, 0)) < 4:
            raise RuntimeError('insufficient source-disjoint tasks for ' + kind)
    f, c, m, t, seeds, kinds = zip(*allrows)
    np.savez_compressed(out / 'labels.npz', features=np.stack(f), cards=np.stack(c),
                        masks=np.stack(m), targets=np.stack(t), seeds=np.array(seeds, np.int64),
                        dev=np.array([s % 5 == 0 for s in seeds], bool), kinds=np.array(kinds))
    anchors, anchor_summary = collect_anchors(args.checkpoint, args.anchor_games)
    af, ac, am = zip(*anchors)
    np.savez_compressed(out / 'anchors.npz', features=np.stack(af), cards=np.stack(ac), masks=np.stack(am))
    atomic_json(out / 'manifest.json', {
        'schema': 'hsai.cycle-process-dataset/v1', 'args': vars(args),
        'source': source_identity(Path(__file__).resolve().parents[1]), 'engine': engine_info(),
        'bank_manifest': file_identity(bank / 'manifest.json'),
        'anchor_parent': file_identity(args.checkpoint), 'anchor_summary': anchor_summary,
        'counts': counts, 'rows': len(allrows), 'source_games': len(chosen), 'rejected': rejected,
        'split': 'whole source game seed % 5 == 0 is dev; anchor seeds independent of all task sources',
        'mechanism_schema': M.SCHEMA, 'mechanism_schema_sha256': M.SCHEMA_SHA256,
        'scope': 'Synthetic shortest bounded resource/growth tasks; not full-game expert optimality',
        'files': {name: file_identity(out / name) for name in ('tasks.jsonl.gz', 'labels.npz', 'anchors.npz')}})
    print(json.dumps({'counts': counts, 'rows': len(allrows), 'rejected': len(rejected),
                      'anchors': anchor_summary}), flush=True)


if __name__ == '__main__':
    main()
