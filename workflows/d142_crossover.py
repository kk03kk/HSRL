"""T9 policy crossover on one natural prefix per seed and opening policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.evaluation.metrics import _strength_by_seed
from hsbg.az.train.common import engine_info
from hsbg.az.train.eval_policy import make_eval_policy, protocol_from_args
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool

SEED = 87419000
N = 256
ROOT_RUNS = Path('artifacts/runs')
MID = ROOT_RUNS/'nus-d142-midpoint-20261119-1'
ARCHIVES = {'P': ROOT_RUNS/'nus-human-turn-parent-20261118-1',
            'E': ROOT_RUNS/'nus-human-turn-overnight-eval-20261119-1'}
BASE = json.loads(Path('config/evaluation/human-turn-3702-parent.json').read_text())
MODELS = {'P': (BASE['models'][0]['checkpoint'], BASE['models'][0]['sha256']),
          'E': (str(ROOT_RUNS/'nus-human-turn-overnight-20261118-1/segment-04/train/snapshots/final_003265/model.pt'),
                'e1117934c76c37090212ec02e75d8ef52b21280c8d47f7dd5eb6089a228f74ad')}
LIB = BASE['evaluation']['mirror-snaps']
EXPECTED = BASE['expected_protocol_sha256']


def digest(state):
    return hashlib.sha256(json.dumps(state, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


class PrefixComplete(Exception):
    pass


class Fork:
    turn = 9

    def __init__(self, stop=False):
        self.state = None
        self.stop = stop

    def __call__(self, game, policy, strict):
        self.state = json.loads(game.state_json())
        self.opening = digest(self.state)
        self.strict = deepcopy(strict, {id(strict.library): strict.library})
        self.policy = deepcopy(policy, {id(policy.evaluator): policy.evaluator})
        if self.stop:
            raise PrefixComplete


class Evidence:
    def __init__(self, expected=None):
        self.expected = expected
        self.terminal = None

    def begin_game(self, game, gi, seed, seat):
        if self.expected and digest(json.loads(game.state_json())) != self.expected:
            raise RuntimeError('restored game differs before the first decision')

    def before(self, game, seat):
        pass

    def after(self, game, before, decision, action, forced):
        pass

    def end_game(self, game, rank, quality):
        self.terminal = digest(json.loads(game.state_json()))


class ResumedPolicy:
    def __init__(self, fork, evaluator):
        self.fork = fork
        self.evaluator = evaluator

    def reset(self):
        saved = self.fork.policy
        self.policy = deepcopy(saved, {id(saved.evaluator): self.evaluator})

    def decide(self, game, seat):
        return self.policy.decide(game, seat)


def play(pool, policy, seed, *, capture=None, resume=None):
    evidence = Evidence(resume.opening if resume else None)
    config = EvalConfig(games=1, seed=seed, mirror_strict=True, mirror_pool=pool,
                        collect_economy=True, collect_action_trace=True, decision_recorder=evidence,
                        research_capture_strict=capture, research_initial_state=resume.state if resume else None,
                        research_strict_context=resume.strict if resume else None)
    result = evaluate_policy(policy, config)
    mirror = result['mirror']
    if (result['games_finished'] != 1 or result['seed_sequence'] != [seed] or
            any(result[k] for k in ('illegal_actions', 'truncated_games', 'truncated_combats')) or
            any(mirror['mirror_'+k] for k in ('rejected', 'missing', 'errors', 'fallback',
                                              'pool_unbalanced', 'pool_species_unbalanced')) or
            mirror['mirror_attempt'] != mirror['mirror_applied'] or not mirror['mirror_applied']):
        raise RuntimeError(f'invalid evaluation seed={seed}')
    actions = [[x['turn'], x['action']] for x in result['action_trace']]
    strength = _strength_by_seed(result['end_turn_economy']).get(str(seed), {})
    if len(actions) != result['decisions'] or any(str(t) not in strength for t,a in actions if a == 0 and t > 0):
        raise ValueError('missing action or EOT strength')
    return dict(rank=result['rank_sequence'][0], actions=actions, strength=strength,
                terminal=evidence.terminal, mirror_applied=mirror['mirror_applied'])


def load():
    options = argparse.Namespace(**{k.replace('-', '_'): v for k,v in BASE['evaluation'].items()})
    protocol = protocol_from_args(options, engine=engine_info())
    protocol.assert_training_contract()
    if protocol.sha256() != EXPECTED:
        raise ValueError('incompatible evaluation protocol')
    pool = MirrorSnapshotPool(LIB, sample='random')
    if pool.frames != 5477:
        raise ValueError('incomplete held-out library')
    return pool, {k: make_eval_policy(path, protocol=protocol, device='cuda:0')['policy']
                  for k,(path,sha) in MODELS.items()}


def replay_archives(pool, policies):
    # Replay a previously sealed game up to the boundary, then restore and
    # compare the *entire* same-policy suffix to the archived natural game.
    verified = {}
    for name, directory in ARCHIVES.items():
        with gzip.open(next((directory/'shard0').glob('decisions-*.jsonl.gz')), 'rt') as stream:
            suffix = []
            opening = None
            for line in stream:
                row = json.loads(line)
                if row['event'] == 'action' and row['turn'] >= 9:
                    if not suffix:
                        opening = digest(row['state_before'])
                    suffix.append([row['turn'], row['action']])
                if row['event'] == 'game_end' and suffix:
                    seed, rank, terminal = row['seed'], row['rank'], digest(row['state'])
                    break
            else:
                raise ValueError('missing archived control')
        fork = Fork(stop=True)
        try:
            play(pool, policies[name], seed, capture=fork)
        except PrefixComplete:
            pass
        if fork.opening != opening:
            raise ValueError(f'{name} archived T9 opening differs')
        matched = play(pool, ResumedPolicy(fork, policies[name].evaluator), seed, resume=fork)
        if (matched['actions'], matched['terminal'], matched['rank']) != (suffix, terminal, rank):
            raise ValueError(f'{name} archived same-policy replay differs')
        verified[name] = dict(seed=seed, opening=opening, actions=len(suffix), rank=rank, terminal=terminal)
    return verified


def worker(out, index):
    pool, policies = load()
    for seed in range(SEED+index, SEED+N, 8):
        for early, other in [('P','E'), ('E','P')]:
            fork = Fork()
            same = play(pool, policies[early], seed, capture=fork)
            if fork.state is None:
                if any(t >= 9 for t,a in same['actions']):
                    raise ValueError('missed T9 fork')
                other_leg = same
            else:
                other_leg = play(pool, ResumedPolicy(fork, policies[other].evaluator), seed, resume=fork)
                other_leg['actions'] = [p for p in same['actions'] if p[0] < 9] + other_leg['actions']
                other_leg['strength'].update({t:v for t,v in same['strength'].items() if int(t) < 9})
                with gzip.open(out/f'{seed}-{early}.state.json.gz', 'xt') as stream:
                    json.dump(fork.state, stream, separators=(',', ':'))
            atomic_json(out/f'{seed}-{early}.json', dict(seed=seed, early=early,
                        opening=fork.opening if fork.state else None, legs={early:same, other:other_leg}))


def verify_inputs():
    sealed = json.loads((MID/'complete.json').read_text())
    if sealed['valid'] is not True or file_identity(MID/'panel.json') != sealed['panel']:
        raise ValueError('midpoint must complete first')
    original = json.loads((MID/'run.json').read_text())['inputs']
    files = {str(path) for path in Path(LIB).rglob('*') if path.is_file()}
    expected_files = {k for k in original if k.startswith(LIB+'/')}
    if files != expected_files or any(file_identity(path) != original[path] for path in files):
        raise ValueError('held-out library changed')
    if any(file_identity(path)['sha256'] != sha for path,sha in MODELS.values()):
        raise ValueError('checkpoint identity mismatch')
    return digest({k:original[k] for k in files})


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    source = source_identity(ROOT)
    if source['tracked_dirty']:
        raise RuntimeError('dirty release')
    field = verify_inputs()
    atomic_json(out/'plan.json', dict(seed=SEED, games=N, logical_trajectories=4*N, switch_turn=9,
                models=MODELS, protocol=EXPECTED, source=source, field_files_sha256=field,
                midpoint=file_identity(MID/'panel.json'), early_deaths='retain in both suffix assignments'))
    pool, policies = load()
    atomic_json(out/'controls.json', replay_archives(pool, policies))
    del policies, pool
    def launch(index):
        with (out/f'worker-{index}.log').open('x') as log:
            subprocess.run([sys.executable, __file__, str(out), '--worker', str(index)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(launch, range(8)))
    if verify_inputs() != field:
        raise ValueError('input changed during run')
    rows = []
    for seed in range(SEED, SEED+N):
        for early in 'PE':
            row = json.loads((out/f'{seed}-{early}.json').read_text())
            if (row['seed'], row['early'], set(row['legs'])) != (seed, early, set('PE')):
                raise ValueError('invalid crossover coverage')
            if row['opening']:
                with gzip.open(out/f'{seed}-{early}.state.json.gz', 'rt') as stream:
                    if digest(json.load(stream)) != row['opening']:
                        raise ValueError('opening checksum mismatch')
            rows.append(row)
    with gzip.open(out/'trajectories.json.gz', 'xt') as stream:
        json.dump(rows, stream, separators=(',', ':'))
    atomic_json(out/'complete.json', dict(valid=True, seeds=N, logical_trajectories=2*len(rows),
                openings=sum(bool(r['opening']) for r in rows),
                files={p:file_identity(out/p) for p in ('trajectories.json.gz','controls.json','plan.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is None:
        run(args.out)
    else:
        worker(args.out, args.worker)
