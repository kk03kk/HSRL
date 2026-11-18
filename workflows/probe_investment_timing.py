"""Paired complete games testing one public investment decision across two families."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.common import engine_info
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity


MODEL = 'artifacts/models/arena-v1.4-r1/model.pt'
FIELDS = {'development': 'artifacts/libraries/arena-cal-v14-snap',
          'transfer': 'artifacts/libraries/arena-cal-v14hold-snap'}
SEEDS = {'development': 74210000, 'transfer': 75210000}
ARMS = ('parent', 'upgrade', 'delay', 'retain')
CACHE = {}


class TimingPolicy:
    """Change the evaluator's prior only. The normal controller accounts for every action."""
    name = 'timing-research'

    def __init__(self, policy, arm, late):
        self.policy = policy
        self.original = policy.evaluator
        self.model = self.original.model
        self.arm = arm
        self.late = late

    def reset(self):
        self.policy.reset()
        self.events = {}
        self.turn = None
        self.blocks = 0
        self.eot = {}
        self.actions = []

    def decide(self, game, seat):
        self.game = game
        self.current = int(game.turn)
        window = 7 <= self.current <= 12
        if window:
            self.player = json.loads(game.player_json(seat))
            if self.turn != self.current:
                self.opening = {m['eid'] for m in self.player['board']}
        self.turn = self.current
        self.policy.evaluator = self if window else self.original
        try:
            dec = self.policy.decide(game, seat)
        finally:
            self.policy.evaluator = self.original
        self.actions.append(int(dec.action))
        if dec.action == 0 and self.current >= 1:
            p = self.player if window else json.loads(game.player_json(seat))
            self.eot[self.current] = [sum(m['atk']+m['hp'] for m in p['board']), p['health'], p['tier']]
        return dec

    def record(self, kind, index, prior):
        p = self.player
        board = p['board']
        event = dict(turn=self.current, tier=p['tier'], health=p['health'], armor=p['armor'],
                     gold=p['gold'], cost=p['upgrade_cost'], hand=len(p['hand']), size=len(board),
                     strength=sum(m['atk']+m['hp'] for m in board), hero=p['hero'],
                     index=len(self.actions), baseline=int(np.argmax(prior)),
                     state=hashlib.sha256(self.game.state_json().encode()).hexdigest())
        if kind == 'sell':
            item = board[index]
            event.update(eid=item['eid'], card=item['card'], retained_strength=item['atk']+item['hp'])
        self.events[kind] = event

    def __call__(self, features, cards, mask, seat):
        prior, value = self.original(features, cards, mask, seat)
        natural = int(np.argmax(prior))
        p = self.player
        turn = self.current
        due = ((turn == 7 and p['tier'] < 4) or (turn == 8 and p['tier'] < 5))
        eligible = (turn in (9, 10) and p['tier'] == 5) if self.late else due
        if 'tier' not in self.events and eligible and mask[3] and mask[0]:
            self.record('tier', None, prior)
        if ('sell' not in self.events and 8 <= turn <= 12 and 16 <= natural < 23 and
                len(p['board']) == 7 and p['board'][natural-16]['eid'] in self.opening and mask[0]):
            self.record('sell', natural-16, prior)
        tier = self.events.get('tier', {})
        sale = self.events.get('sell', {})
        candidate = None
        if self.arm == 'upgrade' and tier.get('index') == len(self.actions):
            candidate = 3
        elif self.arm == 'delay' and tier.get('turn') == turn and natural == 3:
            candidate = self.alternative(prior, mask, natural)
        elif self.arm == 'retain' and sale.get('turn') == turn and 16 <= natural < 23:
            if p['board'][natural-16]['eid'] == sale['eid']:
                candidate = self.alternative(prior, mask, natural)
        if candidate is None or candidate == natural:
            return prior, value
        self.blocks += 1
        result = np.zeros_like(prior)
        result[candidate] = 1
        return result, value

    @staticmethod
    def alternative(prior, mask, blocked):
        legal = np.array(mask, copy=True)
        legal[blocked] = False
        if not legal.any():
            raise ValueError('no alternative to public intervention')
        return int(np.argmax(np.where(legal, prior, -1.)))


def run_seed(task):
    import torch
    torch.set_num_threads(1)
    split, seed, out = task
    field = FIELDS[split]
    if field not in CACHE:
        proto = EvalProtocol(mirror_snaps=field, mirror_strict=1, engine=engine_info())
        CACHE[field] = (make_eval_policy(MODEL, protocol=proto, device='cpu')['policy'],
                        MirrorSnapshotPool(field, sample='p50band'))
    policy, pool = CACHE[field]
    rows = {}
    for arm in ARMS:
        wrapper = TimingPolicy(policy, arm, seed % 2 == 1)
        r = evaluate_policy(wrapper, EvalConfig(games=1, seed=seed, mirror_pool=pool, mirror_strict=True))
        if (r['seed_sequence'] != [seed] or r['illegal_actions'] or r['truncated_games'] or
                r['truncated_combats'] or r['mirror']['mirror_rejected'] or r['mirror']['mirror_errors']):
            raise ValueError('invalid natural game')
        kind = 'sell' if arm == 'retain' else 'tier'
        event = wrapper.events.get(kind)
        if arm != 'parent':
            parent = rows['parent']
            if event != parent['events'].get(kind):
                raise ValueError('different pre-intervention state')
            offset = len(wrapper.actions) if event is None else event['index']
            if parent['actions'][:offset] != wrapper.actions[:offset]:
                raise ValueError('different pre-intervention actions')
            if wrapper.blocks == 0 and (parent['actions'] != wrapper.actions or parent['rank'] != r['rank_sequence'][0]):
                raise ValueError('unaffected game diverged')
        rows[arm] = dict(rank=r['rank_sequence'][0], events=wrapper.events, changed=wrapper.blocks,
                         actions=wrapper.actions, eot=wrapper.eot, mirror=r['mirror'])
    atomic_json(Path(out)/f'{seed}.json', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('split', choices=FIELDS)
    p.add_argument('out', type=Path)
    p.add_argument('--games', type=int, default=112)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path('config/research/natural12m-20261118.json').read_text())
    paths = [MODEL, str(Path(MODEL).with_name('manifest.json'))]
    paths.extend(str(Path(FIELDS[args.split])/f) for f in ('manifest.json','snap_cal.json.gz'))
    def identities():
        return {path:file_identity(path)['sha256'] for path in paths}
    bound = identities()
    if any(sha != plan['asset_sha256'][path] for path,sha in bound.items()):
        raise ValueError('asset identity differs')
    engine = engine_info()
    if engine['source_hash'] != plan['engine_fingerprint']:
        raise ValueError('engine differs')
    start = SEEDS[args.split]
    import subprocess
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','-uno']):
        raise ValueError('tracked source changed')
    proto = EvalProtocol(mirror_snaps=FIELDS[args.split], mirror_strict=1, engine=engine)
    run = dict(revision=revision, engine=engine, assets=bound, split=args.split, arms=ARMS,
               seed=start, games=args.games, protocol=proto.as_dict(), protocol_sha256=proto.sha256())
    atomic_json(args.out/'run.json', run)
    tasks = [(args.split, seed, str(args.out)) for seed in range(start, start+args.games)]
    with ProcessPoolExecutor(args.workers) as executor:
        for n,_ in enumerate(executor.map(run_seed, tasks),1):
            if n % 16 == 0:
                print('completed seeds', n, flush=True)
    if identities() != bound:
        raise ValueError('asset changed during study')
    atomic_json(args.out/'complete.json', dict(seeds=args.games, full_games=4*args.games,
                                              run=file_identity(args.out/'run.json')))


if __name__ == '__main__':
    main()
