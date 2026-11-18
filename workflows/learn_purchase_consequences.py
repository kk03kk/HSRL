"""Paired natural shops for learning two-turn economic outcomes of alternative purchases."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.runtime.artifacts import atomic_json
from tools.assess_investment_timing import UTILITY
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.natural_resource_frontiers import played
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy


ROOT = Path('artifacts/runs/nus-purchase-consequences-20261118-1')
TERMS = ('battlecry', 'deathrattle', 'end of', 'start of', 'whenever', 'after you',
         'each turn', 'gold', 'discount', 'gain', 'give', 'double', 'discover',
         'your hand', 'tavern spell', 'magnetic', 'divine shield', ' avenge',
         'permanent', 'next turn', 'refresh', 'sell', 'play', 'attack', 'summon')
LABELS = ('created_used', 'repeat_source', 'cash_reinvested', 'full_board_actions',
          'hp_after', 'strength_after', 'terminal_utility')


@lru_cache(maxsize=None)
def semantics(field, cid):
    card = native(field)[2][cid]
    text = ' ' + str(card.get('text') or '').lower()
    return np.array([term in text for term in TERMS], dtype=float)


def features(field, player, turn, action, prior, info):
    shop = player['shop'][action - 8]
    cid = item_card(shop)
    card = info[cid]
    board = player['board']
    total = sum(m['atk'] + m['hp'] for m in board)
    state = [turn, player['health'] + player['armor'], player['gold'], np.log1p(total)]
    own = [m['card'] for m in board] + [item_card(q) for q in player['hand']]
    owned_text = np.mean([semantics(field, c) for c in own], axis=0) if own else np.zeros(len(TERMS))
    flags = semantics(field, cid)
    races = set(card['races']) - {'ALL'}
    allies = sum(bool(races.intersection(info[m['card']]['races'])) for m in board)
    unit = shop['item'].get('Minion', {})
    offer = [float(prior[action]), card['tier'] or 0, unit.get('atk', 0), unit.get('hp', 0),
             card['kind'] == 'minion', allies, own.count(cid), len(board), len(player['hand']),
             player['tier'], player['upgrade_cost']]
    return [float(x) for x in [*state, *offer, *flags, *owned_text, *(flags * owned_text),
                               *(flags * (len(board) == 7)), *(flags * min(allies, 4))]]


class Purchase(TimingPolicy):
    def __init__(self, field, target, arm):
        policy, self.pool, self.info = native(field)
        super().__init__(policy, 'parent', False)
        self.field, self.target, self.arm = field, target, arm

    def reset(self):
        super().reset()
        self.sample = None

    def __call__(self, f, c, mask, seat):
        prior, value = self.original(f, c, mask, seat)
        if self.sample is not None or self.current != self.target or self.player['pending']:
            return prior, value
        first = int(prior.argmax())
        if not 8 <= first < 16 or not mask[first]:
            return prior, value
        shop = self.player['shop']
        others = [8 + i for i, q in enumerate(shop[:8]) if mask[8 + i]
                  and item_card(q) != item_card(shop[first - 8])]
        if not others:
            return prior, value
        second = max(others, key=lambda a: prior[a])
        race = Counter(r for m in self.player['board'] for r in self.info[m['card']]['races'] if r != 'ALL')
        top = race.most_common(2)
        family = top[0][0] if top and top[0][1] >= 3 and (len(top) == 1 or top[0][1] > top[1][1]) else 'MIXED'
        self.sample = dict(turn=self.current, step=len(self.actions), family=family, choices=[first, second],
                           cards=[item_card(shop[a - 8]) for a in (first, second)],
                           x=[features(self.field, self.player, self.current, a, prior, self.info)
                              for a in (first, second)])
        if self.arm == 0:
            return prior, value
        out = np.zeros_like(prior)
        out[second] = 1
        return out, value


class Ledger:
    def __init__(self, policy):
        self.policy = policy

    def begin_game(self, game, gi, seed, seat):
        self.live = {}
        self.uses = Counter()
        self.spent = self.earned = self.crowded = 0
        self.turn = -1
        self.first = None

    def before(self, game, seat):
        t = int(game.turn)
        if t not in (self.policy.target, self.policy.target + 1):
            return None
        if t != self.turn:
            self.turn, self.earned = t, 0
        return json.loads(game.player_json(seat))

    def after(self, game, before, dec, action, forced):
        if before is None or self.policy.sample is None:
            return
        if self.first is None:
            self.first = action
        if action == 0 or int(game.turn) != self.turn:
            return
        post = json.loads(game.player_json(0))
        step = played(before, action)
        if step and step['hid'] in self.live:
            self.uses[self.live.pop(step['hid'])] += 1
        # Count different physical outputs of the same source card type separately.
        if step or action in (4, 5):
            source = item_card(step) if step else -action
            old = {q['hid'] for q in before['hand']}
            self.live.update({q['hid']: source for q in post['hand'] if q['hid'] not in old})
        delta = post['gold'] - before['gold']
        if delta > 0 and not 16 <= action < 23:
            self.earned += delta
        elif delta < 0:
            # Spend other funds first. This is a conservative within-turn quantity.
            used = max(0, self.earned - post['gold'])
            self.spent += used
            self.earned -= used
        self.crowded += int(len(before['board']) == 7)

    def end_game(self, *args):
        pass


def collect_game(i):
    field = 'development' if i < 1024 else 'transfer'
    turn = 7 + i % 5
    seed = 85100000 + i
    outcomes = []
    for arm in (0, 1):
        policy = Purchase(field, turn, arm)
        ledger = Ledger(policy)
        result = evaluate_policy(policy, EvalConfig(games=1, seed=seed, mirror_pool=policy.pool,
                                                    decision_recorder=ledger))
        check(result)
        if policy.sample is None:
            if arm:
                raise ValueError('lost the paired opportunity')
            return dict(i=i, seed=seed, field=field, turn=turn, sample=None, rank=result['rank_sequence'][0])
        if ledger.first != policy.sample['choices'][arm]:
            raise ValueError('purchase identity mismatch')
        if arm == 0:
            sample = policy.sample
        elif sample != policy.sample:
            raise ValueError('unequal state before intervention')
        rank = result['rank_sequence'][0]
        after = policy.eot.get(turn + 1, [0, 0, 0])
        outcomes.append(dict(rank=rank, y=[sum(ledger.uses.values()), sum(max(n-1, 0) for n in ledger.uses.values()),
                                           ledger.spent, ledger.crowded, after[1], np.log1p(after[0]), UTILITY[rank-1]]))
    return dict(i=i, seed=seed, field=field, turn=turn, sample=sample, outcomes=outcomes)


if __name__ == '__main__':
    shard = int(sys.argv[1])
    out = ROOT / f'shard-{shard:02}'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'run.json', provenance())
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
    with ProcessPoolExecutor(workers) as pool, (out / 'games.jsonl').open('x') as stream:
        for row in pool.map(collect_game, range(64 * shard, 64 * (shard + 1))):
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    atomic_json(out / 'complete.json', dict(shard=shard, games=64, labels=LABELS))
