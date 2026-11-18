"""Learn how selling each eligible unit changes the next recruit's rank forecast."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.common import mask_from_bytes, obs_arrays
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.runtime.artifacts import atomic_json
from tools.assess_investment_timing import UTILITY
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.learn_purchase_consequences import semantics
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy
from workflows.replicate_investment_returns import Captured, GUARDS

ROOT = Path('artifacts/runs/nus-sale-afterstates-20261118-1')


def sale_features(field, player, opening, prior, sales, info, turn):
    board = player['board']
    totals = np.array([m['atk'] + m['hp'] for m in board])
    ids = Counter(m['card'] for m in board)
    traits = np.array([semantics(field, m['card']) for m in board])
    mean_traits = traits.mean(axis=0)
    rows = []
    for a in sales:
        i = a-16
        m = board[i]
        card = info[m['card']]
        races = set(card['races']) - {'ALL'}
        support = sum(bool(races.intersection(info[q['card']]['races'])) for q in board if q['eid'] != m['eid'])
        share = totals[i] / max(1, totals.sum())
        old = m['eid'] in opening
        rich = [np.log(max(prior[a], 1e-9)), np.log1p(totals[i]), share, m['atk']/max(1, totals[i]),
                card['tier'] or 0, bool(card.get('normal')), ids[m['card']], old, support,
                i/max(1, len(board)-1), share*turn, share*player['health'], old*len(player['hand'])]
        rows.append(np.r_[rich, traits[i], traits[i]*mean_traits, traits[i]*support, traits[i]*old].tolist())
    return rows


class SalePolicy(TimingPolicy):
    def __init__(self, field, target=None, source=None, action=None, head=None):
        policy, self.pool, self.info = native(field)
        super().__init__(policy, 'parent', False)
        self.field, self.target, self.source, self.first, self.head = field, target, source, action, head

    def reset(self):
        super().reset()
        self.changed = []
        self.considered = set()
        if self.source:
            for k, v in self.source['guards'].items():
                setattr(self.policy, k, deepcopy(v))

    def __call__(self, f, c, mask, seat):
        prior, value = self.original(f, c, mask, seat)
        default = int(prior.argmax())
        if self.source and not self.actions:
            if default != self.source['default'] or not mask[self.first]:
                raise ValueError('restored decision drift')
            choice = self.first
        elif self.source or not 16 <= default < 23 or self.current in self.considered:
            return prior, value
        else:
            sales = [a for a in range(16, 23) if mask[a]]
            if len(sales) < 3 or self.player['pending']:
                return prior, value
            self.considered.add(self.current)
            if self.target is None and self.head is None:
                return prior, value
            x = sale_features(self.field, self.player, self.opening, prior, sales, self.info, self.current)
            if self.current == self.target:
                require_clean_engine(self.game)
                self.saved = dict(state=json.loads(self.game.state_json()),
                                  guards={k: deepcopy(getattr(self.policy, k)) for k in GUARDS},
                                  default=default, turn=self.current, x=x, sales=sales)
                raise Captured()
            if self.head is None:
                return prior, value
            score = np.asarray(x) @ self.head
            j = int(score.argmax())
            if score[j] - score[sales.index(default)] <= .035:
                return prior, value
            choice = sales[j]
            if choice != default:
                self.changed.append([self.current, default, choice])
        out = np.zeros_like(prior)
        out[choice] = 1
        return out, value


class NextRecruit:
    def __init__(self, policy, turn):
        self.policy, self.turn = policy, turn

    def begin_game(self, *args):
        self.first = None

    def before(self, game, seat):
        if int(game.turn) <= self.turn:
            return
        require_clean_engine(game)
        f, c = obs_arrays(game.observe(seat))
        _, rank = self.policy.original.logits(f, c, mask_from_bytes(game.legal_mask(seat)))
        prob = np.exp(rank[0] - rank[0].max())
        self.payoff = float(prob @ UTILITY / prob.sum())
        raise Captured()

    def after(self, game, before, dec, action, forced):
        if self.first is None:
            self.first = action

    def end_game(self, *args):
        pass


def collect(i):
    import hsbg_engine as engine
    turn, seed = 7 + i % 6, 86100000+i
    w = SalePolicy('development', target=turn)
    try:
        check(evaluate_policy(w, EvalConfig(games=1, seed=seed, mirror_pool=w.pool)))
    except Captured:
        src = w.saved
    else:
        return dict(i=i, turn=turn, opportunity=False)
    outcomes = []
    for rep in range(3):
        future = 86400000 + 4*i + rep
        fresh = json.loads(engine.Game(seed=future).state_json())
        state = dict(src['state'], seed=fresh['seed'], rng=fresh['rng'])
        values = []
        for action in src['sales']:
            policy = SalePolicy('development', source=src, action=action)
            horizon = NextRecruit(policy, turn)
            try:
                result = evaluate_policy(policy, EvalConfig(games=1, seed=future, mirror_pool=policy.pool,
                                         research_initial_state=state, decision_recorder=horizon))
                check(result)
                value = UTILITY[result['rank_sequence'][0]-1]
            except Captured:
                value = horizon.payoff
            if horizon.first != action:
                raise ValueError('sale was not executed')
            values.append(value)
        outcomes.append(values)
    return dict(i=i, turn=turn, opportunity=True, x=src['x'], sales=src['sales'],
                default=src['default'], outcomes=outcomes)


def train():
    rows = []
    for shard in range(16):
        group = json.loads((ROOT / f'source-{shard:02}' / 'complete.json').read_text())['rows']
        if [r['i'] for r in group] != list(range(48*shard, 48*(shard+1))):
            raise ValueError('incorrect source set')
        rows.extend(group)
    result = []
    for arm in range(2):
        # Interleaved blocks give both independent training splits the full turn distribution.
        selected = [r for r in rows if r['opportunity'] and r['i']//6%2 == arm]
        xs, ys = [], []
        for r in selected:
            x = np.array(r['x'])
            y = np.mean(r['outcomes'], axis=0)
            xs.append((x-x.mean(axis=0))/np.sqrt(len(x)))
            ys.append((y-y.mean())/np.sqrt(len(y)))
        x, y = np.concatenate(xs), np.concatenate(ys)
        scale = np.maximum(x.std(axis=0), .02)
        z = x/scale
        weight = np.linalg.solve(z.T@z + 64*np.eye(z.shape[1]), z.T@y)/scale
        result.append(dict(weight=weight.tolist(), sources=len(selected), labels=len(y)))
    if (ROOT / 'heads.json').exists():
        raise FileExistsError('fixed heads already exist')
    atomic_json(ROOT / 'heads.json', result)


def panel(i):
    heads = json.loads((ROOT / 'heads.json').read_text())
    rows = []
    for h in [None, *[np.array(a['weight']) for a in heads]]:
        w = SalePolicy('transfer', head=h)
        result = evaluate_policy(w, EvalConfig(games=1, seed=86700000+i, mirror_pool=w.pool))
        check(result)
        rows.append(dict(seed=86700000+i, rank=result['rank_sequence'][0], eot=w.eot, changes=len(w.changed)))
    return rows


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'train':
        train()
    elif stage == 'assess':
        from workflows.learn_recruit_residual import paired_endpoints
        rows = []
        for shard in range(8):
            cases = json.loads((ROOT / f'panel-{shard:02}' / 'complete.json').read_text())['rows']
            rows.extend(r for case in cases for r in case)
        atomic_json(ROOT / 'assessment.json', paired_endpoints(rows, 86700000, 256))
    elif stage in ('source', 'panel'):
        shard = int(sys.argv[2])
        n = 48 if stage == 'source' else 32
        out = ROOT / f'{stage}-{shard:02}'
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / 'run.json', provenance())
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 2))) as pool:
            rows = list(pool.map(collect if stage == 'source' else panel, range(n*shard, n*(shard+1))))
        atomic_json(out / 'complete.json', dict(rows=rows))
    else:
        raise ValueError('unknown stage')
