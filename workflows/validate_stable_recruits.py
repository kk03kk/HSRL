"""Evaluate whether successive online updates agree on useful natural recruit changes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.group_relative_recruit import checkpoint, evaluator
from workflows.learn_recruit_residual import paired_endpoints
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy


ROOT = Path('artifacts/runs/nus-stable-recruits-20261118-1')
FIRST = 89600000


@lru_cache(maxsize=4)
def actor(arm, iteration):
    return evaluator('transfer', checkpoint(arm, iteration))


class StableRecruit(TimingPolicy):
    def __init__(self, arm, require_agreement):
        base, self.pool, _ = native('transfer')
        super().__init__(base, arm, False)
        self.require_agreement = require_agreement

    def reset(self):
        super().reset()
        self.changes = Counter()
        self.refusals = Counter()

    def __call__(self, f, c, mask, seat):
        parent = self.original(f, c, mask, seat)
        if self.arm is None:
            return parent
        final = actor(self.arm, 4)(f, c, mask, seat)
        decision = int(final[0].argmax())
        if decision == int(parent[0].argmax()):
            return parent
        if self.require_agreement:
            previous = actor(self.arm, 3)(f, c, mask, seat)
            if int(previous[0].argmax()) != decision:
                self.refusals[self.current] += 1
                return parent
        self.changes[self.current] += 1
        return final


def game(i):
    seed = FIRST + i
    rows = []
    for arm, agreement in ((None, False), (0, True), (1, True), (0, False), (1, False)):
        policy = StableRecruit(arm, agreement)
        result = evaluate_policy(policy, EvalConfig(games=1, seed=seed, mirror_pool=policy.pool))
        check(result)
        rows.append(dict(seed=seed, rank=result['rank_sequence'][0], eot=policy.eot,
                         changes=sum(policy.changes.values()), accepted=policy.changes, refused=policy.refusals))
    return rows


def panel(shard):
    directory = ROOT/f'panel-{shard:02}'
    directory.mkdir(parents=True, exist_ok=False)
    identities = {f'{a}-{it}': file_identity(checkpoint(a,it)) for a in (0,1) for it in (3,4)}
    atomic_json(directory/'run.json', dict(**provenance(), endpoints=identities))
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as pool:
        rows = list(pool.map(game, range(32*shard,32*(shard+1))))
    atomic_json(directory/'rows.json', rows)


def assess():
    cases = [case for s in range(8) for case in json.loads((ROOT/f'panel-{s:02}'/'rows.json').read_text())]
    out = {}
    for label, idx in [('stable',(0,1,2)), ('unrestricted',(0,3,4))]:
        report = paired_endpoints([case[j] for case in cases for j in idx], FIRST, 256)
        report['changed_games'] = [sum(case[j]['changes'] > 0 for case in cases) for j in idx[1:]]
        out[label] = report
    for arm in (0,1):
        idx = (3+arm,1+arm,1+arm)
        out[f'stable_minus_unrestricted_{arm}'] = paired_endpoints(
            [case[j] for case in cases for j in idx], FIRST, 256)['report'][1]
    atomic_json(ROOT/'assessment.json', out)


if __name__ == '__main__':
    if sys.argv[1] == 'panel':
        panel(int(sys.argv[2]))
    elif sys.argv[1] == 'assess':
        assess()
    else:
        raise ValueError('unknown stage')
