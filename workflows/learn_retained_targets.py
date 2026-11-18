"""Teach target selection using entity survival and the native, visible effect on each board."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.runtime.artifacts import atomic_json
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.learn_purchase_consequences import semantics
from workflows.learn_recruit_residual import paired_endpoints
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy

ROOT = Path('artifacts/runs/nus-retained-targets-20261118-1')


def unit_features(field, p, t):
    board = p['board']
    total = max(1, sum(m['atk']+m['hp'] for m in board))
    cards = Counter(m['card'] for m in board)
    return np.array([np.r_[t, p['health']+p['armor'], p['tier'], len(board), m['tier'], m['golden'],
                           np.log1p(max(0,m['atk'])), np.log1p(max(0,m['hp'])), (m['atk']+m['hp'])/total,
                           min(m['shield_hits'], 3), cards[m['card']], len(m['attached']),
                           semantics(field, m['card'])] for m in board])


def strength(board):
    current = np.array([m['atk']+m['hp'] for m in board], dtype=float)
    permanent = current - np.array([m['temp_atk']+m['temp_hp'] for m in board])
    return current, permanent


def contract(p):
    # Effects may redistribute stats. Board identities, keywords, attachments,
    # hand and immediate resources must agree for two targets to be comparable.
    return ([(m['eid'],m['card'],m['kw'],m['golden'],m['attached']) for m in p['board']],
            p['hand'],p['gold'],p['health'],p['armor'],p['pending'])


@lru_cache(maxsize=8)
def random_stream(rep):
    import hsbg_engine as engine
    f = json.loads(engine.Game(seed=88190000+rep).state_json())
    return f['seed'],f['rng']


class Durability(TimingPolicy):
    def __init__(self, field, collect=False, head=None):
        parent,self.pool,_ = native(field)
        super().__init__(parent,'parent',False)
        self.field,self.collect,self.head = field,collect,head

    def reset(self):
        super().reset()
        self.frames = {}
        self.counter = Counter()
        self.sources = Counter()

    def decide(self, game, seat):
        dec = super().decide(game, seat)
        t = int(game.turn)
        if self.collect and dec.action == 0 and 7<=t<=14:
            p = json.loads(game.player_json(seat))
            self.frames[t] = dict(ids=[m['eid'] for m in p['board']],
                                  x=unit_features(self.field,p,t).tolist() if t<=12 else [])
        return dec

    def __call__(self,f,c,mask,seat):
        prior,value = self.original(f,c,mask,seat)
        if self.head is None or not self.player['pending']:
            return prior,value
        pending = self.player['pending'][-1]
        if 'Target' not in pending:
            return prior,value
        default = int(prior.argmax())
        targets = pending['Target']['options']
        own = {m['eid'] for m in self.player['board']}
        opts = [30+i for i,e in enumerate(targets[:16]) if e in own and mask[30+i]]
        if default not in opts or len(opts)<2:
            return prior,value
        self.counter['eligible'] += 1
        import hsbg_engine as engine
        st = json.loads(self.game.state_json())
        board = self.player['board']
        before = strength(board)
        x = unit_features(self.field,self.player,self.current)
        z = (x-self.head['mean'])/self.head['scale']
        retention = np.clip(self.head['intercept'] + z @ self.head['weight'], .05, .95)
        reference = contract(self.player)
        estimates = []
        for rep in (0,1):
            seed,rng = random_stream(rep)
            serialized = json.dumps(dict(st,seed=seed,rng=rng), separators=(',',':'))
            by_action = {}
            for a in opts:
                branch = engine.Game.from_json(serialized)
                branch.step(seat,a)
                require_clean_engine(branch)
                if int(branch.turn)!=self.current or branch.is_over():
                    continue
                p = json.loads(branch.player_json(seat))
                cons = contract(p)
                if cons[0] != reference[0] or p['pending']:
                    continue
                now,perm = strength(p['board'])
                dnow,dperm = now-before[0],perm-before[1]
                score = float(.35*dnow.sum() + retention@dperm)
                magnitude = float(max(np.abs(dnow).sum(), np.abs(dperm).sum()))
                by_action[a] = (score,magnitude,cons)
            estimates.append(by_action)
        if any(default not in e for e in estimates):
            return prior,value
        # Require the conservative improvement in both reseeded simulations.
        best,best_margin=default,0.
        for a in opts:
            if any(a not in e or e[a][2]!=e[default][2] for e in estimates):
                continue
            margin=min(e[a][0]-e[default][0]-max(2., .20*max(e[a][1],e[default][1])) for e in estimates)
            if margin>best_margin:
                best,best_margin=a,margin
        if best==default:
            return prior,value
        self.counter['changed']+=1
        self.sources[int(c[4])]+=1
        out=np.zeros_like(prior)
        out[best]=1
        return out,value


def source(i):
    w=Durability('development',collect=True)
    r=evaluate_policy(w,EvalConfig(games=1,seed=88100000+i,mirror_pool=w.pool))
    check(r)
    rows=[]
    for t,f in w.frames.items():
        if t>12 or t+2 not in w.frames:
            continue
        later=set(w.frames[t+2]['ids'])
        rows.extend((x,int(e in later)) for e,x in zip(f['ids'],f['x']))
    return dict(i=i,rows=rows)


def fit():
    games=[r for s in range(8) for r in json.loads((ROOT/f'source-{s}'/'rows.json').read_text())]
    if [r['i'] for r in games]!=list(range(512)):
        raise ValueError('incomplete durability sources')
    heads=[]
    for arm in (0,1):
        rows=[v for r in games if r['i']%2==arm for v in r['rows']]
        x=np.array([v[0] for v in rows]); y=np.array([v[1] for v in rows])
        mean,scale=x.mean(0),np.maximum(x.std(0),.03)
        z=(x-mean)/scale
        weight=np.linalg.solve(z.T@z+150*np.eye(z.shape[1]),z.T@(y-y.mean()))
        heads.append(dict(weight=weight.tolist(),mean=mean.tolist(),scale=scale.tolist(),
                          intercept=float(y.mean()),n=len(y)))
    if (ROOT/'heads.json').exists():
        raise FileExistsError('heads already exist')
    atomic_json(ROOT/'heads.json',heads)


def panel(i):
    heads=json.loads((ROOT/'heads.json').read_text())
    rows=[]
    for h in (None,*heads):
        w=Durability('transfer',head=h)
        r=evaluate_policy(w,EvalConfig(games=1,seed=88300000+i,mirror_pool=w.pool))
        check(r)
        rows.append(dict(seed=88300000+i,rank=r['rank_sequence'][0],eot=w.eot,changes=w.counter['changed'],
                         eligible=w.counter['eligible'],sources=w.sources))
    return rows


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='fit':
        fit()
    elif stage=='assess':
        rows=[r for s in range(8) for case in json.loads((ROOT/f'panel-{s}'/'rows.json').read_text()) for r in case]
        report=paired_endpoints(rows,88300000,256)
        atomic_json(ROOT/'assessment.json',report)
    elif stage in ('source','panel'):
        s=int(sys.argv[2]); n=64 if stage=='source' else 32
        out=ROOT/f'{stage}-{s}';out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'run.json',provenance())
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            rows=list(pool.map(source if stage=='source' else panel,range(s*n,(s+1)*n)))
        atomic_json(out/'rows.json',rows)
