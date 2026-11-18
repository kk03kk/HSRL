"""Look for executable gains before discarding a shop, using public candidates and two reseeded futures."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.common import mask_from_bytes,obs_arrays
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import atomic_json
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy


ROOT=Path('artifacts/runs/nus-public-recruit-trades-20261118-1')


def score(before,after,mode):
    if after['pending'] or any(after[k]<before[k] for k in
            ('health','armor','tier','free_refreshes','gold_next_turn','max_gold_bonus')):
        return None
    hands={h['hid']:h for h in after['hand']}
    if any(hands.get(h['hid'])!=h for h in before['hand']):
        return None
    ids={m['eid'] for m in before['board']}
    original=[m for m in after['board'] if m['eid'] in ids]
    if [m['eid'] for m in original]!=[m['eid'] for m in before['board']]:
        return None
    for b,a in zip(before['board'],original):
        if any(b[k]!=a[k] for k in ('card','golden','kw','bonus_kw','attached')) or a['shield_hits']<b['shield_hits']:
            return None
        for stat,temp in (('atk','temp_atk'),('hp','temp_hp')):
            if a[stat]<b[stat] or a[stat]-a[temp]<b[stat]-b[temp]:
                return None
    board=lambda p:sum(m['atk']+m['hp']-m['temp_atk']-m['temp_hp'] for m in p['board'])
    gain=board(after)-board(before)
    delta=after['gold']-before['gold']
    if mode=='refresh' and (delta<0 or len(before['board'])!=len(after['board'])):
        return None
    if (delta>=1 or gain>=max(8,.05*board(before))):
        return dict(coin=delta,power=gain,extra_hand=len(after['hand'])-len(before['hand']))
    return None


def advance(g,seat,action,t):
    if not g.needs_action(seat) or not g.legal_mask(seat)[action]:
        return False
    g.step(seat,int(action))
    require_clean_engine(g)
    return not g.is_over() and g.turn==t and g.needs_action(seat)


def pick(g,seat,allowed,actor):
    mask=mask_from_bytes(g.legal_mask(seat)).copy()
    local=np.zeros_like(mask)
    local[allowed]=True
    local &= mask
    if not local.any():
        return None
    if local.sum()==1:
        return int(np.flatnonzero(local)[0])
    f,c=obs_arrays(g.observe(seat))
    prior,_=actor(f,c,mask,seat)
    return int(np.where(local,prior,-1.).argmax())


def choose_pending(g,seat,t,actor):
    for _ in range(5):
        p=json.loads(g.player_json(seat))
        if not p['pending']:
            return p
        a=pick(g,seat,np.arange(30,46),actor)
        if a is None or not advance(g,seat,a,t):
            return None
    return None


@lru_cache(maxsize=2)
def stream(rep):
    import hsbg_engine as engine
    st=json.loads(engine.Game(seed=91281000+rep).state_json())
    return st['seed'],st['rng']


def after_buy(serialized,seat,action,before,mode,actor):
    import hsbg_engine as engine
    g=engine.Game.from_json(serialized);t=g.turn
    if not advance(g,seat,action,t):
        return {}
    p=choose_pending(g,seat,t,actor)
    if p is None:
        return {}
    old={h['hid'] for h in before['hand']}
    acquired=[i for i,h in enumerate(p['hand']) if h['hid'] not in old]
    if len(acquired)!=1 or len(p['hand'])!=len(old)+1:
        return {}
    h=acquired[0]
    a=pick(g,seat,np.r_[46+h*8:54+h*8,126+h*7:133+h*7],actor)
    if a is None or not advance(g,seat,a,t):
        return {}
    p=choose_pending(g,seat,t,actor)
    if p is None:
        return {}
    result={}
    v=score(before,p,mode)
    if v:
        result['keep']=v
    old_board={m['eid'] for m in before['board']}
    new=[i for i,m in enumerate(p['board']) if m['eid'] not in old_board]
    if len(new)==1 and advance(g,seat,16+new[0],t):
        p=choose_pending(g,seat,t,actor)
        v=score(before,p,mode) if p is not None else None
        if v:
            result['sell']=v
    return result


class Shadow(TimingPolicy):
    def __init__(self,field):
        base,self.pool,_=native(field)
        super().__init__(base,'parent',False)

    def reset(self):
        super().reset()
        self.seen=set();self.gains=[];self.offers=0

    def __call__(self,f,c,mask,seat):
        prior,value=self.original(f,c,mask,seat)
        default=int(prior.argmax());key=(self.current,default)
        if (default not in (0,1) or key in self.seen or self.player['pending']
                or self.player['actions_this_turn']>85):
            return prior,value
        self.seen.add(key)
        p=self.player;mode='end' if default==0 else 'refresh'
        options=[8+i for i,s in enumerate(p['shop']) if mask[8+i] and not s['frozen']]
        if not options:
            return prior,value
        state=json.loads(self.game.state_json())
        futures=[json.dumps(dict(state,seed=stream(rep)[0],rng=stream(rep)[1]),separators=(',',':')) for rep in (0,1)]
        for a in options:
            self.offers+=1
            first=after_buy(futures[0],seat,a,p,mode,self.original)
            if not first:
                continue
            second=after_buy(futures[1],seat,a,p,mode,self.original)
            for route in sorted(first.keys() & second.keys()):
                item=next(iter(p['shop'][a-8]['item'].values()))
                self.gains.append(dict(turn=self.current,mode=mode,card=item['card'],route=route,
                         coin=min(first[route]['coin'],second[route]['coin']),
                         power=min(first[route]['power'],second[route]['power'])))
        return prior,value


def game(i):
    w=Shadow('development')
    r=evaluate_policy(w,EvalConfig(games=1,seed=91200000+i,mirror_pool=w.pool))
    check(r)
    return dict(i=i,rank=r['rank_sequence'][0],opportunities=len(w.seen),offers=w.offers,gains=w.gains)


if __name__=='__main__':
    if sys.argv[1]=='source':
        s=int(sys.argv[2]);d=ROOT/f'source-{s:02}';d.mkdir(parents=True,exist_ok=False)
        atomic_json(d/'run.json',provenance())
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            rows=list(pool.map(game,range(16*s,16*(s+1))))
        atomic_json(d/'rows.json',rows)
    elif sys.argv[1]=='assess':
        rows=[r for s in range(8) for r in json.loads((ROOT/f'source-{s:02}'/'rows.json').read_text())]
        if [r['i'] for r in rows]!=list(range(128)):
            raise ValueError('incomplete natural screen')
        result={}
        for name in ('cash','closing'):
            def qualifies(g):
                return g['coin']>=0 if name=='cash' else (g['mode']=='end' and g['coin']<0)
            gains=[a for r in rows for a in r['gains'] if qualifies(a)]
            cards=Counter(a['card'] for a in gains)
            ng=sum(any(qualifies(a) for a in r['gains']) for r in rows)
            result[name]=dict(games=ng,routes=len(gains),cards=cards,
                              passes=ng>=20 and len(cards)>=3)
        atomic_json(ROOT/'assessment.json',dict(games=len(rows),offers=sum(r['offers'] for r in rows),classes=result))
    else:
        raise ValueError('unknown stage')
