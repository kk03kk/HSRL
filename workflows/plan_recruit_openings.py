"""Learn and test broad recruit-opening choices with disjoint paired future streams."""
import gzip
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.assess_investment_timing import UTILITY
from tools.compare_research_panels import bootstrap
from tools.profile_lineup_use import read_cards
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.probe_investment_timing import TimingPolicy, MODEL, FIELDS
from workflows.replicate_investment_returns import GUARDS, Captured

ROOT = Path('artifacts/runs/nus-opening-plans-20261118-1')
CACHE = {}
ROLES = ('upgrade','refresh','tempo','invest')


def native(field):
    if field not in CACHE:
        import hsbg_engine as engine
        torch.set_num_threads(1)
        pool = MirrorSnapshotPool(FIELDS[field],sample='p50band')
        policy = make_eval_policy(MODEL,protocol=EvalProtocol(mirror_snaps=FIELDS[field]),device='cpu')['policy']
        cards = read_cards()
        info = {i:cards[engine.card_info(i+1)[0]] for i in range(policy.evaluator.model.n_cards-1)
                if engine.card_info(i+1)[0] in cards}
        CACHE[field] = policy,pool,info
    return CACHE[field]


def offers(p,mask,info):
    owned=[info[item_card(q)] for q in p['board']+p['hand']]
    board=[info[item_card(q)] for q in p['board']]
    result=[]
    for slot,q in enumerate(p['shop'][:8]):
        a=8+slot
        cid=item_card(q)
        c=info[cid]
        if not mask[a] or c['kind']!='minion':
            continue
        race=set(c['races'])-{'ALL'}
        support=sum(bool(race.intersection(m['races'])) for m in board)
        base=c.get('normal') or c['id']
        copies=sum((m.get('normal') or m['id'])==base for m in owned)
        unit=q['item']['Minion']
        stats=unit['atk']+unit['hp']
        kw=set(c['keywords'])
        tempo=np.log1p(max(0,stats))+0.55*('DIVINE_SHIELD' in kw)+0.5*('VENOMOUS' in kw)
        invest=1.25*min(2,copies)+0.4*min(5,support)+.3*c['tier']+.2*('MAGNETIC' in kw)
        result.append(dict(action=a,card=cid,support=support,copies=copies,tempo=float(tempo),invest=float(invest)))
    return result


def proposals(p,mask,info):
    choices=offers(p,mask,info)
    routes=[dict(role='parent',action=None,card=None)]
    if mask[3]:
        routes.append(dict(role='upgrade',action=3,card=None))
    elif mask[1]:
        routes.append(dict(role='refresh',action=1,card=None))
    taken=set()
    for role in ('tempo','invest'):
        eligible=[q for q in choices if q['card'] not in taken]
        if not eligible:
            break
        best=max(eligible,key=lambda q:q[role])
        taken.add(best['card'])
        routes.append(dict(role=role,**best))
    if len(routes)<4 and mask[1] and not any(r['action']==1 for r in routes):
        routes.append(dict(role='refresh',action=1,card=None))
    return routes


def route_features(p,route):
    strength=sum(m['atk']+m['hp'] for m in p['board'])
    state=[p['health']+p['armor'],p['tier'],p['gold'],p['upgrade_cost'],len(p['board']),len(p['hand']),np.log1p(strength)]
    role=np.array([float(route['role']==k) for k in ROLES])
    return [*role,*state,*(role[:,None]*np.array(state)).ravel(),
            *(route.get(k,0) for k in ('tempo','invest','support','copies'))]


class Opening(TimingPolicy):
    def __init__(self,field,source=None,route=0,first_only=False):
        policy,self.pool,self.info=native(field)
        super().__init__(policy,'parent',False)
        self.source,self.route,self.first_only=source,route,first_only

    def reset(self):
        super().reset()
        self.alterations=[]
        self.executed=[]
        self.focus=None
        self.until=-1
        self.diverged={}
        if self.source:
            self.until=self.source['turn']+1
            for k,v in self.source['guards'].items():
                setattr(self.policy,k,deepcopy(v))

    def decide(self,game,seat):
        dec=super().decide(game,seat)
        if self.source and self.current<=self.until:
            self.executed.append([self.current,int(dec.action)])
        return dec

    def __call__(self,f,c,mask,seat):
        prior,value=self.original(f,c,mask,seat)
        if not self.source or not self.route or self.current>self.until:
            return prior,value
        spec=self.source['routes'][self.route]
        a=int(np.argmax(prior))
        if not self.actions:
            b=spec['action']
            if not mask[b] or (spec['card'] is not None and item_card(self.player['shop'][b-8])!=spec['card']):
                raise ValueError('sealed purchase or investment is unavailable')
            self.focus=spec['card']
        elif self.first_only or not mask[0] or self.diverged.get(self.current,0)>=3:
            return prior,value
        else:
            p=self.player
            b=a
            # Buy routes can make space and deploy the acquired unit. A triple or a
            # transformed card no longer matches and automatically returns control.
            play=[]
            if self.focus is not None:
                for i,q in enumerate(p['hand'][:10]):
                    if item_card(q)==self.focus:
                        play.extend(j for j in range(46+8*i,54+8*i) if mask[j])
                        play.extend(j for j in range(126+7*i,133+7*i) if mask[j])
                in_hand=any(item_card(q)==self.focus for q in p['hand'])
                if play:
                    b=max(play,key=lambda j:prior[j])
                elif in_hand and len(p['board'])==7:
                    sales=[16+i for i,m in enumerate(p['board']) if mask[16+i] and m['card']!=self.focus]
                    if sales:
                        b=max(sales,key=lambda j:float(np.log(max(prior[j],1e-16)))-.3*np.log1p(
                            p['board'][j-16]['atk']+p['board'][j-16]['hp']))
                elif spec['role']=='invest' and self.current==self.source['turn'] and 16<=a<23 and p['board'][a-16]['card']==self.focus:
                    legal=np.flatnonzero(mask)
                    b=max((j for j in legal if j!=a),key=lambda j:prior[j],default=a)
            # After investing in access, use a newly seen shop when the parent
            # would roll or pass. Re-read offers before committing to the unit.
            elif spec['role'] in ('upgrade','refresh') and a in (0,1) and p['gold']>=3:
                avail=offers(p,mask,self.info)
                if avail:
                    pick=max(avail,key=lambda q:q['tempo']+.25*q['invest'])
                    b,self.focus=pick['action'],pick['card']
        if a==b:
            return prior,value
        self.diverged[self.current]=self.diverged.get(self.current,0)+1
        self.alterations.append([self.current,len(self.actions),a,int(b)])
        result=np.zeros_like(prior)
        result[b]=1
        return result,value


def capture(i):
    from hsbg.az.train.common import mask_from_bytes
    field='development' if i%2==0 else 'transfer'
    turn=7+(i//2)%5
    w=Opening(field)
    def before(game,seat):
        if int(game.turn)!=turn or not game.needs_action(seat):
            return
        p=json.loads(game.player_json(seat))
        mask=mask_from_bytes(game.legal_mask(seat))
        if p['pending'] or not mask[0] or p['gold']<3:
            return
        routes=proposals(p,mask,w.info)
        if len(routes)<3:
            return
        require_clean_engine(game)
        cnt=Counter(t for m in p['board'] for t in w.info[m['card']]['races'] if t!='ALL')
        top=cnt.most_common(2)
        family=top[0][0] if top and top[0][1]>=3 and (len(top)==1 or top[0][1]>top[1][1]) else 'MIXED'
        w.captured=dict(state=json.loads(game.state_json()),guards={k:deepcopy(getattr(w.policy,k)) for k in GUARDS},
                        field=field,turn=turn,family=family,hero=p['hero'],prefix=w.actions[:],routes=routes,
                        features=[route_features(p,r) for r in routes])
        raise Captured()
    for offset in range(120):
        seed=82100000+i*200+offset
        try:
            check(evaluate_policy(w,EvalConfig(games=1,seed=seed,mirror_pool=w.pool,research_before_opponents=before)))
        except Captured:
            break
    else:
        raise ValueError('no eligible opening in assigned seed block')
    src=dict(**w.captured,i=i,seed=seed,screened=offset+1)
    path=ROOT/f'{i}-state.pkl.gz'
    with gzip.open(path,'xb') as stream:
        pickle.dump(src,stream)
    return dict(i=i,seed=seed,turn=turn,field=field,routes=src['routes'],family=src['family'],
                screened=offset+1,file=file_identity(path))


def restore(i):
    s=json.loads((ROOT/'sources.json').read_text())[i]
    path=ROOT/f'{i}-state.pkl.gz'
    if file_identity(path)!=s['file']:
        raise ValueError('opening changed')
    with gzip.open(path,'rb') as f:
        return pickle.load(f)


def rollout(args):
    import hsbg_engine as engine
    stage,i,rep=args
    src=restore(i)
    future=82300000+i*32+rep+(8 if stage=='test' else 0)
    fresh=json.loads(engine.Game(seed=future).state_json())
    state=dict(src['state'],seed=fresh['seed'],rng=fresh['rng'])
    if stage=='search':
        specs=[(j,False) for j in range(len(src['routes']))]
    else:
        pick=json.loads((ROOT/'choices.json').read_text())['choices'][i]
        # Deduplicate equivalent interventions. No unused simulations.
        specs=list(dict.fromkeys([(0,False),(pick['search'],False),(pick['student'],False)]+
                                 ([(pick['search'],True)] if pick['search'] else [])))
    rows=[]
    path=ROOT/f'{stage}-{i}-{rep}.jsonl'
    with path.open('x') as out:
        for route,first_only in specs:
            w=Opening(src['field'],src,route,first_only)
            r=evaluate_policy(w,EvalConfig(games=1,seed=future,mirror_pool=w.pool,research_initial_state=state,
                                           research_restore_opening=True))
            check(r)
            if route and w.actions[0]!=src['routes'][route]['action']:
                raise ValueError('assigned opening not executed')
            row=dict(i=i,future=future,route=route,first=first_only,rank=r['rank_sequence'][0],eot=w.eot,
                     changed=w.alterations,actions=w.executed)
            out.write(json.dumps(row)+'\n')
            out.flush()
            rows.append(row)
    return len(rows)


def read_rows(stage,i,rep):
    path=ROOT/f'{stage}-{i}-{rep}.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if any(r['i']!=i or r['future']!=82300000+i*32+rep+8*(stage=='test') for r in rows):
        raise ValueError('incorrect source/future')
    return {(r['route'],r['first']):r for r in rows}


def fit():
    sources=[restore(i) for i in range(64)]
    targets=[]
    for i,s in enumerate(sources):
        sets=[read_rows('search',i,rep) for rep in range(8)]
        payoff=np.array([[UTILITY[row[j,False]['rank']-1] for j in range(len(s['routes']))] for row in sets])
        targets.append((payoff-payoff[:,[0]]).mean(0))
    choices=[]
    for i,s in enumerate(sources):
        # Hold out complete source games. Folds mix both libraries and all turn windows.
        train=[j for j in range(64) if j//2%4!=i//2%4]
        x=np.vstack([sources[j]['features'][1:] for j in train])
        y=np.concatenate([targets[j][1:] for j in train])
        center,scale=x.mean(0),x.std(0)
        scale[scale<1e-6]=1
        z=(x-center)/scale
        coef=np.linalg.solve(z.T@z+50*np.eye(z.shape[1]),z.T@(y-y.mean()))
        prediction=np.r_[0,y.mean()+((np.asarray(s['features'][1:])-center)/scale)@coef]
        choices.append(dict(i=i,search=int(np.argmax(targets[i])),student=int(np.argmax(prediction)),
                            selection=targets[i].tolist(),prediction=prediction.tolist()))
    atomic_json(ROOT/'choices.json',dict(choices=choices,phase='frozen before independent test futures'))


def assess():
    choices=json.loads((ROOT/'choices.json').read_text())['choices']
    data=[]
    total=0
    select_total=0
    for i,pick in enumerate(choices):
        s=restore(i)
        sel=pick['search']
        st=pick['student']
        metrics=[]
        for rep in range(8):
            a=read_rows('test',i,rep)
            select_total+=len(read_rows('search',i,rep))
            total+=len(a)
            base,full,first,student=a[0,False],a[sel,False],a[sel,bool(sel)],a[st,False]
            def u(r):
                return UTILITY[r['rank']-1]
            metrics.append([u(full)-u(base),u(first)-u(base),u(full)-u(first),u(student)-u(base),
                            full['rank']-base['rank'],int(full['actions'][0]!=base['actions'][0]),
                            int(any(c[1]>0 for c in full['changed'])),int(any(c[0]>s['turn'] for c in full['changed']))])
        data.append(dict(i=i,family=s['family'],turn=s['turn'],field=s['field'],search=s['routes'][sel]['role'],
                         student=s['routes'][st]['role'],scores=np.mean(metrics,axis=0).tolist()))
    names=('search','first','follow','student','rank','different_first','later_change','next_turn_change')
    def group(rows):
        x=np.array([r['scores'] for r in rows])
        return {k:bootstrap(x[:,j]) for j,k in enumerate(names)}
    result=group(data)
    gate=result['search']['ci95'][0]>0
    if select_total+total>4096:
        raise ValueError('budget exceeded')
    atomic_json(ROOT/'result.json',dict(advance=gate,overall=result,search_suffixes=select_total,test_suffixes=total,
                by_field={k:group([r for r in data if r['field']==k]) for k in FIELDS},sources=data,
                plan=file_identity(Path('config/research/recruit-planning-20261118.json')),
                choices=file_identity(ROOT/'choices.json'),manifest=file_identity(ROOT/'sources.json')))


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='capture':
        ROOT.mkdir(parents=True,exist_ok=False)
        atomic_json(ROOT/'run.json',provenance())
    if stage in ('fit','assess'):
        fit() if stage=='fit' else assess()
    else:
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            if stage=='capture':
                atomic_json(ROOT/'sources.json',list(pool.map(capture,range(64))))
            elif stage in ('search','test'):
                shard=int(sys.argv[2])
                if not 0<=shard<8:
                    raise ValueError('shard')
                list(pool.map(rollout,[(stage,i,r) for i in range(8*shard,8*shard+8) for r in range(8)]))
            else:
                raise ValueError(stage)
