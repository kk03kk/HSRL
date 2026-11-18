"""Seal natural trade disagreements and compare parent, first action, and a two-turn plan."""
import gzip
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
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
from workflows.d132_trade_plan import FIT, Representation, TradeHead, provenance
from workflows.human_trade_options import MODEL, FIELDS
from workflows.replicate_investment_returns import GUARDS, Captured

ROOT=Path('artifacts/runs/nus-d132-plans-20261118-1')
CACHE={}


def native(field):
    if field not in CACHE:
        torch.set_num_threads(1)
        proto=EvalProtocol(mirror_snaps=FIELDS[field])
        policy=make_eval_policy(MODEL,protocol=proto,device='cpu')['policy']
        cards=Representation(policy)
        fit=json.loads((FIT/'complete.json').read_text())
        path=FIT/'history.pt'
        if fit['history']!=file_identity(path):
            raise ValueError('proposal checkpoint mismatch')
        ck=torch.load(path,map_location='cpu',weights_only=True)
        head=TradeHead(ck['width'])
        head.load_state_dict(ck['state'])
        CACHE[field]=(policy,MirrorSnapshotPool(FIELDS[field],sample='p50band'),cards,head.eval())
    return CACHE[field]


class Plan:
    name='d132-trade-plan'

    def __init__(self,field,window,mode,source=None):
        self.policy,self.pool,self.cards,self.head=native(field)
        self.original=self.policy.evaluator
        self.model=self.original.model
        self.window,self.mode,self.source=window,mode,source

    def reset(self):
        self.policy.reset()
        self.history=defaultdict(list)
        self.pending=None
        self.actions=[]
        self.eot={}
        self.changes=[]
        self.event=None
        if self.source is not None:
            self.history=deepcopy(self.source['history'])
            for k,v in self.source['guards'].items():
                setattr(self.policy,k,deepcopy(v))

    def cid(self,item):
        return self.cards.reverse[item_card(item)+1]

    def decide(self,game,seat):
        self.turn=int(game.turn)
        self.game=game
        active=6<=self.turn<=13
        if active:
            self.player=json.loads(game.player_json(seat))
            if self.pending is not None:
                turn,kind,k,before=self.pending
                delta=self.player['gold']-before
                if self.turn==turn and delta==(1 if kind=='sell' else -3):
                    self.history[turn,kind].append(self.cards.base[k])
                self.pending=None
            if self.mode=='capture' and self.window[0]<=self.turn<=self.window[1]:
                self.guards={k:deepcopy(getattr(self.policy,k)) for k in GUARDS}
        self.policy.evaluator=self if active else self.original
        try:
            d=self.policy.decide(game,seat)
        finally:
            self.policy.evaluator=self.original
        a=int(d.action)
        if active:
            p=self.player
            if 46<=a<126:
                k=self.cid(p['hand'][(a-46)//8])
                if self.cards.db[k]['kind']=='minion':
                    self.history[self.turn,'play_minion'].append(self.cards.base[k])
            if 8<=a<16 and a-8<len(p['shop']):
                k=self.cid(p['shop'][a-8])
                if self.cards.db[k]['kind']=='minion':
                    self.pending=(self.turn,'buy_minion',k,p['gold'])
            if 16<=a<23 and len(p['board'])==7:
                self.pending=(self.turn,'sell',self.cid(p['board'][a-16]),p['gold'])
        self.actions.append(a)
        if a==0 and self.turn>=1:
            p=self.player if active else json.loads(game.player_json(seat))
            self.eot[self.turn]=[sum(m['atk']+m['hp'] for m in p['board']),p['health'],p['tier']]
        return d

    def __call__(self,f,c,mask,seat):
        prior,value=self.original(f,c,mask,seat)
        if not self.window[0]<=self.turn<=self.window[1]:
            return prior,value
        if self.policy._turn_counts.get((seat,self.turn),0)>=99:
            return prior,value
        first=self.source is not None and not self.actions
        if self.mode in ('parent','first') and not first:
            return prior,value
        a=int(np.argmax(prior))
        sell=16<=a<23
        p=self.player
        if (not sell and not 8<=a<16) or (sell and len(p['board'])!=7):
            return prior,value
        offset,items=(16,p['board']) if sell else (8,p['shop'])
        slots=[(offset+i,self.cid(it)) for i,it in enumerate(items[:7 if sell else 8])
               if mask[offset+i] and self.cards.db[self.cid(it)]['kind']=='minion']
        if a not in dict(slots) or len({self.cards.base[k] for _,k in slots})<2:
            return prior,value
        board=[self.cid(q) for q in p['board']]
        hand=[self.cid(q) for q in p['hand']]
        x=[self.cards.features(k,board,hand,self.turn,p['gold'],sell,self.history) for _,k in slots]
        with torch.inference_mode():
            raw,_=self.head(torch.tensor(x,dtype=torch.float32),float(sell))
        logits=raw.numpy()
        adj=np.clip(logits-logits.mean(),-1.5,1.5)
        score=np.array([np.log(max(float(prior[i]),1e-30)) for i,_ in slots])+adj
        chosen=int(np.argmax(score))
        b,k=slots[chosen]
        if first and [a,b]!=self.source['event']['choices']:
            raise ValueError('paired source action or plan changed')
        if a==b:
            return prior,value
        if self.mode=='capture':
            require_clean_engine(self.game)
            strength=sum(m['atk']+m['hp'] for m in p['board'])
            sa=next(i for i,row in enumerate(slots) if row[0]==a)
            tribes=Counter(t for q in board for t in self.cards.tribes.get(q,()))
            top=tribes.most_common(2)
            label=top[0][0] if top and top[0][1]>=3 and (len(top)==1 or top[0][1]>top[1][1]) else 'MIXED'
            share=0.
            if sell:
                lhs,rhs=items[a-offset],items[b-offset]
                share=(rhs['atk']+rhs['hp']-lhs['atk']-lhs['hp'])/max(strength,1)
            support=sum(bool(self.cards.tribes[k]&self.cards.tribes.get(q,set())) for q in board)-int(sell and bool(self.cards.tribes[k]))
            hp=p['health']+p['armor']
            self.event=dict(choices=[a,b],turn=self.turn,kind='sell' if sell else 'buy',board_family=label,
                            health=hp,support=support,hero=p['hero'],cards=[slots[sa][1],k],
                            features=[self.turn,hp,p['gold'],p['tier'],len(hand),len(board),np.log1p(strength),
                                      float(sell),share,support,float(adj[chosen]-adj[sa]),
                                      float(np.log(max(prior[a],1e-30)/max(prior[b],1e-30)))])
            self.initial=json.loads(self.game.state_json())
            raise Captured()
        if self.mode=='parent':
            return prior,value
        self.changes.append([self.turn,len(self.actions),a,b])
        changed=np.zeros_like(prior)
        changed[b]=1
        return changed,value


def check(r):
    if any(r[k] for k in ('illegal_actions','truncated_games','truncated_combats')) or (
            r['mirror']['mirror_errors'] or r['mirror']['mirror_rejected']):
        raise ValueError('invalid game')


def capture(i):
    field='development' if i%2==0 else 'transfer'
    start=7+2*(i//2%3)
    for j in range(256):
        seed=81100000+i*300+j
        w=Plan(field,(start,start+1),'capture')
        try:
            r=evaluate_policy(w,EvalConfig(games=1,seed=seed,mirror_pool=w.pool))
            check(r)
        except Captured:
            break
    else:
        raise ValueError('source block exhausted')
    raw=dict(i=i,field=field,seed=seed,screened=j+1,state=w.initial,event=w.event,
             history=w.history,guards=w.guards,prefix=w.actions)
    path=ROOT/f'{i:03}-source.pkl.gz'
    with gzip.open(path,'xb') as f:
        pickle.dump(raw,f,protocol=5)
    return dict(i=i,field=field,seed=seed,screened=j+1,event=w.event,file=file_identity(path))


def suffix(i):
    import hsbg_engine as engine
    manifest=json.loads((ROOT/'sources.json').read_text())['sources'][i]
    path=ROOT/f'{i:03}-source.pkl.gz'
    if file_identity(path)!=manifest['file']:
        raise ValueError('source changed')
    with gzip.open(path,'rb') as f:
        src=pickle.load(f)
    t=src['event']['turn']
    rows=[]
    for rep in range(8):
        future=81300000+8*i+rep
        rng=json.loads(engine.Game(seed=future).state_json())
        state=dict(src['state'],seed=rng['seed'],rng=rng['rng'])
        arms=[]
        for mode in ('parent','first','plan'):
            w=Plan(src['field'],(t,t+1),mode,src)
            r=evaluate_policy(w,EvalConfig(games=1,seed=future,mirror_pool=w.pool,research_initial_state=state))
            check(r)
            if w.actions[0]!=src['event']['choices'][int(mode!='parent')]:
                raise ValueError('first action differs')
            arms.append(dict(rank=r['rank_sequence'][0],actions=w.actions,eot=w.eot,changes=w.changes))
        if len(arms[2]['changes'])==1 and (arms[1]['actions'],arms[1]['rank'])!=(arms[2]['actions'],arms[2]['rank']):
            raise ValueError('identical interventions diverged')
        rows.append(arms)
    atomic_json(ROOT/f'{i:03}-outcomes.json',dict(i=i,futures=rows))


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='capture':
        ROOT.mkdir(parents=True,exist_ok=False)
        atomic_json(ROOT/'run.json',dict(**provenance(),fit=file_identity(FIT/'complete.json')))
    workers=int(os.environ.get('SLURM_CPUS_PER_TASK',4))
    with ProcessPoolExecutor(workers) as pool:
        if stage=='capture':
            sources=list(pool.map(capture,range(256)))
            atomic_json(ROOT/'sources.json',dict(sources=sources))
        elif stage=='suffix':
            shard=int(sys.argv[2])
            if not 0<=shard<8:
                raise ValueError('invalid shard')
            list(pool.map(suffix,range(shard*32,(shard+1)*32)))
        else:
            raise ValueError(stage)
