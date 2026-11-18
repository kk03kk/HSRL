"""Measure natural resource deployment from receipt and from one earlier recruit opening."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy
from workflows.replicate_investment_returns import GUARDS

ROOT=Path('artifacts/runs/nus-natural-resource-frontiers-20261118-1')


def played(p,action):
    if 46<=action<126:
        return p['hand'][(action-46)//8]
    if 126<=action<196:
        return p['hand'][(action-126)//7]
    return None


class Tracker:
    def __init__(self,policy,mode,source=None,restore=None):
        self.policy=policy
        self.mode=mode
        self.source=source
        self.restore=restore

    def begin_game(self,game,gi,seed,seat):
        self.turn=-1
        self.prior_open=None
        self.opening=None
        self.pending=None
        self.event=None if self.source is None else self.source['event']
        self.receipt=None
        self.live={}
        self.hit=False
        self.same_pair=False
        self.generated=False
        if self.mode=='receipt':
            e=self.event
            self.live[e['output_hid']]=(e['generator'],e['output'])
            self.generated=True

    def snapshot(self,game):
        require_clean_engine(game)
        return dict(state=json.loads(game.state_json()),guards={k:deepcopy(getattr(self.policy,k)) for k in GUARDS})

    def before(self,game,seat):
        t=int(game.turn)
        if self.mode!='natural' and t>self.event['turn']+1:
            return None
        if self.mode=='natural' and not 6<=t<=12:
            return None
        p=json.loads(game.player_json(seat))
        if self.mode=='natural' and t!=self.turn and self.receipt is None:
            self.prior_open,self.opening=self.opening,self.snapshot(game)
        self.turn=t
        if self.pending and self.receipt is None and self.mode=='natural':
            if t==self.pending['turn'] and any(q['hid']==self.pending['output_hid'] for q in p['hand']):
                self.receipt=self.snapshot(game)
                self.event=self.pending
                self.earlier=self.prior_open
            self.pending=None
        return (t,p)

    def after(self,game,before,dec,action,forced):
        if before is None:
            return
        t,p=before
        if self.mode=='natural' and self.event and t>self.event['turn']+1:
            return
        step=played(p,action)
        if step is not None and step['hid'] in self.live:
            pair=self.live.pop(step['hid'])
            if self.event:
                if self.mode=='receipt' or (self.mode=='natural' and step['hid']==self.event['output_hid']):
                    self.hit=True
                if self.mode=='earlier':
                    self.hit=True
                if pair==(self.event['generator'],self.event['output']):
                    self.same_pair=True
        if not (7<=t<=11 or self.mode=='earlier') or (step is None and action!=4):
            return
        if self.mode=='receipt' or self.mode=='natural' and (self.pending or self.event):
            return
        # Match the original evaluation horizon: the earlier-start path must
        # generate during the target turn, then may deploy until the next EOT.
        if self.mode=='earlier' and t!=self.event['turn']:
            return
        if int(game.turn)!=t:
            return
        post=json.loads(game.player_json(0))
        old={q['hid'] for q in p['hand']}
        created=[q for q in post['hand'] if q['hid'] not in old]
        if not created:
            return
        generator=item_card(step) if step is not None else -1
        for q in created:
            self.live[q['hid']]=(generator,item_card(q))
        self.generated=True
        if self.mode=='natural' and self.prior_open is not None:
            q=created[0]
            self.pending=dict(turn=t,generator=generator,output=item_card(q),output_hid=q['hid'],
                              multi=len(created),health=p['health']+p['armor'])

    def end_game(self,*args):
        pass


class Parent(TimingPolicy):
    def __init__(self,policy,start=None):
        super().__init__(policy,'parent',False)
        self.start=start

    def reset(self):
        super().reset()
        if self.start:
            for k,v in self.start['guards'].items():
                setattr(self.policy,k,deepcopy(v))

    def __call__(self,f,c,mask,seat):
        return self.original(f,c,mask,seat)


def produce(i):
    field='development' if i%2==0 else 'transfer'
    base,pool,info=native(field)
    seed=84100000+i
    w=Parent(base)
    rec=Tracker(base,'natural')
    r=evaluate_policy(w,EvalConfig(games=1,seed=seed,mirror_pool=pool,decision_recorder=rec))
    check(r)
    row=dict(i=i,field=field,game=seed,rank=r['rank_sequence'][0],found=rec.receipt is not None)
    if rec.receipt:
        e=rec.event
        source=dict(i=i,field=field,event=e,receipt=rec.receipt,earlier=rec.earlier)
        path=ROOT/f'{i}-source.pkl.gz'
        with gzip.open(path,'xb') as f:
            pickle.dump(source,f)
        row.update(file=file_identity(path),event=e,used=rec.hit,output_kind=info[e['output']]['kind'],
                   generator_kind='hero_power' if e['generator']==-1 else info[e['generator']]['kind'])
    return row


def probe(args):
    import hsbg_engine as engine
    row,rep=args
    i=row['i']
    path=ROOT/f'{i}-source.pkl.gz'
    if row['file']!=file_identity(path):
        raise ValueError('resource source changed')
    with gzip.open(path,'rb') as f:
        src=pickle.load(f)
    base,pool,_=native(row['field'])
    seed=84300000+8*i+rep
    fresh=json.loads(engine.Game(seed=seed).state_json())
    data={}
    for mode in ('receipt','earlier'):
        start=src[mode]
        state=dict(start['state'],seed=fresh['seed'],rng=fresh['rng'])
        w=Parent(base,start)
        rec=Tracker(base,mode,src)
        r=evaluate_policy(w,EvalConfig(games=1,seed=seed,mirror_pool=pool,research_initial_state=state,
                                       decision_recorder=rec))
        check(r)
        data[mode]=dict(generated=rec.generated,used=rec.hit,same=rec.same_pair)
    return dict(i=i,rep=rep,field=row['field'],**data)


if __name__=='__main__':
    stage=sys.argv[1]
    workers=int(os.environ.get('SLURM_CPUS_PER_TASK',4))
    if stage=='collect':
        ROOT.mkdir(parents=True,exist_ok=False)
        atomic_json(ROOT/'run.json',dict(**provenance(),rule='160 natural games, first creation by proactive play or hero power at T7-11; identity-tracked deployment by end of following turn. Keep unused examples. First 24 eligible per field, four new future streams each from receipt and preceding opening. No game outcome filtering.'))
        with ProcessPoolExecutor(workers) as pool:
            rows=list(pool.map(produce,range(160)))
        atomic_json(ROOT/'manifest.json',rows)
    elif stage=='probe':
        rows=json.loads((ROOT/'manifest.json').read_text())
        sample=[r for field in ('development','transfer') for r in [x for x in rows if x['field']==field and x['found']][:24]]
        with ProcessPoolExecutor(workers) as pool:
            data=list(pool.map(probe,[(r,i) for r in sample for i in range(4)]))
        groups={}
        for field in ('development','transfer'):
            part=[r for r in rows if r['field']==field and r['found']]
            probes=[r for r in data if r['field']==field]
            groups[field]=dict(games=80,found=len(part),used=sum(r['used'] for r in part),
                generators=len(set(r['event']['generator'] for r in part)),outputs=len(set(r['event']['output'] for r in part)),
                generator_kinds=dict(Counter(r['generator_kind'] for r in part)),probes=len(probes),
                receipt_used=sum(r['receipt']['used'] for r in probes),earlier_generated=sum(r['earlier']['generated'] for r in probes),
                earlier_any_used=sum(r['earlier']['used'] for r in probes),earlier_same_used=sum(r['earlier']['same'] for r in probes))
        atomic_json(ROOT/'complete.json',dict(groups=groups,manifest=file_identity(ROOT/'manifest.json'),probes=data))
