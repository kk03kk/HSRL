"""Distill a native target-consequence teacher using only public card and entity features."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import file_identity,atomic_json
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.learn_recruit_residual import paired_endpoints
from workflows.learn_retained_targets import Durability, ROOT as TEACHERS, unit_features
from workflows.plan_recruit_openings import native


ROOT=Path('artifacts/runs/nus-target-distillation-20261118-1')
TERMS=('double','give','health','attack','until','permanent','blood gem','each','set ','steal','consume',
       'golden','gain','highest','lowest','other','copy','triple','start of','end of','bonus','random')


@lru_cache(maxsize=1)
def heads():
    p=TEACHERS/'heads.json'
    if file_identity(p)['sha256'] != '9c0a8487719673b08c0defd39daf9ce7170f5f35085311d7c7116a767c1e5b2b':
        raise ValueError('durability teacher changed')
    return json.loads(p.read_text())


@lru_cache(maxsize=512)
def source_features(field,source):
    card=native(field)[2].get(source-1,{})
    desc=str(card.get('text') or '').lower()
    signed=np.zeros(48)
    for word in re.findall(r'[a-z]+|\d+',desc):
        h=hashlib.blake2s(word.encode(),digest_size=2).digest()
        signed[h[0]%48]+=(1 if h[1]%2 else -1)
    signed/=max(1,np.linalg.norm(signed))
    numbers=[int(x) for x in re.findall(r'\d+',desc)][:3]
    return np.r_[signed,[term in desc for term in TERMS],np.log1p(numbers+[0]*(3-len(numbers)))]


def target_rows(policy,source,prior,opts,base,retention):
    p=policy.player
    board=p['board']
    by_id={m['eid']:i for i,m in enumerate(board)}
    ids=p['pending'][-1]['Target']['options']
    info=source_features(policy.field,source)
    def features(a):
        i=by_id[ids[a-30]]
        m=board[i]
        return np.r_[retention[i],np.log1p(max(0,m['atk'])),np.log1p(max(0,m['hp'])),m['tier']/6,
                     int(m['golden']),min(3,m['shield_hits']),len(m['attached']),i/6,
                     np.log(max(1e-7,prior[a]))/6]
    base_x=features(base)
    rows=[]
    for a in opts:
        x=features(a);diff=x-base_x
        rows.append(np.r_[info,x,base_x,diff,info[-25:]*diff[0],policy.current/12,p['tier']/6,len(board)/7])
    return np.asarray(rows,dtype=np.float32)


class Teacher(Durability):
    def reset(self):
        super().reset()
        self.examples=[]

    def observe_comparison(self,c,prior,opts,base,best,retention,scores):
        src=int(c[4])
        x=target_rows(self,src,prior,opts,base,retention)
        self.examples.append((x,np.array([np.clip(scores.get(a,-1),-1,1) for a in opts],dtype=np.float32),
                              opts.index(base),opts.index(best),src))


def samples(i):
    arm=i%2
    w=Teacher('development',head=heads()[arm])
    r=evaluate_policy(w,EvalConfig(games=1,seed=89100000+i,mirror_pool=w.pool))
    check(r)
    return i,w.examples


def collect(shard):
    d=ROOT/f'source-{shard:02}'; d.mkdir(parents=True,exist_ok=False)
    atomic_json(d/'run.json',dict(**provenance(),teachers=file_identity(TEACHERS/'heads.json')))
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
        rows=list(pool.map(samples,range(shard*64,(shard+1)*64)))
    cases=[(i,*example) for i,game in rows for example in game]
    x=np.concatenate([r[1] for r in cases]);y=np.concatenate([r[2] for r in cases])
    meta=np.array([(i,len(vector),base,best,src) for i,vector,_,base,best,src in cases],np.int64)
    np.savez_compressed(d/'examples.npz',x=x,y=y,meta=meta)
    atomic_json(d/'complete.json',dict(games=[i for i,_ in rows],contexts=len(cases),file=file_identity(d/'examples.npz')))


def network(width):
    return nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))


def cases_for(arm):
    rows=[]
    for s in range(32):
        d=ROOT/f'source-{s:02}'
        receipt=json.loads((d/'complete.json').read_text())
        if receipt['games']!=list(range(s*64,(s+1)*64)) or receipt['file']!=file_identity(d/'examples.npz'):
            raise ValueError('distillation data drift')
        with np.load(d/'examples.npz') as arr:
            x,y,meta=arr['x'],arr['y'],arr['meta']
        off=0
        for i,n,base,best,src in meta:
            if i%2==arm:
                rows.append((int(i),x[off:off+n],y[off:off+n],int(base),int(best),int(src)))
            off+=n
        if off!=len(x):
            raise ValueError('incomplete context offsets')
    return rows


def train_one(rows,seed):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    width=rows[0][1].shape[1]
    net=network(width)
    all_x=np.concatenate([r[1] for r in rows])
    raw_y=np.concatenate([r[2] for r in rows])
    mean=all_x.mean(0);scale=np.maximum(all_x.std(0),.05)
    x=torch.from_numpy((all_x-mean)/scale)
    y=torch.from_numpy(raw_y)
    # Source rows include every legal target. Oversample truly beneficial alternatives.
    pos=np.flatnonzero(raw_y>0)
    if len(pos)<30:
        raise ValueError('insufficient positive examples')
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0005)
    for _ in range(18):
        order=rng.permutation(len(x))
        for k in range(0,len(order),384):
            idx=np.r_[order[k:k+384],rng.choice(pos,96,replace=True)]
            pred=net(x[idx]).squeeze(1)
            loss=nn.functional.smooth_l1_loss(pred,y[idx],beta=.10)
            opt.zero_grad();loss.backward();opt.step()
    return net.eval(),mean,scale


def predictions(net,mean,scale,rows):
    x=np.concatenate([r[1] for r in rows])
    with torch.no_grad():
        scores=net(torch.from_numpy((x-mean)/scale)).squeeze(1).numpy()
    return np.split(scores,np.cumsum([len(r[1]) for r in rows])[:-1])


def quality(rows,scores,threshold):
    n=good=possible=exact=0
    for r,sc in zip(rows,scores):
        _,_,y,base,best,_=r
        possible+=int(base!=best)
        pick=int(sc.argmax())
        if pick==base or sc[pick]<=threshold:
            continue
        n+=1;good+=int(y[pick]>0);exact+=int(pick==best)
    return dict(contexts=len(rows),teacher_changes=possible,calls=n,beneficial=good,exact=exact,
                precision=good/max(1,n),coverage=good/max(1,possible))


def fit(arm):
    torch.set_num_threads(2)
    out=ROOT/f'arm-{arm}';out.mkdir(parents=True,exist_ok=False)
    rows=cases_for(arm)
    train=[r for r in rows if (r[0]//2)%5!=4];cal=[r for r in rows if (r[0]//2)%5==4]
    net,mean,scale=train_one(train,190515+arm)
    pred=predictions(net,mean,scale,cal)
    grid=[float(t) for t in np.arange(-.10,.301,.025)]
    stats=[quality(cal,pred,t) for t in grid]
    qualified=[j for j,v in enumerate(stats) if v['precision']>=.70 and v['beneficial']>=8]
    # An uncertain student is allowed to abstain on all targets.
    threshold=grid[max(qualified,key=lambda j:stats[j]['beneficial'])] if qualified else 2.
    torch.save(dict(state=net.state_dict(),mean=mean.tolist(),scale=scale.tolist(),threshold=threshold),out/'head.pt')
    # Mechanism generalization: a separate public-text head with a deterministic held-out source bucket.
    excluded=lambda r:int.from_bytes(hashlib.blake2s(str(r[5]).encode(),digest_size=2).digest(),'little')%4==0
    keep=[r for r in train if not excluded(r)];hidden=[r for r in cal if excluded(r)]
    audit=None
    if keep and hidden and sum(np.count_nonzero(r[2]>0) for r in keep)>=30:
        other,mu,sd=train_one(keep,190515+arm)
        audit=quality(hidden,predictions(other,mu,sd,hidden),threshold)
    atomic_json(out/'complete.json',dict(train_games=len(set(r[0] for r in train)),train_contexts=len(train),
                calibration=quality(cal,pred,threshold),threshold=threshold,unseen_sources=audit,head=file_identity(out/'head.pt')))


@lru_cache(maxsize=2)
def student_head(arm):
    ck=torch.load(ROOT/f'arm-{arm}'/'head.pt',map_location='cpu',weights_only=True)
    net=network(len(ck['mean']));net.load_state_dict(ck['state']);net.eval()
    return net,np.array(ck['mean'],np.float32),np.array(ck['scale'],np.float32),ck['threshold']


class Student(Durability):
    def __init__(self, field, arm):
        super().__init__(field,head=heads()[arm])
        self.approx=student_head(arm)

    def __call__(self,f,c,mask,seat):
        prior,value=self.original(f,c,mask,seat)
        pending=self.player['pending']
        if not pending or 'Target' not in pending[-1]:
            return prior,value
        base=int(prior.argmax())
        own={m['eid'] for m in self.player['board']}
        opts=[30+i for i,e in enumerate(pending[-1]['Target']['options'][:16]) if e in own and mask[30+i]]
        if base not in opts or len(opts)<2:
            return prior,value
        self.counter['eligible']+=1
        x=unit_features(self.field,self.player,self.current)
        ret=np.clip(self.head['intercept']+((x-self.head['mean'])/self.head['scale'])@self.head['weight'],.05,.95)
        public=target_rows(self,int(c[4]),prior,opts,base,ret)
        net,mean,scale,threshold=self.approx
        with torch.no_grad():
            scores=net(torch.from_numpy((public-mean)/scale)).squeeze(1).numpy()
        best=int(scores.argmax())
        if opts[best]==base or scores[best]<=threshold:
            return prior,value
        self.counter['changed']+=1;self.sources[int(c[4])]+=1
        out=np.zeros_like(prior);out[opts[best]]=1
        return out,value


def final_game(i):
    base=89300000+i
    rows=[]
    for w in (Durability('transfer'),Durability('transfer',head=heads()[0]),Durability('transfer',head=heads()[1]),
              Student('transfer',0),Student('transfer',1)):
        r=evaluate_policy(w,EvalConfig(games=1,seed=base,mirror_pool=w.pool))
        check(r)
        rows.append(dict(seed=base,rank=r['rank_sequence'][0],eot=w.eot,changes=w.counter['changed'],
                         sources=w.sources,eligible=w.counter['eligible']))
    return rows


def assess():
    cases=[v for s in range(16) for v in json.loads((ROOT/f'panel-{s:02}'/'rows.json').read_text())]
    out={}
    for label,idx in [('teacher',(0,1,2)),('student',(0,3,4))]:
        rows=[v[j] for v in cases for j in idx]
        out[label]=paired_endpoints(rows,89300000,512)
        for j in idx[1:]:
            counts=Counter()
            for v in cases:
                counts.update(v[j]['sources'])
            out[label].setdefault('coverage',[]).append(dict(games=sum(v[j]['changes']>0 for v in cases),
                                                           sources=dict(counts)))
    atomic_json(ROOT/'assessment.json',out)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='source':
        collect(int(sys.argv[2]))
    elif stage=='fit':
        fit(int(sys.argv[2]))
    elif stage=='panel':
        s=int(sys.argv[2]);d=ROOT/f'panel-{s:02}';d.mkdir(parents=True,exist_ok=False)
        atomic_json(d/'run.json',provenance())
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            rows=list(pool.map(final_game,range(32*s,32*(s+1))))
        atomic_json(d/'rows.json',rows)
    elif stage=='assess':
        assess()
