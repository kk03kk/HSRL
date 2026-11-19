"""D-145: natural choice-quilboar teaching, outcome optimization and blind tests."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.az.train.quilboar_option import basis,roles,TERMS,QuilboarOption
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.evaluate import evaluate_policy,EvalConfig
from hsbg.az.train.eval_policy import protocol_from_args,make_eval_policy
from hsbg.az.train.common import engine_info
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.model import save_checkpoint,load_checkpoint
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.env import obs_std as O

PLAN=json.loads((ROOT/os.environ.get('HSAI_QUILL_PLAN','config/research/quilboar-choice-study.json')).read_text())
OUT=Path(os.environ.get('HSAI_QUILL_OUT',PLAN['out']))
BASE=PLAN['base']
U=np.array([1,.35,-1,-1.15,-1.3,-1.45,-1.6,-1.75])
EXPERT=np.array([7,7,17,15,14,19,16,22,17,16,8,7,4,11,13,5,6,2.2,3.0])


def read(path):
    return json.loads(path.read_text())


@lru_cache(None)
def config(field, *, allow_tail_variant=False):
    opts=read(Path('config/evaluation/human-turn-3702-parent.json'))['evaluation']
    opts['mirror-snaps']='artifacts/libraries/human-turn-3702-20261118/'+field
    proto=protocol_from_args(argparse.Namespace(**{k.replace('-','_'):v for k,v in opts.items()}),engine=engine_info())
    proto.assert_training_contract()
    from dataclasses import replace
    exact=replace(proto,human_tail="exact") if allow_tail_variant else proto
    if field=='hold' and exact.sha256()!=PLAN['protocol']:
        raise ValueError('hold protocol changed')
    pool=MirrorSnapshotPool(opts['mirror-snaps'],sample='random')
    if pool.frames!={'hold':5477,'train':40659}[field]:
        raise ValueError('incomplete library')
    return proto,pool


class Teacher:
    def __init__(self,original):
        self.original=original
        self.model=original.model
        self.samples=[]
        self.calls=0

    def __call__(self,f,c,mask,seat):
        base,value=self.original(f,c,mask,seat)
        x=basis(f,c)
        if x is None or not x[mask].any():
            return base,value
        ids=np.flatnonzero(mask)
        # Equivalent insert positions keep the base's relative preference.
        group,inv=np.unique(x[ids],axis=0,return_inverse=True)
        if len(group)<2:
            return base,value
        lp=np.log(np.maximum(base[ids],1e-35))
        top=np.full(len(group),-90.)
        np.maximum.at(top,inv,lp)
        score=group@EXPERT
        # Prioritize realizing an offered payoff. A lone starter is optional
        # when the current board is strong or health cannot fund investment.
        if f[5]*40+f[6]*20<16:
            score-=4*(group[:,0]+group[:,1]+group[:,17])
        choice=int(np.argmax(top+score))
        base_choice=int(np.argmax(top))
        self.calls+=choice!=base_choice
        self.samples.append((group.astype(np.float16),top.astype(np.float16),choice))
        return np.where(mask, np.exp(np.clip(np.log(np.maximum(base,1e-35))+
                        (x@EXPERT)-(4*(x[:,0]+x[:,1]+x[:,17]) if f[5]*40+f[6]*20<16 else 0),-80,65)),0),value


class Recorder:
    def begin_game(self,game,index,seed,seat):
        self.lobby=int(json.loads(game.state_json())['lobby_races'])
        self.events=[]

    def before(self,game,seat):
        if not self.lobby & 128 or int(game.turn)<3:
            return None
        p=json.loads(game.player_json(seat))
        table=roles()[0]
        riders={m['eid'] for m in p['board'] if table[m['card']+1,0]}
        if not riders:
            return None
        return int(game.turn),riders,{m['eid']:m['gems'] for m in p['board'] if table[m['card']+1,3]},p['hand'],p['pending']

    def after(self,game,before,decision,a,forced):
        if before is None:
            return
        t,riders,gems,hand,pending=before
        if 46<=a<126:
            card=item_card(hand[(a-46)//8])
        elif 30<=a<46 and pending and 'ChooseOne' in pending[-1]:
            card=pending[-1]['ChooseOne']['card']
        else:
            return
        if not roles()[0][card+1,2]:
            return
        p=json.loads(game.player_json(0))
        gains={m['eid']:m['gems']-gems[m['eid']] for m in p['board'] if m['eid'] in gems}
        if any(gains.get(i,0)>=2 for i in riders) and sum(v>=2 for v in gains.values())>=2:
            self.events.append([t,card,sum(max(0,v) for v in gains.values())])

    def end_game(self,game,rank,quality):
        pass


def play(policy,pool,seed,rec=None):
    if rec is None:
        rec=Recorder()
    r=evaluate_policy(policy,EvalConfig(games=1,seed=seed,mirror_pool=pool,mirror_strict=True,
                                        decision_recorder=rec,collect_economy=True))
    if (r['games_finished']!=1 or r['seed_sequence']!=[seed] or any(r[k] for k in
        ('truncated_games','truncated_combats','illegal_actions')) or r['mirror']['mirror_fallback']):
        raise ValueError(f'invalid trajectory {seed}')
    eot={}
    for row in r['end_turn_economy']:
        if row['turn']<1:
            continue
        if row['eot_board'] is None:
            raise ValueError('missing EOT board')
        eot[row['turn']]=[row['eot_attack']+row['eot_board_hp'],[m['card_id'] for m in row['eot_board']]]
    return dict(seed=seed,lobby=rec.lobby,rank=r['rank_sequence'][0],eot=eot,events=rec.events,decisions=r['decisions'])


def seeds(stage,count,quill):
    import hsbg_engine as E
    start=PLAN['seeds'][stage]
    if not quill:
        return list(range(start,start+count))
    result=[]
    for seed in range(start,start+10000):
        if int(json.loads(E.Game(seed=seed).state_json())['lobby_races'])&128:
            result.append(seed)
        if len(result)==count:
            return result
    raise RuntimeError('seed scan exhausted')


def policy(path,proto):
    return make_eval_policy(str(path),protocol=proto,device='cuda:0')['policy']


def shard(stage,i):
    torch.set_num_threads(1)
    field='hold' if stage in ('natural','special') else 'train'
    proto,pool=config(field)
    seq=seeds(stage,PLAN['sizes'][stage],stage!='natural')[i::PLAN['workers']]
    d=OUT/stage
    if stage=='teach':
        p=policy(BASE,proto)
        teacher=Teacher(p.evaluator)
        p.evaluator=teacher
        games=[]
        for seed in seq:
            games.append(play(p,pool,seed))
        xs=[];ps=[];labels=[];lengths=[]
        for x,pr,y in teacher.samples:
            xs.append(x);ps.append(pr);labels.append(y);lengths.append(len(pr))
        np.savez_compressed(d/f'{i}.npz',x=np.concatenate(xs),prior=np.concatenate(ps),
                            y=np.array(labels),lengths=np.array(lengths))
        atomic_json(d/f'{i}.json',dict(calls=teacher.calls,games=games))
    else:
        choices=read(d/'arms.json')
        models={k:policy(path,proto) for k,path in choices.items()}
        results=[]
        for seed in seq:
            results.append({k:play(p,pool,seed) for k,p in models.items()})
        atomic_json(d/f'{i}.json',results)


def launch(stage,arms=None):
    d=OUT/stage
    d.mkdir(exist_ok=False)
    if arms:
        atomic_json(d/'arms.json',arms)
    def work(i):
        with (d/f'worker-{i}.log').open('x') as f:
            subprocess.run([sys.executable,__file__,'shard',stage,str(i)],stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(PLAN['workers']) as executor:
        list(executor.map(work,range(PLAN['workers'])))
    rows=[row for i in range(PLAN['workers']) for row in
          (read(d/f'{i}.json')['games'] if stage=='teach' else read(d/f'{i}.json'))]
    expected=set(seeds(stage,PLAN['sizes'][stage],stage!='natural'))
    if stage=='teach':
        seen=[r['seed'] for r in rows]
    else:
        seen=[next(iter(r.values()))['seed'] for r in rows]
        if any(set(r)!=set(arms) or len({(g['seed'],g['lobby']) for g in r.values()})!=1 for r in rows):
            raise ValueError('unpaired arms')
    if set(seen)!=expected or len(seen)!=len(expected):
        raise ValueError('incomplete panel')
    atomic_json(d/'complete.json',dict(seeds=sorted(seen),arms=arms,source_shards=[
        file_identity(d/f'{i}.json') for i in range(PLAN['workers'])]))
    return rows


def fit():
    # Distill the expert's discrete preference, with a frozen parent logit offset.
    batches=[]
    for i in range(PLAN['workers']):
        with np.load(OUT/'teach'/f'{i}.npz') as z:
            x=z['x'].astype(np.float32);p=z['prior'].astype(np.float32)
            off=0
            for n,y in zip(z['lengths'],z['y']):
                batches.append((x[off:off+n],p[off:off+n],y))
                off+=n
    width=max(len(p) for x,p,y in batches)
    x=np.zeros((len(batches),width,len(TERMS)),np.float32)
    p=np.full((len(batches),width),-1000,np.float32)
    y=np.empty(len(batches),np.int64)
    for j,(features,prior,label) in enumerate(batches):
        x[j,:len(prior)]=features;p[j,:len(prior)]=prior;y[j]=label
    x,p,y=map(torch.from_numpy,(x,p,y))
    w=torch.zeros(len(TERMS),requires_grad=True)
    optim=torch.optim.LBFGS([w],max_iter=150,line_search_fn='strong_wolfe')
    changed=y!=p.argmax(1)
    def loss():
        optim.zero_grad()
        ce=torch.nn.functional.cross_entropy(p+x@w,y,reduction='none')
        l=(ce*torch.where(changed,2.,1.)).mean()+.00012*w.square().sum()
        l.backward()
        return l
    optim.step(loss)
    correct=(p+x@w).argmax(1)==y
    atomic_json(OUT/'fit.json',dict(examples=len(y),divergent=int(changed.sum()),
                accuracy=float(correct.float().mean()),changed_accuracy=float(correct[changed].float().mean()),
                weights=w.detach().tolist()))
    return w.detach().numpy()


def checkpoint(tag,w):
    p=OUT/'models'/tag/'model.pt'
    if p.exists():
        raise FileExistsError(p)
    base,meta=load_checkpoint(BASE,'cpu')
    model=QuilboarOption(base,meta,w)
    save_checkpoint(model,str(p))
    atomic_json(p.parent/'manifest.json',dict(name=tag,base=PLAN['base_sha'],weights=w.tolist(),model=file_identity(p),
                terms=TERMS,engine=PLAN['engine'],protocol=PLAN['protocol']))
    return str(p)


def score(r):
    # A bounded strength target alongside true placement, without rewarding a
    # purchase, race count or professor's preferred action.
    strength=np.mean([min(np.log1p(r['eot'].get(str(t),[0])[0]),10)/8 for t in (10,11,12,13)])
    return float(U[r['rank']-1]+.45*strength)


def scale(bc,theta):
    groups=np.array([0,0,1,1,1,1,2,2,2,1,1,1,1,2,1,2,0,0,0])
    return bc*np.exp(theta[groups])


def optimize(stage,center,bc):
    rng=np.random.default_rng(PLAN['seeds'][stage])
    directions=np.linalg.qr(rng.normal(size=(3,3)))[0]
    population=[center]+[center+s*.60*v for v in directions for s in (1,-1)]
    arms={str(j):checkpoint(f'{stage}-{j}',scale(bc,theta)) for j,theta in enumerate(population)}
    rows=launch(stage,arms)
    scores=np.array([[score(row[k]) for k in arms] for row in rows])
    advantages=(scores-scores[:,0,None]).mean(0)
    best=np.argsort(-advantages,kind='stable')[:2]
    updated=np.clip(np.mean([population[j] for j in best],axis=0),-1.3,1.3)
    atomic_json(OUT/stage/'update.json',dict(center=center.tolist(),directions=directions.tolist(),
                advantages=advantages.tolist(),next=updated.tolist(),population=[v.tolist() for v in population]))
    return updated


def main():
    from hsbg.az.train.personal import library_digest
    OUT.mkdir(parents=True,exist_ok=False)
    source=source_identity(ROOT)
    if source['tracked_dirty'] or file_identity(BASE)['sha256']!=PLAN['base_sha']:
        raise ValueError('source or checkpoint differs')
    for f in ('train','hold'):
        if library_digest(Path('artifacts/libraries/human-turn-3702-20261118')/f)!=PLAN['libraries'][f]:
            raise ValueError('field fingerprint mismatch')
    atomic_json(OUT/'plan.json',dict(plan=PLAN,source=source,engine=engine_info(),job=os.environ.get('SLURM_JOB_ID')))
    torch.set_num_threads(2)
    launch('teach')
    bc=fit()
    imitation=checkpoint('imitation',bc)
    theta=np.zeros(3)
    for stage in ('search1','search2'):
        theta=optimize(stage,theta,bc)
    optimized=checkpoint('outcome',scale(bc,theta))
    arms=dict(parent=BASE,imitation=imitation,outcome=optimized)
    dev=launch('develop',arms)
    quality={k:float(np.mean([score(row[k]) for row in dev])) for k in arms}
    chosen=max(('imitation','outcome'),key=lambda k:quality[k])
    final=arms[chosen]
    atomic_json(OUT/'selection.json',dict(development_scores=quality,candidate=chosen,path=final,
                                         model=file_identity(final)))
    for stage in ('natural','special'):
        launch(stage,dict(parent=BASE,student=final))
    atomic_json(OUT/'complete.json',dict(valid=True,candidate=chosen,selected=file_identity(final),
                                         panels={s:file_identity(OUT/s/'complete.json') for s in ('natural','special')}))


if __name__=='__main__':
    if sys.argv[1]=='run':
        main()
    else:
        shard(sys.argv[2],int(sys.argv[3]))
