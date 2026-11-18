"""Self-critical policy gradients from sparse interventions in otherwise greedy natural games."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.assess_investment_timing import UTILITY
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.group_relative_recruit import GroupPolicy,checkpoint,output,update,GROUPS,UPDATES
from workflows.learn_recruit_residual import paired_endpoints
from workflows.probe_investment_timing import MODEL


ROOT=Path('artifacts/runs/nus-sparse-recruit-20261118-1')
FIRST=91100000


class SparsePolicy(GroupPolicy):
    def __call__(self,f,c,mask,seat):
        if self.current<self.start:
            return self.original(f,c,mask,seat)
        if self.origin is None:
            self.origin=hashlib.sha256(f.tobytes()+c.tobytes()+mask.tobytes()).hexdigest()
        if self.noise is None or mask.sum()<2 or self.noise.random()>=.25:
            return self.actor(f,c,mask,seat)
        return super().__call__(f,c,mask,seat)


def source(task):
    arm,it,i,d=task
    seed=90800000+arm*100000+it*10000+i
    path=checkpoint(arm,it,ROOT)
    ranks=[];traces=[];starts=[];counts=[]
    for replica in range(4):
        noise=(np.random.default_rng(np.random.SeedSequence([310516,arm,it,i,replica])) if replica else None)
        w=SparsePolicy('development',path,7+i%3,noise)
        r=evaluate_policy(w,EvalConfig(games=1,seed=seed,mirror_pool=w.pool))
        check(r)
        ranks.append(r['rank_sequence'][0]);starts.append(w.origin);counts.append(len(w.actions))
        if replica:
            traces.append(w.trajectory)
    if len(set(starts))!=1:
        raise ValueError('greedy counterfactual did not share the start')
    y=np.array(UTILITY)[np.array(ranks)-1]
    other=(y[1:].sum()-y[1:])/2
    relative=y[1:]-.5*(y[0]+other)
    lengths=[len(t) for t in traces]
    rows=[v for t in traces for v in t]
    if rows:
        fs,cs,ms,actions,lp=zip(*rows)
        np.savez_compressed(d/f'group-{i}.npz',f=np.stack(fs),c=np.stack(cs),m=np.stack(ms),
                            a=np.array(actions,np.int16),logp=np.array(lp,np.float32),
                            adv=np.repeat(relative,lengths).astype(np.float32))
    return dict(i=i,seed=seed,start=7+i%3,origin=starts[0],ranks=ranks,actions=lengths,game_decisions=counts)


def collect(arm,it,shard):
    d=output(arm,it,ROOT)/f'rollout-{shard:02}'
    d.mkdir(parents=True,exist_ok=False)
    atomic_json(d/'run.json',dict(**provenance(),behavior=file_identity(checkpoint(arm,it,ROOT))))
    tasks=[(arm,it,i,d) for i in range(GROUPS*shard,GROUPS*(shard+1))]
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
        rows=list(pool.map(source,tasks))
    atomic_json(d/'groups.json',rows)


def game(i):
    rows=[]
    for path in (MODEL,checkpoint(0,UPDATES,ROOT),checkpoint(1,UPDATES,ROOT)):
        w=GroupPolicy('transfer',path,7)
        r=evaluate_policy(w,EvalConfig(games=1,seed=FIRST+i,mirror_pool=w.pool))
        check(r)
        rows.append(dict(seed=FIRST+i,rank=r['rank_sequence'][0],eot=w.eot,changes=0))
    return rows


if __name__=='__main__':
    stage=sys.argv[1];args=list(map(int,sys.argv[2:]))
    if stage=='collect':
        collect(*args)
    elif stage=='update':
        update(*args,root=ROOT)
    elif stage=='panel':
        s,=args;d=ROOT/f'panel-{s:02}';d.mkdir(parents=True,exist_ok=False)
        atomic_json(d/'run.json',provenance())
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            rows=list(pool.map(game,range(32*s,32*(s+1))))
        atomic_json(d/'rows.json',rows)
    elif stage=='assess':
        cases=[g for s in range(8) for row in json.loads((ROOT/f'panel-{s:02}'/'rows.json').read_text()) for g in row]
        report=paired_endpoints(cases,FIRST,256)
        for entry in report['report']:
            entry.pop('changes')
        atomic_json(ROOT/'assessment.json',report)
    else:
        raise ValueError('unknown stage')
