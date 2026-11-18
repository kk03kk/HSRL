"""Fit a whole-recruit-path preference policy from sealed group-relative trajectories."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.az.train.ppo import masked_log_softmax_torch
from hsbg.env.obs_std import GLOBAL_FIELD_INDEX, MINION_ATK, MINION_F, MINION_HP, N_BOARD, SLICE_BOARD
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.d132_paired_plans import check
from workflows.group_relative_recruit import GroupPolicy, ROOT as GROUP_ROOT, SHARDS, TEMPERATURE, UPDATES, output
from workflows.learn_recruit_residual import paired_endpoints
from workflows.probe_investment_timing import MODEL

ROOT=Path('artifacts/runs/nus-prefer-paths-20261118-1')


def preferences(arr, counts, ranks):
    ends=np.flatnonzero(arr['a']==0)
    f=arr['f'][ends]
    turn=np.rint(20*f[:,GLOBAL_FIELD_INDEX['turn']]).astype(int)
    board=f[:,SLICE_BOARD].reshape(-1,N_BOARD,MINION_F)
    stat=np.rint(10*(board[:,:,MINION_ATK]+board[:,:,MINION_HP]).sum(1))
    changes=[]
    offset=0
    for count in counts:
        keep=(ends>=offset)&(ends<offset+count)
        eot=dict(zip(turn[keep],stat[keep]))
        changes.append(np.log1p(eot[12])-np.log1p(eot[9]) if 9 in eot and 12 in eot else None)
        offset+=count
    pairs=[]
    for i,j in combinations(range(len(counts)),2):
        if min(counts[i],counts[j])==0:
            continue
        if changes[i] is None or changes[j] is None:
            if ranks[i]!=ranks[j]:
                pairs.append((i,j,'rank') if ranks[i]<ranks[j] else (j,i,'rank'))
        elif ranks[i]<=ranks[j] and changes[i]>changes[j]+.10:
            pairs.append((i,j,'joint' if ranks[i]<ranks[j] else 'tie'))
        elif ranks[j]<=ranks[i] and changes[j]>changes[i]+.10:
            pairs.append((j,i,'joint' if ranks[j]<ranks[i] else 'tie'))
    return pairs


def eligible(arm):
    groups=[]
    for it in range(UPDATES):
        for s in range(SHARDS):
            d=output(arm,it)/f'rollout-{s:02}'
            rows=json.loads((d/'groups.json').read_text())
            if len(rows)!=16:
                raise ValueError('incomplete grouped rollout')
            for r in rows:
                if not sum(r['actions']):
                    continue
                path=d/f"group-{r['i']}.npz"
                with np.load(path) as arr:
                    pairs=preferences(arr,r['actions'],r['ranks'])
                if pairs:
                    groups.append((path,r['actions'],pairs))
    return groups


def update(arm):
    out=ROOT/f'arm-{arm}'
    out.mkdir(parents=True,exist_ok=False)
    groups=eligible(arm)
    device='cuda'
    torch.manual_seed(187500+arm)
    rng=np.random.default_rng(187500+arm)
    net,meta=load_checkpoint(MODEL,device)
    reference,_=load_checkpoint(MODEL,device)
    reference.eval();net.eval()
    opt=torch.optim.AdamW(net.parameters(),lr=5e-6,weight_decay=0.)
    reference_logp={}
    metrics=[]
    types={k:sum(t==k for _,_,pairs in groups for _,_,t in pairs) for k in ('rank','joint','tie')}
    for epoch in range(2):
        order=rng.permutation(len(groups))
        for st in range(0,len(groups),8):
            selected=order[st:st+8]
            opt.zero_grad(set_to_none=True)
            for g in selected:
                path,counts,pairs=groups[g]
                with np.load(path) as arr:
                    f=torch.as_tensor(arr['f'],device=device)
                    cards=torch.as_tensor(arr['c'].astype(np.int64),device=device)
                    mask=torch.as_tensor(np.unpackbits(arr['m'],axis=1)[:,:245].astype(bool),device=device)
                    a=torch.as_tensor(arr['a'].astype(np.int64),device=device)
                if meta.get('norm_mean') is not None:
                    f=f-torch.as_tensor(meta['norm_mean'],device=device)
                if meta.get('norm_std') is not None:
                    f=f/torch.as_tensor(meta['norm_std'],device=device)
                def action_logp(model):
                    chunks=[]
                    for start in range(0,len(f),256):
                        sl=slice(start,start+256)
                        logits=model(f[sl],cards[sl],mask[sl])[0]
                        chunks.append(masked_log_softmax_torch(logits/TEMPERATURE,mask[sl]).gather(1,a[sl,None]).flatten())
                    return torch.cat(chunks)
                if g not in reference_logp:
                    with torch.no_grad():
                        reference_logp[g]=action_logp(reference).detach().cpu()
                delta=action_logp(net)-reference_logp[g].to(device)
                path_score=torch.stack([v.sum()/np.sqrt(max(n,1)) for v,n in zip(torch.split(delta,counts),counts)])
                margin=torch.stack([path_score[i]-path_score[j] for i,j,_ in pairs])
                loss=F.softplus(-.5*margin).mean()+.025*delta.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite pair loss')
                (loss/len(selected)).backward()
                metrics.append(float(margin.detach().mean()))
            torch.nn.utils.clip_grad_norm_(net.parameters(),.5,error_if_nonfinite=True)
            opt.step()
    save_checkpoint(net,str(out/'model.pt'),meta=meta,norm_mean=meta.get('norm_mean'),norm_std=meta.get('norm_std'))
    torch.save(opt.state_dict(),out/'adam.pt')
    atomic_json(out/'complete.json',dict(training_roots=len(groups),preferences=types,avg_margin=float(np.mean(metrics)),
                                         weights=file_identity(out/'model.pt'),reference=file_identity(MODEL)))


def game(i):
    rows=[]
    for arm in (0,1):
        w=GroupPolicy('transfer',str(ROOT/f'arm-{arm}'/'model.pt'),7)
        result=evaluate_policy(w,EvalConfig(games=1,seed=87900000+i,mirror_pool=w.pool))
        check(result)
        rows.append(dict(seed=87900000+i,rank=result['rank_sequence'][0],eot=w.eot,changes=0))
    return rows


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='update':
        update(int(sys.argv[2]))
    elif stage=='panel':
        s=int(sys.argv[2])
        d=ROOT/f'panel-{s}';d.mkdir(parents=True,exist_ok=False)
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            rows=list(pool.map(game,range(32*s,32*(s+1))))
        atomic_json(d/'rows.json',rows)
    elif stage=='assess':
        rows=[]
        for s in range(8):
            original=json.loads((GROUP_ROOT/f'panel-{s:02}'/'rows.json').read_text())
            learned=json.loads((ROOT/f'panel-{s}'/'rows.json').read_text())
            if len(original)!=32 or len(learned)!=32:
                raise ValueError('incomplete natural panel')
            for base,new in zip(original,learned):
                rows.extend([base[0],*new])
        report=paired_endpoints(rows,87900000,256)
        for r in report['report']:
            r.pop('changes')
        atomic_json(ROOT/'assessment.json',report)
