"""Collect fresh on-policy data and diagnose credit/gradient signals without updates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hsbg.az.train.common import engine_info, set_rank_utility_table
from hsbg.az.train.model import load_checkpoint
from hsbg.az.train.ppo import (CollectConfig, PPOConfig, PPOEvaluator,
    build_batch_from_records, compute_gae, ppo_losses)
from hsbg.az.train.parallel_collect import collect_ppo_lockstep
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.collect_diagnostics import require_valid_mirror_collect
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity, environment_identity

UTILITY = [1, .35, -1, -1.15, -1.3, -1.45, -1.6, -1.75]


def credit_summary(batch):
    mc, _ = compute_gae(batch.rewards, batch.values, batch.dones, gamma=1, lam=1)
    observed = np.asarray(UTILITY)[batch.ranks - 1]
    if not np.allclose(mc, observed - batch.values, atol=2e-6):
        raise ValueError('Monte Carlo telescope disagrees with terminal utility')
    rows = []
    for turn in sorted(set(map(int, batch.turns))):
        ix = np.flatnonzero(batch.turns == turn)
        first = ix[np.r_[True, batch.episode_ids[ix][1:] != batch.episode_ids[ix][:-1]]]
        residual = observed[first] - batch.values[first]
        variance = np.var(observed[first])
        rows.append({'turn': turn, 'games': len(first), 'actions': len(ix),
            'critic_mean': float(batch.values[first].mean()),
            'observed_utility_mean': float(observed[first].mean()),
            'critic_bias': float(-residual.mean()),
            'critic_rmse': float(np.sqrt(np.mean(residual**2))),
            'explained_variance': float(1-np.var(residual)/variance) if variance > 0 else None,
            'gae_abs_mean': float(np.abs(batch.advantages[ix]).mean()),
            'mc_abs_mean': float(np.abs(mc[ix]).mean()),
            'raw_sign_disagreement': float(np.mean((mc[ix]>0)!=(batch.advantages[ix]>0))),
            'teacher_labels': int(np.sum(batch.teacher_action[ix]>=0)) if batch.teacher_action is not None else 0})
    return rows, mc


def gradient_summary(model, batch, mc, device, seed, samples):
    if not samples:
        return []
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    cfg = PPOConfig(device=device, clip=.2, value_coef=.5, entropy_coef=.002,
                    forced_teacher_coef=.1, teacher_loss_normalize='labeled')
    result=[]
    # The production loss normalizes advantages per 1024-row microbatch.
    # These are unaccumulated diagnostic gradients, not a reconstructed Adam step.
    for group, eligible in [('all',np.arange(len(batch))),
                            ('late',np.flatnonzero(batch.turns>=9))]:
        if not len(eligible):
            continue
        for replicate in range(samples):
            ix=rng.choice(eligible,size=min(1024,len(eligible)),replace=False)
            losses=ppo_losses(model,batch,cfg,indices=ix)
            teacher=losses['total']-losses['policy_loss']-.5*losses['value_loss']+.002*losses['entropy']
            terms={'policy_gae':losses['policy_loss'],
                   'value_weighted':.5*losses['value_loss'], 'teacher_weighted':teacher}
            gradients={}
            for key,term in terms.items():
                gs=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
                gradients[key]=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                                         for p,g in zip(params,gs)])
            original=batch.advantages
            try:
                batch.advantages=mc
                mc_loss=ppo_losses(model,batch,cfg,indices=ix)['policy_loss']
                gs=torch.autograd.grad(mc_loss,params,allow_unused=True)
                gradients['policy_mc']=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                                                 for p,g in zip(params,gs)])
            finally:
                batch.advantages=original
            norms={k:float(torch.linalg.vector_norm(v)) for k,v in gradients.items()}
            base=gradients['policy_gae']
            cos={k:float(torch.dot(base,v)/(torch.linalg.vector_norm(base)*torch.linalg.vector_norm(v)).clamp_min(1e-20))
                 for k,v in gradients.items() if k!='policy_gae'}
            result.append({'group':group,'replicate':replicate,'samples':len(ix),
                'teacher_labels':int(np.sum(batch.teacher_action[ix]>=0)),
                'norms':norms,'cosine_with_policy_gae':cos,
                'ratio_mean':float(losses['ratio_mean'].detach()),
                'clip_frac':float(losses['clip_frac'].detach()),
                'indices':ix.tolist()})
            del losses,terms,gradients,gs,mc_loss,teacher
    return result


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    set_rank_utility_table(UTILITY)
    identity=file_identity(args.checkpoint)
    library=Path(args.library)
    library_identity={p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}
    if not library_identity:raise ValueError('missing training snapshot field')
    model,_=load_checkpoint(args.checkpoint);model.to(args.device);model.eval()
    cfg=CollectConfig(games=args.games,seed=args.seed,opponent='bot',behavior='net',lockstep=True,
        mirror_snaps=args.library,mirror_seats=7,mirror_p=1,mirror_sample='p50band',
        forced_tier_plan='2:2,5:3,7:4,8:5',forced_tier_prob=0,forced_tier_teacher=True)
    atomic_json(out/'run.json',{'schema':'hsai.credit-probe/v1','model':identity,'engine':engine_info(),
        'config':asdict(cfg),'utility':UTILITY,'device':args.device,
        'snapshot_files':library_identity,'source':source_identity(Path(__file__).resolve().parents[1]),
        'environment':environment_identity(),
        'scope':'fresh stochastic on-policy diagnostic; no optimizer update, no ability comparison'})
    records,meta=collect_ppo_lockstep(model,cfg,eval_fn=PPOEvaluator(model,args.device),device=args.device,
        seed=args.seed,games=args.games,gamemode='bot',
        mirror_pool=MirrorSnapshotPool(args.library,sample='p50band'),mirror_seats_count=7,mirror_p=1)
    require_valid_mirror_collect(meta,str(out),iteration=0,config=asdict(cfg))
    batch=build_batch_from_records(records,gamma=1,lam=.995,meta=meta)
    errors=batch.validate()
    if errors:raise ValueError(errors)
    batch.save(str(out/'batch.npz'))
    rows,mc=credit_summary(batch)
    gradients=gradient_summary(model,batch,mc,args.device,args.seed+1,args.gradient_samples)
    if file_identity(args.checkpoint)!=identity:raise ValueError('source weights changed')
    if {p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}!=library_identity:
        raise ValueError('snapshot field changed')
    atomic_json(out/'complete.json',{'valid':True,'games':args.games,'decisions':len(batch),
        'run':file_identity(out/'run.json'),'batch':file_identity(out/'batch.npz'),
        'turns':rows,'gradients':gradients,'collect':meta,
        'limitations':['MC is a noisy sampled return, not a true advantage oracle',
            'Sign differences are raw advantages, not final normalized PPO signs',
            'Gradient probes are first-step microbatch gradients, not accumulated Adam updates',
            'Anchor has zero gradient at the unchanged parent, and is omitted']})
    print(json.dumps({'valid':True,'games':args.games,'decisions':len(batch)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--library',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=11920000);p.add_argument('--games',type=int,default=128)
    p.add_argument('--gradient-samples',type=int,default=3)
    main(p.parse_args())
