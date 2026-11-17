"""Bounded process BC experiment; all outputs are research candidates, not promotion."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import hsbg_engine as engine
from hsbg.az.train.model import load_checkpoint,save_checkpoint
from hsbg.az.train.ppo import PPOEvaluator,PPOGreedyPolicy,forward_model,split_model_outputs
from hsbg.az.train.process_learning import process_loss,log_policy,forward_kl
from hsbg.az.train.economic_tasks import own,wealth_remaining,score
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity


def tensors(data,model,device):
    width=getattr(model,'input_floats',3227)
    return {k:torch.as_tensor(np.array(data[k],copy=True),device=device,
             dtype=torch.float32 if k=='features' else torch.long if k=='cards' else torch.bool)
            for k in ('features','cards','masks','targets') if k in data} | {
        'features':torch.as_tensor(np.array(data['features'][:,:width],copy=True),device=device)}


def forward(model,data,indices):
    out=forward_model(model,data['features'][indices],data['cards'][indices],data['masks'][indices])
    policy,rank,_=split_model_outputs(out)
    return policy.float(),rank.float()


@torch.no_grad()
def assess_rows(model,data,indices,names):
    groups={}
    for offset in range(0,len(indices),64):
        ix=indices[offset:offset+64]
        logits,_=forward(model,data,ix)
        probs=log_policy(logits,data['masks'][ix]).exp()
        mass=(probs*data['targets'][ix]).sum(-1).cpu().tolist()
        correct=data['targets'][ix].gather(1,probs.argmax(-1)[:,None]).flatten().cpu().tolist()
        for i,m,c in zip(ix,mass,correct):
            for key in ('all',str(names[i])):groups.setdefault(key,[]).append((m,c))
    return {k:{'rows':len(v),'optimal_mass':float(np.mean([x[0] for x in v])),
               'greedy_optimal':float(np.mean([x[1] for x in v]))} for k,v in groups.items()}


@torch.no_grad()
def closed_loop(model,tasks,device,out):
    """Full legal policy, not the oracle's restricted candidate set. Stop before EOT."""
    evaluator=PPOEvaluator(model,device=device)
    results=[]
    with gzip.open(out,'xt',encoding='utf-8',compresslevel=3) as stream:
        for task in tasks:
            game=engine.Game.from_json(json.dumps(task['initial']))
            policy=PPOGreedyPolicy(evaluator)
            original={m['eid'] for m in own(game)['board']}
            actions=[];stop='budget';board_retained=True
            for step in range(12):
                if wealth_remaining(game)==0:stop='completed';break
                before=json.loads(game.state_json())
                decision=policy.decide(game,0)
                action=int(decision.action)
                if action==0:
                    stop='end_turn';break
                if not bytes(game.legal_mask(0))[action]:raise RuntimeError('illegal skill action')
                game.step(0,action)
                after=json.loads(game.state_json())
                actions.append({'action':action,'before':before,'after':after})
                if not original<={m['eid'] for m in own(game)['board']}:
                    board_retained=False;stop='sold_original';break
                if any(game.diagnostic_counts()):raise RuntimeError('engine diagnostic in skill rollout')
            if wealth_remaining(game)==0 and board_retained:stop='completed'
            value=score(game,original)
            row={'seed':task['seed'],'kind':task['kind'],'bounties':task['bounties'],
                 'stop':stop,'actions':len(actions),'wealth_remaining':wealth_remaining(game),
                 'gold':own(game)['gold'],'value':value,'oracle_value':task['objective'],
                 'board_retained':board_retained,
                 'optimal_completed':bool(stop=='completed' and value>=task['objective']-1e-6)}
            results.append(row)
            stream.write(json.dumps({'result':row,'initial':task['initial'],'steps':actions},separators=(',',':'))+'\n')
    def summary(rows):
        return {'tasks':len(rows),'optimal_completed':sum(r['optimal_completed'] for r in rows)/max(1,len(rows)),
                'completed':sum(r['stop']=='completed' for r in rows)/max(1,len(rows)),
                'sold_original':sum(not r['board_retained'] for r in rows)}
    return {'all':summary(results),**{k:summary([r for r in results if r['kind']==k])
                                   for k in sorted({r['kind'] for r in results})},'rows':results}


def run(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    root=Path(args.dataset)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,identity in manifest['files'].items():
        if file_identity(root/name)!=identity:raise ValueError('dataset identity mismatch: '+name)
    model,meta=load_checkpoint(args.checkpoint,device=args.device)
    if meta.get('norm_mean') is not None or meta.get('norm_std') is not None:
        raise ValueError('this experiment requires an internally normalized entity model')
    reference=copy.deepcopy(model).eval()
    for parameter in reference.parameters():parameter.requires_grad_(False)
    raw=np.load(root/'labels.npz',allow_pickle=False)
    labels=tensors(raw,model,args.device)
    anchors=tensors(np.load(root/'anchors.npz',allow_pickle=False),model,args.device)
    train=np.flatnonzero(~raw['dev']);dev=np.flatnonzero(raw['dev'])
    if not len(train) or not len(dev):raise ValueError('missing source-disjoint split')
    with gzip.open(root/'tasks.jsonl.gz','rt',encoding='utf-8') as stream:
        tasks=[x for line in stream if (x:=json.loads(line))['split']=='dev']
    identity={'args':vars(args),'source':source_identity(Path(__file__).resolve().parents[1]),
              'environment':environment_identity(),'parent':file_identity(args.checkpoint),
              'dataset_manifest':file_identity(root/'manifest.json')}
    if identity['source']['tracked_dirty']:raise RuntimeError('committed source required')
    atomic_json(out/'run.json',identity)
    model.eval()
    baseline={'labels_dev':assess_rows(model,labels,dev,raw['kinds']),
              'closed_loop_dev':closed_loop(model,tasks,args.device,out/'before.jsonl.gz')}
    atomic_json(out/'baseline.json',baseline);print(json.dumps({'baseline':baseline['labels_dev']}),flush=True)
    if args.updates==0:return
    rng=np.random.default_rng(args.seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    history=[];started=time.monotonic()
    # Eval mode disables dropout, but gradients remain active. All parameters may learn.
    for update in range(1,args.updates+1):
        ix=rng.choice(train,size=args.batch,replace=True)
        ai=rng.integers(len(anchors['features']),size=args.batch)
        plog,_=forward(model,labels,ix)
        skill=process_loss(plog,labels['masks'][ix],labels['targets'][ix])
        current,rank=forward(model,anchors,ai)
        with torch.no_grad():rp,rr=forward(reference,anchors,ai)
        kl=forward_kl(log_policy(rp,anchors['masks'][ai]),log_policy(current,anchors['masks'][ai]))
        rank_kl=forward_kl(rr.log_softmax(-1),rank.log_softmax(-1))
        loss=skill+args.anchor_kl*kl+rank_kl
        if not torch.isfinite(loss):raise RuntimeError('non-finite process loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if update==1 or update%50==0 or update==args.updates:
            row={'update':update,'loss':float(loss.detach()),'skill':float(skill.detach()),
                 'anchor_policy_kl':float(kl.detach()),'anchor_rank_kl':float(rank_kl.detach()),
                 'gradient_norm':float(norm),'elapsed':time.monotonic()-started,
                 'labels_dev':assess_rows(model,labels,dev,raw['kinds'])}
            history.append(row);print(json.dumps(row),flush=True)
            atomic_json(out/'history.json',history)
            if row['anchor_policy_kl']>.1:raise RuntimeError('natural policy KL safety budget exceeded')
    meta.update(process_learning={'parent':identity['parent'],'dataset':identity['dataset_manifest'],
                                  'args':vars(args),'research_only':True})
    save_checkpoint(model,str(out/'model.pt'),meta=meta)
    torch.save({'schema':'hsai.process-bc-state/v1','optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,
                'updates':args.updates,'model':file_identity(out/'model.pt'),
                'note':'Not a PPO Trainer; PPO continuation must declare a fresh optimizer'},out/'bc_state.pt')
    restored,_=load_checkpoint(str(out/'model.pt'),device=args.device)
    with torch.no_grad():
        for a,b in zip(forward(model,labels,dev[:8]),forward(restored,labels,dev[:8])):
            if not torch.equal(a,b):raise RuntimeError('BC checkpoint roundtrip mismatch')
    final={'labels_dev':assess_rows(restored,labels,dev,raw['kinds']),
           'closed_loop_dev':closed_loop(restored,tasks,args.device,out/'after.jsonl.gz'),
           'model':file_identity(out/'model.pt'),'bc_state':file_identity(out/'bc_state.pt'),
           'roundtrip_exact':True}
    atomic_json(out/'result.json',final)
    print(json.dumps({'final':final['labels_dev'],'closed_loop':final['closed_loop_dev']['all']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--updates',type=int,default=200);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--anchor-kl',type=float,default=3.0);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--seed',type=int,default=1181802)
    run(p.parse_args())
