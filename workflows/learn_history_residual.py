"""Fixed-budget paired public-history ablation; base PPO remains frozen.

The saved residual and optimizer are research composite assets, not PPO Trainers.
All source splits remain the original teacher split. No endpoint selection.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from hsbg.az.train.history_residual import HistoryResidual,current_spec,save_adapter,load_adapter
from hsbg.az.train.public_turn_history import SCHEMA_SHA256 as HISTORY_SHA
from hsbg.az.train.eval_policy import make_eval_policy,EvalProtocol
from workflows.learn_payoff_process import mixed_event_sampling,validate_dataset


def validate_history(root,dataset,raw):
    root=Path(root);dataset=Path(dataset)
    result=json.loads((root/'complete.json').read_text());run=json.loads((root/'run.json').read_text())
    if not result['passed'] or result['probe'] or result['run']!=file_identity(root/'run.json'):
        raise ValueError('complete sealed history annotations required')
    if run['history_schema_sha256']!=HISTORY_SHA or run['original_manifest']!=file_identity(dataset/'manifest.json'):
        raise ValueError('history schema or original dataset differs')
    if result['history']!=file_identity(root/'history.npz'):raise ValueError('history payload changed')
    with np.load(root/'history.npz',allow_pickle=False) as h:
        for key in ('seeds','dev','turns'):
            if not np.array_equal(h[key],raw[key]):raise ValueError('history row alignment differs')
        if not np.array_equal(h['original_row'],np.arange(len(raw['dev']))):raise ValueError('history row index differs')
        marks=h['played_by_slot'].copy()
    if marks.shape!=(len(raw['dev']),7) or not np.isin(marks,(0,1)).all():raise ValueError('invalid public history marks')
    return marks


@torch.no_grad()
def assess(model,data,indices,reasons):
    correct=[];nll=[]
    for start in range(0,len(indices),512):
        ix=indices[start:start+512]
        p=data['base_logits'][ix]+model(data['features'][ix],data['cards'][ix],data['base_logits'][ix],data['history'][ix])
        lp=p.masked_fill(~data['masks'][ix],-1e9).log_softmax(-1)
        target=data['target'][ix]
        correct.extend((lp.argmax(-1)==target).cpu().tolist())
        nll.extend((-lp.gather(1,target[:,None]).squeeze(1)).cpu().tolist())
    correct=np.asarray(correct);nll=np.asarray(nll)
    def summary(sel):return {'rows':int(sel.sum()),'agreement':float(correct[sel].mean()),'nll':float(nll[sel].mean())}
    return {'all':summary(np.ones(len(indices),bool)),
        'reasons':{str(r):summary(reasons[indices]==r) for r in sorted(set(reasons[indices]))}}


def main(args):
    torch.set_num_threads(1)
    plan=json.loads(Path(args.plan).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    root=Path(plan['dataset']);history=Path(plan['history']);base=file_identity(plan['base'])
    if base['sha256']!=plan['base_sha256']:raise ValueError('frozen base differs')
    if file_identity(root/'manifest.json')['sha256']!=plan['dataset_manifest_sha256']:raise ValueError('dataset differs')
    admission=validate_dataset(root,plan['teacher_parent'])
    manifest=json.loads((root/'manifest.json').read_text())
    from hsbg.az.train import mechanism_obs as M
    if manifest.get('mechanism_schema_sha256')!=M.SCHEMA_SHA256:raise ValueError('dataset mechanism schema differs')
    for name,item in manifest['files'].items():
        if file_identity(root/name)!=item:raise ValueError('dataset payload changed')
    with np.load(root/'labels.npz',allow_pickle=False) as archive:raw={k:archive[k] for k in archive.files}
    marks=validate_history(history,root,raw)
    groups,sizes=mixed_event_sampling(raw)
    if sum(sizes.values())!=96 or any(raw['dev'][v].any() for v in groups.values()):raise ValueError('sampling split differs')
    source=source_identity(Path(__file__).resolve().parents[1])
    if source['tracked_dirty']:raise ValueError('committed source required')
    identity={'plan':file_identity(args.plan),'base':base,'source':source,'environment':environment_identity(),
        'dataset_manifest':file_identity(root/'manifest.json'),'history_complete':file_identity(history/'complete.json'),
        'history_payload':file_identity(history/'history.npz'),'admission':admission,'args':vars(args)}
    atomic_json(out/'run.json',identity)
    evaluator=make_eval_policy(plan['base'],protocol=EvalProtocol(),device=args.device)['policy'].evaluator
    if M.schema_of(evaluator.model)!=M.SCHEMA:raise ValueError('frozen base observation schema differs')
    for parameter in evaluator.model.parameters():parameter.requires_grad_(False)
    evaluator.model.eval();frozen={k:v.detach().cpu().clone() for k,v in evaluator.model.state_dict().items()}
    cached=[]
    for start in range(0,len(marks),256):
        logits,_=evaluator.logits(raw['features'][start:start+256],raw['cards'][start:start+256],raw['masks'][start:start+256])
        if not np.isfinite(logits).all():raise ValueError('nonfinite frozen logits')
        cached.append(logits)
    cached=np.concatenate(cached);np.save(out/'base_logits.npy',cached,allow_pickle=False)
    data={k:torch.as_tensor(raw[k],device=args.device,dtype=dtype) for k,dtype in
        (('features',torch.float32),('cards',torch.long),('masks',torch.bool))}
    data.update(base_logits=torch.as_tensor(cached,device=args.device),history=torch.as_tensor(marks,device=args.device),
        target=torch.as_tensor(raw['targets'].argmax(-1),device=args.device))
    if not (raw['targets'].sum(-1)==1).all() or np.any(raw['targets']&~raw['masks']):raise ValueError('invalid targets')
    results={};initial=None
    for arm,use_history in (('no_history',False),('history',True)):
        dest=out/arm;dest.mkdir();torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
        model=HistoryResidual(current_spec(use_history)).to(args.device)
        if initial is None:initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):raise ValueError('paired initial weights differ')
        optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate']);started=time.monotonic()
        for step in range(1,plan['updates']+1):
            ix=np.concatenate([rng.choice(v,sizes[k],replace=True) for k,v in groups.items()])
            logits=data['base_logits'][ix]+model(data['features'][ix],data['cards'][ix],data['base_logits'][ix],data['history'][ix])
            loss=torch.nn.functional.cross_entropy(logits.masked_fill(~data['masks'][ix],-1e9),data['target'][ix])
            if not torch.isfinite(loss):raise ValueError('nonfinite residual loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step==1 or step%500==0:
                row={'arm':arm,'update':step,'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':time.monotonic()-started}
                atomic_json(dest/'checks'/f'{step:06d}.json',row);print(json.dumps(row),flush=True)
        model.eval();save_adapter(dest/'model.pt',model,base,{'identity':file_identity(out/'run.json'),'updates':plan['updates']})
        torch.save({'schema':'hsai.history-residual-bc-state/v1','optimizer':optimizer.state_dict(),
            'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'updates':plan['updates'],
            'model':file_identity(dest/'model.pt'),'note':'Not a PPO Trainer; base is separately frozen'},dest/'bc_state.pt')
        restored=load_adapter(dest/'model.pt',base,args.device);ix=np.arange(16)
        with torch.no_grad():
            if not torch.equal(model(data['features'][ix],data['cards'][ix],data['base_logits'][ix],data['history'][ix]),
                    restored(data['features'][ix],data['cards'][ix],data['base_logits'][ix],data['history'][ix])):raise ValueError('roundtrip differs')
        result={'model':file_identity(dest/'model.pt'),'bc_state':file_identity(dest/'bc_state.pt'),'roundtrip_exact':True,
            'parameters':sum(p.numel() for p in model.parameters()),'updates':plan['updates'],
            'train':assess(restored,data,np.flatnonzero(~raw['dev']),raw['reasons']),
            'dev':assess(restored,data,np.flatnonzero(raw['dev']),raw['reasons'])}
        results[arm]=result;atomic_json(dest/'result.json',result)
    if any(not torch.equal(v.cpu(),frozen[k]) for k,v in evaluator.model.state_dict().items()):raise ValueError('frozen model changed')
    if file_identity(plan['base'])!=base:raise ValueError('base file changed')
    atomic_json(out/'complete.json',{'passed':True,'run':file_identity(out/'run.json'),'arms':results,
        'frozen_base_exact':True,'matched_initial_weights':True,'cached_logits':file_identity(out/'base_logits.npy'),
        'scope':'Fixed endpoints; label fit is not closed-loop ability; composite requires frozen base plus adapter'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())
