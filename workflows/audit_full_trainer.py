"""核验完整Trainer的来源、卡表、Adam、调度器、RNG及真实前向；不更新权重。"""
import argparse, gc, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def audit(source, expected_version=None, expected_hashes=None, device='cpu'):
    import numpy as np
    import torch
    from hsbg.az.train.model import load_checkpoint
    from hsbg.az.train.entity_model import load_card_tables
    from hsbg.az.train.personal import fingerprint,sha256
    from hsbg.az.train.ppo import PPOConfig,forward_model,split_model_outputs
    from hsbg.az.train.ppo_trainer import PPOTrainer
    import hsbg_engine as E
    source=Path(source); hashes={name:sha256(source/name) for name in ('model.pt','trainer.pt')}
    if expected_hashes and hashes!=expected_hashes:raise ValueError('源权重SHA不符')
    model,meta=load_checkpoint(str(source/'model.pt'),device='cpu')
    state=torch.load(source/'trainer.pt',map_location='cpu',weights_only=False)
    if expected_version is not None and state['updates']!=expected_version:raise ValueError('Trainer版本不符')
    if state['updates']!=meta['trainer_updates']:raise ValueError('导出版本与Trainer不符')
    if state['model'].keys()!=model.state_dict().keys() or any(not torch.equal(v,state['model'][k]) for k,v in model.state_dict().items()):raise ValueError('model导出与Trainer不一致')
    for key in ('optimizer','scheduler','anchor_model','torch_rng','cuda_rng','python_rng','rng','bc_rng','dpo_rng','problem_rng','extra'):
        if key not in state:raise ValueError('缺少完整训练状态:'+key)
    count=0
    def finite(value):
        nonlocal count
        if isinstance(value,torch.Tensor):
            count+=1
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError('非有限张量')
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(tuple,list)):
            for v in value:finite(v)
    finite(state)
    for v in state['optimizer']['state'].values():
        if not bool((v['exp_avg_sq']>=0).all()):raise ValueError('Adam二阶矩非法')
    tables=load_card_tables()
    for key,table in [('card_static',tables.static),('card_text',tables.text)]:
        if not torch.equal(state['model'][key],torch.from_numpy(table)):raise ValueError('卡表buffer与当前引擎不符:'+key)
    cfg=PPOConfig(**state['config']);cfg.device=device;cfg.compile_train=0
    trainer=PPOTrainer(model,cfg,state=state)
    if trainer.scheduler.state_dict()!=state['scheduler'] or trainer.rng.bit_generator.state!=state['rng']:raise ValueError('调度器或采样RNG恢复不符')
    if random.getstate()!=state['python_rng'] or not torch.equal(torch.get_rng_state(),state['torch_rng']):raise ValueError('Python/Torch RNG恢复不符')
    if device.startswith('cuda'):
        if len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG设备数不同，需要显式迁移')
        if any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng'])):raise ValueError('CUDA RNG恢复不符')
    game=E.Game(seed=7302);obs=game.observe(0);mask=np.frombuffer(game.legal_mask(0),dtype=np.uint8).astype(bool)
    f=torch.from_numpy(np.frombuffer(obs[0],dtype=np.float32).copy()[None]).to(device)
    c=torch.from_numpy(np.frombuffer(obs[1],dtype=np.int32).copy().astype(np.int64)[None]).to(device)
    with torch.inference_mode():outputs=split_model_outputs(forward_model(model,f,c,torch.from_numpy(mask[None]).to(device)))
    if any(not bool(torch.isfinite(x).all()) for x in outputs if isinstance(x,torch.Tensor)):raise ValueError('前向非有限')
    steps=[float(v['step']) for v in state['optimizer']['state'].values()]
    return dict(passed=True,version=state['updates'],cum_decisions=state['extra']['cum_decisions'],sha256=hashes,parameters=sum(p.numel() for p in model.parameters()),adam_states=len(steps),adam_step_min=min(steps),adam_step_max=max(steps),actual_lr=trainer.optimizer.param_groups[0]['lr'],scheduler=state['scheduler'],finite_tensors=count,cuda_rng_states=len(state['cuda_rng']),cuda_rng_verified=device.startswith('cuda'),fingerprint=fingerprint(),device=device,config=state['config'],extra_keys=sorted(state['extra']),scope='full_state_restore_and_forward_no_optimizer_update')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--version',type=int);p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cpu');a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    from hsbg.az.train.personal import write_json
    result=audit(a.source,a.version,device=a.device);write_json(a.out,result);print(json.dumps(result,ensure_ascii=False))
