#!/usr/bin/env python3
"""实际硬件上比较同精度新旧actor：动作、行为logprob、value必须一致。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def device_inventory(torch):
    """先初始化CUDA运行时再枚举；Slurm的GRES数量不保证等于进程可用设备数。"""
    hardware=[];errors=[];reported=0
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
            reported=torch.cuda.device_count()
            for i in range(reported):
                p=torch.cuda.get_device_properties(i)
                hardware.append(dict(index=i,name=p.name,sm=p.multi_processor_count,
                    memory=p.total_memory,uuid=str(getattr(p,'uuid',''))))
    except Exception as exc:
        errors.append(str(exc))
    raw={}
    for name,flags in [('devices',['-L']),('driver',['--query-gpu=uuid,name,driver_version,mig.mode.current','--format=csv'])]:
        try:
            result=subprocess.run(['nvidia-smi',*flags],capture_output=True,text=True,timeout=5)
            raw[name]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raw[name]=dict(error=str(exc))
    identities=[]
    if hardware and os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('MIG-'):
        from workflows.launch_nus_refinement import driver_identity
        try:
            identities=driver_identity()
        except Exception as exc:
            errors.append(str(exc))
    return dict(hardware=hardware,reported_cuda_count=reported,accessible_cuda_count=len(hardware),
        driver_instance_uuids=identities,
        errors=errors,torch=torch.__version__,cuda=torch.version.cuda,nvidia_smi=raw,
        allocation={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','HSRL_ORIGINAL_CUDA_VISIBLE_DEVICES','SLURM_JOB_ID',
            'SLURM_JOB_NODELIST','SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE')})


def select_device(device,count,required=0):
    if required<0:
        raise ValueError('require-gpus必须非负')
    if count<required:
        raise RuntimeError(f'CUDA设备可见性不满足要求：需要至少{required}个，实际可访问{count}个；'
            'GRES记账不等于CUDA枚举，查看*.devices.json。禁止取消Slurm隔离或假冒双设备运行。')
    if device=='cpu':
        return device
    if device=='auto':
        if not count:
            raise RuntimeError('CUDA没有可访问设备，查看*.devices.json')
        return 'cuda:1' if count>=2 else 'cuda:0'
    if device=='cuda':
        device='cuda:0'
    if not device.startswith('cuda:') or not device[5:].isdigit() or int(device[5:])>=count:
        raise ValueError(f'检查设备{device}超出可访问CUDA范围（数量{count}），尚未加载模型')
    return device


def check(checkpoint,out,device='auto',require_gpus=0):
    import torch
    from workflows.run_learning_experiment import write,digest
    inventory=device_inventory(torch)
    diagnostic=Path(out).with_suffix('.devices.json')
    write(diagnostic,inventory)
    print(f'[devices] accessible={inventory["accessible_cuda_count"]} diagnostic={diagnostic}',flush=True)
    device=select_device(device,inventory['accessible_cuda_count'],require_gpus)
    if device.startswith('cuda'):
        from workflows.launch_nus_refinement import verify_mig_identity
        verify_mig_identity(inventory,os.environ.get('CUDA_VISIBLE_DEVICES',''))
        if require_gpus:
            allocated=[]
            try:
                for i in range(require_gpus):
                    value=torch.tensor([float(i+1)],device=f'cuda:{i}')
                    if float((value*2).cpu()[0])!=2*(i+1):
                        raise RuntimeError(f'cuda:{i}分配/计算校验失败')
                    allocated.append(i)
            finally:
                inventory['allocation_probe_devices']=allocated
                write(diagnostic,inventory)
    import hsbg_engine as E
    from hsbg.az.train.model import load_checkpoint
    from hsbg.az.train.common import obs_arrays,mask_from_bytes
    from hsbg.az.train.batch_eval import BatchLeafServer
    from workflows.mirror_hard_gates import _drive_to_turn
    model,meta=load_checkpoint(checkpoint,device=device)
    rows=[]
    for turn in (1,8,12):
        game=_drive_to_turn(E.Game(seed=45000000+turn),turn)
        for seat in range(8):
            mask=mask_from_bytes(game.legal_mask(seat))
            if mask.any():
                f,c=obs_arrays(game.observe(seat));rows.append((f,c,mask))
    if not rows:
        raise RuntimeError('缺合法观测，不能空测通过')
    old=BatchLeafServer(model,device=device,max_wait_ms=0,norm_mean=meta.get('norm_mean'),norm_std=meta.get('norm_std'))
    new=BatchLeafServer(model,device=device,max_wait_ms=0,fused_act=True,norm_mean=meta.get('norm_mean'),norm_std=meta.get('norm_std'))
    checks=[]
    try:
        for amp in (('off','bf16') if device.startswith('cuda') else ('off',)):
            old.evaluator.amp=new.evaluator.amp=amp
            for n in (1,8,32,64):
                f,c,m=(np.stack([rows[i%len(rows)][k] for i in range(n)]) for k in range(3))
                for bias_on in (False,True):
                    bias=np.random.default_rng(42).normal(0,.1,m.shape).astype(np.float32) if bias_on else None
                    a=old.act_batch(f,c,m,seed=42,temperature=.7,bias=bias)
                    b=new.act_batch(f,c,m,seed=42,temperature=.7,bias=bias)
                    if not np.array_equal(a[0],b[0]) or not np.allclose(a[1],b[1],atol=1e-6,rtol=0) or not np.allclose(a[2],b[2],atol=1e-6,rtol=0):
                        raise RuntimeError('actor动作/行为logprob/value不一致，禁止进入PPO基准')
                    checks.append(dict(amp=amp,rows=n,bias=bias_on,max_logp_error=float(np.max(np.abs(a[1]-b[1])))))
    finally:
        old.close();new.close()
    hardware=inventory['hardware'] if device.startswith('cuda') else []
    write(out,dict(passed=True,checkpoint_sha256=digest(checkpoint),checks=checks,hardware=hardware,
        torch=torch.__version__,cuda=torch.version.cuda,visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checked_device=device,driver_instance_uuids=inventory.get('driver_instance_uuids',[])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--require-gpus',type=int,default=0);a=p.parse_args()
    check(a.checkpoint,a.out,a.device,a.require_gpus)
