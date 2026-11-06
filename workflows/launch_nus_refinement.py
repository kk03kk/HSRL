#!/usr/bin/env python3
"""在Torch导入前，将可证明属于当前Slurm作业的数字MIG列表转换为UUID。"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def mig_mapping(text,env,driver_uuids=None):
    original=env.get('CUDA_VISIBLE_DEVICES','')
    tokens=original.split(',')
    devices=[];parent=None
    for line in text.splitlines():
        match=re.match(r'GPU \d+:.*\(UUID: (GPU-[^)]+)\)',line)
        if match:
            parent=match[1]
        match=re.match(r'\s+MIG .* Device\s+(\d+): \(UUID: (MIG-[^)]+)\)',line)
        if match:
            devices.append(dict(index=int(match[1]),uuid=match[2],parent=parent))
    if not env.get('SLURM_JOB_ID') or not original or len(tokens)!=len(set(tokens)):
        raise ValueError('需要Slurm分配及无重复的CUDA可见设备列表')
    if not devices:
        # 无MIG时同样核对原始Driver证明；GRES标签不能证明两个独立设备。
        if driver_uuids is None:
            raise ValueError('非MIG设备缺少原始CUDA Driver枚举证明')
        identities=[str(uuid.UUID(value)) for value in driver_uuids]
        if len(identities)!=len(tokens) or len(set(identities))!=len(identities):
            raise ValueError('非MIG原始CUDA Driver设备数量/唯一性不匹配；不可扩展可见设备')
        return original,dict(mode='unchanged-non-mig',original=original,selected=original,
            original_driver_instance_uuids=identities)
    available={d['uuid'] for d in devices}
    if all(t.startswith('MIG-') for t in tokens):
        if not set(tokens)<=available:
            raise ValueError('继承的MIG UUID未出现在当前设备清单')
        selected=original
    else:
        # 由原CVD下CUDA Driver的真实枚举解析；Device 0可在不同物理卡重复。
        # 不猜单卡/多卡、全局GRES、局部编号，也不从nvidia-smi补选设备。
        if not all(t.isdigit() for t in tokens) or driver_uuids is None:
            raise ValueError('数字MIG映射需要原始可见性下的CUDA Driver实例UUID')
        identities=[str(uuid.UUID(value)) for value in driver_uuids]
        if len(identities)!=len(tokens) or len(set(identities))!=len(identities):
            raise ValueError('原始CUDA Driver实例数量/唯一性不匹配；不可扩展可见设备')
        selected=','.join('MIG-'+value for value in identities)
        if not set(selected.split(','))<=available:
            raise ValueError('原始CUDA Driver实例UUID未出现在MIG清单中')
    return selected,dict(mode='driver-resolved-mig-uuid',original=original,selected=selected,
        original_driver_instance_uuids=driver_uuids,
        devices=devices,slurm_job_gpus=env.get('SLURM_JOB_GPUS'),job_id=env['SLURM_JOB_ID'])


def driver_identity():
    """v2 UUID识别MIG CI；普通device properties可能仅返回父物理卡UUID。"""
    lib=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
    class Uuid(ctypes.Structure):
        _fields_=[('data',ctypes.c_ubyte*16)]
    def call(name,args,types):
        fn=getattr(lib,name);fn.argtypes=types;fn.restype=ctypes.c_int
        code=fn(*args)
        if code:
            raise RuntimeError(f'{name}错误码{code}')
    call('cuInit',[0],[ctypes.c_uint])
    count=ctypes.c_int()
    call('cuDeviceGetCount',[ctypes.byref(count)],[ctypes.POINTER(ctypes.c_int)])
    identities=[]
    for index in range(count.value):
        dev=ctypes.c_int();value=Uuid()
        call('cuDeviceGet',[ctypes.byref(dev),index],[ctypes.POINTER(ctypes.c_int),ctypes.c_int])
        call('cuDeviceGetUuid_v2',[ctypes.byref(value),dev.value],[ctypes.POINTER(Uuid),ctypes.c_int])
        identities.append(str(uuid.UUID(bytes=bytes(value.data))))
    return identities


def verify_mig_identity(inventory,visibility):
    if not visibility.startswith('MIG-'):
        return
    expected=[str(uuid.UUID(t[4:])) for t in visibility.split(',')]
    actual=inventory.get('driver_instance_uuids',[])
    if actual!=expected or len(set(actual))!=len(actual):
        raise RuntimeError('CUDA Driver实际MIG实例UUID与分配列表不一致，禁止开始训练')


def visibility_evidence():
    """分别留存batch/step的真实可见性；不导入Torch，不改任何隔离变量。"""
    keys=('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','NVIDIA_VISIBLE_DEVICES','SLURM_JOB_ID',
          'SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_GPUS','SLURM_STEP_ID',
          'SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE','SLURM_GPUS','SLURM_GPUS_PER_NODE',
          'SLURM_GPUS_PER_TASK','SLURM_TRES_PER_NODE','SLURM_TRES_PER_TASK','SLURM_GPU_BIND')
    evidence=dict(job_id=os.environ.get('SLURM_JOB_ID','missing'),
                  original=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  allocation={k:os.environ.get(k) for k in keys})
    for name,flags in [('devices',['-L']),('topology',[
        '--query-gpu=uuid,name,mig.mode.current,memory.total,pci.bus_id,driver_version',
        '--format=csv,noheader'])]:
        try:
            result=subprocess.run(['nvidia-smi',*flags],capture_output=True,text=True,timeout=10)
            evidence['nvidia_smi' if name=='devices' else 'gpu_topology']=result.stdout
            evidence[name+'_command']=dict(returncode=getattr(result,'returncode',0),
                                           stderr=getattr(result,'stderr',''))
        except (OSError,subprocess.TimeoutExpired) as exc:
            evidence[name+'_error']=str(exc)
    try:
        evidence['original_driver_instance_uuids']=driver_identity()
    except Exception as exc:
        evidence['original_driver_error']=str(exc)
    try:
        evidence['process_cgroup']=Path('/proc/self/cgroup').read_text()
    except OSError:
        pass
    return evidence


def main():
    from workflows.run_learning_experiment import write
    tasks = {
        'refine': ('refine_nus_scaling.py', 'nus-refine-005'),
        'curriculum': ('prepare_calibrated_training.py', 'nus-curriculum-001'),
        'long': ('run_curriculum_long.py', 'nus-human-long-001'),
        'ppo-profile': ('profile_nus_ppo.py', 'nus-ppo-profile'),
        'pending-audit': ('audit_3702_recovery.py', 'nus-pending-audit-001'),
        'long-4159': ('run_4159_long.py', 'nus-4159-economy-long-002'),
        'p50-preflight': ('run_7357_preflight.py', 'p50-7357-preflight'),
        'p50-long': ('run_7357_long.py', 'p50-7357-long-001'),
    }
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=tasks, default='refine')
    p.add_argument('--out')
    p.add_argument('--probe-only',action='store_true',help='仅记录batch可见性，不授权训练或更改CVD')
    args = p.parse_args()
    runner, default_out = tasks[args.task]
    args.out = args.out or 'artifacts/runs/' + default_out
    out=Path(args.out);job=os.environ.get('SLURM_JOB_ID','missing')
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('CUDA设备检查必须在Slurm计算节点执行')
    evidence=visibility_evidence()
    # 独立留存原数字可见性下的Driver结果，区分Torch/NVML误计数与真实枚举不足。
    # 此进程不导入Torch；后继exec重新初始化CUDA并使用UUID。
    if args.probe_only:
        evidence.update(diagnostic_only=True,training_authorized=False)
        write(out/f'visibility-batch-{job}.json',evidence)
        print(f'[visibility:batch] driver={evidence.get("original_driver_instance_uuids")} '
              f'out={out}',flush=True)
        return
    inventory_failed=('devices_error' in evidence
                      or evidence.get('devices_command',{}).get('returncode',1) != 0)
    try:
        if inventory_failed:
            raise RuntimeError('nvidia-smi设备清单不可用，已保留诊断，禁止猜测GPU映射')
        selected,mapping=mig_mapping(evidence.get('nvidia_smi',''),os.environ,
                                     evidence.get('original_driver_instance_uuids'))
        if args.task in ('pending-audit','long-4159','p50-preflight','p50-long') and len(evidence.get('original_driver_instance_uuids', [])) != 2:
            raise ValueError('恢复工程验收需要两个Driver设备；当前作业步并未完整分配双GPU，禁止进入模型加载')
        evidence.update(mapping=mapping)
    except Exception as exc:
        evidence['error']=str(exc)
        evidence['error_category']=('gpu_inventory_unavailable' if inventory_failed
                                    else 'cuda_allocation_visibility_mismatch')
        write(out/f'visibility-{job}.json',evidence)
        raise
    write(out/f'visibility-{job}.json',evidence)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=selected,
        HSRL_ORIGINAL_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
    print('[visibility] '+json.dumps(mapping,ensure_ascii=False),flush=True)
    # exec为全新进程入口，Torch/模型/常驻worker均继承相同已核验的UUID列表。
    os.execve(sys.executable,[sys.executable,'-u',str(ROOT/'workflows'/runner),'--out',str(out)],env)


if __name__=='__main__':
    main()
