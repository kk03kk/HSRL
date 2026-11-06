#!/usr/bin/env python3
"""有界单/双卡扩展基准：真实完整状态分叉，按有效处理样本吞吐择优，不启动长训。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_learning_experiment import read, write, digest, writer_lock, training_report
from workflows.prepare_opponent_research import MODELS, TRAINER_SHA


def configurations(base, count, *, cpu_budget=32):
    """保持有效minibatch1024、epochs1、全部FP32 records；只扩大并发和计算批量。"""
    common = dict(base)
    common.update(iterations=8, minutes=10, epochs=1, **{'snapshot-every':1,'keep-snapshots':3,
        'records-dtype':'fp32','leaf-amp':'off','anchor-amp':0,'pipeline-guard':'abort',
        'leaf-wait-ms':0,'workers':1,'threads':1})
    arms = []
    def add(name, **overrides):
        cfg = dict(common, **overrides)
        arms.append(dict(name=name, config=cfg))
    add('baseline-fp32-16', games=16, processes=1, pipeline=0, amp='off',
        minibatch=256, **{'grad-accum':4,'local-batch-inference':1})
    add('single-bf16-64', games=64, processes=1, pipeline=0, amp='bf16',
        minibatch=1024, **{'grad-accum':1,'local-batch-inference':1,'anchor-amp':1})
    actors = ','.join(str(i) for i in range(1, count)) if count > 1 else '0'
    # 先用更大的worker内batch摊薄前向成本，再试更多CPU并发；最多16个CUDA上下文。
    for workers, games in ((2,128), (4,256), (8,512), (12,384), (16,512)):
        if workers > max(2, cpu_budget//2):
            continue
        add(f'pool-{workers}-games-{games}', games=games, processes=workers,
            pipeline=1, amp='bf16', minibatch=1024, **{'grad-accum':1,
                'local-batch-inference':0,'gpus':actors,'gpu-index-space':'visible','ipc-mode':'shm','anchor-amp':1})
    return arms


def measure(train, warmup=2):
    from workflows.report_personal_run import summarize
    records = [json.loads(line) for line in (Path(train)/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    records = [r for r in records if r.get('iter') is not None]
    if len(records) <= warmup:
        raise RuntimeError('完成轮数不足，不能把预热当稳态吞吐')
    report = summarize(records[warmup:])
    t = report['totals']
    if t['iteration_s'] <= 0 or t['processed_samples'] <= 0:
        raise RuntimeError('没有有效PPO更新样本，不能推荐该配方')
    if report['unexpected_mirror_rejections'] or any(t.get(k, 0) for k in
            ('illegal_actions','truncated_games','mirror_errors','mirror_pool_unbalanced','mirror_pool_species_unbalanced')):
        raise RuntimeError('扩展基准工程质量失败')
    pipeline = [r.get('pipeline', {}) for r in records]
    if any(p.get('lag', 0) > 1 or p.get('guard_tripped', False) for p in pipeline):
        raise RuntimeError('流水线版本/比率异常')
    report['effective_samples_per_s'] = min(t['processed_samples'], t['decisions'])/t['iteration_s']
    report['processed_fraction'] = t['processed_samples']/t['decisions']
    report['checkpoint_seconds'] = sum(r['profile'].get('checkpoint_save_publish_s', 0) for r in records[warmup:])
    report['warmup_iterations'] = warmup
    return report


def execute(argv, log, deadline, telemetry=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    remaining = deadline-time.monotonic()
    if remaining < 60:
        raise TimeoutError('扩展基准预算耗尽')
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(list(map(str, argv)), cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
        monitor=None
        if telemetry is not None:
            from workflows.scaling_telemetry import ResourceMonitor
            monitor=ResourceMonitor(process.pid,telemetry,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
            monitor.start()
        try:
            code = process.wait(timeout=remaining-15)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError('基准进程组超过预算，保留已提交状态')
        finally:
            if monitor:
                monitor.stop()
                write(Path(telemetry).with_suffix('.summary.json'),monitor.summary())
    if code:
        text = log.read_text(encoding='utf-8', errors='replace')
        if any(key in text for key in ('采集质量门失败','PPO非有限','流水线滞后异常',
                                      '流水线策略比率越界','batch invalid','non-finite','FloatingPointError')):
            raise RuntimeError(f'基准工程质量失败，禁止按OOM跳过：{log}')
        if 'CUDA out of memory' in text or 'torch.OutOfMemoryError' in text:
            return False
        raise RuntimeError(f'基准执行/工程失败，不能当性能不足跳过：{log}，退出{code}')
    return True


def run(args):
    import torch
    from hsbg.az.train.personal import checked_library, training_source_digest, validate_config
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count < args.require_gpus:
        raise RuntimeError(f'需要至少{args.require_gpus}个可见CUDA设备，实际{count}；保留Slurm隔离，不使用物理卡号')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('仅接受Slurm已分配GPU的计算节点')
    out = Path(args.out).resolve()
    source = ROOT/MODELS['b10m'][0]
    trainer = source.with_name('trainer.pt')
    if digest(source) != MODELS['b10m'][1] or digest(trainer) != TRAINER_SHA:
        raise ValueError('B10M原件摘要不符')
    library = Path(args.library).resolve()
    manifest = checked_library(library)
    if manifest['split'] != 'train' or not manifest.get('validation', {}).get('accepted'):
        raise ValueError('必须用已验证Train库，不读取Dev/Test作优化目标')
    base = read(args.config)
    cpu_budget = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    arms = configurations(base, count, cpu_budget=cpu_budget)
    deadline = time.monotonic()+args.minutes*60
    hardware = [dict(index=i,name=torch.cuda.get_device_properties(i).name,
        sm=torch.cuda.get_device_properties(i).multi_processor_count,
        memory=torch.cuda.get_device_properties(i).total_memory) for i in range(count)]
    with writer_lock(out):
        plan = dict(schema='hsrl.scaling_plan/v1',source=training_source_digest(), runner=digest(__file__),
            model=digest(source),trainer=digest(trainer),library=manifest['digest'],arms=arms,
            torch=torch.__version__,cuda=torch.version.cuda,hardware=hardware,cpu_budget=cpu_budget)
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise ValueError('扩展计划/硬件漂移，请使用新目录')
        write(out/'plan.json',plan)
        results = []
        def one(arm):
            name, cfg = arm['name'], arm['config']
            validate_config(cfg)
            root = out/name
            done = root/'measurement.json'
            if done.exists():
                return read(done)
            config = root/'config.json'
            write(config,cfg)
            train = root/'train'
            state = ['--resume'] if (train/'latest.json').exists() else [
                '--checkpoint',source,'--initial-trainer-state',trainer]
            start = time.monotonic()
            ok = execute([sys.executable,'-m','hsbg.az.train.personal','train','--config',config,
                '--library',library,'--out',train,'--device','cuda:0',*state],root/'run.log',deadline)
            if not ok:
                result = dict(name=name,capacity_failure='CUDA OOM',eligible=False)
            else:
                quality = training_report(train,root,read(train/'latest.json')['version'])
                if quality.get('unexpected_mirror_rejections'):
                    raise RuntimeError('采集非预期拒绝，停止扩展基准')
                if len(quality['iterations']) < 6:
                    result = dict(name=name,eligible=False,capacity_failure='10分钟窗口内不足6轮',
                        quality=quality,process_seconds=time.monotonic()-start)
                    write(done,result)
                    return result
                report = measure(train)
                result = dict(name=name,eligible=True,config=cfg,measurement=report,
                    process_seconds=time.monotonic()-start)
            write(done,result)
            return result
        capacity_hit = False
        stop_reason = 'sweep_complete'
        for arm in arms:
            if deadline-time.monotonic() < 600:
                stop_reason = 'time_budget'
                break  # 留出最终确认窗口，已经完成的有效基准仍报告
            if capacity_hit and arm['config']['processes'] > 1:
                continue
            result = one(arm)
            results.append(result)
            if not result['eligible']:
                capacity_hit = True
                stop_reason = 'capacity'
            write(out/'progress.json',dict(results=results))
            # 连续两次扩展未超过之前峰值5%时不再加上下文；不是棋力止训。
            pools = [r for r in results if r.get('eligible') and r['config']['processes'] > 1]
            if len(pools) >= 3:
                rates = [r['measurement']['effective_samples_per_s'] for r in pools]
                if max(rates[-2:]) <= max(rates[:-2])*1.05:
                    stop_reason = 'plateau'
                    break
        valid = [r for r in results if r.get('eligible')]
        if not valid:
            raise RuntimeError('没有通过质量门的性能样本')
        winner = max(valid,key=lambda r:r['measurement']['effective_samples_per_s'])
        confirm = None
        if deadline-time.monotonic() >= 600:
            confirm = one(dict(name='confirmation',config=winner['config']))
            if not confirm.get('eligible'):
                raise RuntimeError('最优配置确认出现资源不足')
        if digest(source) != plan['model'] or digest(trainer) != plan['trainer']:
            raise RuntimeError('源模型/完整Trainer被改变')
        recommendation = dict(winner['config'],iterations=0,minutes=30)
        baseline = next((r for r in valid if r['name'] == 'baseline-fp32-16'),None)
        write(out/'recommended.config.json',recommendation)
        report = dict(schema='hsrl.scaling_result/v1',complete=True,hardware=hardware,
            results=results,winner=winner['name'],confirmation=confirm,
            stop_reason=stop_reason,tested_arms=len(results),planned_arms=len(arms),
            recommended_gpu_count=1 if winner['config']['processes'] == 1 else count,
            speedup_vs_baseline=winner['measurement']['effective_samples_per_s']/baseline['measurement']['effective_samples_per_s']
                if baseline else None,
            recommended_config_sha256=digest(out/'recommended.config.json'),long_training_submitted=False,
            note='按有效处理样本/含保存迭代秒择优；不因棋力波动终止。批量/并行/精度变化为新实验，非原轨迹逐位恢复。')
        write(out/'report.json',report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='artifacts/runs/nus-scaling-001')
    p.add_argument('--library',default='artifacts/runs/nus-overnight-001/libraries/train-x1.5')
    p.add_argument('--config',default='artifacts/runs/nus-sustained-001/train.config.json')
    p.add_argument('--require-gpus',type=int,default=2)
    p.add_argument('--minutes',type=int,default=75)
    a = p.parse_args()
    if not 1 <= a.minutes <= 90 or not 1 <= a.require_gpus <= 8:
        p.error('minutes 1..90，require-gpus 1..8')
    run(a)
