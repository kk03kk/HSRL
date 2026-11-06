#!/usr/bin/env python3
"""固定512局的有界精调：并发、设备端采样、BF16及完整恢复确认。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows.run_learning_experiment import read,write,digest,writer_lock,training_report,freeze_checkpoint
from workflows.benchmark_nus_scaling import execute
from workflows.prepare_opponent_research import MODELS,TRAINER_SHA


def recipes(base):
    common=dict(base,games=512,iterations=8,minutes=12)
    common.update(**{'leaf-fused':0,'leaf-amp':'off','pipeline':1,'gpus':'1','processes':8})
    return [dict(name=name,config=dict(common,**extra)) for name,extra in [
        ('control-p8',{}),('matched-p12',dict(processes=12)),
        ('tensor-p8',{'leaf-fused':1}),('tensor-bf16-p8',{'leaf-fused':1,'leaf-amp':'bf16'})]]


def records(train):
    p=Path(train)/'history.jsonl'
    return [r for line in p.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line)).get('iter') is not None] if p.exists() else []


def details(rows):
    if not rows:return dict(available=False,note='缺少原始history，不能把缺测当0开销')
    profile={k:sum(float(r.get('profile',{}).get(k,0)) for r in rows)
        for k in ('collect_s','ppo_s','parent_read_s','build_batch_s','checkpoint_save_publish_s')}
    worker={}
    for key in ('collect_s','save_s','reload_s','forward_s','inference_total_s'):
        vals=[float(w.get(key,0)) for r in rows for w in r.get('collect',{}).get('workers',[])]
        worker[key]=dict(sum=sum(vals),max=max(vals,default=0))
    ratios=[r.get('pipeline',{}).get('pre_ratio_mean',1) for r in rows]
    return dict(profile=profile,worker_service=worker,pre_ratio_range=[min(ratios,default=1),max(ratios,default=1)],
        learner_peak_bytes=max((r.get('learner_gpu_memory',{}).get('peak_allocated_bytes',0) for r in rows),default=0),
        actor_single_process_peak_bytes=max((w.get('cuda_peak_allocated_bytes',0) for r in rows for w in r.get('collect',{}).get('workers',[])),default=0),
        note='worker sum跨并行进程，不能除以wall当GPU利用率。tensor路径forward_s为异步启动，推理比较用inference_total_s或采集wall。')


def preserve_first_segment(train,segment):
    # 已完成16轮后重启控制器时，仍验证原8轮副本，不用最新进度改写它。
    record=segment/'checkpoint.json'
    target=read(record)['target_decisions'] if record.exists() else read(train/'latest.json')['cum_decisions']
    return freeze_checkpoint(train,segment,target)


def run(args):
    from hsbg.az.train.personal import checked_library,training_source_digest,validate_config
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('需要Slurm计算节点与继承的GPU隔离')
    parent=Path(args.parent).resolve();out=Path(args.out).resolve()
    previous=read(parent/'report.json')
    if not previous['complete']:
        raise ValueError('父性能测试没有完成')
    base=read(parent/'recommended.config.json')
    if base.get('games')!=512 or base.get('processes')!=8:
        raise ValueError('此精调以已确认8进程512局配方为控制')
    source=ROOT/MODELS['b10m'][0];trainer=source.with_name('trainer.pt')
    if digest(source)!=MODELS['b10m'][1] or digest(trainer)!=TRAINER_SHA:
        raise ValueError('B10M原件摘要不同')
    initial_version=int(read(source.with_name('checkpoint.json'))['version'])
    library=Path(args.library).resolve();manifest=checked_library(library)
    if manifest['split']!='train' or not manifest.get('validation',{}).get('accepted'):
        raise ValueError('需要验证过的Train控制库')
    deadline=time.monotonic()+args.minutes*60
    with writer_lock(out):
        probe=out/f"actor-parity-{os.environ['SLURM_JOB_ID']}.json"
        if not probe.exists():
            execute([sys.executable,'workflows/check_actor_tensor_path.py','--checkpoint',source,
                '--device','auto','--require-gpus','2','--out',probe],out/'actor-parity.log',deadline)
        parity=read(probe)
        if not parity['passed'] or len(parity['hardware'])!=2 or parity['checkpoint_sha256']!=digest(source):
            raise RuntimeError('新旧actor或设备检查不通过，此有界配方需要恰好2个分配实例')
        plan=dict(schema='hsrl.refinement_plan/v1',source=training_source_digest(),
            scripts={name:digest(ROOT/'workflows'/name) for name in ('refine_nus_scaling.py','scaling_telemetry.py',
                'check_actor_tensor_path.py','benchmark_nus_scaling.py','launch_nus_refinement.py')},
            parent_report=digest(parent/'report.json'),model=digest(source),trainer=digest(trainer),
            library=manifest['digest'],base=base,cpu_budget=int(os.environ.get('SLURM_CPUS_PER_TASK','1')),
            hardware=parity['hardware'],driver_instance_uuids=parity.get('driver_instance_uuids',[]),
            runtime=dict(torch=parity['torch'],cuda=parity['cuda']))
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:
            raise ValueError('精调计划漂移，请用新目录保留旧证据')
        write(out/'plan.json',plan)
        write(out/'prior-details.json',{r['name']:details(records(parent/r['name']/'train')[2:])
            for r in previous['results']+[previous['confirmation']] if r})
        def one(arm,target=8):
            name,cfg=arm['name'],arm['config'];validate_config(cfg)
            root=out/name;train=root/'train';done=root/f'measurement-{target}.json'
            if done.exists():
                return read(done)
            config=root/'config.json';write(config,cfg)
            current=int(read(train/'latest.json')['version']) if (train/'latest.json').exists() else initial_version
            remaining=initial_version+target-current
            if remaining>0:
                before=len(records(train));attempts=read(root/'attempts.json') if (root/'attempts.json').exists() else []
                attempts.append(dict(state='running',before_records=before))
                write(root/'attempts.json',attempts)
                state=['--resume'] if current!=initial_version else ['--checkpoint',source,'--initial-trainer-state',trainer]
                started=time.monotonic();status='failed'
                try:
                    ok=execute([sys.executable,'-m','hsbg.az.train.personal','train','--config',config,
                        '--library',library,'--out',train,'--device','cuda:0','--run-iterations',min(8,remaining),*state],
                        root/f'run-{target}.log',deadline,telemetry=root/f'telemetry-{target}.jsonl')
                    status='success' if ok else 'CUDA OOM'
                    if not ok:
                        value=dict(name=name,eligible=False,capacity_failure='CUDA OOM')
                        write(done,value);return value
                finally:
                    attempts[-1].update(state=status,seconds=time.monotonic()-started)
                    write(root/'attempts.json',attempts)
            if int(read(train/'latest.json')['version'])<initial_version+target:
                raise RuntimeError('没有完成有界验证轮数；保留现场，可同源码恢复')
            full=training_report(train,root/'quality',read(train/'latest.json')['version'])
            if full.get('unexpected_mirror_rejections'):
                raise RuntimeError('工程质量门失败')
            rows=records(train);attempts=read(root/'attempts.json')
            excludes={i for a in attempts for i in (a['before_records'],a['before_records']+1)}
            steady=[r for i,r in enumerate(rows) if i not in excludes]
            if len(steady)<4:
                raise RuntimeError('稳态记录不足，不能宣布性能收益')
            from workflows.report_personal_run import summarize
            stats=summarize(steady);t=stats['totals']
            score=min(t['processed_samples'],t['decisions'])/t['iteration_s']
            if score<=0:
                raise RuntimeError('没有实际PPO更新')
            active=2 if cfg['pipeline'] else 1
            measured_wall=sum(a.get('seconds',0) for a in attempts)
            all_known=all(a['state']=='success' for a in attempts)
            value=dict(name=name,eligible=True,config=cfg,measurement=stats,effective_samples_per_s=score,
                processed_fraction=t['processed_samples']/t['decisions'],detail=details(steady),
                active_gpu_hours_per_million=active*1e6/score/3600,
                reserved_gpu_hours_per_million=len(parity['hardware'])*1e6/score/3600,
                full_process_effective_samples_per_s=full['totals']['processed_samples']/measured_wall if all_known and measured_wall else None,
                process_seconds=measured_wall,active_gpu_count=active,
                resource=[read(p) for p in sorted(root.glob('telemetry-*.summary.json'))])
            write(done,value);return value
        results=[]
        for arm in recipes(base):
            if deadline-time.monotonic()<1200:
                break
            result=one(arm);results.append(result);write(out/'progress.json',dict(results=results))
            if not result['eligible']:
                break
        valid=[r for r in results if r.get('eligible')]
        p8=[r for r in valid if r['config']['processes']==8]
        if not p8:
            raise RuntimeError('没有有效控制组')
        best_backend=max(p8,key=lambda r:r['effective_samples_per_s'])
        candidates=[r for r in valid if r['config']['processes']!=8]
        if candidates and deadline-time.monotonic()>1500:
            candidate=max(candidates,key=lambda r:r['effective_samples_per_s'])
            if candidate['effective_samples_per_s']>p8[0]['effective_samples_per_s']*1.05:
                combined=dict(candidate['config'],**{'leaf-fused':best_backend['config']['leaf-fused'],
                                                      'leaf-amp':best_backend['config']['leaf-amp']})
                result=one(dict(name='combined',config=combined));results.append(result)
                if result.get('eligible'):valid.append(result)
        winner=max(valid,key=lambda r:r['effective_samples_per_s'])
        confirmation=None
        if deadline-time.monotonic()>900:
            arm=dict(name='confirmation',config=winner['config'])
            first=one(arm,8)
            if not first.get('eligible'):raise RuntimeError('确认配置容量失败')
            preserve_first_segment(out/'confirmation/train',out/'confirmation/segment-1')
            confirmation=one(arm,16)
        if digest(source)!=plan['model'] or digest(trainer)!=plan['trainer']:
            raise RuntimeError('源模型被修改')
        recommended=dict(winner['config'],iterations=0,minutes=30)
        write(out/'recommended.config.json',recommended)
        write(out/'report.json',dict(schema='hsrl.refinement_result/v1',complete=True,
            hardware=parity['hardware'],results=results,winner=winner['name'],confirmation=confirmation,
            device_proof=dict(driver_instance_uuids=parity.get('driver_instance_uuids',[]),
                visibility=parity['visibility'],checked_device=parity.get('checked_device')),
            recommended_gpu_count=winner['active_gpu_count'],
            gain_vs_current_control=winner['effective_samples_per_s']/p8[0]['effective_samples_per_s']-1,
            confirmation_speed_retention=confirmation['effective_samples_per_s']/winner['effective_samples_per_s']
                if confirmation and confirmation.get('eligible') else None,
            further_sweep_recommended=False,ready_for_training=bool(confirmation and confirmation.get('eligible')),
            long_training_submitted=False,
            note='MIG GPU利用率若N/A则未知；双卡同时改变批量和流水线，非逐位等价。性能优化到此原则上收敛，后继用已确认配方。'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',default='artifacts/runs/nus-scaling-001')
    p.add_argument('--out',default='artifacts/runs/nus-refine-005')
    p.add_argument('--library',default='artifacts/runs/nus-overnight-001/libraries/train-x1.5')
    p.add_argument('--minutes',type=int,default=60);a=p.parse_args()
    if not 30<=a.minutes<=90:p.error('minutes必须30..90')
    run(a)
