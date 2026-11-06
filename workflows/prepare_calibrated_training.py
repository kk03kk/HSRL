#!/usr/bin/env python3
"""性能收敛后的单次有界准备：新曲线库、战斗审计、动作预算和两轮完整状态PPO。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows.run_learning_experiment import read,write,digest,writer_lock,training_report
from workflows.prepare_opponent_research import MODELS,TRAINER_SHA,HUMAN_SHA,ensure_human_train


def exposure_summary(result):
    from collections import defaultdict,Counter
    groups=defaultdict(list)
    for seat in result['per_seat']:
        for row in seat.get('combat_exposure',[]):groups[int(row['turn'])].append(row)
    turns={}
    for turn,rows in sorted(groups.items()):
        turns[str(turn)]=dict(encounters=len(rows),mirrored=sum(bool(r['mirror_applied']) for r in rows),
            ghosts=sum(bool(r['ghost']) for r in rows),
            learner_score=sum(float(r['win'])+.5*float(r['tie']) for r in rows)/len(rows),
            sources=dict(Counter(r['source'] if r['mirror_applied'] else 'ghost' if r['ghost'] else 'fallback' for r in rows)))
    return dict(valid=result['valid'],mean_rank=result['mean_rank'],top4_rate=result['top4_rate'],
        win_rate=result['win_rate'],turns=turns,protocol=result['protocol'],raw_file='dev-exposure.json')


def pressure_summary(report):
    return [dict(turn=r['turn'],source=r['source'],human_games=r['human_games'],
        source_games=len({p['game'] for p in r['pairs']}),pairs=len(r['pairs']),
        original_score=sum(p['original']['score'] for p in r['pairs'])/len(r['pairs']),
        calibrated_score=sum(p['calibrated']['score'] for p in r['pairs'])/len(r['pairs'])) for r in report['results']]


def run(args):
    from workflows.build_human_curve_bank import build,pressure_audit
    from workflows.prepare_validated_library import prepare
    from workflows.run_overnight_learning import run_process
    from hsbg.az.train.personal import checked_library,training_source_digest
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('只在计算节点执行')
    import torch
    out=Path(args.out).resolve();parent=Path(args.parent).resolve()
    refiner=Path(args.refinement).resolve();perf=read(refiner/'report.json')
    if not perf['complete'] or not perf['ready_for_training']:raise ValueError('需要完成性能与恢复确认')
    config=read(refiner/'recommended.config.json')
    source=ROOT/MODELS['b10m'][0];trainer=source.with_name('trainer.pt')
    human=ROOT/'artifacts/libraries/firestone-20261102-v1/train.jsonl';ensure_human_train(human)
    if digest(source)!=MODELS['b10m'][1] or digest(trainer)!=TRAINER_SHA or digest(human)!=HUMAN_SHA:
        raise ValueError('冻结B10M完整状态/真人Train摘要不符')
    captured=read(parent/'capture-report.json')
    sources={split:{name:str(parent/'sources'/name/split) for name in (*MODELS,'bot')} for split in ('train','dev')}
    manifests={split:{name:checked_library(path) for name,path in values.items()} for split,values in sources.items()}
    for split in manifests:
        for name,m in manifests[split].items():
            if m['digest']!=captured['libraries'][name][split]:raise ValueError('源库与已完成采集不符')
    deadline=time.monotonic()+args.minutes*60
    def command(*argv):
        if deadline-time.monotonic()<90:raise TimeoutError('准备预算不足，保留阶段，不启动长训')
        run_process([sys.executable,*map(str,argv)],deadline)
    with writer_lock(out):
        plan=dict(schema='hsrl.calibrated_preparation/v1',source=training_source_digest(),
            scripts={name:digest(ROOT/'workflows'/name) for name in ('build_human_curve_bank.py','probe_action_budget.py',
                'prepare_calibrated_training.py','launch_nus_refinement.py')},
            parent_capture=digest(parent/'capture-report.json'),performance=digest(refiner/'report.json'),
            model=digest(source),trainer=digest(trainer),human_train=digest(human),
            sources={split:{k:v['digest'] for k,v in values.items()} for split,values in manifests.items()},
            config=config,runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,platform=sys.platform))
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('计划漂移，保留旧目录')
        write(out/'plan.json',plan)
        # 每个新分配实例先验证实际设备/同精度actor路径，不重跑全量测试或性能扫参。
        command('workflows/check_actor_tensor_path.py','--checkpoint',source,'--out',out/'actor-parity.json',
            '--device','auto','--require-gpus',2)
        for split in ('train','dev'):
            raw=out/'raw'/split;bank=out/'libraries'/split
            if not (raw/'manifest.json').exists():build(sources[split],human,raw,split)
            if not (bank/'manifest.json').exists():
                validation=prepare(raw,bank,derived_dev=split=='dev')
            if read(bank/'validation.json')['quarantined']:raise RuntimeError('校准后出现新无效帧，保留现场')
            if not (out/f'pressure-{split}.json').exists():
                pressure_audit(sources[split],bank,human,out/f'pressure-{split}.json',split)
        audit=out/'mirror-audit.json'
        if not audit.exists():
            command('workflows/mirror_hard_gates.py','--snaps',out/'libraries/train','--sample','random',
                '--seeds',16,'--min-injections',1,'--p-values',1,'--seat-values',7,
                '--baseline-games',16,'--workers',4,'--out',audit)
        if not read(audit)['zero_violations']:raise RuntimeError('新库完整对局工程门失败')
        # 只观察B10M在新Dev的真实遭遇压力；沿用历史greedy/40动作口径，不作棋力止训。
        exposure=out/'dev-exposure.json'
        if not exposure.exists():
            command('-m','hsbg.az.train.personal','eval','--checkpoint',source,'--library',out/'libraries/dev',
                '--device','cuda:1','--games',8,'--seed',64000000,'--exposure',1,'--out',exposure)
        from workflows.run_overnight_learning import check_eval
        check_eval(read(exposure),digest(source))
        protocol=read(exposure)['protocol']
        if protocol['library_digest']!=checked_library(out/'libraries/dev')['digest'] or protocol['seed']!=64000000:
            raise ValueError('真实遭遇评测缓存协议不符')
        if not (out/'action-budget/report.json').exists():
            command('workflows/probe_action_budget.py','--checkpoint',source,'--library',out/'libraries/train',
                '--out',out/'action-budget','--device','cuda:1','--games',64)
        action=read(out/'action-budget/report.json')
        if action['checkpoint']!=digest(source) or action['library']!=checked_library(out/'libraries/train')['digest']:
            raise ValueError('动作预算缓存模型/库摘要不符')
        next_config=dict(config,iterations=0,minutes=30,**{'max-actions-per-turn':80})
        write(out/'next-training.config.json',next_config)
        check_config=dict(next_config,iterations=2,minutes=12)
        write(out/'ppo-check.config.json',check_config)
        train=out/'ppo-check'
        initial=int(read(source.with_name('checkpoint.json'))['version'])
        if not (train/'summary.json').exists():
            current=int(read(train/'latest.json')['version']) if (train/'latest.json').exists() else initial
            remaining=initial+2-current
            if remaining>0:
                state=['--resume'] if current>initial else ['--checkpoint',source,'--initial-trainer-state',trainer]
                command('-m','hsbg.az.train.personal','train','--config',out/'ppo-check.config.json',
                    '--library',out/'libraries/train','--out',train,'--device','cuda:0','--run-iterations',remaining,*state)
        if int(read(train/'latest.json')['version'])<initial+2:raise RuntimeError('两轮完整状态PPO尚未完成')
        quality=training_report(train,out/'quality',read(train/'latest.json')['version'])
        if quality.get('unexpected_mirror_rejections'):raise RuntimeError('PPO注入非预期拒绝')
        if digest(source)!=plan['model'] or digest(trainer)!=plan['trainer']:raise RuntimeError('B10M原件改变')
        write(out/'report.json',dict(schema='hsrl.calibrated_preparation_result/v1',complete=True,
            performance_confirmed=perf['confirmation']['effective_samples_per_s'],
            libraries={s:checked_library(out/'libraries'/s)['digest'] for s in ('train','dev')},
            calibration={s:read(out/'raw'/s/'calibration.json') for s in ('train','dev')},
            pressure_files=[f'pressure-{s}.json' for s in ('train','dev')],action_budget=read(out/'action-budget/report.json'),
            pressure={s:pressure_summary(read(out/f'pressure-{s}.json')) for s in ('train','dev')},
            dev_exposure=exposure_summary(read(exposure)),
            ppo_quality=quality,ready_for_time_budgeted_training=True,long_training_submitted=False,
            next=dict(start='原B10M完整Trainer在新目录分叉，不从2轮资格模型择优',
                config='next-training.config.json',wall_hours=9,bank='libraries/train',
                note='下一步固定小时级预算，不因正常名次波动终止；先依据本次压力/动作报告固定正式计划。')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='artifacts/runs/nus-curriculum-001')
    p.add_argument('--parent',default='artifacts/runs/opponent-research-001')
    p.add_argument('--refinement',default='artifacts/runs/nus-refine-005');p.add_argument('--minutes',type=int,default=100)
    a=p.parse_args()
    if not 30<=a.minutes<=100:p.error('minutes必须30..100')
    run(a)
