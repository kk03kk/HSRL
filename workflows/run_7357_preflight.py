"""一次GPU工程验收：完整状态→更新→保存恢复→显式课程换库→继续更新。

``--profile`` 选择用哪一版长训入口的 ``inputs()``/``defaults()``：

- ``long``（默认）：上一轮 P50 四阶段（``run_7357_long``）；
- ``forced``：本轮强制升本（``run_forced_tier_long``），并额外验收强制升本的
  执行统计（``forced_tier`` 段必须出现，且 5 本目标 on_time 占比达标）。
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from workflows.run_7357_long import MODEL_SHA,TRAINER_SHA,read

RUNNERS={'long':'workflows/run_7357_long.py','forced':'workflows/run_forced_tier_long.py'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=sorted(RUNNERS),default='long')
    from workflows.run_7357_long import defaults as long_defaults
    p=long_defaults(p);p.add_argument('--out',type=Path,default='artifacts/runs/p50-7357-preflight');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('GPU计算节点专用')
    if a.out.exists() and any(not f.name.startswith('visibility-') for f in a.out.iterdir()):p.error('预检输出必须新目录，保留上次证据')
    a.preflight_mode=True
    # 按 profile 选择来源/配方校验逻辑（forced 版不再要求"配方等于上一轮"）
    if a.profile=='forced':
        from workflows.run_forced_tier_long import inputs
    else:
        from workflows.run_7357_long import inputs
    from hsbg.az.train.personal import write_json,fingerprint,sha256,training_source_digest
    from workflows.run_overnight_learning import run_process
    from workflows.run_learning_experiment import training_report
    from workflows.audit_full_trainer import audit
    from workflows.report_firestone_learning import panel_diagnostics
    deadline=time.monotonic()+35*60
    def run(*args):run_process([sys.executable,*map(str,args)],deadline)
    run('workflows/check_actor_tensor_path.py','--checkpoint',a.source_7357/'model.pt','--device','auto','--require-gpus',2,'--out',a.out/'actor.json')
    parent,source,inherited,factor,paths,manifests,cfg,previous=inputs(a)
    write_json(a.out/'train.config.json',cfg);train=a.out/'train';versions=[]
    for i,key in enumerate(('warm0','warm1','train')):
        extra=['--initial-trainer-state',source/'trainer.pt'] if i==0 else ['--resume','--library-transition-from',manifests[('warm0','warm1','train')[i-1]]['digest']]
        run('-m','hsbg.az.train.personal','train','--config',a.out/'train.config.json','--checkpoint',source/'model.pt','--library',paths[key],'--out',train,'--device','cuda:0','--run-iterations',1,*extra)
        latest=read(train/'latest.json');versions.append(latest['version'])
        if latest['version']!=7358+i:raise ValueError('更新版本不连续')
    quality=training_report(train,a.out/'health',versions[-1]);tot=quality['totals'];strict=quality['mirror_strict_counts']
    if not tot['mirror_attempt']==tot['mirror_applied']==strict['eligible_live_seats'] or any(tot.get(k,0) for k in ('illegal_actions','truncated_games','mirror_errors','mirror_missing','mirror_fallback')):raise ValueError('工程注入验收失败')
    if tot['optimizer_steps']<=0 or tot['processed_samples']/tot['decisions']<.8:raise ValueError('样本有效学习率过低，保留诊断')
    forced_tier=None
    if a.profile=='forced':
        forced_tier=check_forced_tier(train,versions[-1],int(tot.get('games') or 0))
    final=audit(train/'snapshots'/f"step_{versions[-1]:06d}",versions[-1],device='cuda:0')
    initial=read(parent/'source-fork.json')['restore_audit']
    if final['adam_step_min']<=initial['adam_step_min']:raise ValueError('Adam未连续推进')
    # 八局只验收诊断链路，不作棋力资格门；正式长训仍有96局冻结起点。
    run('-m','hsbg.az.train.personal','eval','--checkpoint',source/'model.pt','--library',paths['hard'],'--device','cuda:0','--games',1,'--seed',14000000,'--out',a.out/'evaluation.json','--mirror-strict','--economy','--exposure',1,'--max-actions-per-turn',80,'--move-guard',1,'--move-cap',5,'--move-repeat-limit',2)
    evaluation=read(a.out/'evaluation.json');write_json(a.out/'diagnostics.json',panel_diagnostics(evaluation,a.out/'evaluation.json'))
    result=dict(passed=True,profile=a.profile,runner_sha256=sha256(ROOT/RUNNERS[a.profile]),source_sha256=inherited['sha256'],fingerprint=fingerprint(),training_source_sha256=training_source_digest(),libraries={k:v['digest'] for k,v in read(a.bot_3702/'report.json')['splits'].items()},versions=versions,training=quality,final_restore=final,source_adam_step_min=initial['adam_step_min'],forced_tier=forced_tier,probe_weights_not_used_for_long_training=True)
    write_json(a.out/'report.json',result);print(json.dumps({'passed':True,'profile':a.profile,'versions':versions,'training':quality['rates'],'forced_tier':forced_tier},ensure_ascii=False))


def check_forced_tier(train,version,games=0,max_missed_ratio=0.25):
    """强制升本验收：统计必须存在、5 本目标可达、未达比例受限。

    ``games`` 用来判断「样本是不是太少」——如果大部分局在 T8 前就结束，说明训练场
    难度或计划有问题，必须在长训前发现（这正是本轮最需要早发现的一类失败）。
    """
    forced=None
    with (train/'history.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line:continue
            rec=json.loads(line)
            if rec.get('type')=='header':continue
            forced=rec.get('forced_tier') or forced
    if not forced:raise ValueError('强制升本统计缺失（history 没有 forced_tier 段）')
    if int(forced.get('forced_actions') or 0)<=0:raise ValueError('强制升本一次都没执行')
    targets=forced.get('targets') or {}
    five=targets.get('5') or {}
    total=(int(five.get('on_time') or 0)+int(five.get('deferred') or 0)
           +int(five.get('missed') or 0)+int(five.get('self') or 0))
    if total<=0:raise ValueError('5 本目标没有任何样本')
    on_time=float(five.get('on_time') or 0)/total
    deferred=float(five.get('deferred') or 0)/total
    missed=float(five.get('missed') or 0)/total
    if games and total < 0.25*float(games):
        raise ValueError(f'5 本目标样本过少({int(total)}/{games})：多数局在 T8 前结束，'
                         '先查训练场难度/计划，不要直接长训')
    if on_time+deferred<0.60:raise ValueError(f'5 本目标到达率过低: on_time={on_time:.2f} deferred={deferred:.2f}')
    if missed>max_missed_ratio:raise ValueError(f'5 本目标未达比例过高: {missed:.2f}')
    return dict(version=int(version),plan=forced.get('plan'),prob=forced.get('prob'),
                teacher=forced.get('teacher'),forced_actions=int(forced.get('forced_actions') or 0),
                blocked=int(forced.get('blocked') or 0),targets=targets,
                tier5_samples=int(total),tier5_on_time=round(on_time,4),
                tier5_deferred=round(deferred,4),tier5_missed=round(missed,4))


if __name__=='__main__':main()
