"""一次GPU工程验收：完整状态→更新→保存恢复→显式课程换库→继续更新。

``--profile`` 选择用哪一版长训入口的 ``inputs()``/``defaults()``：

- ``long``（默认）：上一轮 P50 四阶段（``run_7357_long``）；
- ``forced``：本轮强制升本（``run_forced_tier_long``），并额外验收强制升本的
  执行统计（``forced_tier`` 段必须出现、5 本目标到达率达标、未达比例受限）。

失败时**先写 report.json（passed=false + 各项检查明细）再抛错**：一次失败就能看到
"注入/效率/强制升本/Adam" 四类指标，不必逐项重跑。
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from workflows.run_7357_long import MODEL_SHA,TRAINER_SHA,read

RUNNERS={'long':'workflows/run_7357_long.py','forced':'workflows/run_forced_tier_long.py'}

#: 样本有效学习率下限（早期 KL 早停会丢掉部分 minibatch；0.5 是"还能学"的安全线）
MIN_PROCESSED_RATIO=0.5
#: 5 本目标 on_time+deferred 下限、missed 上限、样本下限（占局数比例）
MIN_TIER5_REACH=0.60
MAX_TIER5_MISSED=0.25
MIN_TIER5_SAMPLE_RATIO=0.25
#: ── 吞吐门禁（2026-11-07 实测：作业 1091647 两轮 33m48s，而 1085078 三轮 4m54s）──
#: 快速基线（1085131 长训 2,647 轮）：collect ≈11.5 s / 8,600 dec·s⁻¹ / ppo ≈9 s，
#: 单轮（1 iteration，512 局）含进程启动 ≈98 s。这里把"单轮墙钟"和"采样速率"都上界化：
#: 采样链路退化（worker 没上 GPU、批量叶服务失效、镜像库加载异常等）必须**当场失败**，
#: 而不是静默吃掉 12 小时长训。阈值是基线的 3 倍/1/3，只拦"病态"，不拦正常波动。
MAX_ROUND_SECONDS=300.0
MIN_DECISIONS_PER_S=3000.0
#: 单轮**进程内**预算（分钟）：``personal train --run-minutes`` 只缩短本次调用、不改配方。
#: 一轮卡住时最多浪费这么久，而不是把 35 分钟的验收窗口整块吃掉。
ROUND_MINUTES=8.0


def history_tail(train):
    """``train/history.jsonl`` 最后一条迭代记录（不含 header）；缺失返回 ``None``。"""
    path=Path(train)/'history.jsonl'
    last=None
    if not path.is_file():return None
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line:continue
            rec=json.loads(line)
            if rec.get('type')=='header':continue
            last=rec
    return last


def check_throughput(timings):
    """单轮墙钟与采样速率门禁；返回 ``_check``，失败时带着逐轮拆解。"""
    rounds=timings.get('rounds') or []
    if not rounds:
        return _check('throughput',False,'没有拿到任何一轮的计时',rounds=rounds)
    slow=[r for r in rounds if float(r.get('wall_s') or 0.0)>MAX_ROUND_SECONDS]
    thin=[r for r in rounds if float(r.get('decisions_per_s') or 0.0)<MIN_DECISIONS_PER_S]
    detail=dict(rounds=rounds,max_round_s=MAX_ROUND_SECONDS,
                min_decisions_per_s=MIN_DECISIONS_PER_S,
                slow_rounds=[r.get('round') for r in slow],
                thin_rounds=[r.get('round') for r in thin])
    if slow:
        return _check('throughput',False,
                      f'单轮墙钟超过 {MAX_ROUND_SECONDS:.0f}s：'+', '.join(
                          f"{r.get('key')}={float(r.get('wall_s') or 0):.0f}s" for r in slow),
                      **detail)
    if thin:
        return _check('throughput',False,
                      f'采样速率低于 {MIN_DECISIONS_PER_S:.0f} dec/s：'+', '.join(
                          f"{r.get('key')}={float(r.get('decisions_per_s') or 0):.0f}"
                          for r in thin),
                      **detail)
    return _check('throughput',True,'',**detail)


def _check(name,ok,reason='',**detail):
    return dict(name=name,ok=bool(ok),reason=str(reason),**detail)


def _sha_lf(path):
    """按 LF 归一化的 SHA256——与 ``tools/nus_submit._sha_lf`` / 训练源码指纹同口径。"""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def check_forced_tier(train,version,games=0):
    """返回 ``_check(...)``：强制升本统计是否存在、是否真的执行、5 本是否可达。

    ``games`` 必须是本轮的**完整对局数**（``engine_quality_checked_games``）；拿不到就
    直接判失败——样本比例检查是这条门禁的核心，不能因为缺字段而静默跳过。
    """
    forced=None
    with (train/'history.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line:continue
            rec=json.loads(line)
            if rec.get('type')=='header':continue
            forced=rec.get('forced_tier') or forced
    if not forced:
        return _check('forced_tier',False,'history 没有 forced_tier 段')
    targets=forced.get('targets') or {}
    five=targets.get('5') or {}
    total=(int(five.get('on_time') or 0)+int(five.get('deferred') or 0)
           +int(five.get('missed') or 0)+int(five.get('self') or 0))
    detail=dict(version=int(version),plan=forced.get('plan'),prob=forced.get('prob'),
                teacher=forced.get('teacher'),
                forced_actions=int(forced.get('forced_actions') or 0),
                teacher_labels=int(forced.get('teacher_labels') or 0),
                blocked=int(forced.get('blocked') or 0),
                games=int(games),targets=targets,tier5_samples=int(total))
    if int(games)<=0:
        return _check('forced_tier',False,'无法确定本轮局数（totals 缺少 engine_quality_checked_games）',**detail)
    if int(forced.get('forced_actions') or 0)<=0:
        return _check('forced_tier',False,'一次强制都没执行（检查计划是否被平移成累计决策）',**detail)
    if total<=0:
        return _check('forced_tier',False,'5 本目标没有任何样本',**detail)
    on_time=float(five.get('on_time') or 0)/total
    deferred=float(five.get('deferred') or 0)/total
    missed=float(five.get('missed') or 0)/total
    detail.update(tier5_on_time=round(on_time,4),tier5_deferred=round(deferred,4),
                  tier5_missed=round(missed,4))
    if total < MIN_TIER5_SAMPLE_RATIO*float(games):
        return _check('forced_tier',False,
                      f'5 本目标样本过少({total}/{games})：多数局在 T8 前结束，先查训练场难度',**detail)
    if on_time+deferred<MIN_TIER5_REACH:
        return _check('forced_tier',False,
                      f'5 本到达率过低 on_time={on_time:.2f} deferred={deferred:.2f}',**detail)
    if missed>MAX_TIER5_MISSED:
        return _check('forced_tier',False,f'5 本未达比例过高 {missed:.2f}',**detail)
    return _check('forced_tier',True,'',**detail)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=sorted(RUNNERS),default='long')
    from workflows.run_7357_long import defaults as long_defaults
    p=long_defaults(p);p.add_argument('--out',type=Path,default='artifacts/runs/p50-7357-preflight');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('GPU计算节点专用')
    if a.out.exists() and any(not f.name.startswith('visibility-') for f in a.out.iterdir()):p.error('预检输出必须新目录，保留上次证据')
    a.preflight_mode=True
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
    t_inputs=time.time()
    parent,source,inherited,factor,paths,manifests,cfg,previous=inputs(a)
    write_json(a.out/'train.config.json',cfg);train=a.out/'train';versions=[]
    timings={'inputs_s':round(time.time()-t_inputs,2),'rounds':[]}
    base=dict(profile=a.profile,versions=versions,checks=[],timings=timings)
    for i,key in enumerate(('warm0','warm1','train')):
        extra=['--initial-trainer-state',source/'trainer.pt'] if i==0 else ['--resume','--library-transition-from',manifests[('warm0','warm1','train')[i-1]]['digest']]
        t0=time.time()
        run('-m','hsbg.az.train.personal','train','--config',a.out/'train.config.json','--checkpoint',source/'model.pt','--library',paths[key],'--out',train,'--device','cuda:0','--run-iterations',1,'--run-minutes',ROUND_MINUTES,*extra)
        wall=round(time.time()-t0,2)
        latest=read(train/'latest.json');versions.append(latest['version'])
        if latest['version']!=7358+i:raise ValueError('更新版本不连续')
        tail=history_tail(train) or {}
        collect=tail.get('collect') or {};ppo=tail.get('ppo') or {}
        row=dict(round=i,key=key,wall_s=wall,
                 collect_wall_s=round(float(collect.get('wall_s') or 0.0),2),
                 decisions=int(collect.get('decisions') or 0),
                 decisions_per_s=round(float(collect.get('decisions_per_s') or 0.0),1),
                 ppo_s=round(float(ppo.get('seconds') or 0.0),2),
                 samples=int(ppo.get('samples') or 0),
                 updates=int(ppo.get('updates') or 0))
        timings['rounds'].append(row)
        print(f"[preflight] round {i} ({key}): wall={row['wall_s']}s "
              f"collect={row['collect_wall_s']}s {row['decisions_per_s']} dec/s "
              f"ppo={row['ppo_s']}s samples={row['samples']} updates={row['updates']}",flush=True)
    quality=training_report(train,a.out/'health',versions[-1]);tot=quality['totals'];strict=quality['mirror_strict_counts']
    checks=[]
    inj_ok=bool(tot['mirror_attempt']==tot['mirror_applied']==strict['eligible_live_seats']
                and not any(tot.get(k,0) for k in ('illegal_actions','truncated_games','mirror_errors','mirror_missing','mirror_fallback')))
    checks.append(_check('injection',inj_ok,'' if inj_ok else '工程注入验收失败',
                         mirror_attempt=int(tot['mirror_attempt']),mirror_applied=int(tot['mirror_applied']),
                         eligible_live_seats=int(strict['eligible_live_seats']),
                         illegal_actions=int(tot.get('illegal_actions',0)),
                         truncated_games=int(tot.get('truncated_games',0))))
    ratio=float(tot['processed_samples'])/max(float(tot['decisions']),1.0)
    eff_ok=bool(int(tot['optimizer_steps'])>0 and ratio>=MIN_PROCESSED_RATIO)
    checks.append(_check('efficiency',eff_ok,
                         '' if eff_ok else f'样本有效学习率过低 processed/decisions={ratio:.3f}',
                         optimizer_steps=int(tot['optimizer_steps']),processed_ratio=round(ratio,4),
                         kl_early_stop_iterations=int(tot.get('kl_early_stop_iterations',0))))
    if a.profile=='forced':
        # 局数只能用 engine_quality_checked_games（totals 里没有 'games' 键）
        checks.append(check_forced_tier(train,versions[-1],
                                        int(tot.get('engine_quality_checked_games') or 0)))
    # 吞吐门禁：每轮墙钟与采样速率（基线 98 s / 8,600 dec·s⁻¹）
    checks.append(check_throughput(timings))
    failed=[c for c in checks if not c['ok']]
    base.update(versions=versions,checks=checks,timings=timings,
                training=quality,forced_tier=next((c for c in checks if c['name']=='forced_tier'),None),
                fingerprint=fingerprint(),training_source_sha256=training_source_digest(),
                runner_sha256=_sha_lf(ROOT/RUNNERS[a.profile]))
    if failed:
        write_json(a.out/'report.json',dict(passed=False,failed=[c['name'] for c in failed],**base))
        raise ValueError('预检未通过: '+'; '.join(f"{c['name']}({c['reason']})" for c in failed)
                         +f" -> 明细见 {a.out/'report.json'}")
    final=audit(train/'snapshots'/f"step_{versions[-1]:06d}",versions[-1],device='cuda:0')
    initial=read(parent/'source-fork.json')['restore_audit']
    if final['adam_step_min']<=initial['adam_step_min']:raise ValueError('Adam未连续推进')
    # 八局只验收诊断链路，不作棋力资格门；正式长训仍有96局冻结起点。
    run('-m','hsbg.az.train.personal','eval','--checkpoint',source/'model.pt','--library',paths['hard'],'--device','cuda:0','--games',1,'--seed',14000000,'--out',a.out/'evaluation.json','--mirror-strict','--economy','--exposure',1,'--max-actions-per-turn',80,'--move-guard',1,'--move-cap',5,'--move-repeat-limit',2)
    evaluation=read(a.out/'evaluation.json');write_json(a.out/'diagnostics.json',panel_diagnostics(evaluation,a.out/'evaluation.json'))
    result=dict(passed=True,source_sha256=inherited['sha256'],libraries={k:v['digest'] for k,v in read(a.bot_3702/'report.json')['splits'].items()},
                final_restore=final,source_adam_step_min=initial['adam_step_min'],
                probe_weights_not_used_for_long_training=True,**base)
    write_json(a.out/'report.json',result)
    print(json.dumps({'passed':True,'profile':a.profile,'versions':versions,
                      'processed_ratio':round(ratio,4),
                      'forced_tier':result['forced_tier']},ensure_ascii=False))


if __name__=='__main__':main()
