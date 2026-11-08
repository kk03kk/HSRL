#!/usr/bin/env python3
"""充分使用有界预算的持续PPO；棋力评测仅记录，工程异常仍失败关闭。"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_learning_experiment import read, write, digest, writer_lock, freeze_checkpoint, training_report
from workflows.run_overnight_learning import run_process, check_eval


def validate_eval_cache(protocol, library, seed, games, source):
    # 帧文件不变不代表引擎规则没变；两边同为旧缓存也可能通过配对检查。
    if (protocol.get('fingerprint') != library['fingerprint']
            or protocol['library_digest'] != library['digest'] or protocol['seed'] != seed
            or protocol['games_per_seat'] != games or protocol['training_source_sha256'] != source):
        raise ValueError('评测缓存协议不一致（含引擎指纹）')


def curriculum_segment(current, base, target, paths, schedule):
    """固定课程只在已提交批次之间换库，下一边界缩短本次采样预算。"""
    if schedule and ([p[0] for p in schedule] != sorted(set(p[0] for p in schedule))
                     or schedule[0][0] != 0):
        raise ValueError('课程必须从0开始且边界严格递增')
    selected, end = paths['train'], target
    for offset, key in schedule:
        boundary = base + offset
        if current >= boundary:
            selected = paths[key]
        elif boundary < end:
            end = boundary
    return selected, end


def learning_activity(train):
    rows=[r for line in (Path(train)/'history.jsonl').read_text(encoding='utf-8').splitlines()
          if (r:=json.loads(line)).get('iter') is not None]
    trailing=0
    for row in reversed(rows):
        if int(row['ppo'].get('updates',0))>0:break
        trailing+=1
    sampled=sum(int(r['ppo'].get('samples',0)) for r in rows)
    processed=sum(int(r['ppo'].get('processed_samples',0)) for r in rows)
    decisions=sum(int(r.get('collect',{}).get('move_guard_decisions',0)) for r in rows)
    moves=sum(int(r.get('collect',{}).get('move_guard_moves',0)) for r in rows)
    return dict(optimizer_steps=sum(int(r['ppo'].get('updates',0)) for r in rows),
        zero_update_iterations=sum(int(r['ppo'].get('updates',0))==0 for r in rows),
        trailing_zero_update_iterations=trailing,processed_fraction=processed/max(1,sampled),
        pipeline_fallbacks=sum(bool(r.get('pipeline',{}).get('fallback_zero_step')) for r in rows),
        recorded_games=sum(int(r.get('collect',{}).get('games',0)) for r in rows),
        guarded_moves=moves,guarded_move_fraction=moves/max(1,decisions))


def parent_inputs(args):
    """选出本轮的 ``inputs()``（决定冻结哪个父状态、用哪套课程表与哪个库）。

    **为什么需要显式绑定**：强制升本入口同样带 ``--source-7357``，靠参数嗅探会把
    ``p50-forced`` 判成旧的 P50 四阶段入口，于是冻结错父状态、用错课程表，并在
    ``run_7357_long.check_inputs`` 里撞上"曲线摘要写死旧曲线"（2026-11-08 长训 1093247
    9 秒失败）。启动器（``launch_nus_refinement.py``）按 task→runner 关系写入
    ``HSAI_INPUTS_RUNNER=workflows.<runner 模块名>``，这里优先采用它。
    """
    explicit = str(getattr(args, 'inputs_runner', '')
                   or os.environ.get('HSAI_INPUTS_RUNNER', '')).strip()
    if explicit:
        import importlib
        try:
            module = importlib.import_module(explicit)
        except ImportError as exc:
            raise ValueError(f'HSAI_INPUTS_RUNNER={explicit!r} 无法导入: {exc}') from exc
        if not callable(getattr(module, 'inputs', None)):
            raise ValueError(f'HSAI_INPUTS_RUNNER={explicit!r} 没有 inputs()')
        print(f'[inputs] 使用显式绑定的 {explicit}.inputs', flush=True)
        return module.inputs(args)
    from hsbg.az.train.personal import checked_library, validate_config
    print('[inputs][warn] 未设置 HSAI_INPUTS_RUNNER，回退到按参数嗅探（旧行为）', flush=True)
    if getattr(args, 'source_7357', None):
        from workflows.run_7357_long import inputs
        return inputs(args)
    if getattr(args, 'source_4159_audit', None):
        from workflows.run_4159_long import inputs
        return inputs(args)
    if getattr(args, 'bot_3702', None):
        from workflows.run_3702_preflight import inputs
        return inputs(args)
    if getattr(args,'curriculum_parent',None):
        from workflows.prepare_opponent_research import MODELS,TRAINER_SHA
        from workflows.fork_trainer_lr import materialize
        parent=Path(args.curriculum_parent).resolve();report=read(parent/'report.json')
        if not report['complete'] or not report['ready_for_time_budgeted_training']:
            raise ValueError('新库准备工程门未完成')
        origin=ROOT/Path(MODELS['b10m'][0]).parent
        if digest(origin/'model.pt')!=MODELS['b10m'][1] or digest(origin/'trainer.pt')!=TRAINER_SHA:
            raise ValueError('原B10M完整状态摘要不符')
        paths=dict(train=parent/'libraries/train',hard=parent/'libraries/dev',
            bot=Path(getattr(args,'bot_library',ROOT/'artifacts/runs/nus-overnight-001/libraries/dev-control')))
        manifests={k:checked_library(p) for k,p in paths.items()}
        if not all(m.get('validation',{}).get('accepted',0)>0 for m in manifests.values()):
            raise ValueError('长训只使用已逐帧验证的库')
        if any(manifests[k]['digest']!=report['libraries'][s] for k,s in [('train','train'),('hard','dev')]):
            raise ValueError('新库准备后被修改')
        if manifests['train']['split']!='train' or any(manifests[k]['split']!='dev' for k in ('hard','bot')):
            raise ValueError('Train/Dev分割不符')
        strict = bool(getattr(args, 'strict_mirror', False))
        original_libraries = {k: v['digest'] for k, v in manifests.items()}
        firestone_eot = getattr(args, 'firestone_eot', None)
        if firestone_eot:
            if not strict:
                raise ValueError('Firestone 全库续训要求严格镜像')
            bank = Path(firestone_eot).resolve()
            bank_report = read(bank / 'report.json')
            if (bank_report['schema'] != 'hsrl.firestone_eot/v1' or not bank_report['complete']
                    or bank_report['build_number'] != 255790):
                raise ValueError('Firestone 255790 建库未完成')
            for key, split in (('train', 'train'), ('hard', 'dev')):
                paths[key] = bank / split
                manifest = checked_library(paths[key])
                if (manifest['driver'] != 'firestone_visible_eot' or manifest['split'] != split
                        or manifest['digest'] != bank_report['libraries'][split]):
                    raise ValueError(f'Firestone {split} 库或摘要不符')
                manifests[key] = manifest
        continuation=getattr(args,'continue_from',None)
        if continuation:
            old=Path(continuation).resolve()
            old_plan=read(old/'plan.json')
            old_reference = MODELS['b10m'][1]
            if old_plan['config'].get('mirror-strict'):
                old_reference = old_plan['inherited']['sha256']['model.pt']
            expected_libraries = {k: v['digest'] for k, v in manifests.items()}
            if firestone_eot and old_plan.get('factor') == 'human_curve_v1':
                expected_libraries = original_libraries
            if (old_plan['libraries'] != expected_libraries or old_plan['reference_sha256'] != old_reference):
                raise ValueError('接续实验的库/参考模型不同')
            latest=read(old/'train/latest.json')
            origin=Path(args.out).resolve()/'continued-source'
            freeze_checkpoint(old/'train',origin,int(latest['cum_decisions']))
        source=Path(args.out).resolve()/'initial-state'
        record=materialize(origin,source,2e-6)
        reference_sha = record['sha256']['model.pt'] if strict else MODELS['b10m'][1]
        if digest(args.reference_checkpoint) != reference_sha:
            raise ValueError('主参考与冻结训练起点不符')
        cfg=read(parent/'next-training.config.json')
        cfg.update(lr=2e-6,iterations=0,minutes=30,**{'max-actions-per-turn':80,'move-guard':1,
            'move-cap':14,'move-repeat-limit':2,'pipeline-zero-step-fallback':1})
        previous_config = dict(old_plan['config']) if continuation else None
        if strict:
            from hsbg.az.train.personal import require_calibrated_library
            from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
            from hsbg.az.train.strict_mirror import StrictLibrary
            for key in ('train', 'hard'):
                require_calibrated_library(manifests[key])
                StrictLibrary(MirrorSnapshotPool(str(paths[key]), sample='random'))
            if previous_config is not None:
                previous_config.pop('mirror-strict', None)
        if continuation and previous_config != validate_config(cfg):
            raise ValueError('接续训练配置不同，禁止隐式改变配方')
        if strict:
            cfg['mirror-strict'] = 1
        factor = 'firestone_visible_eot_v1' if firestone_eot else 'human_curve_v1'
        return parent,source,record,factor,paths,manifests,validate_config(cfg),dict(checkpoint=MODELS['b10m'][1])
    parent = Path(args.parent).resolve()
    report = read(parent/'report.json')
    quality = read(parent/'training-report.json')
    if not report['complete']:
        raise ValueError('父实验未完整报告，请先核对最后提交状态')
    if any(quality['totals'].get(k, 0) for k in ('illegal_actions', 'truncated_games', 'mirror_errors',
            'mirror_pool_unbalanced', 'mirror_pool_species_unbalanced')) or quality.get('unexpected_mirror_rejections'):
        raise ValueError('父实验工程质量门未通过，不能自动衔接')
    version = int(report['latest']['version'])
    source = parent/'last-committed'/f'step_{version:06d}'
    record = read(source/'checkpoint.json')
    if int(record['version']) != version or int(record['actual_decisions']) != int(report['latest']['cum_decisions']):
        raise ValueError('父实验最后完整副本与报告不一致')
    for name, expected in [('model.pt', getattr(args, 'parent_model_sha', None)),
                            ('trainer.pt', getattr(args, 'parent_trainer_sha', None))]:
        sha = digest(source/name)
        if sha != record['sha256'][name] or (expected and sha != expected):
            raise ValueError('父实验完整状态摘要不符')
    factor = float(report['selected']['factor'])
    paths = dict(train=parent/'libraries'/f'train-x{factor:g}',
                 bot=parent/'libraries/dev-control', hard=parent/'libraries'/f'dev-x{factor:g}')
    manifests = {k: checked_library(p) for k, p in paths.items()}
    if manifests['train']['split'] != 'train' or any(manifests[k]['split'] != 'dev' for k in ('bot', 'hard')):
        raise ValueError('训练/Dev分割不符，禁止读取Test')
    if not all(m.get('validation', {}).get('accepted', 0) > 0 for m in manifests.values()):
        raise ValueError('需要父实验已验证的三套派生库')
    cfg = validate_config(read(parent/'train.config.json'))
    # 沿用成熟anchor调度器：适应期后减弱约束，完整anchor权重/RNG/Adam仍继承。
    if getattr(args, 'anchor_iterations', None) is not None:
        cfg['anchor-iterations'] = args.anchor_iterations
    if getattr(args, 'anchor_beta_tail', None) is not None:
        cfg['anchor-beta-tail'] = args.anchor_beta_tail
    if cfg.get('iterations') != 0 or cfg.get('snapshot-every') != 1 or not 0 < cfg.get('minutes', 0) <= 30:
        raise ValueError('需要每轮快照、最多30分钟分段的原配方')
    parent_plan = read(parent/'plan.json')
    if digest(args.reference_checkpoint) != parent_plan['checkpoint']:
        raise ValueError('参考模型不是父实验记录的B100k')
    return parent, source, record, factor, paths, manifests, cfg, parent_plan


def run(args, execute=None):
    args._writer_acquired = False
    from hsbg.az.train.personal import training_source_digest, checked_library
    from hsbg.az.train.runlog import recover_committed_record
    from workflows.paired_panel_delta import compare_personal
    import torch
    start = time.monotonic()
    deadline = start+args.wall_minutes*60
    train_deadline = deadline-args.reserve_minutes*60
    out = Path(args.out).resolve()
    economy_3702 = bool(getattr(args, 'bot_3702', None))

    def command(*argv, timeout_minutes=None):
        full = [sys.executable, *map(str, argv)]
        if execute:
            execute(full)
        else:
            limit = deadline if timeout_minutes is None else min(deadline, time.monotonic()+timeout_minutes*60)
            run_process(full, limit)
    def evaluate(model, library, path, games, seed,legacy=False):
        profile=[]
        if (getattr(args,'curriculum_parent',None) or economy_3702) and not legacy:
            if getattr(args, 'bot_3702', None):
                from workflows.run_3702_preflight import action_args
                profile=action_args()
            else:
                profile=['--max-actions-per-turn',80,'--move-guard',1,'--move-cap',14,'--move-repeat-limit',2]
        strict = bool(getattr(args, 'strict_mirror', False) and not legacy and library == paths['hard'])
        if strict:
            profile += ['--mirror-strict', '--exposure', 1, '--economy']
        if not path.exists():
            command('-m', 'hsbg.az.train.personal', 'eval', '--checkpoint', model, '--library', library,
                '--device', args.device, '--games', games, '--seed', seed, '--out', path,*profile,
                timeout_minutes=20 if economy_3702 and seed == 11000000 else None)
        result = read(path)
        check_eval(result, digest(model))
        p = result['protocol']
        validate_eval_cache(p, checked_library(library), seed, games, training_source_digest())
        if (getattr(args,'curriculum_parent',None) or economy_3702) and (p['max_actions_per_turn']!=(40 if legacy else 80) or
            p.get('move_guard',0)!=(0 if legacy else 1)):
            raise ValueError('动作守卫评测协议不符')
        if bool(p.get('mirror_strict')) != strict:
            raise ValueError('评测缓存严格镜像协议不符')
        if getattr(args, 'bot_3702', None):
            from workflows.report_firestone_learning import by_turn
            from workflows.run_3702_preflight import validate_eval_moves
            validate_eval_moves(result)
            write(path.with_suffix('.economy.json'), dict(protocol_sha256=result['protocol_sha256'],
                  checkpoint_sha256=result['checkpoint']['sha256'], economy_stage='before_end_turn',
                  turns=by_turn(result)))
        return result
    with writer_lock(out):
        args._writer_acquired = True
        parent, source, inherited, factor, paths, manifests, config, previous_plan = parent_inputs(args)
        runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, os=platform.system())
        if previous_plan.get('runtime', runtime) != runtime:
            raise ValueError('运行环境与父实验不同，请先核对')
        monitor_first = getattr(args, 'monitor_first', args.monitor_every)
        parent_plan_file = 'source-fork.json' if (getattr(args,'source_4159_audit',None) or getattr(args,'source_7357',None)) else 'plan.json'
        plan = dict(schema='hsrl.sustained/v1', parent_report_sha256=digest(parent/'report.json'),
            parent_plan_sha256=digest(parent/parent_plan_file), inherited=inherited,
            reference_sha256=digest(args.reference_checkpoint), libraries={k:v['digest'] for k,v in manifests.items()},
            config=config, factor=factor, source_sha256=training_source_digest(), runner_sha256=digest(__file__),
            runtime=runtime, device=args.device, budget_decisions=args.budget_decisions,
            monitor_every=args.monitor_every, monitor_games=args.monitor_games, eval_games=args.eval_games,
            wall_minutes=args.wall_minutes, reserve_minutes=args.reserve_minutes,
            continuation_policy='engineering_only; all finite performance trajectories continue',
            monitor_seed=11000000, final_seed=13000000)
        if getattr(args,'source_4159_audit',None):
            plan['parent_plan_file'] = parent_plan_file
        if monitor_first != args.monitor_every:
            plan['monitor_first'] = monitor_first
        if getattr(args, 'bot_3702', None):
            plan['baseline_before_training'] = True
        if getattr(args, 'source_7357', None):
            plan['train_schedule'] = args.train_schedule
            plan['parent_plan_file'] = parent_plan_file
        if getattr(args,'continue_from',None):
            old=Path(args.continue_from).resolve()
            plan['continued_from']=dict(run=str(old),plan_sha256=digest(old/'plan.json'),
                frozen_checkpoint_sha256=digest(out/'continued-source/checkpoint.json'))
        if getattr(args, 'recover_from', None):
            plan['recovery_sha256'] = digest(out/'recovery.json')
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise ValueError('计划/来源/源码/库/环境已改变，必须新目录')
        write(out/'plan.json', plan)
        write(out/'train.config.json', config)
        train = out/'train'
        base = int(inherited['actual_decisions'])
        end = base+args.budget_decisions
        targets = list(range(base+monitor_first, end, args.monitor_every))+[end]
        stages, champion = [], None
        reason = 'decision_budget'
        stop_file = out/'STOP_REQUESTED'
        def pause_if_requested():
            if not stop_file.exists():
                return None
            latest = read(train/'latest.json') if (train/'latest.json').exists() else None
            if latest:
                freeze_checkpoint(train, out/'last-committed'/f"step_{int(latest['version']):06d}",
                                  int(latest['cum_decisions']))
            paused = dict(schema='hsrl.sustained_result/v1', complete=False, state='paused',
                          stop_reason='user_request', latest=latest, stages=stages,
                          stopped_for_performance=False, promote_best=False)
            write(out/'report.json', paused)
            write(out/'status.json', paused)
            return paused
        paused = pause_if_requested()
        if paused:
            return paused
        def health():
            latest = read(train/'latest.json')
            ver = int(latest['version'])
            recover_committed_record(train/'history.jsonl', train/'snapshots'/f'step_{ver:06d}', ver)
            report = training_report(train, out/'health'/f'step_{ver:06d}', ver)
            if report.get('unexpected_mirror_rejections') or report['totals']['decisions']+base != int(latest['cum_decisions']):
                raise RuntimeError('训练质量门或累计决策账本不一致')
            if getattr(args,'curriculum_parent',None) or economy_3702:
                activity=learning_activity(train);write(out/'learning-health.json',activity)
                if activity['optimizer_steps']==0 or activity['trailing_zero_update_iterations']>=3:
                    raise RuntimeError('没有实际优化进展或连续三轮零更新，保留状态；不能将纯采样计为学习')
            if getattr(args, 'strict_mirror', False):
                counts = report.get('mirror_strict_counts', {})
                totals = report['totals']
                if (totals['mirror_fallback'] or totals['mirror_applied'] != counts.get('eligible_live_seats')
                        or not (counts.get('live_combats', 0) + counts.get('ghost_combats', 0))):
                    raise RuntimeError('严格镜像训练缺少覆盖验收')
            write(out/'training-report.json', report)
            return latest
        firestone_only = bool(getattr(args, 'firestone_eot', None) or getattr(args, 'bot_3702', None))
        if economy_3702:
            from workflows.report_firestone_learning import panel_diagnostics
            baseline_path = out/'baselines'/f'hard-{args.monitor_games}-11000000.json'
            write(out/'status.json', dict(state='baseline_evaluation', inherited=inherited,
                  stopped_for_performance=False))
            baseline = evaluate(args.reference_checkpoint, paths['hard'], baseline_path, args.monitor_games, 11000000)
            baseline_detail = panel_diagnostics(baseline, baseline_path)
            write(out/'baselines'/'diagnostics.json', baseline_detail)

        def pair(model, tag, games, seed):
            pairs = {}
            for name in (('hard',) if firestone_only else ('bot', 'hard')):
                before = baseline if economy_3702 else evaluate(
                    args.reference_checkpoint, paths[name], out/'baselines'/f'{name}-{games}-{seed}.json', games, seed)
                candidate_path = tag/f'{name}.json'
                after = evaluate(model, paths[name], candidate_path, games, seed)
                pairs[name] = compare_personal(before, after, n_boot=2000, seed=20261103)
                if economy_3702:
                    write(tag/'diagnostics.json', dict(paired=pairs[name], reference=baseline_detail,
                          candidate=panel_diagnostics(after, candidate_path)))
            return pairs
        write(out/'status.json', dict(state='training', inherited=inherited, selected_factor=factor,
            stopped_for_performance=False, latest=None, stages=[]))
        for target in targets:
            paused = pause_if_requested()
            if paused:
                return paused
            milestone = out/'milestones'/f'decisions-{target:09d}'
            if not (milestone/'checkpoint.json').exists():
                while True:
                    current = int(read(train/'latest.json')['cum_decisions']) if (train/'latest.json').exists() else base
                    if current >= target:
                        break
                    remaining = train_deadline-time.monotonic()
                    if remaining < 120:
                        reason = 'time_budget'
                        break
                    minutes = min(float(config['minutes']), (remaining-45)/60)
                    resume = (train/'latest.json').exists()
                    selected_library, segment_target = curriculum_segment(
                        current, base, target, paths, getattr(args, 'train_schedule', []))
                    transition_args = []
                    if resume and getattr(args, 'train_schedule', None):
                        previous_digest = read(train/'provenance.json')['library_digest']
                        selected_digest = checked_library(selected_library)['digest']
                        if previous_digest != selected_digest:
                            transition_args = ['--library-transition-from', previous_digest]
                    previous_stop = os.environ.get('HSRL_STOP_FILE')
                    os.environ['HSRL_STOP_FILE'] = str(stop_file)
                    try:
                        command('-m', 'hsbg.az.train.personal', 'train', '--config', out/'train.config.json',
                        '--checkpoint', source/'model.pt', '--library', selected_library, '--out', train,
                        '--device', args.device, '--stop-at-decisions', segment_target, '--run-minutes', minutes,
                        *(['--resume'] if resume else ['--initial-trainer-state', source/'trainer.pt']),
                        *transition_args,
                        *(['--run-iterations',3] if getattr(args,'curriculum_parent',None) and not resume else []),
                            timeout_minutes=minutes+10 if economy_3702 else None)
                    finally:
                        if previous_stop is None:
                            os.environ.pop('HSRL_STOP_FILE', None)
                        else:
                            os.environ['HSRL_STOP_FILE'] = previous_stop
                    latest = health()
                    if getattr(args,'curriculum_parent',None):
                        freeze_checkpoint(train,out/'segment-checkpoints'/f"step_{int(latest['version']):06d}",int(latest['cum_decisions']))
                    paused = pause_if_requested()
                    if paused:
                        return paused
                    if int(latest['cum_decisions']) <= current:
                        raise RuntimeError('训练无进展，工程异常停止')
                    write(out/'status.json', dict(state='training', latest=latest, stages=stages,
                        stopped_for_performance=False, target_decisions=end, champion=champion))
                if not (train/'latest.json').exists() or int(read(train/'latest.json')['cum_decisions']) < target:
                    break
            record = freeze_checkpoint(train, milestone, target)
            health()
            if time.monotonic()+10*60 >= train_deadline:
                stage = dict(target=target, checkpoint=record, stopped_for_performance=False,
                    monitor_skipped='保留剩余学习窗口，终评仍记录最新模型')
                write(milestone/'report.json', stage)
                stages.append(stage)
                continue
            pairs = pair(milestone/'model.pt', milestone, args.monitor_games, 11000000)
            stage = dict(target=target, checkpoint=record, paired=pairs,
                stopped_for_performance=False, observation='记录升降与CI，任何有限名次均不影响继续训练')
            if getattr(args, 'bot_3702', None):
                stage['diagnostics'] = str(milestone/'diagnostics.json')
                stage['economy_reports'] = {
                    'reference': str(out/'baselines'/f'hard-{args.monitor_games}-11000000.economy.json'),
                    'candidate': str(milestone/'hard.economy.json'),
                }
            write(milestone/'report.json', stage)
            stages.append(stage)
            if champion is None or pairs['hard']['mean_rank_after'] <= champion['mean_rank']:
                champion = dict(path=str(milestone/'model.pt'), target=target,
                    sha256=record['sha256']['model.pt'], mean_rank=pairs['hard']['mean_rank_after'])
            write(out/'status.json', dict(state='training', latest=read(train/'latest.json'), stages=stages,
                stopped_for_performance=False, target_decisions=end, champion=champion))
        paused = pause_if_requested()
        if paused:
            return paused
        latest = health() if (train/'latest.json').exists() else None
        candidates = {}
        if latest:
            frozen = out/'last-committed'/f"step_{int(latest['version']):06d}"
            record = freeze_checkpoint(train, frozen, int(latest['cum_decisions']))
            candidates['latest'] = dict(path=str(frozen/'model.pt'), sha256=record['sha256']['model.pt'])
        if champion and not getattr(args,'curriculum_parent',None) and not getattr(args,'source_4159_audit',None) and not getattr(args,'source_7357',None) and (not candidates or champion['sha256'] != candidates['latest']['sha256']):
            candidates['dev_champion'] = champion
        final = {}
        incomplete = []
        write(out/'status.json', dict(state='final_evaluation', latest=latest, stages=stages,
            stopped_for_performance=False, champion=champion))
        # 无论最新模型表现如何都评测最新模型；不只展示择优结果。
        for label, candidate in candidates.items():
            result = {}
            for name in (('hard',) if firestone_only else ('hard', 'bot')):
                paused = pause_if_requested()
                if paused:
                    return paused
                games = args.eval_games if name == 'hard' else min(args.eval_games, args.monitor_games)
                try:
                    before_path = out/'baselines'/f'{name}-{games}-13000000.json'
                    after_path = out/'final'/candidate['sha256']/f'{name}.json'
                    before = evaluate(args.reference_checkpoint, paths[name], before_path, games, 13000000)
                    after = evaluate(candidate['path'], paths[name], after_path, games, 13000000)
                    result[name] = compare_personal(before, after, n_boot=5000, seed=20261103)
                    if economy_3702:
                        write(after_path.parent/'diagnostics.json', dict(paired=result[name],
                              reference=panel_diagnostics(before, before_path),
                              candidate=panel_diagnostics(after, after_path)))
                except TimeoutError:
                    incomplete.append(f'{label}/{name}')
                    break
            final[label] = result
            if incomplete:
                break
        legacy_result=None
        if getattr(args,'curriculum_parent',None) and 'latest' in candidates and not firestone_only:
            try:
                before=evaluate(args.reference_checkpoint,paths['bot'],out/'baselines/legacy-bot-40.json',
                    args.eval_games,13000000,legacy=True)
                after=evaluate(candidates['latest']['path'],paths['bot'],out/'final/legacy-bot-40.json',
                    args.eval_games,13000000,legacy=True)
                legacy_result=compare_personal(before,after,n_boot=5000,seed=20261103)
            except TimeoutError:incomplete.append('legacy-bot-40')
        report = dict(schema='hsrl.sustained_result/v1', complete=True, stop_reason=reason,
            stopped_for_performance=False, budget_complete=bool(latest and int(latest['cum_decisions']) >= end),
            inherited=inherited, requested_cumulative_decisions=end, latest=latest,
            selected_factor=factor, stages=stages, champion=champion, candidates=candidates,
            final_evaluations=final,legacy_bot_40=legacy_result,final_evaluation_incomplete=incomplete,
            learning_activity=learning_activity(train) if (getattr(args,'curriculum_parent',None) or economy_3702) and latest else None,
            elapsed_minutes=(time.monotonic()-start)/60, promote_best=False,
            note='阶段棋力波动只记录；最新模型和开发榜候选分开报告。Test未查询，旧模型未覆盖。')
        write(out/'reports'/f"version-{int(latest['version']) if latest else 0:06d}.json", report)
        write(out/'report.json', report)
        write(out/'status.json',dict(state='complete',latest=latest,stop_reason=reason,stopped_for_performance=False))
        return report


def main():
    settings = read(ROOT/'config/sustained.json')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', default='artifacts/runs/nus-overnight-001')
    p.add_argument('--out', default='artifacts/runs/nus-sustained-001')
    p.add_argument('--reference-checkpoint', default='artifacts/runs/nus-learning-001/experiment/target-000100000/seed-200000000/model.pt')
    p.add_argument('--parent-model-sha', default='867bec277742f65eb2720a9ad2180ac5c19a2f85dcea78fb3ba58846b6facbc8')
    p.add_argument('--parent-trainer-sha', default='f94484b3f7ebac909ba51820ee5cd802f51bf817786cd3f843abbdfeecb588ff')
    p.add_argument('--device', default='cuda')
    p.add_argument('--anchor-iterations', type=int, default=settings['anchor_iterations'])
    p.add_argument('--anchor-beta-tail', type=float, default=settings['anchor_beta_tail'])
    for name in ('budget_decisions', 'monitor_every', 'monitor_games', 'eval_games'):
        p.add_argument('--'+name.replace('_', '-'), type=int, default=settings[name])
    for name in ('wall_minutes', 'reserve_minutes'):
        p.add_argument('--'+name.replace('_', '-'), type=float, default=settings[name])
    a = p.parse_args()
    if a.anchor_iterations < 0 or not 0 <= a.anchor_beta_tail <= 0.1:
        p.error('anchor迭代>=0，尾部系数0..0.1')
    if not 0 < a.reserve_minutes < a.wall_minutes <= 540 or not 1 <= a.budget_decisions <= 15000000 or a.monitor_every <= 0:
        p.error('运行<=9小时，预留终评时间，预算1..15m，监控间隔>0')
    if any(not 1 <= v <= 256 for v in (a.monitor_games, a.eval_games)):
        p.error('每座评测1..256局')
    try:
        run(a)
    except Exception as exc:
        if getattr(a, '_writer_acquired', False):
            error = dict(error=repr(exc), category='engineering_or_execution', note='棋力升降不会触发此路径；保留最后原子快照')
            write(Path(a.out)/'failures'/f'{time.time_ns()}.json', error)
            write(Path(a.out)/'FAILED.json', error)
        raise


if __name__ == '__main__':
    main()
