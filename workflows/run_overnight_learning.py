#!/usr/bin/env python3
"""有总时间上限的夜间PPO：先回归/校准/短训，再分段训练与独立终评。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_learning_experiment import read, write, digest, writer_lock, freeze_checkpoint, training_report


def choose_level(rows):
    """Train校准选首次进入适中压力的档；这些工程窗口不是已验证最优超参。"""
    safe = [r for r in rows if r['mean_rank'] <= 4.0]
    if not safe:
        raise RuntimeError('最轻增强也过难，停止而非盲目夜训')
    return next((r for r in safe if r['mean_rank'] >= 2.25), safe[-1])


def run_process(argv, deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 30:
        raise TimeoutError('总时间预算不足，保留原子checkpoint')
    print('[overnight]', ' '.join(map(str, argv)), flush=True)
    p = subprocess.Popen(list(map(str, argv)), cwd=ROOT, start_new_session=os.name != 'nt')
    try:
        if p.wait(timeout=remaining-15):
            raise subprocess.CalledProcessError(p.returncode, argv)
    except subprocess.TimeoutExpired as exc:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], check=False, capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        raise TimeoutError('整组子进程超过夜间总预算，保留已提交快照') from exc


def check_eval(result, sha):
    from hsbg.az.train.mirror_snapshots import EXPECTED_REJECT_REASONS
    if not result['valid'] or result['checkpoint']['sha256'] != sha:
        raise ValueError('缓存评测无效或模型不匹配')
    for row in result['per_seat']:
        m = row.get('mirror', {})
        if any(v for k, v in m.get('mirror_rejected_by_reason', {}).items() if k not in EXPECTED_REJECT_REASONS):
            raise ValueError('评测存在非预期拒绝')


def run(args, execute=None):
    args._writer_acquired = False
    from hsbg.az.train.personal import checked_library, training_source_digest, scale_library, validate_config
    from workflows.prepare_validated_library import prepare
    from workflows.paired_panel_delta import compare_personal
    from workflows.report_personal_run import summarize
    import torch
    import platform
    out = Path(args.out).resolve()
    start = time.monotonic()
    deadline = start+args.wall_minutes*60
    # 留足两个512局终评；较慢节点提前结束训练，绝不把10小时当额外训练预算。
    train_deadline = deadline-args.reserve_minutes*60
    def command(*argv):
        full = [sys.executable, *map(str, argv)]
        if execute:
            execute(full)
        else:
            run_process(full, deadline)
    def evaluate(model, library, path, games, seed):
        if not path.exists():
            command('-m', 'hsbg.az.train.personal', 'eval', '--checkpoint', model,
                '--library', library, '--device', args.device, '--games', games, '--seed', seed, '--out', path)
        result = read(path)
        check_eval(result, digest(model))
        if result['protocol']['library_digest'] != checked_library(library)['digest'] or result['protocol']['seed'] != seed or result['protocol']['games_per_seat'] != games:
            raise ValueError('缓存评测库不符')
        return result
    with writer_lock(out):
        args._writer_acquired = True
        train_source = checked_library(args.train_library)
        dev_source = checked_library(args.dev_library)
        if train_source['split'] != 'train' or not train_source.get('validation') or dev_source['split'] != 'dev':
            raise ValueError('需要已逐帧验证Train及原冻结Dev，不允许Test')
        if digest(args.checkpoint) != args.expected_sha:
            raise ValueError('起点不是指定的冻结B100k模型')
        if getattr(args, 'expected_trainer_sha', None) and digest(args.trainer) != args.expected_trainer_sha:
            raise ValueError('起点trainer与已冻结B100k摘要不符')
        config = validate_config(read(args.config))
        if config.get('iterations') != 0 or config.get('snapshot-every') != 1 or config.get('minutes', 100) > 30:
            raise ValueError('夜训要求iterations=0、每轮快照、每段不超过30分钟')
        plan = dict(schema='hsrl.overnight/v1', checkpoint=digest(args.checkpoint), trainer=digest(args.trainer),
            source=training_source_digest(), runner=digest(__file__), config=config,
            train_digest=train_source['digest'], dev_digest=dev_source['digest'], factors=args.factors,
            probe_games=args.probe_games, monitor_games=args.monitor_games, eval_games=args.eval_games,
            device=args.device, runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, os=platform.system()),
            budget_decisions=args.budget_decisions, wall_minutes=args.wall_minutes, reserve_minutes=args.reserve_minutes)
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise ValueError('夜训计划/源码/库/环境已改变，必须使用新目录')
        write(out/'plan.json', plan)
        probe_cfg = out/'probe.config.json'
        write(probe_cfg, {**config, 'games': args.probe_games, 'seed': 399000000})
        # 失败迭代的seed段回归：优先用用户保存的v25，保持失败现场只读。
        regression_model = Path(args.regression_checkpoint)
        regression_sha = digest(regression_model) if regression_model.is_file() else digest(args.checkpoint)
        regression_source = str(regression_model) if regression_model.is_file() else args.checkpoint
        regcfg = out/'regression.config.json'
        write(regcfg, {**config, 'seed': 200000000, 'games': 16})
        reg = out/'regression'
        if not (reg/'probe.json').exists():
            command('-m', 'hsbg.az.train.personal', 'collect-probe', '--config', regcfg,
                '--checkpoint', regression_source, '--library', args.train_library, '--device', args.device,
                '--iteration', 25, '--out', reg)
        if read(reg/'probe.json')['checkpoint']['sha256'] != regression_sha:
            raise ValueError('失败迭代回归缓存模型不符')
        if read(reg/'probe.json')['library_digest'] != train_source['digest']:
            raise ValueError('失败迭代回归缓存库不符')
        # 构建结果采用临时目录后原子改名，失败产物不覆盖。
        def derived(source, name, factor, dev=False):
            dest = out/'libraries'/name
            if dest.exists():
                checked_library(dest)
                return dest
            scaled = out/'library-build'/f'{name}-scaled'
            valid = out/'library-build'/f'{name}-validated'
            scale_library(SimpleNamespace(library=source, out=str(scaled), factor=factor, from_turn=8, level=2))
            prepare(scaled, valid, derived_dev=dev)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(valid, dest)
            return dest
        rows = []
        for factor in args.factors:
            library = derived(args.train_library, f'train-x{factor:g}', factor)
            probe = out/'calibration'/f'x{factor:g}'
            if not (probe/'probe.json').exists():
                command('-m', 'hsbg.az.train.personal', 'collect-probe', '--config', probe_cfg,
                    '--checkpoint', args.checkpoint, '--library', library, '--device', args.device, '--iteration', 0, '--out', probe)
            data = read(probe/'probe.json')
            if data['checkpoint']['sha256'] != args.expected_sha or data['library_digest'] != checked_library(library)['digest']:
                raise ValueError('校准缓存模型摘要不符')
            ranks = [r['rank'] for r in data['collect']['games_meta']]
            rows.append(dict(factor=factor, library=str(library), mean_rank=sum(ranks)/len(ranks),
                games=len(ranks), probe=str(probe/'probe.json')))
        selected = choose_level(rows)
        write(out/'calibration.json', dict(rows=rows, selected=selected,
            scope='固定Train种子、B100k、t8起整团身材倍率；学习者压力校准，非真人分布复现'))
        hard_dev = derived(args.dev_library, f'dev-x{selected["factor"]:g}', selected['factor'], dev=True)
        bot_dev = derived(args.dev_library, 'dev-control', 1., dev=True)
        monitor_seed, final_seed = 11000000, 12000000
        baseline = {name: evaluate(args.checkpoint, lib, out/f'baseline-{name}.json', args.monitor_games, monitor_seed)
                    for name, lib in [('bot', bot_dev), ('hard', hard_dev)]}
        train = out/'train'
        train_cfg = out/'train.config.json'
        write(train_cfg, config)
        state = torch.load(args.trainer, map_location='cpu', weights_only=False)
        base = int(state.get('extra', {}).get('cum_decisions', 0))
        del state
        final_budget = base+args.budget_decisions
        budgets = sorted({base+min(n, args.budget_decisions) for n in (25000, 500000, 1000000, 2000000, 4000000, args.budget_decisions)})
        stages = []
        stopped = False
        best = None
        for target in budgets:
            milestone = out/'milestones'/f'decisions-{target:09d}'
            if not (milestone/'checkpoint.json').exists():
                while time.monotonic()+35*60 < train_deadline:
                    current = int(read(train/'latest.json')['cum_decisions']) if (train/'latest.json').exists() else base
                    if current >= target:
                        break
                    resume = (train/'latest.json').exists()
                    command('-m', 'hsbg.az.train.personal', 'train', '--config', train_cfg,
                        '--checkpoint', args.checkpoint, '--library', selected['library'], '--device', args.device,
                        '--out', train, '--stop-at-decisions', target,
                        *(['--resume'] if resume else ['--initial-trainer-state', args.trainer]))
                    if int(read(train/'latest.json')['cum_decisions']) <= current:
                        raise RuntimeError('训练无进展')
                if not (train/'latest.json').exists() or int(read(train/'latest.json')['cum_decisions']) < target:
                    break
            saved = freeze_checkpoint(train, milestone, target)
            training_report(train, milestone, int(saved['version']))
            if time.monotonic()+15*60 >= train_deadline:
                break
            pairs = {}
            for name, lib in [('bot', bot_dev), ('hard', hard_dev)]:
                result = evaluate(milestone/'model.pt', lib, milestone/f'{name}.json', args.monitor_games, monitor_seed)
                pairs[name] = compare_personal(baseline[name], result, n_boot=2000, seed=20261102)
            stage = dict(target=target, checkpoint=saved, paired=pairs, stopped=stopped)
            write(milestone/'report.json', stage)
            stages.append(stage)
            # 开发榜只用于记录候选，不限制后续学习剂量。
            if not stopped and (best is None or pairs['hard']['mean_rank_after'] < best['rank']):
                best = dict(path=str(milestone/'model.pt'), target=target, rank=pairs['hard']['mean_rank_after'])
            write(out/'status.json', dict(stages=stages, best=best, stopped=stopped))
        if (train/'history.jsonl').exists():
            history = [json.loads(x) for x in (train/'history.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
            report = summarize(history)
            write(out/'training-report.json', report)
        # 即使时间不足未到下一里程碑，保留最后完整状态，不能冒充目标预算已完成。
        latest = read(train/'latest.json') if (train/'latest.json').exists() else None
        if latest:
            from hsbg.az.train.runlog import recover_committed_record
            recover_committed_record(train/'history.jsonl', train/'snapshots'/f"step_{int(latest['version']):06d}", int(latest['version']))
            freeze_checkpoint(train, out/'last-committed'/f"step_{int(latest['version']):06d}", int(latest['cum_decisions']))
        final_pairs = {}
        final_incomplete = []
        if best:
            for name, lib in [('bot', bot_dev), ('hard', hard_dev)]:
                games = args.eval_games if name == 'hard' else min(args.monitor_games, args.eval_games)
                try:
                    before = evaluate(args.checkpoint, lib, out/f'final-baseline-{name}.json', games, final_seed)
                    after = evaluate(best['path'], lib, out/'final-candidates'/str(best['target'])/f'{name}.json', games, final_seed)
                    final_pairs[name] = compare_personal(before, after, n_boot=5000, seed=20261102)
                except TimeoutError:
                    final_incomplete.append(name)
                    break
        summary = dict(schema='hsrl.overnight_result/v1', complete=True, stopped=stopped,
            budget_complete=bool(latest and int(latest['cum_decisions']) >= final_budget),
            requested_cumulative_decisions=final_budget, latest=latest, stages=stages,
            best=best, final_paired=final_pairs, selected=selected,
            final_evaluation_incomplete=final_incomplete,
            elapsed_minutes=(time.monotonic()-start)/60, promote_best=False,
            note='强面板512局、Bot保留128局终评使用未参与中途选择的新种子；Dev源库仍参与开发，不等同封存Test。旧模型不覆盖。')
        write(out/'reports'/f"version-{int(latest['version']) if latest else 0:06d}.json", summary)
        write(out/'report.json', summary)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='artifacts/runs/nus-overnight-001')
    p.add_argument('--checkpoint', default='artifacts/runs/nus-learning-001/experiment/target-000100000/seed-200000000/model.pt')
    p.add_argument('--trainer', default='artifacts/runs/nus-learning-001/experiment/target-000100000/seed-200000000/trainer.pt')
    p.add_argument('--regression-checkpoint', default='artifacts/runs/nus-learning-001/experiment/seed-200000000/train/snapshots/step_000025/model.pt')
    p.add_argument('--expected-sha', default='0858209b4edba906016adcd7584733f4891539372540bcc8562571eb47d64ba0')
    p.add_argument('--expected-trainer-sha', default='33b9dc1fe71edd37cf5f32a5a175397a6a13c18cc6ec3c1ba843a0c865716a31')
    p.add_argument('--config', default='config/overnight.json')
    p.add_argument('--train-library', required=True)
    p.add_argument('--dev-library', default='artifacts/libraries/bot-v1/dev')
    p.add_argument('--device', default='cuda')
    p.add_argument('--factors', type=float, nargs='+', default=[1.25, 1.5, 2., 3.])
    p.add_argument('--probe-games', type=int, default=32)
    p.add_argument('--monitor-games', type=int, default=16)
    p.add_argument('--eval-games', type=int, default=64)
    p.add_argument('--budget-decisions', type=int, default=8000000)
    p.add_argument('--wall-minutes', type=float, default=510)
    p.add_argument('--reserve-minutes', type=float, default=110)
    a = p.parse_args()
    if not 0 < a.reserve_minutes < a.wall_minutes <= 540 or not 1 <= a.budget_decisions <= 12000000:
        p.error('总时长<=9小时，终评预留>0，总新增决策<=12m')
    if not a.factors or sorted(set(a.factors)) != a.factors or any(f <= 1 or f > 4 for f in a.factors):
        p.error('倍率需递增、不重复，范围(1,4]')
    if any(not 1 <= n <= 256 for n in (a.probe_games, a.monitor_games, a.eval_games)):
        p.error('局数1..256')
    try:
        run(a)
    except Exception as exc:
        if getattr(a, '_writer_acquired', False):
            error = dict(error=repr(exc), note='保留证据与最后原子快照；没有自动绕过质量门或晋级')
            write(Path(a.out)/'failures'/f'{time.time_ns()}.json', error)
            write(Path(a.out)/'FAILED.json', error)
        raise


if __name__ == '__main__':
    main()
