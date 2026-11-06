#!/usr/bin/env python3
"""持续 PPO 控制器，支持严格 37.0.2 课程的一天作业。"""
import argparse
from pathlib import Path
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def record_failure(out, exc):
    """原子记录终止状态，保留最后提交点；不把失败伪装成仍在训练。"""
    from workflows.run_learning_experiment import read, write
    failure = dict(error=repr(exc), category='engineering_or_execution',
                   stopped_for_performance=False, time_ns=time.time_ns())
    status = read(out/'status.json') if (out/'status.json').exists() else {}
    latest = out/'train/latest.json'
    if latest.exists():
        status['latest'] = read(latest)
    status.update(state='failed', stop_reason='engineering_or_execution', failure=failure,
                  stopped_for_performance=False)
    write(out/'failures'/f"{failure['time_ns']}.json", failure)
    write(out/'FAILED.json', failure)
    write(out/'status.json', status)


def main():
    from workflows.run_learning_experiment import read,write
    from workflows.run_sustained_learning import run
    from workflows.prepare_opponent_research import MODELS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='artifacts/runs/nus-human-long-001')
    p.add_argument('--curriculum-parent',default='artifacts/runs/nus-curriculum-001')
    p.add_argument('--continue-from',default=os.environ.get('HSRL_LONG_CONTINUE_FROM'),help='代码修复后在新目录从旧实验latest完整状态接续')
    p.add_argument("--firestone-eot", default=os.environ.get("HSRL_FIRESTONE_EOT"))
    p.add_argument("--bot-3702", default=os.environ.get("HSRL_BOT_3702"))
    p.add_argument("--engineering-report", default=os.environ.get("HSRL_3702_ENGINEERING_REPORT"))
    p.add_argument("--engineering-only", action="store_true", default=os.environ.get("HSRL_3702_ENGINEERING_ONLY") == "1")
    p.add_argument("--comparison-checkpoint", default=os.environ.get("HSRL_3702_COMPARISON"))
    p.add_argument("--recover-from", default=os.environ.get("HSRL_LONG_RECOVER_FROM"))
    p.add_argument("--strict-mirror", action="store_true", default=os.environ.get("HSRL_STRICT_MIRROR") == "1")
    p.add_argument('--wall-minutes',type=float,default=os.environ.get('HSRL_LONG_WALL_MINUTES', '540'))
    p.add_argument('--reserve-minutes',type=float,default=45)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('必须在Slurm计算节点')
    if args.firestone_eot and (not args.strict_mirror or not args.continue_from):
        p.error('Firestone 全库续训要求严格镜像和明确的父实验')
    if args.bot_3702 and (args.firestone_eot or not args.strict_mirror or not args.continue_from):
        p.error('37.0.2 Bot 续训要求单独指定 Bot 库、严格镜像和父实验')
    if args.engineering_only and not args.bot_3702:
        p.error('工程验证入口仅用于新版 Bot 库')
    if args.bot_3702 and not args.engineering_only and not args.engineering_report:
        p.error('新版 Bot 长训需要同配置的工程验证报告')
    if args.recover_from and (not args.bot_3702 or args.engineering_only):
        p.error('故障恢复只用于新版 Bot 长训的新目录分叉')
    max_minutes = 1410 if args.bot_3702 else 540
    if not 0 < args.reserve_minutes < args.wall_minutes <= max_minutes:
        p.error(f'总预算最多 {max_minutes} 分钟，且需预留终评')
    out=Path(args.out)
    os.environ['HSRL_FAILURE_DIR'] = str((out/'failure-states').resolve())
    os.environ['HSBG_ENGINE_GUARD_DIR'] = str(out.resolve())
    if (out/'report.json').exists() and read(out/'report.json')['complete']:
        print('本次有界长训已经完成，保留结果');return
    reference = ROOT / MODELS['b10m'][0]
    if args.strict_mirror and args.continue_from:
        old = Path(args.continue_from)
        version = int(read(old / "train/latest.json")["version"])
        reference = old / "last-committed" / f"step_{version:06d}" / "model.pt"
    args.reference_checkpoint = str(reference.resolve())
    args.device = 'cuda:0'
    # 1B 是安全上界，常规按墙钟结束。
    args.budget_decisions=1000000000;args.monitor_every=30000000
    args.monitor_games=8;args.eval_games=64
    if args.bot_3702:
        args.monitor_games = 12
    try:
        from workflows.run_overnight_learning import run_process
        # 在计算节点先跑700次真实shm读取，Linux同时断言FD无持续增长。
        run_process([sys.executable,'-m','pytest','tests/test_shm_owned.py','-q'],time.monotonic()+120)
        if not args.bot_3702:
            run_process([sys.executable,'workflows/check_actor_tensor_path.py','--checkpoint',args.reference_checkpoint,
                '--device','auto','--require-gpus','2','--out',out/'preflight'/f"actor-{os.environ['SLURM_JOB_ID']}.json"],
                time.monotonic()+300)
        if args.engineering_only:
            from workflows.run_3702_preflight import verify
            verify(args)
        else:
            run(args)
    except Exception as exc:
        if getattr(args,'_writer_acquired',False):
            record_failure(out, exc)
        raise


if __name__=='__main__':main()
