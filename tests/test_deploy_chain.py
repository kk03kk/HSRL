"""部署链路回归：时间轴一致性、配置↔CLI 漂移、GPU 验收报告校验、12h 预算、lr 续训。

这些用例对应 2026-11-07 连续出现的故障（Python 导入 / 预检顺序 / 源码指纹 /
配置兼容 / 强制升本时间轴 / KL 早停），目标是把"排队几小时后才发现"的错误挪到本地。
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/p50-forced-tier.json'
SBATCH = ROOT / 'deploy/slurm/p50_forced.sbatch'
TRAINER_SRC = ROOT / 'workflows/train_entity_sync.py'
OLD_TRAINER = ROOT / 'artifacts/models/nus-1080772/step-007357-360m/trainer.pt'
OLD_MODEL = ROOT / 'artifacts/models/nus-1080772/step-007357-360m/model.pt'


# ── 1. 配置 ↔ CLI 漂移 ──────────────────────────────────────────────────


def _cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def test_every_config_key_is_a_known_option_and_has_a_cli_flag():
    from hsbg.az.train.personal import OPTIONS
    src = TRAINER_SRC.read_text(encoding='utf-8')
    cfg = _cfg()
    unknown = sorted(k for k in cfg if k not in OPTIONS)
    assert not unknown, f'配置里有 personal.OPTIONS 不认识的键：{unknown}'
    missing_flag = sorted(k for k in cfg if f"add_argument('--{k}'" not in src
                          and f'add_argument("--{k}"' not in src)
    assert not missing_flag, f'配置键没有对应的 train_entity_sync 参数：{missing_flag}'


# ── 2. 时间轴：强制升本 / 教师截止 / 战团库必须同界 ────────────────────────


def test_static_schedule_is_relative_and_unified_with_library_ladder():
    from hsbg.az.train.forced_tier import parse_decisions, parse_prob_schedule
    from workflows.run_forced_tier_long import SCHEDULE, config
    cfg = config()                      # config() 内含同界断言，能返回即已一致
    sched = parse_prob_schedule(str(cfg['forced-tier-schedule']))
    assert int(sched[0][0]) == 0, '静态配置必须是相对本轮新增决策'
    assert [int(d) for d, _ in sched] == [int(o) for o, _ in SCHEDULE]
    assert int(parse_decisions(str(cfg['forced-teacher-until']))) == int(sched[-1][0])
    assert [p for _, p in sched] == [1.0, 0.6, 0.3, 0.0]


def test_schedule_is_shifted_to_cumulative_and_matches_boundaries():
    from hsbg.az.train.forced_tier import parse_prob_schedule, prob_at
    from workflows.run_forced_tier_long import shift_forced_schedule
    base = 558873732
    cfg = shift_forced_schedule(_cfg(), base)
    sched = parse_prob_schedule(str(cfg['forced-tier-schedule']))
    # 解析器会在 0 处补一个与首段同值的点；真正的时间轴从 base 开始
    assert [int(d) for d, _ in sched if int(d) >= base] == [
        base, base + 40_000_000, base + 80_000_000, base + 120_000_000]
    assert prob_at(sched, base) == 1.0
    assert prob_at(sched, base + 40_000_000 - 1) == 1.0
    assert prob_at(sched, base + 40_000_000) == 0.6
    assert prob_at(sched, base + 80_000_000) == 0.3
    assert prob_at(sched, base + 120_000_000) == 0.0
    assert int(cfg['forced-teacher-until']) == base + 120_000_000


def test_shift_is_deterministic_for_resume():
    """续训会再次从静态配置平移；结果必须与首次完全一致（否则 plan 比对会失败）。"""
    from workflows.run_forced_tier_long import shift_forced_schedule
    base = 558873732
    a = shift_forced_schedule(_cfg(), base)
    b = shift_forced_schedule(_cfg(), base)
    assert a['forced-tier-schedule'] == b['forced-tier-schedule']
    assert a['forced-teacher-until'] == b['forced-teacher-until']


def test_config_rejects_absolute_schedule(tmp_path, monkeypatch):
    """误把绝对累计值写进静态配置时必须 fail-fast，而不是静默跑错。"""
    import workflows.run_forced_tier_long as runner
    bad = _cfg()
    bad['forced-tier-schedule'] = '558873732:1,598873732:0.6,638873732:0.3,678873732:0'
    path = ROOT / 'artifacts/local/_bad-forced-tier.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bad), encoding='utf-8')
    monkeypatch.setattr(runner, 'CONFIG', str(path.relative_to(ROOT)))
    with pytest.raises(ValueError, match='从 0 开始'):
        runner.config()


def test_config_rejects_mismatched_teacher_cutoff(tmp_path, monkeypatch):
    import workflows.run_forced_tier_long as runner
    bad = _cfg()
    bad['forced-teacher-until'] = '90000000'
    path = ROOT / 'artifacts/local/_bad-forced-tier2.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bad), encoding='utf-8')
    monkeypatch.setattr(runner, 'CONFIG', str(path.relative_to(ROOT)))
    with pytest.raises(ValueError, match='teacher-until'):
        runner.config()


# ── 3. GPU 验收报告校验（纯函数，登录节点可跑）──────────────────────────


def _write_audit(tmp_path, **over):
    from tools.nus_submit import _sha_lf, _training_source_sha
    doc = dict(passed=True, profile='forced',
               training_source_sha256=_training_source_sha(ROOT),
               runner_sha256=_sha_lf(ROOT / 'workflows/run_forced_tier_long.py'),
               libraries={'train': 'x'})
    doc.update(over)
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def test_audit_problem_missing(tmp_path):
    from tools.nus_submit import audit_problem
    why = audit_problem(tmp_path / 'nope.json')
    assert '缺少GPU验收报告' in why and 'forced-gpu' in why


def test_audit_problem_ok(tmp_path):
    from tools.nus_submit import audit_problem
    assert audit_problem(_write_audit(tmp_path), expected_libraries={'train': 'x'}) == ''


def test_audit_problem_wrong_profile(tmp_path):
    from tools.nus_submit import audit_problem
    why = audit_problem(_write_audit(tmp_path, profile='long'))
    assert 'profile=long' in why


def test_audit_problem_stale_source(tmp_path):
    from tools.nus_submit import audit_problem
    why = audit_problem(_write_audit(tmp_path, training_source_sha256='deadbeef'))
    assert '训练源码变化' in why


def test_audit_problem_stale_runner(tmp_path):
    from tools.nus_submit import audit_problem
    why = audit_problem(_write_audit(tmp_path, runner_sha256='deadbeef'))
    assert '长训入口改变' in why


def test_audit_problem_library_drift(tmp_path):
    from tools.nus_submit import audit_problem
    why = audit_problem(_write_audit(tmp_path), expected_libraries={'train': 'y'})
    assert '战团库' in why


# ── 4. 12 小时预算与 Slurm 时间 ─────────────────────────────────────────


def _sbatch_minutes(path):
    text = path.read_text(encoding='utf-8')
    m = re.search(r'#SBATCH\s+--time=(\d+):(\d+):(\d+)', text)
    assert m, '解析不到 #SBATCH --time'
    return int(m.group(1)) * 60 + int(m.group(2)) + float(m.group(3)) / 60


def test_long_run_is_twelve_hours_with_slack():
    from workflows.run_forced_tier_long import main  # noqa: F401  仅确保可导入
    text = (ROOT / 'workflows/run_forced_tier_long.py').read_text(encoding='utf-8')
    wall = float(re.search(r'a\.wall_minutes\s*=\s*([\d.]+)', text).group(1))
    reserve = float(re.search(r'a\.reserve_minutes\s*=\s*([\d.]+)', text).group(1))
    budget = int(re.search(r'a\.budget_decisions\s*=\s*(\d+)', text).group(1))
    limit = _sbatch_minutes(SBATCH)
    assert limit == 720, f'Slurm 时间应为 12h，实际 {limit} 分钟'
    assert 0 < wall <= limit - 30, f'控制预算 {wall} 分钟没有给 Slurm 留 ≥30 分钟余量'
    assert 0 < reserve < wall
    assert wall - reserve >= 600, '训练窗口太短（<10h），12h 预算下不划算'
    assert budget >= 300_000_000, '决策预算低于 12h 可跑量，会让时间预算失效'


def test_curriculum_fits_in_twelve_hours_with_autonomous_tail():
    """课程节点必须能在预算内跑完，且自主段占足够比例。"""
    from workflows.run_forced_tier_long import SCHEDULE
    offsets = [int(o) for o, _ in SCHEDULE]
    # 12h 训练窗口 ≈ (wall-reserve) 分钟；按上一轮 7,095 dec/s、扣掉里程碑评测开销
    # 保守按 6,000 dec/s 估算可达决策量。
    text = (ROOT / 'workflows/run_forced_tier_long.py').read_text(encoding='utf-8')
    wall = float(re.search(r'a\.wall_minutes\s*=\s*([\d.]+)', text).group(1))
    reserve = float(re.search(r'a\.reserve_minutes\s*=\s*([\d.]+)', text).group(1))
    reachable = (wall - reserve) * 60 * 6000
    assert offsets[-1] <= 0.6 * reachable, \
        f'撤除强制的节点 {offsets[-1]} 太晚（可达≈{int(reachable)}），自主段不足'
    tail = reachable - offsets[-1]
    assert tail >= 0.35 * reachable, f'自主段只占 {tail / reachable:.0%}，应 ≥35%'


# ── 4b. sbatch ↔ 启动器 ↔ 入口脚本 三者一致 ─────────────────────────────


def _launcher_tasks():
    src = (ROOT / 'workflows/launch_nus_refinement.py').read_text(encoding='utf-8')
    return dict(re.findall(r"'([a-z0-9-]+)':\s*\('([a-z0-9_]+\.py)'", src))


def _sbatch_phases():
    """每个 sbatch 里的 ``--phase X`` 与 ``--task Y``。"""
    out = {}
    for path in sorted((ROOT / 'deploy/slurm').glob('*.sbatch')):
        text = path.read_text(encoding='utf-8')
        phase = re.search(r'--phase\s+([a-z-]+)', text)
        task = re.search(r'--task\s+([a-z0-9-]+)', text)
        out[path.name] = (phase.group(1) if phase else None,
                          task.group(1) if task else None)
    return out


def test_sbatch_phases_and_tasks_exist():
    tasks = _launcher_tasks()
    for name, (phase, task) in _sbatch_phases().items():
        if name in ('p50_forced.sbatch', 'p50_forced_preflight.sbatch'):
            assert phase in ('forced', 'forced-gpu'), f'{name} phase={phase}'
            assert task in tasks, f'{name} task={task} 不在启动器表里'
            assert (ROOT / 'workflows' / tasks[task]).is_file(), f'{task} 的入口脚本不存在'


def test_preflight_uses_short_partition_and_long_uses_twelve_hours():
    pre = (ROOT / 'deploy/slurm/p50_forced_preflight.sbatch').read_text(encoding='utf-8')
    assert '#SBATCH --partition=gpu\n' in pre
    assert '#SBATCH --time=00:30:00' in pre
    long = SBATCH.read_text(encoding='utf-8')
    assert '#SBATCH --partition=gpu-long' in long


# ── 5. 续训：lr 与累计决策都来自 trainer state ───────────────────────────


pytestmark_slow = pytest.mark.skipif(
    not (OLD_TRAINER.is_file() and OLD_MODEL.is_file()),
    reason='需要本地 7357 完整 trainer（artifacts/，不随 Git 发布）')


@pytestmark_slow
def test_effective_lr_comes_from_trainer_state():
    """配置写 2e-6，实际跑 1.7547666666e-6：这是 state 覆盖 lr_min_ratio/max_updates 的结果。"""
    import torch
    from hsbg.az.train.model import load_checkpoint
    from hsbg.az.train.ppo import PPOConfig
    from hsbg.az.train.ppo_trainer import PPOTrainer
    state = torch.load(str(OLD_TRAINER), map_location='cpu', weights_only=False)
    model, _ = load_checkpoint(str(OLD_MODEL), device='cpu')
    cfg = PPOConfig(**state['config']); cfg.device = 'cpu'; cfg.amp = 'off'
    tr = PPOTrainer(model, cfg, lr_min_ratio=1.0, max_updates=20000, state=state)
    # state 覆盖命令行：调度参数与 base_lrs 都来自检查点
    assert tr.lr_min_ratio == pytest.approx(0.8333333333, rel=1e-9)
    assert int(tr.max_updates) == 10000
    assert tr.scheduler.base_lrs == [pytest.approx(2e-06)]
    assert tr.scheduler.last_epoch == 7357
    lr = tr.optimizer.param_groups[0]['lr']
    assert lr == pytest.approx(1.75476666661762e-06, rel=1e-12)
    # 到 iteration 10000 触底为 2e-6 * 0.8333333333，之后恒定
    for _ in range(10000 - 7357):
        tr.scheduler.step()
    floor = 2e-06 * 0.8333333333
    assert tr.optimizer.param_groups[0]['lr'] == pytest.approx(floor, rel=1e-9)
    tr.scheduler.step()
    assert tr.optimizer.param_groups[0]['lr'] == pytest.approx(floor, rel=1e-9)


def test_cum_decisions_survive_save_and_load(tmp_path):
    """暂停恢复后强制升本的绝对时间轴依赖 trainer.extra['cum_decisions']。"""
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo import PPOConfig
    from hsbg.az.train.ppo_trainer import PPOTrainer
    cfg = PPOConfig(device='cpu', amp='off')
    tr = PPOTrainer(build_net(16), cfg, lr_min_ratio=1.0, max_updates=1000)
    tr.extra['cum_decisions'] = 735_123_456
    path = tr.save(str(tmp_path / 'trainer.pt'))
    again = PPOTrainer.load(build_net(16), path, config=cfg, device='cpu')
    assert int(again.extra['cum_decisions']) == 735_123_456
