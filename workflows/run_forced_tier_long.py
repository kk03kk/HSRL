"""强制升本（forced tier-up）长训入口：7357 完整状态分叉 + 加压阶梯 + 逐步解除强制。

与 ``run_7357_long.py`` 的区别（每一处都有机制依据，见 docs/FORCED_TIER.md）：

1. **新配方** ``config/p50-forced-tier.json``（不再断言"必须等于上一轮配方"）：
   lr 量级不变、entropy 0.002→0.008、target-kl 0.05、anchor 0.02→0.01 且
   **重建 anchor**、value-coef 不变；并开启强制升本计划 ``2:2,5:3,6:4,8:5``。
2. **训练场阶梯**从 ``train-start``（上一库实测水平，T12=1135）起步，按
   ``0 / 40M / 80M / 120M`` 逐级升到 ``ramp1 / ramp2 / train``——**与解除强制的
   时间点对齐**：帮助越少，对手越强，逼它把学到的升本能力转成自主决策。
3. 强制概率 ``1.0 → 0.6 → 0.3 → 0``，教师标签（DAgger）到 120M 为止；
   120M 之后 **完全自主**，因此长训后段本身就是"解除规则"的验证窗口。
   12 小时预算下（按 6,000–7,100 dec/s 估算约 230–270M 新决策）自主段约占 45–55%。

安全约束与上一轮相同：只在 Slurm 计算节点运行、输出目录非空必须显式 resume、
``STOP_REQUESTED`` 先改名再恢复、失败写 ``failure-states``、完成不重复训练。
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_SHA = '60594db00a627884d47afd4a326639ab9b1fa7c48eb88618399051d971dcff82'
TRAINER_SHA = '13ef5b03338a7fd31a96a7e024f4431e2dd41183aa49f590d026299a41aca2ab'
SOURCE_CUM = 558873732
#: 训练场阶梯：与解除强制的时间点对齐（0/40M/80M/120M，本轮 12 小时预算）
SCHEDULE = [[0, 'warm0'], [40000000, 'warm1'], [80000000, 'warm2'], [120000000, 'train']]
CONFIG = 'config/p50-forced-tier.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def defaults(p):
    p.add_argument('--source-7357', type=Path,
                   default=os.environ.get('HSAI_SOURCE_7357',
                                          '../HSRL/personal/runs/nus-4159-economy-long-002/milestones/decisions-558819917'))
    p.add_argument('--source-run', type=Path,
                   default=os.environ.get('HSAI_SOURCE_RUN',
                                          '../HSRL/personal/runs/nus-4159-economy-long-002'))
    p.add_argument('--bot-3702', type=Path,
                   default=os.environ.get('HSAI_P50_BANK', 'artifacts/libraries/bot-p50-3702-v1'))
    p.add_argument('--gpu-audit', type=Path,
                   default=os.environ.get('HSAI_P50_AUDIT', 'artifacts/runs/p50-forced-tier-preflight'))
    return p


def shift_forced_schedule(cfg, base):
    """把强制升本的**相对**计划（本轮新增决策）平移成**绝对**累计决策。

    训练器（``train_entity_sync``）拿到的是 ``cum_decisions``，它从 7357 的
    558,873,732 起算；而配置里 ``0:1.0,90M:0.6,...`` 指的是"本轮新增"。
    不平移的话所有阈值都在过去，``prob_at`` 会直接返回 0.0，强制一次都不会发生
    （2026-11-07 的 GPU 预检就是这样空转的）。
    """
    from hsbg.az.train.forced_tier import parse_decisions, parse_prob_schedule
    out = dict(cfg)
    spec = str(cfg.get('forced-tier-schedule', '') or '').strip()
    if spec:
        sched = parse_prob_schedule(spec, default=float(cfg.get('forced-tier-prob', 1.0)),
                                    require_zero_start=True)
        out['forced-tier-schedule'] = ','.join(
            f"{int(base) + int(d)}:{p:g}" for d, p in sched)
    until = str(cfg.get('forced-teacher-until', '') or '').strip()
    if until:
        out['forced-teacher-until'] = str(int(base) + int(parse_decisions(until)))
    return out


def config():
    """本轮配置（结构校验交给 personal.validate_config）。"""
    cfg = read(ROOT / CONFIG)
    from hsbg.az.train.personal import config_drift, validate_config
    # 配置漂移门禁：除本轮显式批准的键以外，必须与快速基线配方完全一致。
    # 2026-11-07 的 GPU 验收（1091647）两轮 33m48s，而基线三轮 4m54s；先把"配方
    # 是否被无意改动"变成机检项，再谈吞吐。
    drift = config_drift(cfg, read(ROOT / 'config/p50-7357-train.json'))
    if drift:
        raise ValueError('配置意外偏离快速基线 config/p50-7357-train.json: '
                         + ','.join(drift))
    cfg = validate_config(cfg)
    if not str(cfg.get('forced-tier-plan', '')).strip():
        raise ValueError('本轮必须给出 forced-tier-plan；空计划等于关闭强制升本')
    if float(cfg.get('anchor-beta', 0.0)) > 0.05:
        raise ValueError('anchor-beta 必须 <= 0.05（anchor 指向更早旧策略，强约束会拖住策略）')
    if int(cfg.get('reset-anchor', 0)) != 1:
        raise ValueError('必须 reset-anchor=1：否则 anchor 仍是继承来的旧策略')
    # ── 时间轴一致性（本轮审计要求）─────────────────────────────────────────
    # 静态配置必须是"相对本轮新增决策"（首段 0），绝对化由 shift_forced_schedule 在
    # fork 时完成；强制概率节点必须与训练场阶梯完全同界，teacher 截止点必须等于撤除
    # 强制的那个节点。三条任一条不满足都在启动期 fail-fast，不让长训跑到一半才发现。
    from hsbg.az.train.forced_tier import parse_decisions, parse_prob_schedule
    sched = parse_prob_schedule(str(cfg.get('forced-tier-schedule', '')),
                                default=float(cfg.get('forced-tier-prob', 1.0)),
                                require_zero_start=True)
    if int(sched[0][0]) != 0:
        raise ValueError('forced-tier-schedule 必须从 0 开始（相对本轮新增决策）；'
                         f'得到首段 {int(sched[0][0])}——是不是误写成了绝对累计值？')
    offsets = [int(d) for d, _ in sched]
    lib_offsets = [int(o) for o, _ in SCHEDULE]
    if offsets != lib_offsets:
        raise ValueError(f'强制升本时间轴 {offsets} 与训练场阶梯 {lib_offsets} 不一致：'
                         '两者必须同界，否则"帮助越少、对手越强"的对齐关系会被破坏')
    until = parse_decisions(str(cfg.get('forced-teacher-until', '') or 0))
    if int(until) != offsets[-1]:
        raise ValueError(f'teacher-until 必须等于撤除强制的节点 {offsets[-1]}，得到 {int(until)}')
    return cfg


def check_inputs(args, require_gpu=True):
    """与上一轮相同的来源/库校验；只把"配方必须不变"换成"新配方自身合法"。"""
    source, old, bank = Path(args.source_7357), Path(args.source_run), Path(args.bot_3702)
    for p in ([source / n for n in ('model.pt', 'trainer.pt', 'checkpoint.json')]
              + [old / n for n in ('plan.json', 'report.json', 'train.config.json')]
              + [bank / 'report.json']):
        if not p.is_file() or not p.stat().st_size:
            raise ValueError('缺少真实输入:' + str(p))
    cp = read(source / 'checkpoint.json')
    if (cp['version'] != 7357 or cp['actual_decisions'] != SOURCE_CUM
            or cp['sha256'] != {'model.pt': MODEL_SHA, 'trainer.pt': TRAINER_SHA}):
        raise ValueError('7357来源元数据不符')
    report = read(bank / 'report.json')
    if (not report.get('complete') or report.get('smoke')
            or report.get('primary_statistic') != 'P50(sum_attack+sum_health)'):
        raise ValueError('需要正式P50库验收报告')
    # 校准档位：新旧库 driver 相同、目录结构相同，只有这两个字段能区分；指错
    # HSAI_P50_BANK 就会拿上一轮"T12–T19 是模型 3–22 倍"的库跑满 12 小时。
    from hsbg.az.train.personal import text_sha256
    curve = Path(os.environ.get('HSAI_P50_CURVE',
                                'config/opponent-p50-slightly-weaker.json'))
    if not curve.is_file():
        raise ValueError('缺少本轮对手曲线:' + str(curve))
    spec = read(curve)
    if report.get('calibration_profile') != spec.get('calibration_profile'):
        raise ValueError(f'训练库校准档不符：库={report.get("calibration_profile")!r}，'
                         f'要求={spec.get("calibration_profile")!r}；'
                         '旧库不是"稍弱于模型"的校准，不能用于本轮')
    if str(report.get('curve_sha256') or '') != text_sha256(curve):
        raise ValueError('训练库不是用当前曲线构建的（curve_sha256 不符）：先重跑 --phase cpu')
    for key in ('train-start', 'train-ramp1', 'train-ramp2', 'train', 'dev', 'test'):
        m = read(bank / key / 'manifest.json')
        if m != report['splits'][key] or m['driver'] != 'bot_firestone_p50_3702':
            raise ValueError('库清单不符:' + key)
        if not next((bank / key).glob('snap_*.json.gz'), None):
            raise ValueError('只有库元数据，没有快照:' + key)
    cfg = config()
    if require_gpu:
        gpu = read(Path(args.gpu_audit) / 'report.json')
        if not gpu.get('passed') or gpu['source_sha256'] != cp['sha256']:
            raise ValueError('GPU验收来源不一致')
        if gpu.get('libraries') != {k: v['digest'] for k, v in report['splits'].items()}:
            raise ValueError('GPU验收库不一致')
    return dict(passed=True, version=7357, scope='metadata_file_precheck_only',
                bank=str(bank), gpu_required=require_gpu,
                config=CONFIG, table=SCHEDULE, forced_tier=cfg['forced-tier-plan'])


def inputs(args):
    from hsbg.az.train.personal import (checked_library, fingerprint, sha256,
                                        write_json, training_source_digest)
    from workflows.audit_full_trainer import audit
    check_inputs(args, require_gpu=not getattr(args, 'preflight_mode', False))
    bank = Path(args.bot_3702).resolve()
    old = Path(args.source_run).resolve()
    source = Path(args.source_7357).resolve()
    out = Path(args.out).resolve()
    keys = {'train': 'train', 'hard': 'dev', 'warm0': 'train-start',
            'warm1': 'train-ramp1', 'warm2': 'train-ramp2'}
    paths = {k: bank / v for k, v in keys.items()}
    manifests = {k: checked_library(p) for k, p in paths.items()}
    actual = fingerprint()
    cfg = config()
    base_decisions = int(read(source / 'checkpoint.json')['actual_decisions'])
    cfg = shift_forced_schedule(cfg, base_decisions)
    proof = audit(source, 7357, {'model.pt': MODEL_SHA, 'trainer.pt': TRAINER_SHA},
                  device='cpu')
    if not getattr(args, 'preflight_mode', False):
        gpu = read(Path(args.gpu_audit) / 'report.json')
        if gpu['fingerprint'] != actual or gpu['training_source_sha256'] != training_source_digest():
            raise ValueError('GPU验收后代码或引擎改变，必须重验')
    frozen = out / 'initial-state'
    frozen.mkdir(parents=True, exist_ok=True)
    for name, digest in proof['sha256'].items():
        target = frozen / name
        if target.exists():
            if sha256(target) != digest:
                raise ValueError('冻结源不同，保留并停止')
        else:
            tmp = target.with_suffix('.pt.tmp')
            shutil.copyfile(source / name, tmp)
            os.replace(tmp, target)
    inherited = read(source / 'checkpoint.json')
    parent = out / 'provenance-source'
    parent.mkdir(exist_ok=True)
    # 真实来源报告与父计划摘要；不伪造上游 complete/plan
    write_json(parent / 'report.json', read(old / 'report.json'))
    write_json(parent / 'source-fork.json', dict(
        source=inherited, source_run=str(old),
        source_plan_sha256=sha256(old / 'plan.json'),
        source_report_sha256=sha256(old / 'report.json'),
        restore_audit=proof, config=cfg, config_file=CONFIG,
        forced_tier_table=SCHEDULE,
        forced_tier_schedule=cfg.get('forced-tier-schedule'),
        forced_teacher_until=cfg.get('forced-teacher-until'),
        forced_tier_base_decisions=base_decisions,
        # 对手库身份：本轮"稍弱于模型"的校准档 + 曲线摘要 + 库报告摘要，
        # 让 12 小时后的复盘能证明训练用的是哪套对手。
        bank=str(bank), bank_report_sha256=sha256(bank / 'report.json'),
        calibration_profile=read(bank / 'report.json').get('calibration_profile'),
        bank_curve_sha256=read(bank / 'report.json').get('curve_sha256'),
        bank_model_reference=read(bank / 'report.json').get('model_reference')))
    args.reference_checkpoint = str(frozen / 'model.pt')
    args.train_schedule = SCHEDULE
    return parent, frozen, inherited, 'p50-forced-tier-v1', paths, manifests, cfg, {}


def main():
    p = defaults(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--out', type=Path,
                   default=os.environ.get('HSAI_LONG_OUT', 'artifacts/runs/p50-forced-tier-long-001'))
    p.add_argument('--resume', action='store_true', default=os.environ.get('HSAI_RESUME') == '1')
    p.add_argument('--check-inputs-only', action='store_true')
    a = p.parse_args()
    if a.check_inputs_only:
        print(json.dumps(check_inputs(a), ensure_ascii=False, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('只能在Slurm计算节点运行')
    out = a.out
    if (out.exists() and any(not x.name.startswith('visibility-') for x in out.iterdir())
            and not a.resume):
        p.error('输出非空，必须显式resume或新目录')
    if a.resume and (out / 'STOP_REQUESTED').exists():
        p.error('先重命名STOP_REQUESTED保留暂停记录')
    if (out / 'report.json').exists() and read(out / 'report.json').get('complete'):
        print('本轮已完成，不重复训练')
        return
    # 12h 预算：wall 690min（11.5h）+ Slurm 30min 余量；45min 终评预留。
    # 训练窗口 ≈10.75h，按上一轮 7,095 dec/s 与 30M 一次的 96 局评测开销估算
    # ≈260M 新决策 ⇒ 自主段（>120M）约占 54%。
    a.wall_minutes = 690
    a.reserve_minutes = 45
    a.budget_decisions = 400000000
    a.monitor_every = 30000000
    a.monitor_games = 12
    a.eval_games = 192
    a.strict_mirror = True
    a.curriculum_parent = None
    a.continue_from = None
    a.device = 'cuda:0'
    a._writer_acquired = False
    os.environ['HSRL_FAILURE_DIR'] = str(out / 'failure-states')
    os.environ['HSBG_ENGINE_GUARD_DIR'] = str(out)
    from workflows.run_sustained_learning import run
    from workflows.run_curriculum_long import record_failure
    try:
        run(a)
    except Exception as exc:
        if a._writer_acquired:
            record_failure(out, exc)
        raise


if __name__ == '__main__':
    main()
