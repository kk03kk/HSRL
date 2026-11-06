"""从4159或故障前6160完整状态做2+1轮PPO；不评棋力、不接长训。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_source(failed, expected_version=6160):
    """仅用标准库核对恢复入口；可在登录节点提交作业前执行，不加载模型。"""
    failed = Path(failed).resolve()
    required = ['train/latest.json', 'train/provenance.json', 'plan.json']
    if expected_version == 6160:
        required.append('FAILED.json')
    paths = [failed/name for name in required]
    evidence = dict(failed_run=str(failed), exists={str(p.relative_to(failed)): p.is_file()
                                                  for p in paths})
    latest = failed/'train/latest.json'
    if latest.is_file():
        evidence['latest'] = json.loads(latest.read_text(encoding='utf-8'))
    missing = [name for name, present in evidence['exists'].items() if not present]
    version = evidence.get('latest', {}).get('version')
    snapshot = failed/'train/snapshots'/f'step_{expected_version:06d}'
    evidence['snapshot_files'] = {name: (snapshot/name).is_file()
                                  for name in ('model.pt', 'trainer.pt')}
    evidence['expected_version'] = expected_version
    evidence['passed'] = not missing and version == expected_version and all(evidence['snapshot_files'].values())
    print('[recovery-source] '+json.dumps(evidence, ensure_ascii=False), flush=True)
    if not evidence['passed']:
        raise ValueError('恢复来源未通过：'+json.dumps(evidence, ensure_ascii=False)+
                         '；请核实原始目录，不要伪造FAILED或改写latest版本')
    return evidence


def optimizer_audit(path):
    """只在有界工程验收读CPU状态，不给正式每轮更新增加逐参数CUDA同步。"""
    import torch
    import math
    state=torch.load(path,map_location='cpu',weights_only=False)
    tensors=list(state['model'].values())
    tensors.extend((state.get('anchor_model') or {}).values())
    steps={}
    for parameter,slot in state['optimizer']['state'].items():
        steps[str(parameter)]=float(slot['step'])
        tensors.extend(v for v in slot.values() if isinstance(v,torch.Tensor))
    if (not steps or any(not math.isfinite(v) or v<0 for v in steps.values())
            or any(not torch.isfinite(v).all().item() for v in tensors)):
        raise ValueError('完整 Trainer 模型/Adam非有限或缺少优化器状态')
    return dict(updates=int(state['updates']),adam_steps=steps,tensors_checked=len(tensors))


def check_optimizer_progress(before,after):
    old,new=before['adam_steps'],after['adam_steps']
    if (set(old)-set(new) or after['updates']<=before['updates']
            or any(new[k]<v for k,v in old.items())
            or not any(new[k]>v for k,v in old.items())):
        raise ValueError('Adam计数未真实接续或没有优化进展')


def audit(args):
    from hsbg.az.train.personal import checked_library, fingerprint, inspect_checkpoint, validate_config
    from workflows.revalidate_3702_bank import check_compatibility
    from workflows.run_3702_preflight import ACTION_PROFILE
    from workflows.run_learning_experiment import digest, freeze_checkpoint, read, write, writer_lock, training_report
    from workflows.run_overnight_learning import run_process

    source_run = getattr(args, 'source_4159', None)
    expected_version = 4159 if source_run else 6160
    out, failed, bank = map(lambda p: Path(p).resolve(), (args.out, source_run or args.failed_run, args.bank))
    if out.exists():
        # UUID 启动器只允许提前写 visibility 文件，不能复用部分训练目录。
        if any(not p.name.startswith('visibility-') for p in out.iterdir()):
            raise FileExistsError('恢复审计请用全新目录')
    os.environ['HSRL_FAILURE_DIR'] = str(out/'failure-states')
    os.environ['HSBG_ENGINE_GUARD_DIR'] = str(out)
    with writer_lock(out):
        try:
            source_evidence = check_source(failed, expected_version)
            write(out/'source-preflight.json', source_evidence)
            old = read(failed/'train/provenance.json')
            plan = read(failed/'plan.json')
            latest = read(failed/'train/latest.json')
            if not source_run:
                check_compatibility(old['fingerprint'], fingerprint())
            else:
                current = fingerprint()
                if any(old['fingerprint'].get(key) != current.get(key) for key in
                       ('vocab_sha256', 'actions', 'obs_floats', 'obs_cards')):
                    raise ValueError('4159旧词表/动作/观测契约与当前引擎不一致')
            manifest = checked_library(bank/'train')
            report = read(bank/'report.json')
            if (not report['complete'] or report['smoke'] or not report.get('fresh_generation')
                    or report.get('calibration_profile') != 'economy-window-v2'
                    or report['splits']['train'] != manifest
                    or old['config'] != plan['config']
                    or old['library_digest'] != plan['libraries']['train']):
                raise ValueError('故障配方来源或新版重建库不一致')
            for split in ('train','dev','test'):
                entry=checked_library(bank/split)
                if (entry!=report['splits'][split] or entry['fingerprint']!=fingerprint()
                        or entry.get('source_max_actions_per_turn')!=80
                        or entry['rules_config'].get('excluded_cards')!=
                            manifest['rules_config'].get('excluded_cards')):
                    raise ValueError(f'新库分割来源或规则不符 {split}')
            write(out/'corpus-fork.json',dict(old=plan['libraries'],
                new={k:v['digest'] for k,v in report['splits'].items()},
                reason='在线禁用池与实体生命周期修复后重新生成；不改学习配方'))
            if source_run:
                # 4159继承完整Trainer；经营规则采用本轮已确认配方。
                recipe_run = Path(args.failed_run).resolve()
                config = validate_config({**read(recipe_run/'plan.json')['config'], **ACTION_PROFILE})
            else:
                config = validate_config(plan['config'])
            if any(config.get(k) != v for k, v in {**ACTION_PROFILE, 'mirror-strict': 1,
                                                   'processes': 8, 'games': 512, 'pipeline': 1}.items()):
                raise ValueError('必须复现故障时真实采集拓扑与动作约束')
            source = out/f'source-{expected_version}'
            frozen = freeze_checkpoint(failed/'train', source, latest['cum_decisions'])
            if source_run:
                from workflows.run_3702_preflight import COMPARISON_SHA
                from workflows.migrate_checkpoint_3702 import migrate
                if frozen['sha256']['model.pt'] != COMPARISON_SHA:
                    raise ValueError('4159权重SHA与已确认模型不一致')
                migration = migrate(source/'model.pt', out/'initial-state', source/'trainer.pt')
                write(out/'source-fork.json', dict(source=frozen, migration=migration,
                      reason='用户选择4159重新起跑；保留Adam/RNG/计数，替换新版卡表buffer',
                      config=config))
                source = out/'initial-state'
            before = {name: digest(source/name) for name in ('model.pt', 'trainer.pt')}
            inspect_checkpoint(source/'model.pt', 'cpu')
            deadline = time.monotonic()+25*60
            run_process([sys.executable, 'workflows/check_actor_tensor_path.py', '--checkpoint',
                         source/'model.pt', '--device', 'auto', '--require-gpus', '2',
                         '--out', out/'actor.json'], deadline)
            write(out/'train.config.json', config)
            base = [sys.executable, '-m', 'hsbg.az.train.personal', 'train', '--config',
                    out/'train.config.json', '--checkpoint', source/'model.pt', '--library',
                    bank/'train', '--out', out/'train', '--device', 'cuda:0']
            run_process(base+['--run-iterations', '2', '--initial-trainer-state', source/'trainer.pt'], deadline)
            first = read(out/'train/latest.json')
            run_process(base+['--run-iterations', '1', '--resume'], deadline)
            final = read(out/'train/latest.json')
            optimizer_states=[optimizer_audit(path) for path in (
                source/'trainer.pt',
                out/'train/snapshots'/f"step_{int(first['version']):06d}"/'trainer.pt',
                out/'train/snapshots'/f"step_{int(final['version']):06d}"/'trainer.pt')]
            check_optimizer_progress(optimizer_states[0],optimizer_states[1])
            check_optimizer_progress(optimizer_states[1],optimizer_states[2])
            write(out/'optimizer-audit.json',optimizer_states)
            if (first['version'] != expected_version+2 or final['version'] != expected_version+3
                    or not frozen['actual_decisions'] < first['cum_decisions'] < final['cum_decisions']):
                raise RuntimeError('完整 Trainer 保存/恢复计数不符')
            summary = training_report(out/'train', out, final['version'])
            t, s = summary['totals'], summary['mirror_strict_counts']
            if (not t['optimizer_steps'] or not t['processed_samples']
                    or t.get('engine_quality_checked_games') != 3*config['games']
                    or summary['unexpected_mirror_rejections'] or t['mirror_fallback'] or t['mirror_missing']
                    or not t['mirror_attempt'] == t['mirror_applied'] == s['eligible_live_seats']
                    or not s.get('live_combats')):
                raise RuntimeError('PPO 或严格镜像验收失败')
            if any(digest(source/name) != sha for name, sha in before.items()):
                raise RuntimeError('冻结源文件被改写')
            write(out/'report.json', dict(passed=True, long_training_authorized=False,
                  schema='hsrl.pending_recovery_audit/v1', source=frozen, latest=final,
                  fingerprint=fingerprint(), bank=report, training=summary))
        except Exception as exc:
            from workflows.run_curriculum_long import record_failure
            record_failure(out, exc)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=os.environ.get('HSRL_AUDIT_OUT', 'artifacts/runs/nus-engine-audit-001'))
    p.add_argument('--failed-run', default=os.environ.get('HSRL_FAILED_RUN', 'artifacts/runs/nus-3702-economy-long-001'))
    p.add_argument('--bank', default=os.environ.get('HSRL_AUDITED_BANK', 'artifacts/libraries/bot-firestone-3702-economy-v2-live-v1'))
    p.add_argument('--check-source-only', action='store_true')
    p.add_argument('--source-4159', default=os.environ.get('HSRL_AUDIT_SOURCE_4159'))
    args = p.parse_args()
    if args.check_source_only:
        check_source(args.source_4159 or args.failed_run, 4159 if args.source_4159 else 6160)
        raise SystemExit(0)
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('必须在 Slurm 计算节点运行')
    audit(args)
