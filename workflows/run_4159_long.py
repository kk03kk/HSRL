"""从已通过NUS验收的4159完整Trainer启动24小时以内长训，不采用4162探针权重。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _force_utf8_stdio():
    """让本脚本的输出与区域设置无关。

    ``--check-inputs-only`` 的 stdout 是给机器解析的 JSON，但 Windows 上通过管道
    运行时 Python 默认用 cp936 编码，调用方按 UTF-8 解码就会炸。这里显式把
    stdout/stderr 钉死为 UTF-8，Linux 上本来就是 UTF-8，因此无副作用。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or '').lower().replace('-', '') != 'utf8':
                stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()


def check_input_files(audit, bank):
    """提交前仅用标准库核对真实产物契约，不加载模型、不伪造父计划。"""
    audit,bank = Path(audit).resolve(),Path(bank).resolve()
    required = [audit/p for p in ('report.json','source-fork.json','train.config.json',
                'initial-state/migration.json','initial-state/model.pt','initial-state/trainer.pt')]
    required += [bank/p for p in ('report.json','train/manifest.json','dev/manifest.json')]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError('长训输入缺失或为空：'+json.dumps(missing,ensure_ascii=False))
    read_json = lambda p: json.loads(p.read_text(encoding='utf-8'))
    report=read_json(audit/'report.json');fork=read_json(audit/'source-fork.json')
    config=read_json(audit/'train.config.json');migration=read_json(audit/'initial-state/migration.json')
    library=read_json(bank/'report.json')
    if (not report.get('passed') or report['source']['version']!=4159
            or fork['source']!=report['source'] or fork['config']!=config
            or fork['migration']!=migration or migration['inputs']!=report['source']['sha256']
            or migration.get('fingerprint')!=report['fingerprint']
            or library!=report['bank'] or not library.get('complete') or library.get('smoke')
            or any(read_json(bank/s/'manifest.json')!=library['splits'][s] for s in ('train','dev'))):
        raise ValueError('长训元数据来源、迁移、配置或库不一致')
    return dict(passed=True,source_version=4159,audit=str(audit),bank=str(bank),
                parent_plan_file='source-fork.json',checked_files=len(required),
                scope='标准库元数据与文件存在性预检；权重摘要/实际引擎/GPU仍在计算节点核对')


def inputs(args):
    from hsbg.az.train.personal import checked_library, fingerprint, validate_config
    from workflows.run_learning_experiment import read, write, digest
    from workflows.run_3702_preflight import ACTION_PROFILE, COMPARISON_SHA

    audit, bank, out = (Path(p).resolve() for p in
                        (args.source_4159_audit, args.bot_3702, args.out))
    check_input_files(audit, bank)
    evidence = read(audit/'report.json')
    fork = read(audit/'source-fork.json')
    config = validate_config(read(audit/'train.config.json'))
    actual = fingerprint()
    totals = evidence['training']['totals']
    strict = evidence['training']['mirror_strict_counts']
    if (not evidence['passed'] or evidence['source']['version'] != 4159
            or evidence['source']['actual_decisions'] != 198819917
            or evidence['source']['sha256']['model.pt'] != COMPARISON_SHA
            or evidence['latest']['version'] != 4162 or evidence['fingerprint'] != actual
            or fork['source'] != evidence['source'] or fork['config'] != config
            or not totals['processed_samples'] or not totals['optimizer_steps']
            or totals['engine_quality_checked_games'] != 1536
            or evidence['training']['unexpected_mirror_rejections']
            or not totals['mirror_attempt'] == totals['mirror_applied'] == strict['eligible_live_seats']
            or any(totals.get(k, 0) for k in ('illegal_actions','truncated_games','mirror_errors',
                   'mirror_missing','mirror_fallback','truncated_combats','truncated_recruit_contexts',
                   'turn_limit_end','engine_stopped','mirror_pool_unbalanced','mirror_pool_species_unbalanced'))):
        raise ValueError('4159完整Trainer工程证据、引擎或严格注入验收不符')
    if any(config.get(k) != v for k,v in {**ACTION_PROFILE, 'games':512,'processes':8,
                  'pipeline':1,'mirror-strict':1,'mirror-sample':'random',
                  'epochs':1,'minibatch':1024,'grad-accum':1,'snapshot-every':1}.items()):
        raise ValueError('不允许隐式改变已验收的4159学习配方')
    bank_report = read(bank/'report.json')
    if (bank_report != evidence['bank'] or not bank_report.get('fresh_generation')
            or not bank_report.get('complete') or bank_report.get('smoke')):
        raise ValueError('长训战团库与GPU验收不同')
    manifests = {key:checked_library(bank/split) for key,split in (('train','train'),('hard','dev'))}
    if any(m != bank_report['splits'][split] or m['fingerprint'] != actual
           for key,split in (('train','train'),('hard','dev')) for m in [manifests[key]]):
        raise ValueError('Train/Dev文件或指纹与验收不符')
    migration = read(audit/'initial-state/migration.json')
    if (migration != fork['migration'] or migration['inputs'] != evidence['source']['sha256']
            or migration.get('fingerprint') != actual):
        raise ValueError('4159迁移来源不符')
    source = out/'initial-state'
    source.mkdir(parents=True, exist_ok=True)
    for name,sha in migration['outputs'].items():
        original = audit/'initial-state'/name
        if name not in ('model.pt','trainer.pt') or digest(original) != sha:
            raise ValueError('已验收4159源文件发生变化')
        target = source/name
        if target.exists():
            if digest(target) != sha:
                raise ValueError('本地冻结源与已验收4159不同')
        else:
            temporary = target.with_suffix('.pt.tmp')
            shutil.copyfile(original, temporary)
            os.replace(temporary, target)
    if set(migration['outputs']) != {'model.pt','trainer.pt'}:
        raise ValueError('必须继承完整Trainer，不能只继承权重')
    inherited = {**evidence['source'], 'sha256':migration['outputs'],
                 'original_sha256':migration['inputs']}
    args.reference_checkpoint = str(source/'model.pt')
    write(out/'engineering-link.json', dict(audit=str(audit), report_sha256=digest(audit/'report.json'),
          source_version=4159, probe_latest_not_used=4162, config=config,
          migration=migration, scope='复用通过的PPO/引擎验收，仅接通控制器与评测记录'))
    return audit, source, inherited, 'economy-window-v2', dict(train=bank/'train',hard=bank/'dev'), manifests, config, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=os.environ.get('HSRL_LONG_OUT','artifacts/runs/nus-4159-economy-long-002'))
    parser.add_argument('--source-4159-audit', default=os.environ.get('HSRL_4159_AUDIT','artifacts/runs/nus-4159-engine-audit-002'))
    parser.add_argument('--bot-3702', default=os.environ.get('HSRL_BOT_3702','artifacts/libraries/bot-firestone-3702-economy-v2-live-v1'))
    parser.add_argument('--resume', action='store_true', default=os.environ.get('HSRL_LONG_RESUME')=='1')
    parser.add_argument('--check-inputs-only',action='store_true')
    args = parser.parse_args()
    if args.check_inputs_only:
        print(json.dumps(check_input_files(args.source_4159_audit,args.bot_3702),ensure_ascii=False,indent=2))
        return
    from workflows.run_learning_experiment import read
    from workflows.run_sustained_learning import run
    from workflows.run_curriculum_long import record_failure
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('只能在Slurm计算节点启动')
    out = Path(args.out).resolve()
    if out.exists() and any(not p.name.startswith('visibility-') for p in out.iterdir()) and not args.resume:
        parser.error('已有实验请显式--resume；不覆盖历史目录')
    if args.resume and (out/'STOP_REQUESTED').exists():
        parser.error('恢复前请移走STOP_REQUESTED，保留暂停记录')
    if (out/'report.json').exists() and read(out/'report.json').get('complete'):
        print('本轮已经完成，不重复训练');return
    args.wall_minutes=1410;args.reserve_minutes=45
    args.budget_decisions=1000000000;args.monitor_every=30000000
    args.monitor_games=12;args.eval_games=64
    args.strict_mirror=True;args.curriculum_parent=None;args.continue_from=None
    args.device='cuda:0';args._writer_acquired=False
    os.environ['HSRL_FAILURE_DIR']=str(out/'failure-states')
    os.environ['HSBG_ENGINE_GUARD_DIR']=str(out)
    try:
        run(args)
    except Exception as exc:
        if args._writer_acquired:
            record_failure(out,exc)
        raise


if __name__=='__main__':main()
