"""37.0.2 引擎及在线禁用审计后复验旧库；战团文件保持原样。"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD_SOURCE = "8bc8d4f0bd397009"
FIXED_SOURCE = "c7b5f27d6f40d1d2"


def check_compatibility(old, current):
    previous, expected = copy.deepcopy(old), copy.deepcopy(current)
    if previous['engine'].pop('source_hash') != OLD_SOURCE:
        raise ValueError('只允许复验已审计的 37.0.2 修复前引擎')
    if expected['engine'].pop('source_hash') != FIXED_SOURCE:
        raise ValueError('只允许当前已审计的饰品修复，不能为其他机制改动重盖指纹')
    if previous != expected or expected['engine']['game_version'] != '37.0.2.256589':
        raise ValueError('卡表、词表、版本、动作或观测发生变化，不能沿用本迁移')


def revalidate(source, out):
    import hsbg_engine as E
    from hsbg.az.train.personal import fingerprint, library_digest, checked_library
    from workflows.build_3702_bot_bank import DRIVER, SPLITS, strict_audit
    from workflows.mirror_hard_gates import _drive_to_turn
    from workflows.run_learning_experiment import digest, read, write

    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('复验使用全新目录，保留之前的失败证据')
    parent = read(source/'report.json')
    if not parent['complete']:
        raise ValueError('源库尚未完成发布')
    fp = fingerprint()
    old = {split: read(source/split/'manifest.json') for split in SPLITS}
    if old != parent['splits']:
        raise ValueError('源库报告与 manifest 不符')
    for split, manifest in old.items():
        check_compatibility(manifest['fingerprint'], fp)
        if (manifest['split'] != split or manifest['driver'] != DRIVER
                or manifest['digest'] != library_digest(source/split)):
            raise ValueError(f'源库内容或分割错误: {split}')
    out.mkdir(parents=True)
    # 任何失败都留下 complete=false，禁止半成品被后继训练接受。
    write(out/'report.json', dict(complete=False, source=str(source)))
    template = _drive_to_turn(E.Game(seed=730000), 1).state_json()
    manifests, validations = {}, {}
    for split, manifest in old.items():
        shutil.copytree(source/split, out/split)
        count = 0
        for path in sorted((out/split).glob('snap_*.json.gz')):
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                shard = json.load(stream)
            for frame in shard['frames']:
                game = E.Game.from_json(template)
                result = game.apply_turn_snapshots([[0, json.dumps(frame['player'])]], False)
                if (result['applied'] != 1 or result['rejected'] or not result['pool_balance_ok']
                        or not result['pool_species_ok'] or game.invariant_violations()):
                    raise RuntimeError(f'新版逐帧注入失败: {split} {path.name} {frame["turn"]} {result}')
                count += 1
        if count != manifest['validation']['accepted'] or not count:
            raise ValueError(f'复验帧数不符: {split}')
        if library_digest(out/split) != manifest['digest']:
            raise ValueError('复制后库内容发生变化')
        validations[split] = dict(accepted=count, content_unchanged=True)
        manifests[split] = {**manifest, 'fingerprint': fp,
            'engine_revalidation': dict(old_fingerprint=manifest['fingerprint'],
                                        parent_manifest_sha256=digest(source/split/'manifest.json'))}
        write(out/split/'manifest.json', manifests[split])
        checked_library(out/split)
    # Test 只校验格式和注入，不运行对局，不提供选模或调参反馈。
    strict = strict_audit(out, manifests, parent['smoke'], splits=('train', 'dev'))
    record = dict(schema='hsrl.engine_bank_revalidation/v1', source=str(source),
                  parent_report_sha256=digest(source/'report.json'), old_source=OLD_SOURCE,
                  fingerprint=fp, validations=validations, strict_audit=strict)
    write(out/'engine-revalidation.json', record)
    write(out/'report.json', {**parent, 'splits': manifests, 'strict_audit': strict,
          'engine_revalidation': record, 'complete': True})
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    revalidate(args.source, args.out)
