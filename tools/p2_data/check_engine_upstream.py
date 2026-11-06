"""离线核对固定上游文件清单及逐项登记的 fork 内容；不会声称远端永远最新。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT/'config/engine-upstream-lock.json'


def source_sha(path):
    raw = Path(path).read_bytes().replace(b'\r\n', b'\n').replace(b'taverngym', b'hsbg_engine')
    return hashlib.sha256(raw).hexdigest()


def check(root=ROOT, lock=LOCK):
    record = json.loads(Path(lock).read_text(encoding='utf-8'))
    same = changed = added = 0
    for group in record['groups']:
        base = Path(root)/group['local']
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*.rs')}
        if actual != set(group['files']):
            raise ValueError(f'源码清单变化，需要重新审计: {base}')
        for name, entry in group['files'].items():
            if source_sha(base/name) != entry['fork_sha256']:
                raise ValueError(f'源码与已审计清单不符: {base/name}')
            if entry['upstream_sha256'] == entry['fork_sha256']:
                same += 1
            else:
                if not entry.get('reason'):
                    raise ValueError(f'未登记 fork 原因: {name}')
                changed += entry['upstream_sha256'] is not None
                added += entry['upstream_sha256'] is None
        test_root = Path(root)/group['local'].removesuffix('/src')/'tests'
        for name in group['upstream_tests']:
            if not (test_root/name).is_file():
                raise ValueError(f'遗漏上游测试: {test_root/name}')
    result = dict(upstream=record['commit'], checked_on=record['checked_on'],
                  identical=same, modified=changed, added=added,
                  remote_checked_now=False)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    check()
