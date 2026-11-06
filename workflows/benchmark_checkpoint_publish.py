#!/usr/bin/env python3
"""实测重复序列化与已提交快照发布；只读源文件，不运行PPO。"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def benchmark(model, trainer, out, device='cpu', repeats=3):
    import torch
    from hsbg.az.train.checkpoint_io import publish_checkpoint
    from hsbg.az.train.personal import sha256, write_json
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payloads = [torch.load(p, map_location=device, weights_only=False) for p in (model, trainer)]
    old, new, modes = [], [], []
    with tempfile.TemporaryDirectory(prefix='checkpoint-bench-', dir=out.parent) as temp:
        root = Path(temp)
        for run in range(repeats):
            # 两臂都含权威快照写入，比较整个保存事务，而非只测别名。
            methods = [('duplicate', old), ('publish', new)]
            for method, times in (methods if run % 2 == 0 else methods[::-1]):
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for name, payload in zip(('model', 'trainer'), payloads):
                    source, alias = root/f'{name}.pt', root/f'{name}-flat.pt'
                    # 永远原子替换，避免重写上一次硬链接共享的inode。
                    torch.save(payload, root/'scratch.tmp')
                    (root/'scratch.tmp').replace(source)
                    if method == 'duplicate':
                        torch.save(payload, root/'flat.tmp')
                        (root/'flat.tmp').replace(alias)
                    else:
                        modes.append(publish_checkpoint(source, alias))
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                for name in ('model', 'trainer'):
                    # 摘要校验移出计时；重复torch.save的zip头可不同，只校验发布臂。
                    if method == 'publish' and sha256(root/f'{name}.pt') != sha256(root/f'{name}-flat.pt'):
                        raise RuntimeError('发布文件摘要不同')
    report = dict(schema='hsrl.checkpoint_benchmark/v1', device=device, torch=torch.__version__,
        source_sha256={str(p):sha256(p) for p in (model, trainer)}, repeats=repeats,
        duplicate_s=old, publish_s=new, median_duplicate_s=statistics.median(old),
        median_publish_s=statistics.median(new), modes=modes,
        note='缓存文件系统微基准；非完整训练加速。两臂权威状态相同，GPU不执行更新。')
    report['saving_fraction'] = 1-report['median_publish_s']/report['median_duplicate_s']
    write_json(out, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--trainer', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    print(json.dumps(benchmark(a.model, a.trainer, a.out, a.device), ensure_ascii=False, indent=2))
