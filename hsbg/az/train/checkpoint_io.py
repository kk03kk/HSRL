"""发布已提交快照的别名；不重复序列化，不就地修改任何已发布文件。"""
import os
from pathlib import Path
import shutil
import tempfile


def publish_checkpoint(source, target):
    """同卷优先硬链接；不支持时复制。临时文件替换保持旧别名可读。"""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=target.name+'.publish-', dir=target.parent)
    os.close(fd)
    os.unlink(temp)
    try:
        try:
            os.link(source, temp)
            mode = 'hardlink'
        except OSError:
            shutil.copyfile(source, temp)
            mode = 'copy'
        os.replace(temp, target)
        return mode
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def publish_final_checkpoint(out, trainer, model, *, meta, norm_mean=None, norm_std=None,
                             record=None, seed=0, elapsed_s=0, history_len=0):
    """Commit the final model/Trainer as one generation, then advance latest.json.

    The pointer is authoritative. Flat files remain compatibility aliases and
    must never be used to infer cross-file transactional consistency.
    """
    from .model import save_checkpoint
    from hsbg.runtime.artifacts import atomic_json, file_identity
    root = Path(out).resolve()
    generation = root/'snapshots'/f'final_{int(trainer.updates):06d}'
    partial = generation.with_name(generation.name+'.partial')
    if generation.exists() or partial.exists():
        raise FileExistsError(generation)
    partial.mkdir(parents=True)
    trainer.save(str(partial/'trainer.pt'))
    save_checkpoint(model, str(partial/'model.pt'), meta=meta,
                    norm_mean=norm_mean, norm_std=norm_std)
    if record is not None:
        atomic_json(partial/'record.json', record)
    files = {name: file_identity(partial/name) for name in ('model.pt', 'trainer.pt')}
    os.rename(partial, generation)
    for name in ('trainer.pt', 'model.pt'):
        publish_checkpoint(generation/name, root/name)
    manifest = {'version': int(trainer.updates), 'trainer': str(generation/'trainer.pt'),
                'model': str(generation/'model.pt'), 'files': files, 'final': True,
                'cum_decisions': int(trainer.extra.get('cum_decisions', 0)),
                'arch': meta.get('arch'), 'seed': int(seed), 'elapsed_s': elapsed_s,
                'history_len': history_len}
    atomic_json(root/'latest.json', manifest, replace=True)
    return manifest
