"""新实验完整状态分叉时缩放学习率；保留Adam/RNG/anchor/进度，严格resume不再缩放。"""
import copy
from pathlib import Path
import shutil


def rescale(state,base_lr):
    result=copy.deepcopy(state)
    old=float(result['scheduler']['base_lrs'][0])
    if not 0<base_lr<=old:raise ValueError('新学习率必须为正且不高于源base_lr')
    factor=base_lr/old
    for group in result['optimizer']['param_groups']:
        for key in ('lr','initial_lr'):
            if key in group:group[key]*=factor
    for key in ('base_lrs','_last_lr'):
        result['scheduler'][key]=[float(v)*factor for v in result['scheduler'][key]]
    result['config']['lr']=base_lr
    return result


def materialize(source,out,base_lr):
    import torch
    from workflows.run_learning_experiment import read,write,digest
    source,out=Path(source),Path(out)
    provenance=dict(model=digest(source/'model.pt'),trainer=digest(source/'trainer.pt'),base_lr=base_lr)
    if (out/'checkpoint.json').exists():
        record=read(out/'checkpoint.json')
        if record['lr_fork']!=provenance:raise ValueError('初始完整状态分叉来源漂移')
        for name,sha in record['sha256'].items():
            if digest(out/name)!=sha:raise ValueError('初始状态副本损坏')
        return record
    if out.exists():raise FileExistsError('保留未完成的初始状态分叉目录')
    out.mkdir(parents=True)
    state=torch.load(source/'trainer.pt',map_location='cpu',weights_only=False)
    torch.save(rescale(state,base_lr),out/'trainer.pt')
    shutil.copyfile(source/'model.pt',out/'model.pt')
    record=dict(read(source/'checkpoint.json'),lr_fork=provenance,
        sha256={name:digest(out/name) for name in ('model.pt','trainer.pt')})
    write(out/'checkpoint.json',record)
    return record
