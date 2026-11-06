"""GPU编号契约：Slurm始终在继承的可见设备空间内使用逻辑编号。"""


def actor_indices(spec, space, *, visible_count, visibility, cuda=True):
    indices = [int(x.strip()) for x in str(spec).split(',') if x.strip()]
    if not indices or any(i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('actor GPU必须是互不重复的非负编号')
    if space == 'physical':
        if visibility:
            raise ValueError('物理编号模式不接受CUDA_VISIBLE_DEVICES；Slurm请使用visible模式，禁止取消资源隔离')
    elif space == 'visible':
        if cuda and any(i >= visible_count for i in indices):
            raise ValueError(f'actor GPU越过当前可见设备范围0..{visible_count-1}')
    else:
        raise ValueError('未知GPU索引空间')
    return indices
