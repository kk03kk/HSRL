"""按设备合并标量指标传输，避免每个loss单独触发CUDA同步。"""
import math


def scalar_metrics(values):
    import torch
    groups = {}
    for key, tensor in values.items():
        groups.setdefault(tensor.device, []).append((key, tensor.detach().reshape(())))
    result = {}
    for device, group in groups.items():
        if device.type == 'cuda':
            packed = torch.stack([v for _, v in group]).cpu().tolist()
        else:
            packed = [float(v) for _, v in group]
        for (key, _), value in zip(group, packed):
            if not math.isfinite(value):
                raise FloatingPointError(f'PPO非有限指标 {key}={value}，拒绝optimizer.step')
            result[key] = value
    return result
