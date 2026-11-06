"""单进程lockstep适配器：复用既有fused_act，不增加线程、IPC或另一套采样算法。"""
import numpy as np

from .ppo import PPOEvaluator


class LocalBatchPPOEvaluator(PPOEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_inference_stats()

    def reset_inference_stats(self):
        self.batch_calls = 0
        self.batch_rows = 0
        self.max_batch_rows = 0

    def act_batch(self, features, cards, mask, temperature=1.0, seed=None, bias=None):
        n = int(np.atleast_2d(features).shape[0])
        if n == 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("需要非空批次和正有限温度")
        result = self.fused_act(features, cards, mask,
            temperatures=np.full(n, temperature, dtype=np.float32),
            seed=0 if seed is None else int(seed), bias=bias)
        self.batch_calls += 1
        self.batch_rows += n
        self.max_batch_rows = max(self.max_batch_rows, n)
        return result

    def act(self, features, cards, mask, temperature=1.0, seed=None, bias=None):
        actions, logprobs, values = self.act_batch(np.atleast_2d(features),
            np.atleast_2d(cards), np.atleast_2d(mask), temperature, seed,
            bias=None if bias is None else np.atleast_2d(bias))
        return int(actions[0]), float(logprobs[0]), float(values[0])

    def inference_stats(self):
        return {"mode": "local_fused_batch", "batch_calls": self.batch_calls,
                "batch_rows": self.batch_rows, "max_batch_rows": self.max_batch_rows}
