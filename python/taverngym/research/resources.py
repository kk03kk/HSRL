"""Disposable optimizer probe; no weights or partial trajectories enter training."""
from dataclasses import replace
import gc
import time
import torch
from taverngym.rl.model import Policy
from taverngym.rl.collect import collect
from taverngym.rl.train import update, rng_state, restore_rng


def probe(cfg):
    saved = rng_state()
    model = optimizer = trajectories = None
    started = time.monotonic()
    try:
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        if cfg.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = Policy(cfg).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        c = replace(cfg,n_games=4,opponent_mode='bots',learner_seats=1,
                    gradient_accumulation=1,epochs=1,target_kl=100.)
        trajectories, stats = collect(c,None,1986100000,teacher=True)
        # Complete source episodes are used, but the model/optimizer are disposable.
        metrics = update(model,optimizer,trajectories,c)
        result = {'parameters':sum(p.numel() for p in model.parameters()),
                  'microbatch_states':cfg.seq_len*cfg.chunks_per_batch,
                  'optimizer_updates':metrics['updates'],'source_decisions':stats['samples'],
                  'seconds':time.monotonic()-started,'weights_discarded':True}
        if cfg.device.startswith('cuda'):
            torch.cuda.synchronize()
            result.update(peak_allocated=torch.cuda.max_memory_allocated(),
                          peak_reserved=torch.cuda.max_memory_reserved(),
                          device_memory=torch.cuda.get_device_properties(0).total_memory)
        return result
    finally:
        del model,optimizer,trajectories
        gc.collect()
        restore_rng(saved)
        if cfg.device.startswith('cuda'):
            torch.cuda.empty_cache()


def qualify_resources(cfg):
    """Reduce microbatch on actual CUDA OOM; preserve the nominal accumulation size."""
    attempts = []
    current = cfg
    while True:
        try:
            report = probe(current)
            return current, {'attempts':attempts,'successful':report,
                             'effective_max_states':current.seq_len*current.chunks_per_batch*current.gradient_accumulation}
        except torch.cuda.OutOfMemoryError:
            attempts.append({'chunks_per_batch':current.chunks_per_batch,'status':'cuda_oom'})
            if current.chunks_per_batch <= 1:
                raise
            if current.chunks_per_batch % 2:
                raise ValueError('automatic resource scaling requires an even microbatch')
            current = replace(current,chunks_per_batch=current.chunks_per_batch//2,
                              gradient_accumulation=current.gradient_accumulation*2)
