"""Fail-closed deployment checks; never installs packages or changes cluster configuration."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

import torch
import taverngym as tg
from .config import Config, identity
from .model import Policy, pack


def check(cfg):
    cfg.validate()
    root = Path.cwd()
    if tg.build_is_current(root) is not True:
        raise RuntimeError("run from the deployed project root and rebuild the native extension (source mismatch)")
    if tg.game_version() != "36.6.3.253932":
        raise RuntimeError("engine version changed: review and version the no-Naga training profile")
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        x = torch.ones((32, 32), device=cfg.device)
        if float((x @ x)[0, 0]) != 32: raise RuntimeError("GPU matmul check failed")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and cfg.threads + cfg.torch_threads > int(allocated):
        raise RuntimeError("environment + Torch threads exceed the allocated CPUs")
    env = tg.BatchEnv(1, controlled=range(8), config=cfg.rules)
    if not hasattr(env._env, "training_extras"):
        raise RuntimeError("training bindings not built")
    from .collect import observation
    b, e = env.observe(), env.training_extras()
    # Preserve RNG so doctor cannot alter the initial learner weights/sampling.
    with torch.random.fork_rng(devices=[]):
        m = Policy(cfg).to(cfg.device).eval()
        with torch.no_grad():
            logits, value, _, _ = m(pack([observation(b, e, 0)], cfg.device),
                                     torch.zeros((1, cfg.hidden), device=cfg.device))
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise RuntimeError("model preflight failed")
    result = {"python": sys.version.split()[0], "platform": platform.platform(),
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "device": cfg.device, "engine": tg.game_version(), "native": tg.build_info(),
              "identity": identity(cfg)["digest"],
              "gpu": torch.cuda.get_device_name() if cfg.device.startswith("cuda") else None,
              "policy_kind": cfg.policy_kind, "parameters": sum(p.numel() for p in m.parameters()),
              "entity_tokens": getattr(m, "token_count", "124 slots plus effects"),
              "profile": "no-Naga 36.6.3 corrected first attack + causal batch; no full live-parity claim"}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p.add_argument("--device")
    a = p.parse_args()
    cfg = Config.load(a.config)
    if a.device: cfg.device = a.device
    result = check(cfg)
    if a.out: Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
