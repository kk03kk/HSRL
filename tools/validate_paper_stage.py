"""一条命令复验完整Python回归、正式模型形状和四候选CPU微型闭环。"""
import argparse
import gc
import json
from pathlib import Path
import time

from local_offline import prepare


def main():
    handles = prepare()
    import pytest
    import torch
    from taverngym.rl.config import Config, identity
    from taverngym.rl.collect import collect
    from taverngym.rl.model import Policy, pack
    from taverngym.research.benchmark import benchmark

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="new result directory; never overwrite a previous run")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    if not out.is_relative_to(root / "runs"):
        p.error("output must be inside this project's runs directory")
    out.mkdir(parents=True, exist_ok=False)
    code = pytest.main(["tests", "-q", f"--junitxml={out / 'tests.xml'}", "-p", "no:cacheprovider"])
    if code:
        raise SystemExit(code)
    torch.set_num_threads(1)
    cfg = Config(device="cpu", n_games=1, learner_seats=1, threads=1,
                 dim=16, layers=1, heads=2, hidden=16)
    tr, _ = collect(cfg, None, 73001, teacher=True)
    obs = max(tr[0]["samples"], key=lambda s: len(s["obs"]["extras"]))["obs"]
    paths = [root / "configs/paper_stage" / f"{n}.json" for n in
             ("legacy128", "compact_gru128", "compact_ff128", "compact_ff192")]
    shapes = []
    for path in paths:
        c = Config.load(path)
        c.device = "cpu"
        started = time.monotonic()
        model = Policy(c)
        inputs = pack([obs, obs], "cpu")
        logits, value, rank, memory = model(inputs, torch.zeros(2, c.hidden))
        assert all(torch.isfinite(v).all() for v in (logits, value, rank, memory))
        (logits[inputs["mask"]].mean() + value.mean() + rank.mean()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        shapes.append({"candidate": path.stem, "parameters": sum(p.numel() for p in model.parameters()),
                       "tokens": getattr(model, "token_count", 125 + len(obs["extras"])),
                       "production_shape_forward_backward": True, "identity": identity(c)["digest"],
                       "seconds_cpu_not_throughput": time.monotonic() - started})
        del model, inputs, logits, value, rank, memory
        gc.collect()
    (out / "production-shapes.json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
    result = benchmark(paths, out / "smoke", device="cpu", smoke=True)
    summary = {"python_suite_passed": True, "production_shapes": len(shapes),
               "micro_candidates_completed": len(result["results"]),
               "gpu_validated": False, "strength_claim": False}
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
