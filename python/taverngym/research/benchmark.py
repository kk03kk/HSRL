"""受控候选GPU验收；小样结果仅作功能/吞吐，不自动宣布棋力胜者。"""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import torch
import taverngym as tg

from taverngym.rl.config import Config, identity
from taverngym.rl.doctor import check
from taverngym.rl.train import train, load_policy
from taverngym.offline.evaluate import import_metrics
from .exam import exam


def benchmark(paths, out, *, device="cuda", smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    names = [Path(p).stem for p in paths]
    if not names or len(set(names)) != len(names):
        raise ValueError("distinct candidate config names required")
    results = []
    for name, path in zip(names, paths):
        c = Config.load(path)
        c = replace(c, device=device, iterations=3, bc_waves=2, decision_budget=0,
                    eval_every=3, eval_games=2, eval_seeds=[900000000])
        if smoke:
            c = replace(c, n_games=1, learner_seats=1, dim=16, layers=1, heads=2, hidden=16,
                        threads=1, torch_threads=1, iterations=1, bc_waves=1, eval_games=1)
        check(c)
        torch.set_num_threads(c.torch_threads)
        if device.startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        run = out / name
        train(c, run)
        metrics = import_metrics(run / "metrics.jsonl")
        rows = metrics["phases"].get("ppo", [])
        decisions = sum(r["collection"]["samples"] for r in rows)
        seconds = sum(r["collection"]["seconds"] + r["training"]["seconds"] for r in rows)
        state = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if state["progress"]["iteration"] != c.iterations or state["progress"]["bc_done"] != c.bc_waves:
            raise InterruptedError(f"candidate incomplete; checkpoint preserved at {run}; no next candidate started")
        model = load_policy(run / "latest.pt", c, identity(c))
        assessment = exam(c, model, games=1 if smoke else 2)
        (run / "distribution-exam.json").write_text(json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"candidate": name, "identity": identity(c)["digest"], "parameters": sum(p.numel() for p in model.parameters()),
                  "policy_kind": c.policy_kind, "config": c.to_dict(), "progress": state["progress"],
                  "unique_ppo_decisions": decisions, "collection_plus_update_seconds": seconds,
                  "effective_decisions_per_second": decisions / seconds if seconds else None,
                  "total_seconds_including_exam": time.monotonic() - started,
                  "prefix_seconds": sum(r["training"]["prefix_seconds"] for r in rows),
                  "optimizer_updates": sum(r["training"]["updates"] for r in rows),
                  "kl_early_stopped_waves": sum(bool(r["training"]["kl_early_stop"]) for r in rows),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else None}
        results.append(result)
        # 每候选完成后原子落盘，后续失败不会丢掉先前测量。
        report = {"format": "taverngym.paper-stage-benchmark.v1", "smoke": smoke, "device": device,
                  "native": tg.build_info(), "strength_claim": False, "auto_select_winner": False,
                  "benchmark_complete": len(results) == len(paths), "results": results}
        temp = out / "report.tmp"
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(out / "report.json")
        del model, state
        gc.collect()
    return report
