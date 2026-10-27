"""成对评测与按大厅统计；独立工具，不更改原有训练选优协议。"""
import json
from pathlib import Path

import numpy as np
import torch

from taverngym.rl.config import Config, identity
from taverngym.rl.train import load_policy
from .capture import capture
from .trace import digest


def checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**state["config"])
    cfg.device = device
    cfg.validate()
    torch.set_num_threads(cfg.torch_threads)
    model = load_policy(path, cfg, identity(cfg))
    return cfg, model


def intervals(values):
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return {"n": 1, "mean": float(x[0]), "ci95": None}
    means = np.random.default_rng(510).choice(x, size=(2000, len(x)), replace=True).mean(1)
    return {"n": len(x), "mean": float(x.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def compare(cfg, policies, seeds, *, out, modes=("greedy", "sample")):
    """每 seed 为一个独立大厅；同 seed/席位/对手进行成对比较。"""
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("distinct nonnegative lobby seeds required")
    if any(m not in ("greedy", "sample") for m in modes):
        raise ValueError("unknown learner mode")
    cfg = Config(**cfg.to_dict())
    cfg.n_games = 1
    results = {}
    for name, model in policies.items():
        for mode in modes:
            key = f"{name}/{mode}"
            records = [capture(cfg, seed=s, actor=model, greedy=mode == "greedy") for s in seeds]
            valid = [r["ranks"][0] for r in records if r["status"] == "completed"]
            results[key] = {"records": records, "rank": intervals(valid),
                            "top4": intervals([int(r <= 4) for r in valid]),
                            "win": intervals([int(r == 1) for r in valid]),
                            "failed_or_truncated": sum(r["status"] != "completed" for r in records)}
    pairs = {}
    keys = list(results)
    for j, a in enumerate(keys):
        for b in keys[j + 1:]:
            pairs[f"{a} minus {b}"] = intervals([
                x["ranks"][0] - y["ranks"][0]
                for x, y in zip(results[a]["records"], results[b]["records"])
                if x["status"] == y["status"] == "completed"])
    report = {"format": "taverngym.comparison.v1", "identity": identity(cfg),
              "protocol": "one rotating learner per lobby vs seven greedy native bots; separate from training selection",
              "seeds": seeds, "results": results, "paired_rank_differences": pairs}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def import_metrics(path):
    """只解析日志已有结果；不 import torch 到服务器登录节点。"""
    phases, parse_errors = {}, []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            parse_errors.append(line_no)
            continue
        if isinstance(row, dict) and "phase" in row:
            phases.setdefault(row["phase"], []).append(row)
    curve = [{"iteration": r.get("iteration"), "loss": r.get("training", {}).get("loss"),
              "collection_rank": r.get("collection", {}).get("mean_rank"),
              "collection_seconds": r.get("collection", {}).get("seconds"),
              "update_seconds": r.get("training", {}).get("seconds")}
             for r in phases.get("ppo", [])]
    validation = [{"iteration": r.get("iteration"), "selection_score": r.get("selection_score"),
                   "panels": {name: panel.get("mean_rank") for name, panel in r.get("results", {}).items()}}
                  for r in phases.get("validation", [])]
    return {"source": str(path), "source_hash": digest(Path(path).read_text(encoding="utf-8")),
            "phases": phases, "invalid_json_lines": parse_errors,
            "bc_waves_recorded": len(phases.get("bc", [])),
            "bc_baseline": next((r for r in validation if r["iteration"] == 0), None),
            "ppo_curve": curve, "validation_curve": validation,
            "checkpoint_inventory": [{"name": p.name, "bytes": p.stat().st_size}
                                     for p in sorted(Path(path).parent.glob("*.pt"))]}


def metrics_html(report):
    """自包含、无脚本的训练曲线；不把训练分布名次冒充固定评测。"""
    import html
    import math
    def plot(rows, key, title):
        points = [(r.get("iteration"), r.get(key)) for r in rows]
        points = [(x, y) for x, y in points if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)]
        if not points:
            return f"<h2>{title}</h2><p>未记录</p>"
        xs, ys = zip(*points)
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        coords = ' '.join(f'{40+(x-xmin)/max(1,xmax-xmin)*700:.2f},{190-(y-ymin)/max(1e-9,ymax-ymin)*150:.2f}' for x, y in points)
        return (f'<h2>{title}</h2><svg viewBox="0 0 800 240" role="img" aria-label="{title}">'
                '<path d="M40 20V200H760" stroke="#94a3b8" fill="none"/>'
                f'<polyline points="{coords}" fill="none" stroke="#5eead4" stroke-width="3"/>'
                f'<text x="40" y="225" fill="white">iteration {xmin} → {xmax}; range {ymin:.4g} → {ymax:.4g}</text></svg>')
    body = plot(report["validation_curve"], "selection_score", "固定验证选优名次（越低越好）")
    body += plot(report["ppo_curve"], "loss", "PPO loss（不等于棋力）")
    body += plot(report["ppo_curve"], "collection_seconds", "每波采集秒数")
    body += '<h2>BC后第0波基线</h2><pre>' + html.escape(json.dumps(report["bc_baseline"], ensure_ascii=False, indent=2)) + '</pre>'
    body += '<h2>Checkpoint清单</h2><pre>' + html.escape(json.dumps(report["checkpoint_inventory"], indent=2)) + '</pre>'
    return ('<!doctype html><meta charset="utf-8"><title>训练结果摘要</title>'
            '<style>body{font:16px system-ui;background:#101827;color:#e2e8f0;max-width:1000px;margin:32px auto}svg{width:100%;background:#1e293b}pre{white-space:pre-wrap}</style>'
            '<h1>训练结果摘要</h1><p>仅展示日志已有数据；空缺表示未记录。训练采样名次不替代独立评测。</p>' + body)


def selection_protocol(cfg, policies, anchor, *, games, split="test"):
    """原训练四分面评测的独立复核入口；不和简化 bot 成对协议混合。

    evaluate 本身恢复 RNG；每个大厅仅一个 learner，原协议 bootstrap 单位为大厅。
    """
    from dataclasses import replace
    from taverngym.rl.train import evaluate
    if anchor is None or any(model is None for model in policies.values()):
        raise ValueError("selection protocol requires network policies and a BC anchor")
    if split not in ("validation", "test") or games < 1:
        raise ValueError("invalid selection protocol split/games")
    c = replace(cfg, eval_games=games)
    if split == "test":
        c.eval_seeds = [1900000000, 1901000000]
    return {"format": "taverngym.selection-comparison.v1", "identity": identity(c), "split": split,
            "protocol": "original training bots/anchor_mix x greedy/sample; one learner per lobby",
            "results": {name: evaluate(c, model, anchor=anchor) for name, model in policies.items()}}
