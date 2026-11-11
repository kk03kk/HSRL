#!/usr/bin/env python3
"""Behaviour audit for checkpoints: action mix, repeats, seat stability, bf16 flips.

This is an audit tool, not an evaluation protocol.  It runs the same
`PPOGreedyPolicy` / `evaluate_policy` path used by the gauntlet, with
`collect_economy` and `collect_action_hist` enabled, then reports:

- mean rank / top4 / win rate and rank distribution per seat;
- action-group and per-action histograms, EndTurn/refresh/buy/play rates;
- consecutive-repeat rate and forced-action count;
- bf16-vs-fp32 greedy argmax flip rate on the collected observations.

Usage:
    python workflows/audit_model_behavior.py \
      --model V1.1=artifacts/models/v1.1/model.pt \
      --model final=artifacts/models/league-final-20261111-step014858/model.pt \
      --games 48 --seats 0,1,7 --out artifacts/runs/audit-behavior-001 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from hsbg.az.train.common import ACTION_SEGMENTS, group_of  # noqa: E402
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy  # noqa: E402
from hsbg.az.train.model import load_checkpoint  # noqa: E402
from hsbg.az.train.personal import inspect_checkpoint  # noqa: E402
from hsbg.az.train.ppo import PPOEvaluator, PPOGreedyPolicy  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def summarize_economy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    group_totals: Counter = Counter()
    turns = actions = repeats = 0
    forced = 0
    for row in rows:
        for turn in row.get("end_turn_economy") or []:
            turns += 1
            actions += int(turn.get("actions") or 0)
            repeats += int(turn.get("consecutive_repeats") or 0)
            forced += int(bool(turn.get("forced")))
            for name, count in (turn.get("action_counts") or {}).items():
                group_totals[str(name)] += int(count)
    return {
        "end_turns": turns,
        "actions_before_end": actions,
        "consecutive_repeats": repeats,
        "repeat_rate": (repeats / actions) if actions else None,
        "forced_end_turns": forced,
        "action_groups": dict(sorted(group_totals.items())),
        "per_end_turn": ({k: (v / turns if turns else None)
                          for k, v in sorted(group_totals.items())} if turns else {}),
    }


def action_entropy(hist: Dict[str, int]) -> Dict[str, Any]:
    counts = np.asarray([int(v) for v in hist.values() if int(v) > 0], dtype=np.float64)
    if counts.size == 0:
        return {"entropy": None, "effective_actions": None, "top": []}
    p = counts / counts.sum()
    return {
        "entropy": float(-(p * np.log(p)).sum()),
        "effective_actions": float(math.exp(float(-(p * np.log(p)).sum()))),
        "top": sorted(((int(k), int(v)) for k, v in hist.items()),
                      key=lambda kv: -kv[1])[:20],
    }


def bf16_flip_rate(model: Any, examples: List[Dict[str, Any]], device: str,
                   batch: int = 256) -> Dict[str, Any]:
    if not examples:
        return {"examples": 0, "flips": None, "flip_rate": None}
    import torch
    fp = PPOEvaluator(model, device=device)
    bf = PPOEvaluator(model, device=device)
    bf.amp = "bf16"
    flips = total = 0
    max_delta = 0.0
    for start in range(0, len(examples), int(batch)):
        chunk = examples[start:start + int(batch)]
        f = np.stack([np.asarray(x["features"], dtype=np.float32) for x in chunk])
        c = np.stack([np.asarray(x["cards"], dtype=np.int64) for x in chunk])
        masks = np.stack([np.asarray(x["mask"], dtype=bool) for x in chunk])
        with torch.no_grad():
            lf, _ = fp.tensor_logits(f, c, None)
            lb, _ = bf.tensor_logits(f, c, None)
        lf = lf.cpu().numpy()
        lb = lb.cpu().numpy()
        neg = np.full_like(lf, -1e30, dtype=np.float32)
        af = np.where(masks, lf, neg).argmax(axis=1)
        ab = np.where(masks, lb, neg).argmax(axis=1)
        flips += int((af != ab).sum())
        total += int(af.size)
        max_delta = max(max_delta, float(np.abs(lf - lb).max()) if lf.size else 0.0)
    return {"examples": total, "flips": flips,
            "flip_rate": (flips / total) if total else None,
            "max_logit_delta": max_delta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="label=path (repeatable)")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--seats", default="0,1,7")
    ap.add_argument("--seed", type=int, default=950000)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--examples", type=int, default=600,
                    help="max observations kept per seat for bf16 flips")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output exists and is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    seats = [int(x) for x in str(args.seats).split(",") if x.strip()]
    models = []
    for spec in args.model:
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"missing checkpoint: {p}")
        models.append((label, p))

    report: Dict[str, Any] = {
        "schema": "hsai.behavior_audit/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "games_per_seat": int(args.games),
        "seats": seats,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "models": {},
    }
    for label, path in models:
        print(f"[behavior] {label} {path}", flush=True)
        audit = inspect_checkpoint(str(path), args.device)
        model, meta = load_checkpoint(str(path), device=args.device)
        per_seat: Dict[str, Any] = {}
        all_examples: List[Dict[str, Any]] = []
        for seat in seats:
            evaluator = PPOEvaluator(model, device=args.device,
                                     norm_mean=meta.get("norm_mean"),
                                     norm_std=meta.get("norm_std"))
            policy = PPOGreedyPolicy(evaluator, max_actions_per_turn=80,
                                     move_guard=True, move_cap=5,
                                     move_repeat_limit=2, forbid_cancel=True)
            cfg = EvalConfig(games=int(args.games), seed=int(args.seed),
                             policy_seat=int(seat), max_steps=int(args.max_steps),
                             collect_economy=True, collect_action_hist=True,
                             collect_examples=int(args.examples))
            t0 = time.time()
            res = evaluate_policy(policy, cfg)
            examples = res.pop("examples", [])
            all_examples.extend(examples[:int(args.examples)])
            econ = summarize_economy([res])
            hist = action_entropy(res.get("action_hist") or {})
            per_seat[str(seat)] = {
                "mean_rank": res.get("mean_rank"),
                "top4_rate": res.get("top4_rate"),
                "win_rate": res.get("win_rate"),
                "rank_counts": res.get("rank_counts"),
                "games": res.get("games"),
                "decisions": res.get("decisions"),
                "illegal_actions": res.get("illegal_actions"),
                "truncated_games": res.get("truncated_games"),
                "forced_actions": int(getattr(policy, "forced_actions", 0)),
                "seconds": res.get("seconds"),
                "ms_per_decision": res.get("ms_per_decision"),
                "action_hist": hist,
                "economy": {k: v for k, v in econ.items() if k != "per_end_turn"},
                "per_end_turn": econ["per_end_turn"],
                "action_group_names": sorted(ACTION_SEGMENTS),
            }
            print(f"[behavior] seat {seat} rank={res.get('mean_rank')} "
                  f"top4={res.get('top4_rate')} win={res.get('win_rate')} "
                  f"illegal={res.get('illegal_actions')} "
                  f"trunc={res.get('truncated_games')} {time.time()-t0:.1f}s", flush=True)
        flips = bf16_flip_rate(model, all_examples, args.device)
        report["models"][label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "audit": audit,
            "checkpoint_meta": {k: meta.get(k) for k in
                                ("arch", "trainer_updates", "norm_mean", "norm_std")},
            "per_seat": per_seat,
            "bf16_vs_fp32": flips,
        }
        print(f"[behavior] {label} bf16 {flips}", flush=True)
        del model

    write_json(args.out / "behavior.json", report)
    def _f(x: Any) -> str:
        return f"{float(x):.4f}" if isinstance(x, (int, float)) else "—"

    lines = ["# Behaviour audit", "",
             f"- games/seat: {args.games}; seats: {seats}; seed: {args.seed}", "",
             "| model | seat | mean rank | top4 | win | illegal | trunc | repeat rate | entropy | bf16 flip |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in report["models"].items():
        for seat, s in row["per_seat"].items():
            lines.append(
                f"| {label} | {seat} | {_f(s['mean_rank'])} | {_f(s['top4_rate'])} | "
                f"{_f(s['win_rate'])} | {s['illegal_actions']} | {s['truncated_games']} | "
                f"{_f(s['economy']['repeat_rate'])} | {_f(s['action_hist']['entropy'])} | "
                f"{_f(row['bf16_vs_fp32']['flip_rate'])} |")
    (args.out / "behavior.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[behavior] -> {args.out / 'behavior.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
