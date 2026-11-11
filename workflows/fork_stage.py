#!/usr/bin/env python3
"""Fork a complete trainer into a new stage with a fresh LR horizon/anchor.

The 36h run showed that ``max-updates`` and the LR schedule are restored from
the trainer state, not from the new config.  A new stage that wants a fresh
``lr -> lr_min`` decay must therefore rewrite the scheduler state explicitly.
This tool preserves Adam, RNG, cumulative decisions and update numbering; only
the LR horizon and the anchor are changed.

Usage:
    python workflows/fork_stage.py \
      --source artifacts/models/league-final-20261111-step014858 \
      --out artifacts/models/stage2-20261112 \
      --lr 2e-6 --lr-min-ratio 0.8333333333 --max-updates 25000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True,
                    help="directory with model.pt + trainer.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lr-min-ratio", type=float, default=0.8333333333)
    ap.add_argument("--max-updates", type=int, default=25000,
                    help="LR decay horizon in trainer iterations from the new stage")
    ap.add_argument("--no-reset-anchor", action="store_true",
                    help="keep the source anchor (default: anchor = source model)")
    args = ap.parse_args()

    model_path = args.source / "model.pt"
    trainer_path = args.source / "trainer.pt"
    if not model_path.is_file() or not trainer_path.is_file():
        raise SystemExit(f"missing model.pt/trainer.pt in {args.source}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output exists and is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    state = torch.load(trainer_path, map_location="cpu", weights_only=False)
    if "scheduler" not in state or "optimizer" not in state:
        raise SystemExit("trainer state missing scheduler/optimizer")
    for group in state["optimizer"]["param_groups"]:
        group["lr"] = float(args.lr)
        if "initial_lr" in group:
            group["initial_lr"] = float(args.lr)
    sched = state["scheduler"]
    sched["base_lrs"] = [float(args.lr)] * len(sched.get("base_lrs", [1]))
    sched["_last_lr"] = [float(args.lr)] * len(sched.get("_last_lr", [1]))
    sched["last_epoch"] = 0
    sched["_step_count"] = 0
    state["max_updates"] = int(args.max_updates)
    state["lr_min_ratio"] = float(args.lr_min_ratio)
    state.setdefault("config", {})["lr"] = float(args.lr)
    if not args.no_reset_anchor:
        state["anchor_model"] = {k: v.detach().clone()
                                 for k, v in state["model"].items()}
        state.setdefault("extra", {})["anchor_set"] = True
    torch.save(state, args.out / "trainer.pt")
    shutil.copyfile(model_path, args.out / "model.pt")
    record = {
        "target_decisions": int(state.get("extra", {}).get("cum_decisions", 0)),
        "actual_decisions": int(state.get("extra", {}).get("cum_decisions", 0)),
        "version": int(state.get("updates", 0)),
        "sha256": {"model.pt": sha256(args.out / "model.pt"),
                   "trainer.pt": sha256(args.out / "trainer.pt")},
        "stage_fork": {
            "source": str(args.source),
            "lr": float(args.lr),
            "lr_min_ratio": float(args.lr_min_ratio),
            "max_updates": int(args.max_updates),
            "anchor_reset": not bool(args.no_reset_anchor),
        },
    }
    write_json(args.out / "checkpoint.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
