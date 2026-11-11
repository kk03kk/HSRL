#!/usr/bin/env python3
"""Validate a checkpoint on a held-out league mirror split (dev/test).

This is the periodic validation/test hook for long league runs. It loads a
checkpoint and evaluates it on the dev or test EOT mirror pool with the strict
mirror gate, using seeds that are disjoint from training and from the library
source games.

Example:
    python workflows/validate_league_checkpoint.py \
        --checkpoint artifacts/runs/<run>/train/model.pt \
        --library artifacts/libraries/league-eot-v1.1-20261109 \
        --split dev --games 32 --seed 930000 \
        --out artifacts/runs/<run>/eval/segment-04-dev.json --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.evaluate import EvalConfig, evaluate_policy  # noqa: E402
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool  # noqa: E402
from hsbg.az.train.model import load_checkpoint  # noqa: E402
from hsbg.az.train.ppo import PPOEvaluator, PPOGreedyPolicy  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def engine_info() -> Dict[str, Any]:
    import hsbg_engine as E

    return {
        "game_version": E.game_version(),
        "server_rules_version": E.server_rules_version(),
        "build_info": E.build_info(),
        "db_info": E.db_info(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Held-out dev/test validation for a league checkpoint")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--library", type=Path,
                    default=ROOT / "artifacts/libraries/league-eot-v1.1-20261109")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample", default="p75",
                    help="mirror sampling mode: p50/p75/p90/random/max/weak")
    ap.add_argument("--policy-seat", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--max-actions-per-turn", type=int, default=80)
    ap.add_argument("--strict", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be > 0")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    split_root = args.library / args.split
    manifest_path = split_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_frames = int(manifest.get("frames", 0))
    pool = MirrorSnapshotPool(str(split_root), sample=args.sample)
    if pool.bad_files:
        raise SystemExit(f"bad mirror files in {split_root}: {pool.bad_files}")
    if expected_frames and pool.frames != expected_frames:
        raise SystemExit(f"frame count mismatch: pool={pool.frames} manifest={expected_frames}")

    model, meta = load_checkpoint(str(args.checkpoint), device=args.device)
    evaluator = PPOEvaluator(model, device=args.device,
                             norm_mean=meta.get("norm_mean"),
                             norm_std=meta.get("norm_std"))
    policy = PPOGreedyPolicy(evaluator, max_actions_per_turn=args.max_actions_per_turn)
    cfg = EvalConfig(
        games=int(args.games),
        seed=int(args.seed),
        policy_seat=int(args.policy_seat),
        max_steps=int(args.max_steps),
        mirror_pool=pool,
        mirror_strict=bool(int(args.strict)),
    )
    started = time.time()
    result = evaluate_policy(policy, cfg)
    result.pop("examples", None)
    protocol = {
        "schema": "hsai.league_validation/v1",
        "split": args.split,
        "library": str(args.library),
        "library_digest": manifest.get("digest"),
        "library_frames": expected_frames,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "seed": int(args.seed),
        "games": int(args.games),
        "policy_seat": int(args.policy_seat),
        "sample": str(args.sample),
        "strict": bool(int(args.strict)),
        "max_steps": int(args.max_steps),
        "max_actions_per_turn": int(args.max_actions_per_turn),
        "engine": engine_info(),
    }
    protocol_bytes = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        **protocol,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": {
            "arch": meta.get("arch"),
            "trainer_updates": meta.get("trainer_updates"),
            "cum_decisions": (meta.get("extra") or {}).get("cum_decisions")
            if isinstance(meta.get("extra"), dict) else None,
        },
        "result": result,
        "seconds": time.time() - started,
    }
    write_json(args.out, payload)
    print(json.dumps({
        "out": str(args.out),
        "split": args.split,
        "games": int(args.games),
        "mean_rank": result.get("mean_rank"),
        "top4_rate": result.get("top4_rate"),
        "win_rate": result.get("win_rate"),
        "illegal_actions": result.get("illegal_actions"),
        "truncated_games": result.get("truncated_games"),
        "mirror": result.get("mirror"),
        "protocol_sha256": payload["protocol_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
