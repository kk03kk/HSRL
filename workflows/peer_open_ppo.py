"""Run the single-seed open-action PPO pilot or its paired holdout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTION_PROFILE = "human-turn-3702-standard"
EVAL_SEED = 257870000


def train(out: Path) -> None:
    from hsbg.az.train.personal import library_digest
    from hsbg.runtime.artifacts import atomic_json, file_identity
    from hsbg.runtime.provenance import source_identity
    from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol
    from workflows.peer_ppo_panel import PEER_SHA, WEIGHT
    from workflows.peer_ppo_pilot import SEED, allocation_end, environment, read, verified_generation

    source = source_identity(ROOT)
    if source["tracked_dirty"] or out.exists():
        raise ValueError("dirty release or occupied pilot directory")
    initial = file_identity(INITIAL)
    if initial["sha256"] != INITIAL_SHA or file_identity(WEIGHT)["sha256"] != PEER_SHA:
        raise ValueError("starting weights do not match D-152")
    protocol, _ = holdout_protocol()
    recipe = read(ROOT / "config/research/human-turn-3702-ppo.json")
    settings = environment()
    library = Path(settings["HSAI_ARENA_SNAPS"]).parent
    if protocol.engine["source_hash"] != recipe["engine_source_hash"] or any(
            library_digest(library / split) != digest
            for split, digest in recipe["library_digests"].items()):
        raise ValueError("engine or library identity mismatch")
    stop = allocation_end() - 150
    if stop <= time.time() + 300:
        raise ValueError("insufficient allocation time")
    settings.update(
        HSRL_ROOT=str(ROOT), HSAI_ACTION_PROFILE=ACTION_PROFILE, HSAI_ARENA_OUT=str(out / "fit"),
        HSAI_ARENA_CKPT=str(INITIAL.resolve()), HSAI_ARENA_SEED=str(SEED),
        HSAI_ARENA_ITERATIONS="0", HSAI_ARENA_MINUTES="175", HSAI_ARENA_STOP_EPOCH=str(stop),
        HSAI_ARENA_ANCHOR="0.06", HSAI_ARENA_ANCHOR_TAIL="0.04",
    )
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "plan.json", dict(
        source=source, peer=PEER_SHA, initial=initial, settings=settings,
        seed=SEED, evaluation_seed=EVAL_SEED, protocol=protocol.as_dict(),
        protocol_sha256=protocol.sha256(), libraries=recipe["library_digests"],
        engine_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT / "config/live/model.json"), job=os.environ["SLURM_JOB_ID"],
    ))
    subprocess.run(["bash", "deploy/slurm/stage_arena.sbatch"], env=os.environ | settings, check=True)
    atomic_json(out / "complete.json", verified_generation(out / "fit", SEED))


def assess(out: Path) -> None:
    from workflows.peer_finetune_holdout import STANDARD_PROTOCOL
    from workflows.peer_ppo_pilot import SEED, read, verified_generation

    if (read(out / "plan.json")["protocol_sha256"] != STANDARD_PROTOCOL
            or read(out / "complete.json") != verified_generation(out / "fit", SEED)):
        raise ValueError("pilot protocol or final seal mismatch")
    subprocess.run([
        sys.executable, "workflows/peer_finetune_holdout.py", "--segment", str(out / "fit"),
        "--out", str(out / "assessment"), "--seed", str(EVAL_SEED),
        "--games", "512", "--protected", "256",
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "assess"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE=ACTION_PROFILE, HSAI_HUMAN_TAIL="last-observed-v1")
    (train if args.stage == "train" else assess)(args.out.resolve())
