"""Start an isolated peer PPO segment from a sealed model and full Trainer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.peer_ppo_pilot import SEED, allocation_end, environment, read, verified_generation
from workflows.peer_finetune_holdout import PROTOCOL


def run(args):
    from hsbg.az.train.personal import library_digest
    from workflows import quilboar_study as quill
    source = source_identity(ROOT)
    if source["tracked_dirty"] or args.out.exists() or args.iterations < 0:
        raise ValueError("dirty release, occupied output or invalid budget")
    parent = verified_generation(args.parent, SEED)
    if args.previous_seal:
        seal = read(args.previous_seal)
        if seal["checkpoint"] != parent or seal["commit"] != source["commit"]:
            raise ValueError("previous segment seal mismatch")
    elif parent["files"]["model.pt"]["sha256"] != args.model_sha or parent["files"]["trainer.pt"]["sha256"] != args.trainer_sha:
        raise ValueError("first model or Trainer differs from the approved input")
    env = {k:v for k,v in os.environ.items() if not k.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_"))}
    env.update(environment())
    os.environ.update(HSAI_ACTION_PROFILE=env["HSAI_ACTION_PROFILE"], HSAI_HUMAN_TAIL=env["HSAI_HUMAN_TAIL"])
    recipe = read("config/research/human-turn-3702-ppo.json")
    if quill.engine_info()["source_hash"] != recipe["engine_source_hash"] or quill.config("hold", allow_tail_variant=True)[0].sha256() != PROTOCOL:
        raise ValueError("engine or protocol differs")
    lib = Path(env["HSAI_ARENA_SNAPS"]).parent
    if any(library_digest(lib / split) != digest for split, digest in recipe["library_digests"].items()):
        raise ValueError("library checksum changed")
    stop = allocation_end() - 150
    if stop <= time.time() + 90:
        raise ValueError("insufficient time for a rollout and final save")
    env.update(HSRL_ROOT=str(ROOT), HSAI_ARENA_SEED=str(SEED), HSAI_ARENA_OUT=str(args.out),
               HSAI_ARENA_CKPT=parent["model"], HSAI_ARENA_PARENT_TRAINER=parent["trainer"],
               HSAI_ARENA_ITERATIONS=str(args.iterations), HSAI_ARENA_STOP_EPOCH=str(stop))
    print(f"Full Trainer resume at v{parent['version']} with {parent['cum_decisions']} decisions", flush=True)
    subprocess.run(["bash", "deploy/slurm/stage_arena.sbatch"], check=True, env=env)
    final = verified_generation(args.out, SEED)
    if final["version"] <= parent["version"] or final["cum_decisions"] <= parent["cum_decisions"]:
        raise ValueError("no forward training progress")
    if args.iterations and final["version"] != parent["version"] + args.iterations:
        raise ValueError("incomplete fixed budget")
    first = next((line for line in (args.out / "train.log").open() if line.startswith("[resume]")), "")
    if f"start_iter={parent['version']} updates={parent['version']}" not in first:
        raise ValueError("incorrect Trainer restore")
    atomic_json(args.out / "complete.json", dict(commit=source["commit"], seed=SEED, protocol=PROTOCOL,
        base_model=parent["files"]["model.pt"], base_trainer=parent["files"]["trainer.pt"],
        previous_version=parent["version"], previous_decisions=parent["cum_decisions"],
        checkpoint=final, stop_epoch=stop, job=os.environ["SLURM_JOB_ID"],
        engine_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT / "config/live/model.json"), libraries=recipe["library_digests"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-sha")
    parser.add_argument("--trainer-sha")
    parser.add_argument("--previous-seal", type=Path)
    parser.add_argument("--iterations", type=int, default=0)
    run(parser.parse_args())
