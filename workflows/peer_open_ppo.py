"""Run the D-155 fixed-dose PPO pilot and its paired assessment."""

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
FIRST_BATCHES = 8
TOTAL_BATCHES = 60


def inspect_trainer(checkpoint):
    import torch
    from workflows.peer_finetune_holdout import INITIAL

    model = torch.load(checkpoint["model"], map_location="cpu", weights_only=False)
    state = torch.load(checkpoint["trainer"], map_location="cpu", weights_only=False)
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)["state_dict"]
    tensors = model["state_dict"]
    if (state["updates"] != checkpoint["version"]
            or state["extra"]["cum_decisions"] != checkpoint["cum_decisions"]
            or state["scheduler"]["last_epoch"] != state["updates"]
            or not state["optimizer"]["state"] or len(state["cuda_rng"]) != 2
            or any(state.get(key) is None for key in ("rng", "python_rng", "torch_rng", "anchor_model"))
            or initial.keys() != state["anchor_model"].keys()
            or any(not torch.equal(value, state["anchor_model"][key]) for key, value in initial.items())
            or tensors.keys() != state["model"].keys()
            or any(not torch.equal(value, state["model"][key]) for key, value in tensors.items())):
        raise ValueError("incomplete or inconsistent Trainer generation")
    return dict(optimizer_states=len(state["optimizer"]["state"]), cuda_rng_devices=2,
                lr=state["scheduler"]["_last_lr"][0], scheduler_step=state["scheduler"]["last_epoch"])


def train(out: Path, *, previous: Path | None = None) -> None:
    from hsbg.az.train.personal import library_digest
    from hsbg.runtime.artifacts import atomic_json, file_identity
    from hsbg.runtime.provenance import source_identity
    from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol
    from workflows.peer_ppo_panel import PEER_SHA, WEIGHT
    from workflows.peer_ppo_pilot import SEED, allocation_end, environment, read, verified_generation

    source = source_identity(ROOT)
    segment = out / ("segment-02" if previous else "segment-01")
    if source["tracked_dirty"] or segment.exists() or (previous is None and out.exists()):
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
    settings.update(HSAI_ACTION_PROFILE=ACTION_PROFILE, HSAI_ARENA_SEED=str(SEED),
                    HSAI_ARENA_MINUTES="175", HSAI_ARENA_ANCHOR="0.06", HSAI_ARENA_ANCHOR_TAIL="0.04")
    identity = dict(commit=source["commit"], settings=settings, protocol_sha256=protocol.sha256(),
                    initial=initial, peer=PEER_SHA, libraries=recipe["library_digests"],
                    engine_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
                    live_pin=file_identity(ROOT / "config/live/model.json"))
    parent = None
    if previous:
        parent = verified_generation(previous / "fit", SEED)
        seal = read(previous / "complete.json")
        if (seal["identity"] != identity or seal["checkpoint"] != parent
                or parent["version"] != FIRST_BATCHES or seal["trainer"] != inspect_trainer(parent)):
            raise ValueError("first segment identity or Trainer restore source mismatch")
    else:
        import torch

        original = torch.load(WEIGHT, map_location="cpu", weights_only=False)["state_dict"]
        wrapped = torch.load(INITIAL, map_location="cpu", weights_only=False)["state_dict"]
        actor = {key.removeprefix("actor."): value for key, value in wrapped.items() if key.startswith("actor.")}
        if original.keys() != actor.keys() or any(not torch.equal(value, actor[key]) for key, value in original.items()):
            raise ValueError("wrapped starting actor does not equal the published weights")
    stop = allocation_end() - 150
    if stop <= time.time() + 300:
        raise ValueError("insufficient allocation time")
    start = parent["version"] if parent else 0
    end = TOTAL_BATCHES if parent else FIRST_BATCHES
    runtime = dict(settings, HSRL_ROOT=str(ROOT), HSAI_ARENA_OUT=str(segment / "fit"),
                   HSAI_ARENA_CKPT=parent["model"] if parent else str(INITIAL.resolve()),
                   HSAI_ARENA_ITERATIONS=str(end - start), HSAI_ARENA_STOP_EPOCH=str(stop))
    if parent:
        runtime["HSAI_ARENA_PARENT_TRAINER"] = parent["trainer"]
    segment.mkdir(parents=True, exist_ok=False)
    atomic_json(segment / "plan.json", dict(
        source=source, identity=identity, protocol=protocol.as_dict(), start=start, end=end,
        evaluation_seed=EVAL_SEED, parent=parent, runtime=runtime, job=os.environ["SLURM_JOB_ID"]))
    subprocess.run(["bash", "deploy/slurm/stage_arena.sbatch"], env=os.environ | runtime, check=True)
    final = verified_generation(segment / "fit", SEED)
    if final["version"] != end or final["cum_decisions"] <= (parent["cum_decisions"] if parent else 0):
        raise ValueError("incomplete fixed-dose budget")
    log = (segment / "fit/train.log").read_text(encoding="utf-8")
    if f"[resume] start_iter={start} updates={start}" not in log or (
            parent and f"[trainer] loaded updates={start}" not in log):
        raise ValueError("unexpected starting Trainer iteration")
    atomic_json(segment / "complete.json", dict(
        identity=identity, checkpoint=final, trainer=inspect_trainer(final), start=start,
        job=os.environ["SLURM_JOB_ID"]))


def assess(out: Path) -> None:
    from workflows.peer_finetune_holdout import STANDARD_PROTOCOL
    from workflows.peer_ppo_pilot import SEED, read, verified_generation
    from hsbg.runtime.artifacts import atomic_json

    first = read(out / "segment-01/complete.json")
    last = out / "segment-02"
    seal = read(last / "complete.json")
    final = verified_generation(last / "fit", SEED)
    if (seal["identity"]["protocol_sha256"] != STANDARD_PROTOCOL
            or first["identity"] != seal["identity"] or seal["checkpoint"] != final
            or seal["start"] != FIRST_BATCHES or final["version"] != TOTAL_BATCHES
            or read(last / "plan.json")["parent"] != first["checkpoint"]):
        raise ValueError("pilot protocol or final seal mismatch")
    subprocess.run([
        sys.executable, "workflows/peer_finetune_holdout.py", "--segment", str(last / "fit"),
        "--out", str(out / "assessment"), "--seed", str(EVAL_SEED),
        "--games", "512", "--protected", "256", "--raw-parent",
    ], check=True)
    result = read(out / "assessment/complete.json")
    utility = result["delta"]["utility"]["ci95"][0] > 0
    growth = result["delta"]["log12"]["ci95"][0] >= -0.12
    atomic_json(out / "decision.json", dict(
        utility_pass=utility, t12_growth_pass=growth, passed=utility and growth,
        eval_seed=EVAL_SEED, games=result["n"], final=final["files"]["model.pt"],
        protocol=STANDARD_PROTOCOL))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "resume", "assess"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE=ACTION_PROFILE, HSAI_HUMAN_TAIL="last-observed-v1")
    out = args.out.resolve()
    if args.stage == "assess":
        assess(out)
    else:
        train(out, previous=out / "segment-01" if args.stage == "resume" else None)
