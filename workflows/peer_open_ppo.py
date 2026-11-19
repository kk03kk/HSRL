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


def train(out: Path, *, previous: Path | None = None, last_block=False, seed=None) -> None:
    from hsbg.az.train.personal import library_digest
    from hsbg.runtime.artifacts import atomic_json, file_identity
    from hsbg.runtime.provenance import source_identity
    from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol
    from workflows.peer_ppo_panel import PEER_SHA, WEIGHT
    from workflows.peer_ppo_pilot import SEED, allocation_end, environment, read, verified_generation
    train_seed = SEED if seed is None else seed

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
    settings.update(HSAI_ACTION_PROFILE=ACTION_PROFILE, HSAI_ARENA_SEED=str(train_seed),
                    HSAI_ARENA_MINUTES="175", HSAI_ARENA_ANCHOR="0.06", HSAI_ARENA_ANCHOR_TAIL="0.04")
    identity = dict(commit=source["commit"], settings=settings, protocol_sha256=protocol.sha256(),
                    initial=initial, peer=PEER_SHA, libraries=recipe["library_digests"],
                    engine_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
                    live_pin=file_identity(ROOT / "config/live/model.json"))
    if last_block:
        identity["adapt_last_block"] = True
    parent = None
    if previous:
        parent = verified_generation(previous / "fit", train_seed)
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
    if last_block and not parent:
        payload = torch.load(INITIAL, map_location="cpu", weights_only=True)
        from hsbg.az.train.peer_actor import load_payload
        original_model, _ = load_payload(payload, "cpu")
        payload["meta"]["adapt_last_block"] = True
        fork = segment / "initial"
        fork.mkdir()
        torch.save(payload, fork / "model.pt")
        changed_model, _ = load_payload(torch.load(fork / "model.pt", weights_only=True), "cpu")
        old = dict(original_model.named_parameters())
        added = [k for k,v in changed_model.named_parameters() if v.requires_grad and not old[k].requires_grad]
        expected = {"actor.encoder.layers.3." + k for k,_ in changed_model.actor.encoder.layers[-1].named_parameters()}
        if (len(expected) != 12 or set(added) != expected
                or any(not torch.equal(v, old[k]) for k,v in changed_model.named_parameters())):
            raise ValueError("unexpected representation fork")
        stamp = dict(origin=initial, model=file_identity(fork / "model.pt"), new_trainable=sorted(added))
        atomic_json(fork / "manifest.json", stamp)
        atomic_json(fork / "checkpoint.json", dict(model=stamp["model"], kind="weight-only fork"))
        (fork / "MODEL_CARD.md").write_text("Unchanged D-155 initial tensors. PPO also updates encoder layer 3.\n", encoding="ascii")
        runtime["HSAI_ARENA_CKPT"] = str(fork / "model.pt")
    atomic_json(segment / "plan.json", dict(
        source=source, identity=identity, protocol=protocol.as_dict(), start=start, end=end,
        evaluation_seed=308410000 if last_block else EVAL_SEED, parent=parent, runtime=runtime, job=os.environ["SLURM_JOB_ID"]))
    subprocess.run(["bash", "deploy/slurm/stage_arena.sbatch"], env=os.environ | runtime, check=True)
    final = verified_generation(segment / "fit", train_seed)
    if final["version"] != end or final["cum_decisions"] <= (parent["cum_decisions"] if parent else 0):
        raise ValueError("incomplete fixed-dose budget")
    log = (segment / "fit/train.log").read_text(encoding="utf-8")
    if f"[resume] start_iter={start} updates={start}" not in log or (
            parent and f"[trainer] loaded updates={start}" not in log):
        raise ValueError("unexpected starting Trainer iteration")
    if last_block:
        import torch
        from hsbg.az.train.peer_actor import load_payload
        fitted, _ = load_payload(torch.load(final["model"], weights_only=True), "cpu")
        base = torch.load(INITIAL, weights_only=True)["state_dict"]
        changed = {k for k,v in fitted.state_dict().items() if not torch.equal(v, base[k])}
        allowed = {k for k,v in fitted.named_parameters() if v.requires_grad}
        if (not fitted.adapt_last_block or not changed <= allowed
                or not any(k.startswith("actor.encoder.layers.3.") for k in changed)):
            raise ValueError("invalid last-block updates")
        atomic_json(segment / "updated-tensors.json", sorted(changed))
    atomic_json(segment / "complete.json", dict(
        identity=identity, checkpoint=final, trainer=inspect_trainer(final), start=start,
        job=os.environ["SLURM_JOB_ID"]))


def assess(out: Path, *, reused: Path | None = None) -> None:
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
    assessment = out / ("assessment-recovered" if reused else "assessment")
    cmd = [
        sys.executable, "workflows/peer_finetune_holdout.py", "--segment", str(last / "fit"),
        "--out", str(assessment), "--seed", str(EVAL_SEED),
        "--games", "512", "--protected", "256", "--raw-parent",
    ]
    if reused:
        cmd.extend(("--reuse-complete", str(reused)))
    subprocess.run(cmd, check=True)
    result = read(assessment / "complete.json")
    utility = result["delta"]["utility"]["ci95"][0] > 0
    growth = result["delta"]["log12"]["ci95"][0] >= -0.12
    atomic_json(out / "decision.json", dict(
        utility_pass=utility, t12_growth_pass=growth, passed=utility and growth,
        eval_seed=EVAL_SEED, games=result["n"], final=final["files"]["model.pt"],
        protocol=STANDARD_PROTOCOL, assessment=str(assessment)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "resume", "assess"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-assessment", type=Path)
    parser.add_argument("--last-block", action="store_true")
    parser.add_argument("--train-seed", type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE=ACTION_PROFILE, HSAI_HUMAN_TAIL="last-observed-v1")
    out = args.out.resolve()
    if args.stage == "assess":
        if args.last_block or args.train_seed is not None:
            parser.error("use peer_finetune_holdout.py with explicit train and assessment seeds")
        assess(out, reused=args.reuse_assessment)
    else:
        train(out, previous=out / "segment-01" if args.stage == "resume" else None,
              last_block=args.last_block, seed=args.train_seed)
