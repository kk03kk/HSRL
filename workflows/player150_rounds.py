"""Prepare or execute one immutable stage of the player150 five-round series."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

RECIPE = ROOT / "config/research/player150-five-rounds.json"
START_SHA = "1cfa3f0d584ad2896b7e1c6f9c1b480e87a01a2be78498a1678324a1febbe094"


def read(path):
    return json.loads(path.read_bytes())


def identity(cfg):
    import hsbg_engine as engine
    from workflows.peer_3703_field import LIB, R1_SHA
    from workflows.player150_start import START
    from workflows.peer_3703_round import field
    from hsbg.az.train.player_observation import SCHEMA_SHA
    source = source_identity(ROOT)
    model = file_identity(START / "model.pt")
    manifest = read(START / "manifest.json")
    if (source["tracked_dirty"] or not engine.build_is_current(ROOT / "engine") or
        engine.build_info()["source_hash"] != cfg["engine"] or model["sha256"] != START_SHA or
        manifest["model"] != model or manifest["parent"]["sha256"] != R1_SHA or
        manifest["observation"] != SCHEMA_SHA or manifest["equivalence_states"] < 8):
        raise ValueError("unqualified source, model or native engine")
    fields = {}
    games = {}
    for split in ("train", "hold", "human"):
        protocol, man, spec, sha = field(split)
        fields[split] = dict(protocol=sha, library=man["digest"], frames=man["frames"])
        games[split] = set(man["games"])
        assert (protocol.max_actions_per_turn, protocol.move_cap, protocol.freeze_cap) == (150, 3, 2)
    if games["train"] & games["hold"] or not (LIB / "complete.json").is_file():
        raise ValueError("opponent field leak or missing seal")
    return dict(source=source, engine=cfg["engine"], fields=fields, model=model, recipe=file_identity(RECIPE),
                observation=SCHEMA_SHA, engine_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
                live_pin=read(ROOT / "config/live/model.json")["sha256"], manifest=file_identity(START / "manifest.json"))


def train(dest, cfg, seed, parent, *, test):
    from workflows.player150_start import START
    from workflows.peer_3703_field import LIB
    from workflows.human_turn_preflight import allocation_end, verified_generation
    from workflows.peer_open_ppo import inspect_trainer
    version = parent["version"] if parent else 0
    stop = allocation_end() - cfg["save_margin_seconds"]
    if stop - time.time() < 90:
        raise ValueError("insufficient remaining allocation")
    env = dict(cfg["environment"], HSRL_ROOT=str(ROOT), HSAI_ARENA_OUT=str(dest / "fit"),
               HSAI_ARENA_SNAPS=str(LIB / "train"), HSAI_ARENA_SEED=str(seed),
               HSAI_ARENA_STOP_EPOCH=str(stop), HSAI_ARENA_MINUTES=str(cfg["minutes"]),
               HSAI_ARENA_ITERATIONS="2" if test and not parent else ("1" if test else "0"),
               HSAI_ARENA_CKPT=parent["model"] if parent else str((START / "model.pt").resolve()))
    if test:
        env["HSAI_ARENA_GAMES"] = "16"
    if parent:
        env["HSAI_ARENA_PARENT_TRAINER"] = parent["trainer"]
    atomic_json(dest / "plan.json", dict(environment=env, parent=parent, job=os.environ["SLURM_JOB_ID"]))
    subprocess.run(["bash","deploy/slurm/stage_arena.sbatch"], env=os.environ | env, check=True)
    seal = verified_generation(dest / "fit", seed)
    log = (dest / "fit/train.log").read_text(encoding="utf-8")
    if (seal["version"] <= version or seal["cum_decisions"] <= (parent["cum_decisions"] if parent else 0) or
        f"[resume] start_iter={version} updates={version}" not in log or
        (parent and f"[trainer] loaded updates={version}" not in log)):
        raise ValueError("invalid Trainer continuation or missing updates")
    info = inspect_trainer(seal, initial_model=START / "model.pt")
    if test and seal["version"] != version + int(env["HSAI_ARENA_ITERATIONS"]):
        raise ValueError("incomplete smoke updates")
    import torch
    initial = torch.load(START / "model.pt", weights_only=True, map_location="cpu")["state_dict"]
    state = torch.load(seal["model"], weights_only=True, map_location="cpu")["state_dict"]
    changed = [k for k in initial if k.startswith("public_input.") and not torch.equal(initial[k],state[k])]
    if not changed or not all(torch.isfinite(p).all() for p in state.values() if p.is_floating_point()):
        raise ValueError("public inputs did not learn or model is non-finite")
    atomic_json(dest / "complete.json", dict(checkpoint=seal, trainer=info, changed=changed))
    return seal


def run(stage, root, qualification, index):
    cfg = read(RECIPE)
    for key in tuple(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_", "HSAI_QUILL_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE=cfg["profile"], HSAI_HUMAN_TAIL="last-observed-v1")
    from workflows.player150_start import START
    from workflows.peer_3703_field import R1
    from workflows.peer_3703_round import panel
    from workflows.human_turn_preflight import verified_generation
    if not 1 <= index <= cfg["rounds"]:
        raise ValueError("invalid round")
    if os.environ["SLURM_JOB_PARTITION"] != ("gpu" if stage in ("smoke","train") else "normal"):
        raise ValueError("incorrect compute allocation")
    stamp = identity(cfg)
    start_model = START / "model.pt"
    if stage in ("baseline", "smoke"):
        dest = qualification / stage
        dest.mkdir(parents=True, exist_ok=False)
        if stage == "baseline":
            for split in ("hold", "human"):
                out = dest / split
                out.mkdir()
                report = panel(split, [R1, start_model], 561730000, 16, out)
                if report["protocol_sha256"] != stamp["fields"][split]["protocol"]:
                    raise ValueError("baseline protocol drift")
        else:
            parent = None
            for attempt in ("fresh", "resume"):
                d = dest / attempt
                d.mkdir()
                parent = train(d, cfg, 567410000, parent, test=True)
        atomic_json(dest / "qualified.json", stamp)
        return
    for key in ("baseline", "smoke"):
        if read(qualification / key / "qualified.json") != stamp:
            raise ValueError("qualification differs from release")
    dest = root / f"round-{index:02d}" / stage
    dest.mkdir(parents=True, exist_ok=False)
    def prior(i):
        directory = root / f"round-{i:02d}"
        seal = verified_generation(directory / "train/fit", cfg["train_seed"])
        if (read(directory / "train/complete.json")["checkpoint"] != seal or
            read(directory / "train/identity.json") != stamp):
            raise ValueError("invalid previous checkpoint")
        return seal
    atomic_json(dest / "identity.json", stamp)
    if stage == "train":
        parent = None
        if index > 1:
            parent = prior(index-1)
            report = read(root / f"round-{index-1:02d}" / "monitor/complete.json")
            if (report["models"] != [stamp["model"], parent["files"]["model.pt"]] or
                report["protocol_sha256"] != stamp["fields"]["hold"]["protocol"] or
                (report["start"], report["n"]) != (cfg["monitor"]["seed"], cfg["monitor"]["pairs"])):
                raise ValueError("missing paired evaluation of the preceding endpoint")
        train(dest, cfg, cfg["train_seed"], parent, test=False)
    else:
        if stage == "external" and index != cfg["rounds"]:
            raise ValueError("external field reserved for final round")
        spec = cfg[stage]
        seal = prior(index)
        panel(spec["split"], [start_model, Path(seal["model"])], spec["seed"], spec["pairs"], dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("baseline","smoke","train","monitor","external"))
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--round", type=int, default=1)
    args = p.parse_args()
    os.chdir(ROOT)
    run(args.stage, args.series.resolve(), args.qualification.resolve(), args.round)
