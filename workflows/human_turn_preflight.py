"""Run one verified segment of the pinned human-field training series."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verified_generation(segment, seed):
    from hsbg.runtime.artifacts import file_identity
    train = (segment / "train").resolve()
    manifest = json.loads((train / "latest.json").read_bytes())
    if manifest["final"] is not True or manifest["seed"] != seed or manifest["history_len"] < 1:
        raise ValueError("expected a finalized, nonempty segment with the same seed")
    generation = train / "snapshots" / f'final_{manifest["version"]:06d}'
    for key in ("model", "trainer"):
        path = Path(manifest[key])
        if path.resolve() != generation / (key + ".pt") or file_identity(path) != manifest["files"][path.name]:
            raise ValueError("checkpoint identity mismatch: " + key)
    return manifest


def allocation_end():
    if os.environ.get("SLURM_JOB_END_TIME"):
        return int(os.environ["SLURM_JOB_END_TIME"])
    job = os.environ["SLURM_JOB_ID"]
    row = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    end = next(field.split("=", 1)[1] for field in row.split() if field.startswith("EndTime="))
    return int(datetime.fromisoformat(end).timestamp())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--recipe", type=Path, default=ROOT/"config/research/human-turn-3702-ppo.json")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.segment < 1:
        parser.error("segment numbers start at 1")
    recipe = json.loads(args.recipe.read_bytes())
    # Inherited staging flags must not silently change a frozen arm.
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_")):
            del os.environ[key]
    os.environ.update(recipe["environment"])
    seed = recipe["seed"]
    series = Path(recipe["series"]).resolve()
    out = series / f"segment-{args.segment:02d}"
    os.environ.update(HSAI_ARENA_SEED=str(seed), HSAI_ARENA_OUT=str(out))
    os.environ["HSRL_ROOT"] = str(ROOT)
    from hsbg.az.train.personal import checked_library, inspect_checkpoint, sha256
    from workflows.evaluate_suite import preflight_protocol
    from hsbg.runtime.provenance import source_identity
    plan = json.loads(Path(recipe["evaluation"]).read_bytes())
    endpoint = json.loads(Path(recipe["endpoint_evaluation"]).read_bytes())
    if any(endpoint[k] != plan[k] for k in ("seed", "games", "evaluation", "expected_protocol_sha256")):
        raise ValueError("unpaired endpoint plan")
    if seed < plan["seed"] + plan["games"] or int(recipe["environment"]["HSAI_ARENA_GAMES"]) > 1000:
        raise ValueError("rollout seed interval is not isolated")
    preflight_protocol(plan)
    source = source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("dirty training release")
    model = Path(os.environ["HSAI_ARENA_CKPT"])
    checked = inspect_checkpoint(str(model), "cpu")
    if (checked["sha256"] != recipe["checkpoint_sha256"] or
            sha256(model.parent/"manifest.json") != recipe["model_manifest_sha256"]):
        raise ValueError("starting weights or manifest changed")
    paths = {"train":os.environ["HSAI_ARENA_SNAPS"],"hold":plan["evaluation"]["mirror-snaps"]}
    sources = {}
    for split, path in paths.items():
        lib = checked_library(path)
        if (lib["digest"] != recipe["library_digests"][split] or lib["split"] != split or
                lib["fingerprint"]["engine"]["source_hash"] != recipe["engine_source_hash"]):
            raise ValueError("library mismatch: "+split)
        sources[split] = set(lib["games"])
    if sources["train"] & sources["hold"]:
        raise ValueError("overlapping source games")
    if out.exists():
        raise FileExistsError(out)
    identity = {"commit": source["commit"], "recipe_sha256": sha256(args.recipe), "seed": seed}
    previous_decisions = 0
    if args.segment > 1:
        previous = series / f"segment-{args.segment - 1:02d}"
        seal = json.loads((previous / "complete.json").read_bytes())
        manifest = verified_generation(previous, seed)
        if seal["identity"] != identity or seal["checkpoint"] != manifest:
            raise ValueError("previous segment provenance mismatch")
        os.environ.update(HSAI_ARENA_CKPT=manifest["model"], HSAI_ARENA_PARENT_TRAINER=manifest["trainer"])
        previous_decisions = manifest["cum_decisions"]
    print(json.dumps({"segment":args.segment,"source":source,"parent":os.environ["HSAI_ARENA_CKPT"],
                      "parameters":checked["parameters"], "forward_ok":checked["forward_ok"],
                      "library_digests":recipe["library_digests"],
                      "protocol":plan["expected_protocol_sha256"]}),flush=True)
    if args.train:
        from hsbg.runtime.artifacts import atomic_json
        cutoff = min(allocation_end(), datetime.fromisoformat(recipe["deadline"]).timestamp())
        stop_epoch = int(cutoff - 90)
        if stop_epoch - time.time() < 90:
            raise RuntimeError("insufficient time for a complete training rollout and final save")
        os.environ["HSAI_ARENA_STOP_EPOCH"] = str(stop_epoch)
        subprocess.run(["bash", recipe["entrypoint"]], check=True)
        manifest = verified_generation(out, seed)
        if manifest["cum_decisions"] <= previous_decisions:
            raise ValueError("segment made no training progress")
        atomic_json(out / "complete.json", {"identity": identity, "checkpoint": manifest,
                    "previous_decisions": previous_decisions, "stop_epoch": stop_epoch})
        link = out / "publish-link"
        link.symlink_to(Path(manifest["model"]).parent)
        os.replace(link, series / "latest-complete")


if __name__ == "__main__":
    main()
