"""Verify the pinned assets and optionally start one specified arm inside its allocation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A","B"], required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    recipe = json.loads((ROOT/"config/research/human-turn-3702-ppo.json").read_bytes())
    # Inherited staging flags must not silently change a frozen arm.
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_")):
            del os.environ[key]
    os.environ.update(recipe["environment"], **recipe["arms"][args.arm])
    os.environ["HSRL_ROOT"] = str(ROOT)
    from hsbg.az.train.personal import checked_library, inspect_checkpoint, sha256
    from workflows.evaluate_suite import preflight_protocol
    from hsbg.runtime.provenance import source_identity
    plan = json.loads(Path(recipe["evaluation"]).read_bytes())
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
    if Path(os.environ["HSAI_ARENA_OUT"]).exists():
        raise FileExistsError(os.environ["HSAI_ARENA_OUT"])
    print(json.dumps({"arm":args.arm,"source":source,"model":checked["sha256"],
                      "parameters":checked["parameters"], "forward_ok":checked["forward_ok"],
                      "library_digests":recipe["library_digests"],
                      "protocol":plan["expected_protocol_sha256"]}),flush=True)
    if args.train:
        os.execvp("bash",["bash",recipe["entrypoint"]])


if __name__ == "__main__":
    main()
