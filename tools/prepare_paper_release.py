"""Prepare an isolated source release using the existing user's Python environment.

Only copies the same-platform native module and writes local execution metadata.
No installation, imports of torch, or modification of the environment source checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def prepare(release, environment):
    release, environment = Path(release).resolve(), Path(environment).resolve()
    if not release.is_relative_to(environment / "runs/releases") or release == environment:
        raise ValueError("release must be inside the named project's runs/releases directory")
    if not (release / "SOURCE_MANIFEST.json").is_file():
        raise ValueError("extract and verify a source release first")
    suffix = ".pyd" if os.name == "nt" else ".so"
    modules = [p for p in (environment / "python/taverngym").glob("_core.*") if p.suffix == suffix]
    if len(modules) != 1:
        raise ValueError("expected exactly one matching-platform native module in environment checkout")
    executable = environment / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not executable.is_file():
        raise ValueError("existing Python environment missing")
    source = modules[0]
    target = release / "python/taverngym" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    meta_path = release / "EXECUTION_ENV.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["environment_root"] != str(environment):
        raise ValueError("release is already bound to another environment")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
        raise ValueError("release already has a different native module; refusing overwrite")
    if not target.exists():
        shutil.copy2(source, target)
    meta = {"environment_root": str(environment), "source_root": str(release),
            "native_sha256": checksum, "native_source": str(source),
            "validation_required": "source manifest + doctor build fingerprint + full tests"}
    temporary = meta_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temporary.replace(meta_path)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--environment", required=True)
    args = p.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.environment)))
