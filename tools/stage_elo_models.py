#!/usr/bin/env python3
"""Stage the 8 Elo-evaluation model.pt files and verify SHA256/bytes.

The source files are large (about 68 MB each).  This tool copies them to
``artifacts/models/elo-8/models/<id>/model.pt`` so a later engine/model audit and
the formal evaluation cannot silently use a different checkpoint.  It never
overwrites a target whose SHA differs unless ``--force`` is explicit.

Usage::

    python tools/stage_elo_models.py
    python tools/stage_elo_models.py --check-only
    python tools/stage_elo_models.py --force       # only after inspecting drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = ROOT / "config/eval-elo-8.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    entries = []
    for model in cfg["models"]:
        model_id = str(model["id"])
        source = ROOT / str(model["source_path"])
        target = ROOT / str(model["path"])
        expected_sha = str(model["file_sha256"])
        expected_bytes = int(model["bytes"])
        if not args.check_only:
            if not source.is_file():
                raise SystemExit(f"缺少源权重 {model_id}: {source}")
            source_sha = sha256_file(source)
            source_bytes = source.stat().st_size
            if source_sha != expected_sha or source_bytes != expected_bytes:
                raise SystemExit(
                    f"源权重与配置不符 {model_id}: {source}\n"
                    f"  sha expected={expected_sha} actual={source_sha}\n"
                    f"  bytes expected={expected_bytes} actual={source_bytes}")
            if target.exists():
                target_sha = sha256_file(target)
                if target_sha != expected_sha and not args.force:
                    raise SystemExit(
                        f"目标已存在但 SHA 不符，保留现场: {target}\n"
                        f"  expected={expected_sha} actual={target_sha}\n"
                        "确认后用 --force 覆盖")
                if target_sha != expected_sha:
                    target.unlink()
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        if not target.is_file():
            raise SystemExit(f"缺少目标权重 {model_id}: {target}")
        target_sha = sha256_file(target)
        target_bytes = target.stat().st_size
        if target_sha != expected_sha or target_bytes != expected_bytes:
            raise SystemExit(
                f"目标权重与配置不符 {model_id}: {target}\n"
                f"  sha expected={expected_sha} actual={target_sha}\n"
                f"  bytes expected={expected_bytes} actual={target_bytes}")
        entries.append({
            "id": model_id,
            "label": model.get("label", ""),
            "source_path": str(model["source_path"]),
            "path": str(model["path"]),
            "file_sha256": target_sha,
            "bytes": target_bytes,
        })
        print(f"[stage] {model_id}: {target_bytes} bytes sha256={target_sha}")

    manifest = {
        "schema": "hsai.elo-models/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path.relative_to(ROOT)) if str(config_path).startswith(str(ROOT))
        else str(config_path),
        "config_sha256": sha256_text(config_path),
        "engine": cfg.get("engine", {}),
        "models": entries,
        "note": "Formal Elo evaluation must use these staged files and matching SHA256.",
    }
    out = ROOT / "artifacts/models/elo-8/MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[stage] manifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
