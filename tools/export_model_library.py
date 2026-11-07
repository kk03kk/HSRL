#!/usr/bin/env python3
"""Pack one model-library cohort into a portable tar.gz for NUS.

The bundle contains only the selected models, their model.json cards, a
filtered manifest and SHA256SUMS.  Source files are copied (not linked), so the
archive is self-contained.

Usage:
    python tools/export_model_library.py --cohort mainline-v1 \
        --out artifacts/exports/HSAI-model-library-mainline-v1.tar.gz
    python tools/export_model_library.py --models entity16m-nus-7357
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "artifacts" / "models" / "library-v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--models", default=None, help="comma separated ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-trainer", action="store_true")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((LIB / "manifest.json").read_text(encoding="utf-8-sig"))
    cohorts = json.loads((LIB / "cohorts" / "cohorts.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in manifest["models"]}
    if args.models:
        ids = [x.strip() for x in args.models.split(",") if x.strip()]
    elif args.cohort:
        ids = cohorts["cohorts"][args.cohort]["members"]
    else:
        raise SystemExit("need --cohort or --models")

    unknown = [m for m in ids if m not in entries]
    if unknown:
        raise SystemExit(f"unknown model ids: {unknown}")
    out = Path(args.out).resolve()
    staging = out.with_suffix("").with_name(out.name.replace(".tar.gz", "") + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "models").mkdir(parents=True)

    sums = []
    kept = []
    for mid in ids:
        e = dict(entries[mid])
        src_dir = LIB / "models" / mid
        if not src_dir.exists():
            print(f"[skip] {mid}: no library dir")
            continue
        dst_dir = staging / "models" / mid
        dst_dir.mkdir(parents=True)
        for name in ("model.pt", "trainer.pt"):
            if name == "trainer.pt" and args.no_trainer:
                continue
            src = src_dir / name
            if not src.exists():
                continue
            shutil.copy2(src, dst_dir / name)
            sums.append((sha256_file(dst_dir / name),
                         f"models/{mid}/{name}"))
        shutil.copy2(src_dir / "model.json", dst_dir / "model.json")
        kept.append(e)

    sub = dict(manifest)
    sub["models"] = kept
    sub["counts"] = {**manifest["counts"], "bundled_models": len(kept)}
    sub["bundle"] = {"cohort": args.cohort, "models": ids,
                     "exported_to": out.name}
    (staging / "manifest.json").write_text(
        json.dumps(sub, indent=1, ensure_ascii=False), encoding="utf-8")
    (staging / "cohorts").mkdir(exist_ok=True)
    (staging / "cohorts" / "cohorts.json").write_text(
        json.dumps(cohorts, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ("README.md", "gaps.json", "index.csv"):
        if (LIB / name).exists():
            shutil.copy2(LIB / name, staging / name)
    with (staging / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as f:
        for h, p in sorted(sums):
            f.write(f"{h}  {p}\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(staging, arcname="library-v1")
    digest = sha256_file(out)
    (out.parent / (out.name + ".sha256")).write_text(
        f"{digest}  {out.name}\n", encoding="utf-8")
    size = out.stat().st_size
    print(f"bundled {len(kept)} models, {len(sums)} weight files")
    print(f"{out}  {size/1e9:.2f} GB  sha256={digest}")
    if not args.keep_staging:
        shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
