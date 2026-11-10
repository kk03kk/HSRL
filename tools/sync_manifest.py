"""Emit a comparable inventory manifest for one HSAI side (local or server).

The local Windows workspace and the NUS login node are never inspected by the
same process, so the only reliable way to compare them is the same *script* run
on both sides with the same arguments, producing two JSON files that are then
diffed in Git.

Design rules
------------
* stdlib only, login-node safe (no torch, no numpy, no project imports);
* never walks multi-GB trees: only counts/sizes for `artifacts/*`, and hashes a
  short, explicit pin list;
* every number is either measured here or explicitly absent (`null`);
* the output is small (tens of KB) so it can live in Git.

Usage
-----
    python tools/sync_manifest.py --side local  --out docs/sync/manifests/local.json
    python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json

Run it again after a big change; the diff between two manifests *is* the drift
report that both agents review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

#: Code/doc trees that must be identical (or intentionally different) on both
#: sides. Sizes are cheap; no hashing of whole trees.
TREES = ("engine", "hsbg", "client", "workflows", "tools", "tests", "deploy",
         "config", "data", "docs")

#: Files that identify a *version* rather than a tree. Hashed on both sides.
PINS = (
    "config/live/model.json",
    "config/live/autoplay-255790.json",
    "config/engine-upstream-lock.json",
    "config/eval-elo-8.json",
    "data/card_zhcn.json",
    "data/bg_cards.json",
    "pyproject.toml",
    "requirements.txt",
    "artifacts/models/v1.1/model.pt",
    "artifacts/models/v1.1/manifest.json",
    "artifacts/models/v1.1/checkpoint.json",
    "artifacts/models/library-v1/manifest.json",
    "artifacts/models/library-v1/SHA256SUMS.txt",
    "artifacts/models/leaderboard-v1/manifest-2026-11-09.json",
)

#: Big local trees whose *shape* matters but whose bytes never enter Git.
ARTIFACT_ROOTS = ("models", "libraries", "datasets", "imports", "archives",
                  "runs", "local", "client")


def sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def tree_stat(path: Path) -> dict:
    if not path.exists():
        return dict(exists=False, files=0, bytes=0, newest=None)
    files = 0
    total = 0
    newest = 0.0
    for root, _dirs, names in os.walk(path):
        for name in names:
            p = Path(root) / name
            try:
                st = p.stat()
            except OSError:
                continue
            files += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return dict(exists=True, files=files, bytes=total,
                newest=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)) if newest else None)


def git_info(root: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        except Exception:  # noqa: BLE001
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    remote = run("remote", "get-url", "origin")
    return dict(head=head, branch=branch, dirty=bool(status), changed=len(status.splitlines()) if status else 0,
                origin=remote)


def env_info() -> dict:
    """Version numbers without importing torch/numpy (login nodes have ~1 GiB).

    ``importlib.metadata`` reads the installed distribution metadata; it never
    loads the package, so this stays safe on the NUS login node.
    """
    from importlib import metadata

    info = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine())
    for name in ("torch", "numpy", "pytest", "maturin", "rapidocr", "onnxruntime"):
        try:
            info[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            info[name] = None
    return info


def workspace_problems(root: Path) -> list[dict]:
    """Known reference points that a missing file would silently break."""
    checks = [
        ("engine manifest", root / "engine" / "crates" / "hsbg-engine" / "Cargo.toml"),
        ("card table", root / "data" / "bg_cards.json"),
        ("zh card table", root / "data" / "card_zhcn.json"),
        ("live profile", root / "config" / "live" / "autoplay-255790.json"),
        ("live model", root / "artifacts" / "models" / "v1.1" / "model.pt"),
        ("bot library", root / "artifacts" / "models" / "library-v1" / "manifest.json"),
        ("engine lock", root / "config" / "engine-upstream-lock.json"),
    ]
    if os.name == "nt":
        checks.append(("game dir", Path(r"D:\Hearthstone")))
    out = []
    for label, path in checks:
        out.append(dict(check=label, path=str(path), exists=path.exists()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--side", choices=("local", "server"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--notes", default="", help="one free-text line recorded next to the manifest")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    pins = {}
    for rel in PINS:
        p = root / rel
        pins[rel] = (dict(bytes=p.stat().st_size, sha256=sha256(p)) if p.is_file() else None)

    manifest = dict(
        schema="hsai.sync-manifest/v1",
        side=args.side,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        root=str(root),
        notes=args.notes,
        git=git_info(root),
        env=env_info(),
        trees={name: tree_stat(root / name) for name in TREES},
        artifacts={name: tree_stat(root / "artifacts" / name) for name in ARTIFACT_ROOTS},
        pins=pins,
        problems=workspace_problems(root),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = [p["check"] for p in manifest["problems"] if not p["exists"]]
    print("wrote %s (head=%s dirty=%s)" % (args.out, (manifest["git"]["head"] or "?")[:10], manifest["git"]["dirty"]))
    if missing:
        print("missing:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
