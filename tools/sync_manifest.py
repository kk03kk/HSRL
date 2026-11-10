"""Emit a comparable inventory manifest for one HSAI side (local or server).

The local Windows workspace and the NUS login node are never inspected by the
same process, so the only reliable way to compare them is the same *script* run
on both sides with the same arguments, producing two JSON files that are then
diffed in Git (or with ``tools/sync_compare.py``).

Design rules
------------
* stdlib only, login-node safe (no torch, no numpy, no project imports);
* never walks multi-GB trees: counts/sizes only, hashes a short explicit pin
  list, and caps the walk so a 19 GB ``records/`` cannot stall the login node;
* every number is either measured here or explicitly absent (``null``);
* **pins carry both the worktree SHA256 and the Git blob hash**. Windows
  checkouts are CRLF while the server is LF, so raw SHA256 differs for files
  that are identical in Git; ``git_blob`` is the cross-side truth (D-014);
* the output is small (tens of KB) so it can live in Git.

Usage
-----
    python tools/sync_manifest.py --side local  --out docs/sync/manifests/local.json
    python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json
    python tools/sync_compare.py docs/sync/manifests/local.json docs/sync/manifests/server.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

#: Code/doc trees that must be identical (or intentionally different) on both
#: sides. Sizes are cheap; no hashing of whole trees.
TREES = ("engine", "hsbg", "client", "workflows", "tools", "tests", "deploy",
         "config", "data", "docs")

#: Big trees whose *shape* matters but whose bytes never enter Git.
ARTIFACT_ROOTS = ("models", "libraries", "datasets", "imports", "archives",
                  "runs", "local", "client", "exports")
RECORD_ROOTS = ("evidence", "archive", "research", "migration", "source-docs",
                "source-snapshots", "nus-audits", "development", "long-runs")

#: Files that identify a *version*. ``scope`` says which side must have them:
#:   both   – must exist and be identical on both sides
#:   local  – local is the source of truth (server may legitimately not have it)
#:   server – server is the source of truth
#:   archived – not at this path anymore; kept only for documentation
PINS = (
    ("config/live/model.json", "both"),
    ("config/live/autoplay-255790.json", "both"),
    ("config/engine-upstream-lock.json", "both"),
    ("config/eval-elo-8.json", "both"),
    ("data/card_zhcn.json", "both"),
    ("data/bg_cards.json", "both"),
    ("pyproject.toml", "both"),
    ("requirements.txt", "both"),
    ("artifacts/models/v1.1/model.pt", "both"),
    ("artifacts/models/v1.1/manifest.json", "both"),
    ("artifacts/models/v1.1/checkpoint.json", "both"),
    ("artifacts/models/v1.1/trainer.pt", "server"),
    ("artifacts/models/library-v1/manifest.json", "local"),
    ("artifacts/models/library-v1/SHA256SUMS.txt", "local"),
    ("artifacts/libraries/league-eot-v1.1-20261109/manifest.json", "server"),
)

#: Hard cap so a huge or cold NFS tree cannot stall the login node.
MAX_FILES_PER_TREE = 200_000
#: Generated trees that exist on one side only and must not look like drift.
SKIP_DIRS = frozenset({"target", "node_modules", "__pycache__", ".git", ".venv",
                       "venv", "dist", "build", ".pytest_cache", ".mypy_cache"})


def sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_blob(root: Path, rel: str) -> str | None:
    """Blob hash *after* CRLF normalisation, or None when untracked/absent."""
    try:
        out = subprocess.run(["git", "hash-object", "--", rel], cwd=root,
                             capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def tree_stat(path: Path) -> dict:
    if not path.exists():
        return dict(exists=False, files=0, bytes=0, newest=None, truncated=False)
    files = 0
    total = 0
    newest = 0.0
    truncated = False
    for root, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            files += 1
            if files > MAX_FILES_PER_TREE:
                truncated = True
                break
            try:
                st = (Path(root) / name).stat()
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
        if truncated:
            break
    return dict(exists=True, files=files, bytes=total, truncated=truncated,
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
    """Version numbers without importing torch/numpy (login nodes have ~1 GiB)."""
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
        ("engine lock", root / "config" / "engine-upstream-lock.json"),
    ]
    if os.name == "nt":
        checks.append(("game dir", Path(r"D:\Hearthstone")))
    return [dict(check=label, path=str(path), exists=path.exists()) for label, path in checks]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--side", choices=("local", "server"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--notes", default="", help="one free-text line recorded next to the manifest")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    pins = {}
    for rel, scope in PINS:
        p = root / rel
        pins[rel] = (dict(scope=scope, bytes=p.stat().st_size, sha256=sha256(p),
                          git_blob=git_blob(root, rel)) if p.is_file()
                     else dict(scope=scope, bytes=None, sha256=None, git_blob=None))

    manifest = dict(
        schema="hsai.sync-manifest/v2",
        side=args.side,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        root=str(root),
        notes=args.notes,
        git=git_info(root),
        env=env_info(),
        trees={name: tree_stat(root / name) for name in TREES},
        artifacts={name: tree_stat(root / "artifacts" / name) for name in ARTIFACT_ROOTS},
        records={name: tree_stat(root / "records" / name) for name in RECORD_ROOTS},
        pins=pins,
        problems=workspace_problems(root),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = [p["check"] for p in manifest["problems"] if not p["exists"]]
    print("wrote %s (head=%s dirty=%s, pins=%d)" % (
        args.out, (manifest["git"]["head"] or "?")[:10], manifest["git"]["dirty"], len(pins)))
    if missing:
        print("missing:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
