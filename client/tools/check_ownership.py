"""Warn when the local workspace edits files that the server owns.

Architecture (docs/ARCHITECTURE.md, 2026-11-10): the NUS server owns the
simulator, the training stack, evaluation and the shared docs; this Windows
box only owns the real-game bridge. This script does not revert anything —
it just makes an out-of-scope edit visible before it lands in a commit.

Usage
-----
    python client/tools/check_ownership.py            # report only
    python client/tools/check_ownership.py --strict    # exit 1 if out of scope

The classification is deliberately explicit rather than heuristic: a path is
local-owned only if it appears below.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Paths the local box may change.
LOCAL_PREFIXES = (
    "client/",
    "config/live/",
    "artifacts/",
    "records/",
    "docs/sync/LOCAL.md",
    "docs/live/",
    "docs/LIVE_V1_1_AUDIT.md",
    "docs/E_PHASE_RUNBOOK.md",
    "docs/analysis/",
    "docs/ARCHITECTURE.md",     # joint: the architecture doc is written by both ends
)
#: Explicit local-owned exceptions inside server-owned trees.
LOCAL_EXCEPTIONS = ("tests/test_live_", "docs/sync/SERVER_HANDOFF_", "docs/sync/DECISIONS.md",
                    "docs/sync/manifests/local.json")

#: Trees the server owns; the local box mirrors them read-only.
SERVER_PREFIXES = (
    "engine/",
    "hsbg/",
    "workflows/",
    "deploy/",
    "data/",
    "tools/",
    "tests/",
    "config/",          # except config/live (checked first)
    "docs/",            # except the local-owned entries above
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "AGENTS.md",
)


def git_status() -> list[tuple[str, str]]:
    out = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT,
                         capture_output=True, text=True, timeout=60)
    rows = []
    for line in out.stdout.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:].strip()
        if " -> " in path:                       # rename: judge the destination
            path = path.split(" -> ", 1)[1]
        if path.endswith("/"):                   # untracked directory
            path += "*"
        rows.append((code, path))
    return rows


def classify(path: str) -> str:
    if any(path.startswith(p) for p in LOCAL_PREFIXES):
        return "local"
    if any(path.startswith(e) for e in LOCAL_EXCEPTIONS):
        return "local"
    if any(path.startswith(p) for p in SERVER_PREFIXES):
        return "server"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 when server-owned paths changed")
    args = ap.parse_args(argv)

    rows = git_status()
    if not rows:
        print("workspace clean - ownership check passed")
        return 0

    buckets: dict[str, list[tuple[str, str]]] = {"local": [], "server": [], "unknown": []}
    for code, path in rows:
        buckets[classify(path)].append((code, path))

    for name, title in (("local", "local-owned (ok)"),
                        ("server", "SERVER-OWNED (should be changed on NUS)"),
                        ("unknown", "unclassified (review by hand)")):
        items = buckets[name]
        if not items:
            continue
        print(f"\n{title}: {len(items)}")
        for code, path in items:
            print(f"  {code:>2}  {path}")

    if buckets["server"]:
        print("\nThese paths belong to the server (docs/ARCHITECTURE.md §1).")
        print("If the change is really needed, land it on the server and pull it back.")
    if args.strict and (buckets["server"] or buckets["unknown"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
