#!/usr/bin/env python3
"""One-line live status for the current run directory (cheap, no big JSON parsing).

Usage:
    python client/tools/live_status.py --out artifacts/runs/live-night-20261112/e0-002
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", action="store_true", help="print the action histogram")
    args = ap.parse_args()
    out = Path(args.out)

    health = read_json(out / "health.json") or {}
    result = read_json(out / "result.json")
    status_age = None
    try:
        status_age = round(time.time() - os.stat(out / "status.json").st_mtime, 1)
    except OSError:
        pass

    decisions = 0
    kinds: collections.Counter = collections.Counter()
    turns: set[int] = set()
    try:
        with (out / "decisions.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                decisions += 1
                kinds[row.get("kind")] += 1
                if isinstance(row.get("turn"), int):
                    turns.add(row["turn"])
    except OSError:
        pass

    print(
        "turn={t} phase={p} hp={hp} gold={g} tier={ti} pending={pd} "
        "decisions={d} turns={lo}-{hi} status_age={age}s".format(
            t=health.get("turn"), p=health.get("phase"), hp=health.get("hero_health"),
            g=health.get("gold"), ti=health.get("tier"), pd=health.get("pending"),
            d=decisions, lo=(min(turns) if turns else "-"), hi=(max(turns) if turns else "-"),
            age=status_age,
        )
    )
    if args.kinds and kinds:
        print("  kinds:", dict(kinds.most_common()))
    if result:
        print(
            "  RESULT: {r} place={p} turn={t} source={s} terminal={term} proof={proof}".format(
                r=result.get("result"), p=result.get("placement"), t=result.get("turn"),
                s=result.get("placement_source"), term=result.get("terminal"),
                proof=result.get("terminal_proof"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
