"""Build a locked post-hotfix visible-board transfer field using 37.0.2 simulation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.mirror_snapshots import engine_stamp, write_snapshot_file
from hsbg.az.train.personal import fingerprint, library_digest
from hsbg.runtime.artifacts import atomic_json
from tools.build_human_turn_library import sha, visible_frame
from tools.inventory_firestone_boards import probe_game, read, unpack

FEED_SHA = "257bf0f9c591dc3e3ac2b59e0d79766bed7a407631f745431de7d14f2953b5b5"


def checked_slots(task):
    path, meta, digest = task
    if sha(path) != digest:
        raise ValueError("raw file identity mismatch")
    _, slots = unpack((path, meta))
    slots = {str(k):v for k,v in slots.items()}
    gid, verdict = probe_game((meta[0], slots))
    return gid, verdict, {k:v for k,v in slots.items() if verdict[k] == "pass"}


def build(catalog, out, workers):
    import hsbg_engine
    if sha(catalog/"feed.json") != FEED_SHA or fingerprint()["engine"]["source_hash"] != "c4c71014e3d41760":
        raise ValueError("unexpected feed or engine")
    rows = {r.get("originalReviewId") or r["reviewId"]:r for r in read(catalog/"feed.json")}
    rows = {k:r for k,r in rows.items() if "2026-11-18" <=
            (r.get("originalCreationDate") or r["creationDate"])[:10] <= "2026-11-19"}
    seen = set()
    for split in ("train", "hold"):
        seen.update(read(ROOT/f"artifacts/libraries/human-turn-3702-20261118/{split}/manifest.json")["games"])
    receipts = read(catalog/"raw-01/receipts.txt")
    if len(rows) != 353 or len(receipts) != len(rows) or {r[0] for r in receipts} != set(rows):
        raise ValueError("missing or duplicated source")
    jobs = []
    for gid, review, digest, size in sorted(receipts):
        row = rows[gid]
        path = catalog/"raw-01"/(gid+".json")
        if (gid in seen or review in seen or row["reviewId"] != review or row["buildNumber"] != 255790
                or path.stat().st_size != size):
            raise ValueError("prior library overlap or source metadata mismatch")
        jobs.append((path, (gid, review), digest))
    target = out.with_name(out.name+".building")
    if out.exists():
        raise FileExistsError(out)
    target.mkdir()
    counts, verdicts, frames, games, source_slots = Counter(), Counter(), [], [], {}
    shard = 0
    stamp = engine_stamp()
    def flush():
        nonlocal shard
        if frames:
            write_snapshot_file(str(target/f"snap_{shard:04d}.json.gz"), frames, engine=stamp)
            frames.clear()
            shard += 1
    with ProcessPoolExecutor(workers) as procs:
        for index, (gid, statuses, slots) in enumerate(procs.map(checked_slots, jobs, chunksize=3)):
            source_slots[gid] = statuses
            verdicts.update(statuses.values())
            if slots:
                games.append(gid)
            for key, slot in sorted(slots.items(), key=lambda x:int(x[0])):
                frame = visible_frame(hsbg_engine, index, gid, key, slot)
                frames.append(frame)
                counts[frame["turn"]] += 1
                if len(frames) == 384:
                    flush()
    flush()
    if not counts or sorted(counts) != list(range(1, max(counts)+1)):
        raise ValueError("transfer field has gaps")
    atomic_json(target/"source-slots.json", source_slots)
    atomic_json(target/"manifest.json", dict(games=games, source_games=len(jobs), frames=sum(counts.values()),
        turns=dict(sorted(counts.items())), statuses=verdicts, digest=library_digest(target),
        sim_engine=fingerprint(), data_patch="37.0.3 server hotfix period", dates=["2026-11-18","2026-11-19"],
        source_feed_sha256=FEED_SHA, raw_receipts_sha256=sha(catalog/"raw-01/receipts.txt"),
        split="external transfer only", fidelity="visible EOT boards in 37.0.2 combat; perfect-games feed"))
    target.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    build(args.catalog, args.out, args.workers)
