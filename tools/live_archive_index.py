"""Index archived Hearthstone logs and collapse duplicate copies (read-only).

Every run stores byte-prefix copies of the same session logs. Auditing a night
therefore needs one entry per *source range*, not one per run directory. This
tool reads the manifests that already carry ``source``/``sha256``/
``source_start_offset`` and writes a deduplicated index:

    artifacts/local/archive-index/<name>.json

Usage:
    python tools/live_archive_index.py artifacts/runs [--out artifacts/local/archive-index]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/local/archive-index"))
    args = ap.parse_args()
    entries: dict[tuple, dict] = {}
    for manifest in sorted(args.runs.rglob("manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for item in data.get("files", []):
            source = item.get("source")
            if not source:
                continue
            key = (source, item.get("source_start_offset"), item.get("sha256"), item.get("bytes"))
            entry = entries.setdefault(key, dict(source=source, start_offset=item.get("source_start_offset"),
                                                bytes=item.get("bytes"), sha256=item.get("sha256"),
                                                consumers=[]))
            entry["consumers"].append(str(manifest.parent.parent.parent.relative_to(args.runs)))
    args.out.mkdir(parents=True, exist_ok=True)
    index = list(entries.values())
    total_copies = sum(len(e["consumers"]) for e in index)
    payload = dict(ranges=len(index), copies=total_copies,
                   duplicated=sum(len(e["consumers"]) - 1 for e in index), entries=index)
    target = args.out / (args.runs.name + ".json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ranges=%d copies=%d redundant=%d -> %s" % (payload["ranges"], payload["copies"],
                                                      payload["duplicated"], target))
    for entry in sorted(index, key=lambda e: -len(e["consumers"]))[:5]:
        if len(entry["consumers"]) > 1:
            print("  %s @%s x%d" % (Path(entry["source"]).name, entry["start_offset"], len(entry["consumers"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
