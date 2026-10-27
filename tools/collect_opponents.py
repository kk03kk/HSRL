"""Collect opponents from existing script/game logs; never launch/control the game."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

from opponent_capture.parsers import Catalog
from opponent_capture.store import ROOT, PARSER_ID, Store, single_writer


def sources(game_logs, hdt_logs, explicit):
    found = {}
    for directory, pattern, kind in ((game_logs, "**/Power.log", "power"),
                                     (hdt_logs, "**/hdt_log*.txt", "hdt")):
        if directory and Path(directory).is_dir():
            for p in Path(directory).glob(pattern):
                found[p.resolve()] = kind
    for p in explicit:
        p = Path(p).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        found[p] = "power" if p.name.casefold() == "power.log" else "hdt"
    return sorted(found.items(), key=lambda x: str(x[0]))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("scan", "watch", "report", "stop"))
    p.add_argument("--out", default=str(ROOT / "runs/opponent-capture"))
    p.add_argument("--game-logs", default="D:/Hearthstone/Logs")
    p.add_argument("--hdt-logs", default=str(Path.home() / "AppData/Roaming/HearthstoneDeckTracker/Logs"))
    p.add_argument("--input", action="append", default=[])
    p.add_argument("--interval", type=float, default=4)
    p.add_argument("--max-polls", type=int, help="bounded watch for verification")
    args = p.parse_args()
    if args.interval < 0.2 or args.max_polls is not None and args.max_polls < 1:
        p.error("invalid interval/max-polls")
    store = Store(args.out)
    if args.command == "stop":
        (store.directory / "STOP").write_text("stop requested\n", encoding="ascii")
        print("Stop requested; game/script is unaffected."); store.close(); return
    if args.command == "report":
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2)); store.close(); return
    catalog = Catalog(json.loads((ROOT / "crates/taverngym/data/cards.json").read_text(encoding="utf-8")))
    stopping = False
    acquired = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        with single_writer(store.directory):
            acquired = True
            if any(v[0] != PARSER_ID for v in store.db.execute("SELECT DISTINCT parser FROM sources")):
                raise RuntimeError("parser changed; preserve the old capture and use a new --out runs directory")
            if args.command == "watch" and (store.directory / "STOP").exists():
                raise RuntimeError("STOP file exists; remove it explicitly before restarting capture")
            polls = 0
            print("Read-only log capture; no game input. Output: " + str(store.directory), flush=True)
            while not stopping:
                count = changed = 0
                for path, kind in sources(args.game_logs, args.hdt_logs, args.input):
                    if stopping:
                        break
                    try:
                        while True:
                            result = store.ingest(path, kind, catalog)
                            count += result["bytes"]; changed += result["changed"]
                            if result["bytes"] == 0 or stopping:
                                break
                    except (OSError, ValueError) as e:
                        # Do not leak raw paths, player strings or malformed line contents.
                        with store.db:
                            store.diagnostic("input", type(e).__name__)
                if changed or polls == 0 or args.command == "scan":
                    summary = store.export()
                    print(json.dumps({"bytes_read": count, "changed_revisions": changed,
                        "encounters": summary["encounters"], "opponent_boards": summary["opponent_boards_recorded"]}), flush=True)
                status = {"format": "taverngym.capture-status.v1", "pid": __import__('os').getpid(),
                          "poll": polls + 1, "time_unix": time.time(), "bytes_read": count,
                          "changed_revisions": changed, "input_enabled": False, "state": "running"}
                tmp = store.directory / "status.tmp"
                tmp.write_text(json.dumps(status), encoding="utf-8"); tmp.replace(store.directory / "status.json")
                polls += 1
                if args.command == "scan" or args.max_polls and polls >= args.max_polls:
                    break
                if (store.directory / "STOP").exists():
                    break
                # Interruptible sleep; no scheduler, OS hooks or game dependency.
                deadline = time.monotonic() + args.interval
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(.2, max(0, deadline - time.monotonic())))
            store.export()
    finally:
        if acquired:
            stopped = store.directory / "status.json"
            state = json.loads(stopped.read_text()) if stopped.exists() else {}
            state.update(state="stopped", stopped_unix=time.time())
            tmp = store.directory / "status.tmp"
            tmp.write_text(json.dumps(state), encoding="utf-8"); tmp.replace(stopped)
        store.close()


if __name__ == "__main__":
    main()
