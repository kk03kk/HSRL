#!/usr/bin/env python3
"""Overnight unattended supervisor for live Battlegrounds batches.

Why this exists (2026-11-11 post-mortem): the batch loop inside
``live_245.py autoplay --games N`` has **no per-match wall-clock bound and no
stall detector**. When the client showed the offline「重新连接」dialog, every match
failed in 8.8 s, and after the failure budget was spent the whole process simply
exited and nothing restarted it. Nine batches died like that and roughly five
hours produced zero games. Leaving one script running is not "unattended"; it is
"unmonitored".

This supervisor is deliberately outside the input hot path. It never sends a
click. It only:

* launches ONE batch at a time in a fresh output directory,
* watches every file under that directory as a liveness signal,
* writes ``STOP`` and, if the child does not exit, terminates it,
* starts the next batch in a NEW directory (a fresh input ledger),
* stops at a deadline, or when a global safety cap is hit.

Terminal-game counting: ``result.json`` is only ever written after the client
reports ``phase == game_over``, so its presence *is* the terminal proof. The
supervisor only counts those, and never a batch whose ``batch-stop.json`` says a
severity S0 stopped it before the game.

Usage:
    python client/tools/live_overnight.py \
      --game-dir D:\\Hearthstone \
      --root artifacts/runs/live-night-20261112 \
      --deadline 08:15 --games-per-batch 6 --match-seconds 3000 --stall-seconds 420
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "client" / "live_245.py"


def _now() -> float:
    return time.time()


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def newest_mtime(directory: Path) -> float:
    """Latest file mtime anywhere under ``directory``; 0.0 when it is empty.

    A healthy match writes status/health/events continuously, so a quiet tree is
    the cheapest possible stuck detector and needs no cooperation from the child.
    """
    newest = 0.0
    try:
        for dirpath, _dirnames, filenames in os.walk(directory):
            for name in filenames:
                try:
                    m = os.stat(os.path.join(dirpath, name)).st_mtime
                except OSError:
                    continue
                if m > newest:
                    newest = m
    except OSError:
        return 0.0
    return newest


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - supervisor must never die on a partial file
        return None


def terminal_games(batch_dir: Path) -> list[dict]:
    """Completed games in one batch. ``result.json`` only appears at game_over."""
    rows = []
    for result in sorted(Path(batch_dir).glob("match-*/result.json")):
        payload = read_json(result) or {}
        rows.append(
            dict(
                match=result.parent.name,
                game_id=payload.get("game_id"),
                result=payload.get("result"),
                placement=payload.get("placement"),
                placement_source=payload.get("placement_source"),
                turn=payload.get("turn"),
                terminal=bool(payload.get("game_id")),
                model_sha256=payload.get("model_sha256"),
            )
        )
    return rows


def summarise_batches(root: Path) -> dict:
    batches = []
    for batch in sorted(root.glob("batch-*")):
        stop = read_json(batch / "batch-stop.json") or {}
        games = terminal_games(batch)
        attempted = len(read_json(batch / "matches.json") or [])
        batches.append(
            dict(
                batch=batch.name,
                attempted=attempted,
                terminal=len(games),
                stop_why=stop.get("why"),
                stop_severity=stop.get("severity"),
                games=games,
            )
        )
    return dict(
        updated=_stamp(),
        batches=batches,
        terminal_total=sum(b["terminal"] for b in batches),
        attempted_total=sum(b["attempted"] for b in batches),
    )


def log(path: Path, **fields) -> None:
    fields.setdefault("at", _stamp())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields, ensure_ascii=False) + "\n")


def parse_deadline(text: str) -> float:
    """``HH:MM`` today, or tomorrow when that clock time has already passed."""
    hour, minute = (int(part) for part in text.split(":"))
    now = datetime.now()
    when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if when <= now:
        when += timedelta(days=1)
    return when.timestamp()


def stop_child(proc: subprocess.Popen, batch: Path, *, grace: float) -> str:
    """Ask nicely via STOP, then terminate. Returns how it ended."""
    try:
        if Path(batch).is_dir():
            (batch / "STOP").write_text("supervisor stop\n", encoding="utf-8")
    except OSError:
        pass
    deadline = _now() + grace
    while _now() < deadline:
        if proc.poll() is not None:
            return "stopped_cleanly"
        time.sleep(2.0)
    proc.terminate()
    deadline = _now() + 20.0
    while _now() < deadline:
        if proc.poll() is not None:
            return "terminated"
        time.sleep(1.0)
    proc.kill()
    return "killed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--deadline", default="08:15", help="local HH:MM to stop")
    ap.add_argument("--games-per-batch", type=int, default=6)
    ap.add_argument("--match-seconds", type=int, default=3000,
                    help="per-match wall-clock bound handed to the child")
    ap.add_argument("--stall-seconds", type=float, default=420.0,
                    help="no file activity under the batch dir for this long = stuck")
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--max-s0", type=int, default=6,
                    help="severity-S0 batches tolerated before giving up for the night")
    ap.add_argument("--max-stalls", type=int, default=6)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--grace-seconds", type=float, default=90.0)
    ap.add_argument("--stop-at-games", type=int, default=0,
                    help="stop after N terminal games (0 = only the deadline stops us)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"refusing to reuse a non-empty overnight root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    events = root / "overnight.jsonl"
    deadline = parse_deadline(args.deadline)
    log(events, event="start", deadline=datetime.fromtimestamp(deadline).isoformat(),
        games_per_batch=args.games_per_batch, match_seconds=args.match_seconds,
        stall_seconds=args.stall_seconds, root=str(root))

    s0_batches = 0
    stalls = 0
    saw_game = 0
    empty_batches = 0
    batch_index = 0
    while batch_index < args.max_batches:
        if _now() >= deadline:
            log(events, event="deadline_reached", terminal_total=summarise_batches(root)["terminal_total"])
            break
        summary = summarise_batches(root)
        if args.stop_at_games and summary["terminal_total"] >= args.stop_at_games:
            log(events, event="game_target_reached", terminal_total=summary["terminal_total"])
            break
        batch_index += 1
        batch = root / f"batch-{batch_index:04d}"
        # Never create this directory: the child refuses a non-empty --out and the
        # supervisor's own log would make it non-empty (that bug burned 7 batches in
        # 90 s the first time this ran). The child creates it.
        command = [
            sys.executable, str(LIVE), "autoplay",
            "--game-dir", args.game_dir,
            "--out", str(batch),
            "--games", str(args.games_per_batch),
            "--seconds", str(args.match_seconds),
            "--max-failures", "4",
            "--max-same-failure", "3",
            "--queue-retries", "4",
            "--queue-timeout", "300",
        ]
        log(events, event="batch_start", batch=batch.name, command=command)
        if args.dry_run:
            print(" ".join(command))
            continue
        logfile = (root / f"{batch.name}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(command, cwd=str(ROOT), stdout=logfile, stderr=subprocess.STDOUT)
        started = _now()
        last_note = 0.0
        outcome = "exited"
        while True:
            code = proc.poll()
            if code is not None:
                outcome = f"exited:{code}"
                break
            if _now() >= deadline:
                outcome = "deadline:" + stop_child(proc, batch, grace=args.grace_seconds)
                break
            quiet_for = _now() - max(newest_mtime(batch), started)
            if quiet_for > args.stall_seconds:
                stalls += 1
                log(events, event="stall_detected", batch=batch.name,
                    quiet_seconds=round(quiet_for, 1), stalls=stalls)
                outcome = "stall:" + stop_child(proc, batch, grace=args.grace_seconds)
                break
            if _now() - last_note >= 300:
                last_note = _now()
                partial = len(terminal_games(batch))
                log(events, event="heartbeat", batch=batch.name,
                    minutes=round((_now() - started) / 60.0, 1),
                    terminal_in_batch=partial,
                    quiet_seconds=round(quiet_for, 1))
            time.sleep(args.poll_seconds)
        logfile.close()
        stop = read_json(batch / "batch-stop.json") or {}
        games = terminal_games(batch)
        saw_game += len(games)
        log(events, event="batch_end", batch=batch.name, outcome=outcome,
            terminal=len(games), stop_why=stop.get("why"),
            stop_severity=stop.get("severity"),
            attempted=len(read_json(batch / "matches.json") or []),
            placements=[g.get("placement") for g in games])
        (root / "status.json").write_text(
            json.dumps(summarise_batches(root), ensure_ascii=False, indent=2), encoding="utf-8")
        if stop.get("severity") == "S0":
            s0_batches += 1
            if s0_batches > args.max_s0:
                log(events, event="give_up", reason="too many S0 batches", s0_batches=s0_batches)
                break
        # A batch that never even started a match means the launcher is broken, not
        # that the client is flaky; without this guard it would burn every batch
        # index in seconds and hide the cause.
        if not games and not (read_json(batch / "matches.json") or []):
            empty_batches += 1
        else:
            empty_batches = 0
        if empty_batches > 3:
            log(events, event="give_up", reason="batches fail before any match",
                empty_batches=empty_batches)
            break
        if stalls > args.max_stalls:
            log(events, event="give_up", reason="too many stalls", stalls=stalls)
            break
    (root / "status.json").write_text(
        json.dumps(summarise_batches(root), ensure_ascii=False, indent=2), encoding="utf-8")
    log(events, event="finish", terminal_total=summarise_batches(root)["terminal_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
