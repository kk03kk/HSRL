"""Transactional, resumable log ingestion. Sources are read-only; output stays local."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from .parsers import VERSION, PowerParser, HDTParser, fingerprint

ROOT = Path(__file__).resolve().parents[2]
PARSER_ID = VERSION + ":" + hashlib.sha256(Path(__file__).read_bytes() +
    Path(__file__).with_name("parsers.py").read_bytes()).hexdigest()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def output_directory(path):
    result = Path(path).resolve()
    if not result.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("capture output must be inside the main project's runs directory")
    result.mkdir(parents=True, exist_ok=True)
    return result


@contextmanager
def single_writer(directory):
    """An OS lock releases on crash; no stale PID lock or forced process killing."""
    path = Path(directory) / "collector.lock"
    with path.open("a+b") as handle:
        handle.seek(0); handle.write(b"0"); handle.flush(); handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("collector already running for this output directory") from e
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class Store:
    def __init__(self, directory):
        self.directory = output_directory(directory)
        self.db = sqlite3.connect(self.directory / "encounters.sqlite3", timeout=20)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, generation INTEGER NOT NULL,
          offset INTEGER NOT NULL, anchor TEXT NOT NULL, state TEXT NOT NULL,
          parser TEXT NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS encounters (
          id TEXT PRIMARY KEY, root TEXT NOT NULL, source TEXT NOT NULL,
          revision INTEGER NOT NULL, digest TEXT NOT NULL, data TEXT NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS revisions (
          id TEXT NOT NULL, revision INTEGER NOT NULL, data TEXT NOT NULL,
          PRIMARY KEY(id, revision));
        CREATE TABLE IF NOT EXISTS diagnostics (
          source TEXT NOT NULL, reason TEXT NOT NULL, count INTEGER NOT NULL,
          PRIMARY KEY(source, reason));
        """)

    def close(self):
        self.db.close()

    def diagnostic(self, source, reason):
        self.db.execute("INSERT INTO diagnostics VALUES(?,?,1) ON CONFLICT(source,reason) DO UPDATE SET count=count+1",
                        (source, reason))

    def put(self, row, catalog=None):
        if catalog:
            for side in ("player", "opponent"):
                for card in row[side]["board"]:
                    definition = catalog.cards.get(card.get("card_id"), {})
                    if definition.get("normal"):
                        card["golden"] = True
                        card["golden_source"] = "reference_card_relationship"
            ids = {c.get("card_id") for side in ("player", "opponent") for c in row[side]["board"] if c.get("card_id")}
            row["reference_catalog"] = catalog.version
            row["unknown_reference_card_ids"] = sorted(ids - set(catalog.cards))
            row["native_compatibility"] = "state_and_rules_adapter_required"
        row["combat_signature"] = fingerprint([row["root"], row.get("turn"), row["player"]["board"], row["opponent"]["board"]])
        data = dump(row); digest = hashlib.sha256(data.encode()).hexdigest()
        previous = self.db.execute("SELECT revision,digest FROM encounters WHERE id=?", (row["encounter_id"],)).fetchone()
        if previous and previous[1] == digest:
            return 0
        revision = previous[0] + 1 if previous else 1
        self.db.execute("INSERT INTO encounters VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,digest=excluded.digest,data=excluded.data,updated=excluded.updated",
                        (row["encounter_id"], row["root"], row["source_id"], revision, digest, data, time.time()))
        self.db.execute("INSERT INTO revisions VALUES(?,?,?)", (row["encounter_id"], revision, data))
        return 1

    @staticmethod
    def anchor(handle, offset):
        handle.seek(max(0, offset - 128))
        return hashlib.sha256(handle.read(min(offset, 128))).hexdigest()

    def ingest(self, path, kind, catalog, *, max_bytes=8 * 1024 * 1024):
        """Commit parser state and emitted rows with exactly the same consumed byte offset.

        Partial UTF-8/lines are not consumed until their newline is present. A changed
        pre-cursor anchor detects truncation/replacement even if the file regrew.
        """
        if kind not in ("power", "hdt") or max_bytes < 1024:
            raise ValueError("invalid source kind/chunk size")
        path = Path(path).resolve()
        # No raw source names or account identifiers in the database.
        sid = fingerprint([str(path).casefold(), kind])
        updated = 0
        with path.open("rb") as handle, self.db:
            old = self.db.execute("SELECT generation,offset,anchor,state,parser FROM sources WHERE id=?", (sid,)).fetchone()
            generation, offset, state = 0, 0, None
            if old:
                generation, offset, old_anchor, raw_state, parser_version = old
                if parser_version != PARSER_ID:
                    raise ValueError("parser version changed; use a new output directory")
                stat = os.fstat(handle.fileno()); size = stat.st_size
                candidate = json.loads(raw_state)
                io = candidate.get("__io", {})
                handle.seek(0)
                head_changed = bool(io) and hashlib.sha256(handle.read(io["head_n"])).hexdigest() != io["head"]
                same_size_rewrite = bool(io) and size == offset and stat.st_mtime_ns != io["mtime_ns"]
                if (size < offset or self.anchor(handle, offset) != old_anchor or head_changed
                        or same_size_rewrite or io and stat.st_ino != io["inode"]):
                    generation += 1; offset = 0
                    self.diagnostic(sid, "log_replaced_or_truncated")
                else:
                    state = candidate
            source = fingerprint([sid, generation])
            parser = PowerParser(source, state) if kind == "power" else HDTParser(source, catalog, state)
            handle.seek(offset)
            chunk = handle.read(max_bytes)
            end = chunk.rfind(b"\n") + 1
            if end == 0:
                if len(chunk) == max_bytes:
                    # Do not silently skip a line and continue as if history were intact.
                    raise ValueError("overlong_or_unterminated_log_line")
                return {"bytes": 0, "changed": 0, "source": sid}
            for line in chunk[:end].splitlines():
                for row in parser.feed(line.decode("utf-8-sig", errors="replace")):
                    updated += self.put(row, catalog)
            pending = parser.pending_record()
            if pending:
                updated += self.put(pending, catalog)
            offset += end
            stat = os.fstat(handle.fileno()); head_n = min(offset, 4096)
            handle.seek(0)
            parser.s["__io"] = {"head_n": head_n, "head": hashlib.sha256(handle.read(head_n)).hexdigest(),
                                  "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
            checkpoint = dump(parser.s)
            self.db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET generation=excluded.generation,offset=excluded.offset,anchor=excluded.anchor,state=excluded.state,updated=excluded.updated",
                            (sid, kind, generation, offset, self.anchor(handle, offset), checkpoint, PARSER_ID, time.time()))
        return {"bytes": end, "changed": updated, "source": sid}

    def summary(self):
        rows = [json.loads(x[0]) for x in self.db.execute("SELECT data FROM encounters")]
        missing = Counter(x for row in rows for x in row["missing_fields"])
        return {"format": "taverngym.encounter-summary.v1", "encounters": len(rows),
                "unit": "source_snapshots_not_unique_battles", "parser_fingerprint": PARSER_ID,
                "opponent_boards_recorded": sum(r["opponent_board_recorded"] for r in rows),
                "completed_boundaries": sum(r["capture_complete"] for r in rows),
                "actual_results": sum(r["actual_result"] is not None for r in rows),
                "solo_training_candidates_needing_adapter": sum(r.get("solo_training_candidate", False) for r in rows),
                "sources": dict(Counter(r["source_kind"] for r in rows)),
                "modes": dict(Counter(str(r.get("mode")) for r in rows)),
                "builds": dict(Counter(str(r["build"]) for r in rows)),
                "turns": dict(Counter(str(r["turn"]) for r in rows)),
                "missing_fields": dict(missing.most_common()),
                "training_import_allowed": False, "policy_input": False,
                "diagnostics": [dict(source=x[0], reason=x[1], count=x[2]) for x in self.db.execute("SELECT * FROM diagnostics")],
                "notice": "Observed opponents, not a complete hidden-state or PPO library environment."}

    def export(self):
        """Reproducible materialized snapshots, atomic replacement; SQLite is authoritative."""
        rows = [json.loads(x[0]) for x in self.db.execute("SELECT data FROM encounters ORDER BY id")]
        bank, quarantine = [], []
        for row in rows:
            opponent = copy_json(row["opponent"])
            target = bank if row.get("solo_training_candidate", False) else quarantine
            target.append({"format": "taverngym.observed-opponent.v1", "encounter_id": row["encounter_id"],
                "root": row["root"], "split": "unassigned", "build": row["build"],
                "rules_patch": row["rules_patch"], "turn": row["turn"], "boundary": row["boundary"],
                "source_kind": row["source_kind"], "opponent": opponent,
                "mode": row.get("mode"), "solo_training_candidate": row.get("solo_training_candidate", False),
                "available_tribes": row["available_tribes"], "readiness": row["readiness"],
                "missing_fields": row["missing_fields"], "training_import_allowed": False,
                "human_calibrated": False, "matchmaking_strength": row["matchmaking_strength"],
                "combat_signature": row["combat_signature"],
                "split_blocker": "link_reconnects_and_cross_source_duplicates_before_game_level_splitting"})
        payloads = {"encounters.jsonl": "".join(dump(r) + "\n" for r in rows),
                    "opponents.jsonl": "".join(dump(r) + "\n" for r in bank),
                    "quarantine.jsonl": "".join(dump(r) + "\n" for r in quarantine),
                    "summary.json": json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n"}
        for name, content in payloads.items():
            target = self.directory / name; tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(content, encoding="utf-8"); os.replace(tmp, target)
        return self.summary()


def copy_json(obj):
    return json.loads(json.dumps(obj))
