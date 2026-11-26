"""Seal and audit one HSRL BC split using the matching installed native extension."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import taverngym
from taverngym._core import bc_teacher_name

A = dict(taverngym.layout())
ROW = np.dtype([
    ("features", "<f4", A["OBS_FLOATS"]),
    ("cards", "<i4", A["OBS_CARDS"]),
    ("mask", "u1", A["N_ACTIONS"]),
    ("action", "<u2"),
    ("turn", "<u2"),
    ("rank", "u1"),
    ("flags", "u1"),
])
BOARD_START = 3 + A["N_POWERS"] + 3 * A["N_QUESTS"] + A["N_SECRETS"]
CORES = {
    "tea": (("BG36_400",), ("BG35_883", "BG36_320")),
    "azerite_brann": (("BG26_537",), ("BG_LOEA_010",)),
    "apm_undead": (("BG33_823",), ("BG33_825",), ("BG34_692",), ("BG36_515",)),
    "apm_magnetic": (("BG33_823",), ("BG33_825",), ("BG36_367",), ("BG26_149",)),
    "apm_deity": (("BG33_823",), ("BG33_825",), ("BG36_111",)),
}


def card_groups(catalogue):
    by_id = {card["id"]: card for card in catalogue}
    groups = {}
    for key, requirements in CORES.items():
        groups[key] = []
        for alternatives in requirements:
            ids = set()
            for name in alternatives:
                card = by_id[name]
                ids.add(card["cid"] + 1)
                if card.get("golden_id") in by_id:
                    ids.add(by_id[card["golden_id"]]["cid"] + 1)
                if name + "_G" in by_id:
                    ids.add(by_id[name + "_G"]["cid"] + 1)
            groups[key].append(tuple(sorted(ids)))
    return groups


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def audit(split, source_commit):
    m = json.loads((split / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "format": 2, "teacher": bc_teacher_name(), "rules": taverngym.server_rules_version(),
        "row_bytes": ROW.itemsize, "floats": A["OBS_FLOATS"], "cards": A["OBS_CARDS"],
        "actions": A["N_ACTIONS"], "turn_action_cap": 100, "allow_move": False,
        "allow_cancel": False, "lobby": "five random of ten",
    }
    if any(m.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{split}: unexpected simulator or schema")
    if m["field"] not in {"original", "mixed"}:
        raise ValueError(f"{split}: unexpected opponent field")

    cores = card_groups(json.loads(taverngym.catalog_json()))
    totals = Counter()
    seed_set = set()
    lobbies = set()
    shards = {}
    for index in sorted(split.glob("games-*.jsonl")):
        raw_index = index.read_bytes()
        ends, places, decision_counts, seeds = [], [], [], []
        offset = 0
        for line in raw_index.splitlines():
            item = json.loads(line)
            game = item["game"]
            seed = game["seed"]
            if (
                seed in seed_set or game["seat"] != seed % 8 or game["rank"] not in range(1, 9)
                or item["offset"] != offset or game["lobby"].bit_count() != 5
                or item["rows"] != (game["actions"] if m["rows"] else 0)
            ):
                raise ValueError(f"{split}: invalid record {seed}")
            seed_set.add(seed)
            lobbies.add(game["lobby"])
            offset += item["rows"]
            ends.append(offset)
            places.append(game["rank"])
            decision_counts.append(game["decisions"] if m["rows"] else 0)
            seeds.append(seed)
            totals["budget_turns"] += game["budget_turns"]
            totals["recruit_turns"] += len(game["rounds"])

        name = index.name.replace("games-", "rows-").replace(".jsonl", ".bin")
        path = split / name
        size = path.stat().st_size
        if size != offset * ROW.itemsize:
            raise ValueError(f"{path}: size does not match its game index")
        digest = hashlib.sha256()
        per_game_decisions = np.zeros(len(seeds), dtype=np.int64)
        core_games = {key: set() for key in cores}
        freezes = set()
        if offset:
            table = np.memmap(path, dtype=ROW, mode="r", shape=(offset,))
            expected_ranks = np.asarray(places, dtype=np.uint8)
            for start in range(0, offset, 4096):
                batch = table[start : start + 4096]
                digest.update(memoryview(batch).cast("B"))
                owners = np.searchsorted(ends, np.arange(start, start + len(batch)), side="right")
                masks = batch["mask"]
                actions = batch["action"]
                flags = batch["flags"]
                choice = masks.sum(axis=1) > 1
                turns = batch["turn"]
                if (
                    (actions >= A["N_ACTIONS"]).any()
                    or not np.all(masks[np.arange(len(batch)), actions] == 1)
                    or (masks > 1).any()
                    or masks[:, A["A_CANCEL"]].any()
                    or masks[:, A["A_MOVE"] :].any()
                    or (flags & ~np.uint8(7)).any()
                    or not np.array_equal(choice, (flags & 4) != 0)
                    or not np.array_equal(batch["rank"], expected_ranks[owners])
                    or (batch["cards"] < 0).any()
                    or (batch["cards"] >= taverngym.N_CARDS).any()
                    or not np.isfinite(batch["features"]).all()
                ):
                    raise ValueError(f"{path}: invalid row at block {start}")
                np.add.at(per_game_decisions, owners, choice.astype(np.int64))
                board = batch["cards"][:, BOARD_START : BOARD_START + A["N_BOARD"]]
                for key, requirements in cores.items():
                    match = np.ones(len(batch), dtype=bool)
                    for group in requirements:
                        match &= np.isin(board, group).any(axis=1)
                    core_games[key].update(owners[match])
                    totals[key + "_decisions"] += int(np.count_nonzero(match & choice))
                for owner, turn in zip(owners[actions == A["A_FREEZE"]], turns[actions == A["A_FREEZE"]], strict=True):
                    token = (owner, turn)
                    totals["freeze"] += 1
                    if token in freezes:
                        totals["repeat_freeze"] += 1
                    freezes.add(token)
                totals["late_decisions"] += int(np.count_nonzero(choice & (turns >= 8)))
                totals["choices"] += int(np.count_nonzero(choice))
                totals["rows"] += len(batch)
        if not np.array_equal(per_game_decisions, decision_counts):
            raise ValueError(f"{path}: decision counts do not match game index")
        for key, hits in core_games.items():
            totals[key + "_games"] += len(hits)
        shards[name] = {
            "sha256": digest.hexdigest(), "index_sha256": hashlib.sha256(raw_index).hexdigest(),
            "bytes": size, "games": len(seeds), "rows": offset,
        }

    if seed_set != set(range(m["first_seed"], m["first_seed"] + m["games"])):
        raise ValueError(f"{split}: seed range mismatch")
    if totals["rows"] != m["rows"]:
        raise ValueError(f"{split}: manifest row count mismatch")
    fingerprint = taverngym.build_info()["source_hash"]
    m.update(source_hash=fingerprint, source_commit=source_commit, shards=shards)
    summary = {
        "split": split.name, "games": len(seed_set), "distinct_lobbies": len(lobbies),
        "bytes": sum(shard["bytes"] for shard in shards.values()), "totals": dict(totals),
    }
    write_json(split / "manifest.json", m)
    write_json(split / "audit.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    audit(args.split, args.source_commit)
