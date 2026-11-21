"""Compare first ordinary attacks in versioned Firestone PMS and raw HSReplay XML."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from workflows.extract_firestone_boards import remove_player_display_names
from workflows.parse_firestone_replay_xml import BLOCK_ATTACK, CardDb, XmlReplay


class Openers(XmlReplay):
    """Keep raw, top-level ATTACK blocks and the original main-player turn mapping."""

    def __init__(self, cards):
        super().__init__(cards)
        self.attacks = defaultdict(list)

    def _on_block_start(self, el):
        kind = el.get("type")
        if kind == BLOCK_ATTACK and not self.block_stack and el.get("entity"):
            self.attacks[self.round].append((int(el.get("entity")), el.get("ts")))
        self.block_stack.append(kind)

    def _on_block_end(self, el):
        self.block_stack.pop()

    def snapshot(self, reason):
        pass

    def result(self):
        return self.attacks


def xml_openers(path, cards):
    try:
        return Openers(cards).parse(path)
    except ET.ParseError:
        with zipfile.ZipFile(path) as archive:
            name = next(name for name in archive.namelist() if name.lower().endswith(".xml"))
            raw = archive.read(name)
        clean = remove_player_display_names(raw)
        if clean == raw:
            raise
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("replay.xml", clean)
            buffer.seek(0)
            return Openers(cards).parse(buffer)


def pms_battles(path, counts):
    data = json.loads(path.read_bytes())
    first = {
        record["turn"]: record["value"]
        for record in data.get("wentFirstInBattleOverTurn", [])
        if type(record.get("turn")) is int and type(record.get("value")) is bool
    }
    face_offs = {row.get("turn"): row for row in data.get("faceOffs", [])}
    seen = set()
    for battle in data.get("battleResultHistory", []):
        info = battle.get("battleInfo") or {}
        turn = (info.get("gameState") or {}).get("currentTurn")
        result = battle.get("actualResult")
        if type(turn) is not int or result not in ("won", "lost", "tied"):
            counts["invalid_turn_or_result"] += 1
            continue
        if turn in seen:
            raise ValueError(f"{path.stem}: duplicate solo battle at turn {turn}")
        seen.add(turn)
        if info.get("playerTeammateBoard") is not None or info.get("opponentTeammateBoard") is not None:
            counts["duos"] += 1
            continue
        sides = [info.get(key) or {} for key in ("playerBoard", "opponentBoard")]
        players = [side.get("player") or {} for side in sides]
        boards = [side.get("board") or [] for side in sides]
        tiers = [player.get("tavernTier") for player in players]
        health = [player.get("hpLeft") for player in players]
        if any(type(value) is not int for value in tiers + health):
            counts["missing_tier_or_health"] += 1
            continue
        face = face_offs.get(turn) or {}
        damage = face.get("damage") if face.get("result") == result else None
        yield {
            "source": path.stem,
            "turn": turn,
            "sizes": [len(board) for board in boards],
            "tiers": tiers,
            "health": health,
            "heroes": [player.get("cardId") for player in players],
            "boards": [
                [[minion.get("entityId"), minion.get("cardId"), minion.get("attack")] for minion in board]
                for board in boards
            ],
            "pms_first": None if turn not in first else int(not first[turn]),
            "result": result,
            "damage": damage,
        }


def record_first(row, side, counts):
    if side is None:
        counts["no_first_actor"] += 1
        return
    if min(row["health"]) <= 0 or min(row["sizes"]) == 0:
        counts["ghost_or_empty"] += 1
        return
    own, other = side, 1 - side
    if row["sizes"][own] != row["sizes"][other]:
        outcome = "more" if row["sizes"][own] > row["sizes"][other] else "fewer"
        key = "unequal_count_" + outcome
    elif row["tiers"][own] != row["tiers"][other]:
        outcome = "higher" if row["tiers"][own] > row["tiers"][other] else "lower"
        key = "equal_count_unequal_tier_" + outcome
    else:
        key = "equal_count_equal_tier_" + ("self" if side == 0 else "opponent")
    counts[key] += 1


def run(pms_dir, xml_dir, card_path):
    cards = CardDb(card_path)
    raw_sources = {p.name.removesuffix(".xml.zip"): p for p in xml_dir.glob("*.xml.zip")}
    cohorts = defaultdict(Counter)
    skipped = Counter()
    rows = []
    pms_sources = sorted(pms_dir.glob("*.json"))
    if not pms_sources or not raw_sources:
        raise ValueError("PMS and XML source folders must both be populated")
    for path in pms_sources:
        uuid = path.stem
        if uuid in raw_sources:
            split = "xml"
            events = xml_openers(raw_sources[uuid], cards)
            if events is None:
                raise ValueError(f"{uuid}: archive contains no XML")
        else:
            split = "validation" if hashlib.sha256(uuid.encode()).digest()[0] < 64 else "discovery"
            events = {}
        for row in pms_battles(path, skipped):
            row["split"] = split
            record_first(row, row["pms_first"], cohorts[split])
            if split == "xml":
                ids = [{m[0] for m in board if type(m[0]) is int} for board in row["boards"]]
                actor = next(iter(events.get(row["turn"], [])), None)
                if actor is not None:
                    sides = [side for side in (0, 1) if actor[0] in ids[side]]
                    if len(sides) == 1:
                        row["xml_first"] = sides[0]
                        row["xml_event"] = actor
                        if row["pms_first"] is not None:
                            key = "agreement" if sides[0] == row["pms_first"] else "disagreement"
                            cohorts["xml_alignment"][key] += 1
                record_first(row, row.get("xml_first"), cohorts["xml_raw"])
            rows.append(row)
    summary = {
        "pms_files": len(pms_sources),
        "xml_files": len(raw_sources),
        "pms_battles": len(rows),
        "cohorts": dict(cohorts),
        "skipped": skipped,
    }
    return dict(summary=summary, battles=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pms-dir", required=True, type=Path)
    parser.add_argument("--xml-dir", required=True, type=Path)
    parser.add_argument("--cards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args.pms_dir, args.xml_dir, args.cards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, separators=(",", ":"))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
