"""Recover human Dark Gift selections from the offered entity order and the actual chosen entity."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.parse_firestone_replay_xml import CardDb, ENT_TAGS, XmlReplay

CARDS = ROOT/"engine/crates/hsbg-engine/data/cards.json"
SOURCE = "BG36_MidGameEffect_010"
BUTTON = "BG36_Button_DarkGift"
RESTRICTED = {"BG35_123", "BG35_342", "BG36_102", "BG36_356"}


def selected_index(offers, chosen):
    if len(offers) != 3 or set(offers.values()) != {0, 1, 2} or chosen not in offers:
        raise ValueError("missing or ambiguous three-way choice")
    return offers[chosen]


def inspect(path):
    replay = XmlReplay(CardDb(str(CARDS)))
    stack = []
    pending = None
    rows = []
    repeated = 0
    with zipfile.ZipFile(path) as archive, archive.open(next(n for n in archive.namelist() if n.endswith(".xml"))) as stream:
        for ev, el in ET.iterparse(stream, events=("start","end")):
            replay.seq += 1
            if ev == "start":
                if el.tag == "Block":
                    card = replay.ent.get(el.get("entity"),{}).get("card")
                    stack.append((el.get("type"),card))
                    replay._on_block_start(el)
                elif el.tag == "TagChange" and el.get("entity"):
                    replay.set_tag(el.get("entity"),el.get("tag"),el.get("value"))
                continue
            if el.tag in ENT_TAGS:
                replay._on_entity_end(el)
                el.clear()
            elif el.tag == "Block":
                replay._on_block_end(el)
                stack.pop()
                el.clear()
            elif el.tag == "Choices" and replay.main is not None and el.get("playerID") == replay.main:
                options = sorted((int(v.get("index")),v.get("entity")) for v in el.findall("Choice"))
                if pending is not None:
                    menu, _, previous = pending
                    if (menu == (el.get("source"), options) and previous["turn"] == replay.round
                            and previous["cards"] == [replay.ent.get(eid,{}).get("card") for _,eid in options]):
                        repeated += 1
                        el.clear()
                        continue
                    raise ValueError("new main-player choice before resolving an open Dark Gift")
                if replay.ent.get(el.get("source"),{}).get("card") == SOURCE:
                    offers = {eid:slot for slot,eid in options}
                    ids = [replay.ent.get(eid,{}).get("card") for _, eid in options]
                    if not all(ids) or [s for s,_ in options] != [0,1,2]:
                        raise ValueError("missing offered entity identity or position")
                    pending = ((el.get("source"),options),offers,dict(turn=replay.round, offer_sequence=replay.seq,
                        manual=("7",BUTTON) in stack, cards=ids,
                        board=[m["card_id"] for m in replay._board(replay.main_ctrl)]))
                el.clear()
            elif el.tag == "ChosenEntities" and el.get("playerID") == replay.main and pending is not None:
                choices = el.findall("Choice")
                if len(choices) != 1 or el.get("count") != "1":
                    raise ValueError("not a single observed main-player selection")
                _, offers, row = pending
                slot = selected_index(offers,choices[0].get("entity"))
                row.update(chosen_slot=slot,native_action=30+slot,chosen=row["cards"][slot],chosen_sequence=replay.seq)
                rows.append(row)
                pending = None
                el.clear()
    if pending is not None:
        raise ValueError("unresolved human Dark Gift")
    return dict(source=file_identity(path),rows=rows,retransmissions=repeated)


def main():
    out = Path("artifacts/runs/nus-peer-gift-selections-20261120-1")
    source = source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("dirty source")
    files = sorted(Path("data/firestone_replays/hotfix-xml-20261120-01").glob("*.xml.zip"))
    if len(files) != 24:
        raise ValueError("independent source count differs")
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],files=[file_identity(p) for p in files],
        cards=file_identity(CARDS),engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),scope="Observed human choices, minion identities only; no outcome or gift-affinity imputation."))
    with ProcessPoolExecutor(6) as pool:
        records = list(pool.map(inspect,files))
    rows = [r for game in records for r in game["rows"]]
    late = [r for r in rows if r["turn"]>9]
    atomic_json(out/"choices.json",records)
    atomic_json(out/"complete.json",dict(games=len(files),games_with_choices=sum(bool(g["rows"]) for g in records),
        choices=len(rows),button=sum(r["manual"] for r in rows),late=len(late),
        late_games=sum(any(r["turn"]>9 for r in g["rows"]) for g in records),
        late_restricted_offers=sum(k.removesuffix("_G") in RESTRICTED for r in late for k in r["cards"]),
        chosen_slots=dict(Counter(r["chosen_slot"] for r in rows)),examples=file_identity(out/"choices.json")))


if __name__ == "__main__":
    main()
