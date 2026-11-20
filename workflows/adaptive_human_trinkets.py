"""Check support for public greater-trinket preferences from actual human offer/choice pairs."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.adaptive_decision_study import RUNS, read
from workflows.parse_firestone_replay_xml import CardDb, ENT_TAGS, XmlReplay
from workflows.peer_human_spell_policy import ARCHIVE, CARDS, OUT as HUMAN

OUT = RUNS / "nus-adaptive-human-trinkets-20261120-1"
BANK = RUNS / "nus-adaptive-branch-20261120-1"


def parse(task):
    original, path, identity = task
    if file_identity(path) != identity:
        raise ValueError("human replay differs from the archived input")
    replay = XmlReplay(CardDb(str(CARDS)))
    pending = None
    found = []
    with zipfile.ZipFile(path) as archive, archive.open(next(n for n in archive.namelist() if n.endswith(".xml"))) as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            replay.seq += 1
            if event == "start":
                if element.tag == "TagChange" and element.get("entity"):
                    replay.set_tag(element.get("entity"), element.get("tag"), element.get("value"))
                continue
            if element.tag in ENT_TAGS:
                replay._on_entity_end(element)
                element.clear()
            elif element.tag == "Block":
                element.clear()
            elif element.tag == "Choices" and replay.main and element.get("playerID") == replay.main:
                listed = sorted((int(x.get("index")), x.get("entity")) for x in element.findall("Choice"))
                ids = [e for _, e in listed]
                cards = [replay.ent.get(e, {}).get("card") for e in ids]
                eligible = (replay.round == 9 and len(ids) >= 2 and
                            all(replay.cdb.kind(c) == "trinket_greater" for c in cards))
                if eligible:
                    if [n for n, _ in listed] != list(range(len(ids))) or len(set(ids)) != len(ids):
                        raise ValueError("invalid offered entities")
                    entry = (ids, cards)
                    if pending is not None and pending != entry:
                        raise ValueError("unresolved previous human offer")
                    pending = entry
                elif pending is not None:
                    raise ValueError("main player's next offer preceded the greater-trinket selection")
                element.clear()
            elif element.tag == "ChosenEntities" and replay.main and element.get("playerID") == replay.main and pending:
                picked = [v.get("entity") for v in element.findall("Choice")]
                ids, cards = pending
                if len(picked) != 1 or picked[0] not in ids:
                    raise ValueError("human pick not uniquely in offer")
                found.append(dict(cards=cards, chosen=cards[ids.index(picked[0])]))
                pending = None
                element.clear()
    if pending:
        raise ValueError("unresolved greater-trinket offer")
    return dict(original=original, menus=found)


def main():
    source = source_identity(ROOT)
    observed = [r for r in read(HUMAN / "observations.json") if "error" not in r]
    expected = {r["original"]: r["file"] for r in observed}
    available = {}
    for game in read(ARCHIVE / "bgs-perfect-games.json"):
        key = game.get("originalReviewId")
        path = ARCHIVE / "xmlz" / (game["reviewId"] + ".xml.zip")
        if key in expected and path.is_file() and key not in available and file_identity(path) == expected[key]:
            available[key] = path
    if len(expected) != 382 or set(expected) != set(available) or source["tracked_dirty"]:
        raise ValueError("incomplete or changed original replay cohort")
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT / "plan.json", dict(source=source, originals=expected, card_table=file_identity(CARDS),
        job=os.environ["SLURM_JOB_ID"], branch=file_identity(BANK / "plan.json")))
    with ProcessPoolExecutor(16) as pool:
        games = list(pool.map(parse, [(k, available[k], identity) for k, identity in expected.items()]))
    atomic_json(OUT / "observations.json", games)
    keys = sorted(expected, key=lambda k: sha256(("hsai-greater-original-v1/" + k).encode()).digest())
    train = set(keys[:256])
    offers, choices = Counter(), Counter()
    hold = []
    for g in games:
        if len(g["menus"]) != 1:
            continue
        menu = g["menus"][0]
        if len(set(menu["cards"])) != len(menu["cards"]):
            continue
        if g["original"] in train:
            offers.update(menu["cards"])
            choices[menu["chosen"]] += 1
        else:
            hold.append(menu)
    rate = {c: (choices[c] + 1) / (n + 4) for c, n in offers.items() if n >= 8}
    predictions = []
    for menu in hold:
        visible = menu["cards"]
        if all(c in rate for c in visible):
            best = max(visible, key=rate.get)
            if sum(rate[c] == rate[best] for c in visible) == 1:
                predictions.append((best == menu["chosen"], 1 / len(visible)))
    names = [c["id"] for c in read(CARDS)["cards"]]
    eligible = []
    for i in range(64):
        e = read(BANK / f"game-{i}.json")["alternatives"]["greater_runner"]["trigger"]
        if e is None:
            continue
        a, b = [names[e["cards"][e[k] - 30]] for k in ("original", "alternative")]
        if a in rate and b in rate and rate[b] - rate[a] >= .20:
            eligible.append(i)
    n = len(predictions)
    hits = sum(r[0] for r in predictions)
    chance = sum(r[1] for r in predictions)
    passed = n >= 32 and hits >= chance + .10 * n and len(eligible) >= 12
    atomic_json(OUT / "complete.json", dict(originals=len(games), menus=Counter(len(g["menus"]) for g in games),
        train_games=len(train), hold_games=len(keys)-len(train), hold_clean_menus=len(hold), rates=rate,
        predictable=n, hits=hits, expected_uniform=chance, eligible_lobbies=eligible, followup=passed))


if __name__ == "__main__":
    main()
