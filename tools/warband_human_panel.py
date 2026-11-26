"""Build a board-only rolling panel without inventing player trajectories."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import taverngym

root, raw, card_table, native_cards = map(Path,sys.argv[1:])
source_db = json.loads(card_table.read_text())["cards"]
target_db = json.loads(native_cards.read_text())["cards"]
targets = {card["id"]:i for i,card in enumerate(target_db)}
for i,card in enumerate(target_db):
    assert taverngym.card_info(i+1)[0] == card["id"]
translated = {i:targets[card["id"]] for i,card in enumerate(source_db) if card["id"] in targets}
turns = defaultdict(list)
seen = set()
drops = Counter()
for path in sorted(raw.glob("snap_*.json.gz")):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        shard = json.load(f)
    assert shard["engine"]["server_rules_version"] == "37.0.3"
    for row in shard["frames"]:
        if row["turn"] > 17:
            continue
        original = row["source_game_id"]
        identity=(original,row.get("source_side"),row["turn"])
        if identity in seen:
            drops["duplicate_origin_side_turn"] += 1
            continue
        seen.add(identity)
        board = row["player"]["board"]
        if not board:
            continue
        try:
            for m in board:
                m["card"] = translated[m["card"]]
                m["attached"] = [translated[x] for x in m.get("attached",[])]
                m["gained_dr"] = [translated[x] for x in m.get("gained_dr",[])]
                m["mods"] = []
                m["killed_by"] = 0
        except KeyError:
            drops["unknown_card"] += 1
            continue
        # Testing covers visible compatible tribes, not an inferred original lobby mask.
        races = tuple(m["races"] for m in board if m["races"].bit_count() in (1,2))
        power = sum(m["atk"]+m["hp"] for m in board)
        p=row["player"]
        for key in ("shop","hand","hand_queue","hand_queue_marks","last_board","seen_board","scouted"):
            p[key]=[]
        turns[row["turn"]].append((races,power,original,p))
for values in turns.values():
    values.sort(key=lambda x:x[1])
randomizer = random.Random(61831)
span = 16
used = defaultdict(set)
unique_sizes = {}
target = root/"human-external.jsonl"
with target.with_suffix(".tmp").open("w") as f:
    with (root/"source-data/dev-script-0.jsonl").open() as data:
        f.write(data.readline())
    for mask in range(1,1<<10):
        if mask.bit_count()!=5:
            continue
        frames = [[] for _ in range(span)]
        unique_sizes[mask] = {}
        for t in range(1,18):
            eligible = [r for r in turns[t] if all(mask & part for part in r[0])]
            randomizer.seed(35_971 + mask*37 + t)
            unique_sizes[mask][t] = len({x[2] for x in eligible})
            if len(eligible) < 3:
                break
            ordered = list(range(span))
            randomizer.shuffle(ordered)
            originals=set()
            for slice_id,carrier in enumerate(ordered):
                available = [x for x in eligible[len(eligible)*slice_id//span:
                    max(len(eligible)*slice_id//span+1,len(eligible)*(slice_id+1)//span)]
                    if x[2] not in originals]
                if not available:
                    available = [x for x in eligible if x[2] not in originals] or eligible
                _,_,original,player = randomizer.choice(available)
                originals.add(original)
                used[t].add(original)
                frames[carrier].append({"turn":t,"player":player})
        if not all(len(x)>=6 for x in frames):
            raise RuntimeError(f"Insufficient boards for lobby {mask}")
        for i,r in enumerate(frames):
            print(json.dumps({"seed":14800000+mask*span+i,"lobby":mask,"source":"human-rolling",
                              "frames":r},separators=(",",":")),file=f)
target.with_suffix(".tmp").replace(target)
receipt={"source_manifest_sha256":hashlib.file_digest((raw/"manifest.json").open("rb"),"sha256").hexdigest(),
         "original_games_by_turn":{t:len(xs) for t,xs in sorted(used.items())},
         "dropped":drops,"mode":"rolling only","source_cards":len(source_db),"translated":len(translated),
         "coverage":{t:min(stats.get(t,0) for stats in unique_sizes.values()) for t in range(1,18)}}
(root/"human-receipt.json").write_text(json.dumps(receipt,indent=2))
(root/"human-plan.json").write_text(json.dumps({"human":{"bank":str(target),
    "seats":7,"start":1,"level":0,"mode":"rolling"}},indent=2))
