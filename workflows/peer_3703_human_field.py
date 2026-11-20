"""Qualify post-hotfix human EOT boards on the current engine, for external evaluation only."""
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.peer_3703_field import ENGINE

LIB = Path("artifacts/libraries/peer-3703-human-external-20261120-01")
ORIGINAL = Path("artifacts/libraries/human-turn-transfer-20261119-01")
SOURCE_MANIFEST = "d1581c4c13f9e8ccadad97b06aefbfe1963b4357dc2c63260d4e09d7b022175e"

def build():
    import hsbg_engine as engine
    from hsbg.az.train.lobby_filter import make_filtered_game
    from hsbg.az.train.mirror_snapshots import engine_stamp,write_snapshot_file
    from hsbg.az.train.personal import fingerprint,library_digest
    from workflows.mirror_hard_gates import _drive_to_turn
    src=ORIGINAL/"manifest.json"
    if file_identity(src)["sha256"]!=SOURCE_MANIFEST:
        raise ValueError("source manifest changed")
    old=json.loads(src.read_bytes())
    fp=fingerprint()
    if (old["digest"]!=library_digest(ORIGINAL) or old["split"]!="external transfer only"
        or old["sim_engine"]["vocab_sha256"]!=fp["vocab_sha256"] or fp["engine"]["source_hash"]!=ENGINE):
        raise ValueError("invalid transfer source or current native vocabulary")
    if LIB.exists():
        raise FileExistsError(LIB)
    building=LIB.with_suffix(".building")
    building.mkdir()
    g,_=make_filtered_game(engine,730000)
    base=json.loads(_drive_to_turn(g,1).state_json())
    counts=Counter()
    templates={}
    native_ids={c["cid"]:c["id"] for c in json.loads(engine.catalog_json())}
    for f in sorted(ORIGINAL.glob("snap_*.json.gz")):
        with gzip.open(f,"rt",encoding="utf8") as stream:
            data=json.load(stream)
        for frame in data["frames"]:
            turn=frame["turn"]
            if turn not in templates:
                templates[turn]=engine.Game.from_json(json.dumps(dict(base,turn=turn)))
            target=templates[turn].copy()
            p=frame["player"]
            if any(p.get(k) for k in ("hand","shop","hand_queue")):
                raise ValueError("nonpublic zones in visible-board field")
            if any(native_ids[m["card"]].removesuffix("_G")=="BG34_320" for m in p["board"]):
                raise ValueError(("hotfix-ban mismatch",frame["source_game_id"],turn))
            rep=target.apply_turn_snapshots([[0,json.dumps(p)]],False)
            if (rep["applied"]!=1 or rep["rejected"] or not rep["pool_species_ok"]
                or not rep["pool_balance_ok"] or target.invariant_violations()):
                raise ValueError(("invalid transfer",f,turn,rep))
            actual=json.loads(target.player_json(0))
            def visible(player):
                keys=("card","atk","hp","kw","bonus_kw","golden","shield_max")
                return player["tier"],[[m[k] for k in keys] for m in player["board"]]
            if visible(p)!=visible(actual):
                raise ValueError("visible board differs after injection")
            counts[turn]+=1
        write_snapshot_file(str(building/f.name),data["frames"],engine=engine_stamp())
    if sum(counts.values())!=6845 or {str(t):v for t,v in counts.items()}!=old["turns"]:
        raise ValueError("missing source frames")
    atomic_json(building/"manifest.json",dict(split="human",fingerprint=fp,digest=library_digest(building),
        turns=counts,games=old["games"],frames=sum(counts.values()),source_manifest=SOURCE_MANIFEST,
        dates=old["dates"],fidelity="post-hotfix public combat boards, seven visible minion fields",usage="external evaluation only"))
    building.rename(LIB)
    print("Qualified",sum(counts.values()),"human frames",flush=True)

if __name__=="__main__":
    os.chdir(ROOT)
    build()
