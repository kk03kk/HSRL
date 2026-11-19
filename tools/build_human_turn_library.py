"""Materialize the previously validated 37.0.2 Firestone slots, without resampling."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inventory_firestone_boards import FIELDS, bucket, collect_sources, template
from workflows.compare_board_libraries import install_board
from hsbg.az.train.mirror_snapshots import STAGE_EOT, engine_stamp, write_snapshot_file
from hsbg.az.train.personal import fingerprint, library_digest, write_json


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def visible_frame(engine, index, gid, key, slot):
    turn, side = divmod(int(key), 2)
    tier, units, _ = slot
    board = [dict(zip(FIELDS, u)) for u in units]
    game = engine.Game.from_json(template(turn))
    install_board(game, 0, {"tier": tier, "board_detail": board})
    player = json.loads(game.player_json(0))
    for field in ("hand", "shop", "hand_queue", "hand_queue_marks"):
        player[field] = []
    player.update(ended_turn=True, gold=0)
    receiver = engine.Game.from_json(template(turn))
    rep = receiver.apply_turn_snapshots([[0, json.dumps(player)]], False)
    actual = json.loads(receiver.player_json(0))
    if (rep["applied"] != 1 or rep["rejected"] or not rep["pool_balance_ok"]
            or not rep["pool_species_ok"] or receiver.invariant_violations()
            or actual["tier"] != tier or [[m[k] for k in FIELDS] for m in actual["board"]] != units):
        raise ValueError(f"round-trip mismatch {gid}/{key}")
    return dict(turn=turn, seat=0, source_turn=turn, source_game_seed=index,
                source_game_id=gid, source_side=side, stage=STAGE_EOT, player=player,
                source="firestone_visible_eot", pool_mode="unknown", held_pool_parts=[])


def build(args):
    import hsbg_engine as engine
    if (sha(args.receipts) != "aa82876ff5c2cab258e4d248b73d026fbae9604599c2dbc246a6045d49c910c7"
            or sha(args.dir/"local.jsonl.gz") != "d3aad923db977ecb70e9635ff8380556bbbc66e50c51fcb1e17754967eaf50e8"):
        raise ValueError("inventory changed")
    fp = fingerprint()
    if fp["engine"]["source_hash"] != "c4c71014e3d41760":
        raise ValueError("wrong engine")
    games, aliases = collect_sources(args)
    trained = set()
    with (args.data/"real_library/firestone-37.0.2-joint/train-joint.jsonl").open() as f:
        for line in f:
            gid = json.loads(line)["game_id"]
            trained.add(aliases.get(gid, gid))
    root = args.out.with_name(args.out.name+".building")
    if args.out.exists():
        raise FileExistsError(args.out)
    root.mkdir()
    stamp = engine_stamp()
    manifests = {}
    for split, size, matches in [("train",40659,2022),("hold",5477,278)]:
        target = root/split
        target.mkdir()
        counts, seen, frames = Counter(), set(), []
        shard = 0
        def flush():
            nonlocal shard
            if frames:
                write_snapshot_file(str(target/f"snap_{shard:04d}.json.gz"), frames, engine=stamp)
                frames.clear()
                shard += 1
        with gzip.open(args.receipts,"rt") as f:
            for index, line in enumerate(f):
                gid, dates, builds, role, slots = json.loads(line)
                if len(dates)!=1 or len(builds)!=1 or bucket(dates[0])!="main":
                    continue
                which = "train" if role in ("train","historic_train") else "hold"
                if which!=split or (split=="hold" and gid in trained):
                    continue
                for key, status in slots.items():
                    if status!="pass":
                        continue
                    frame = visible_frame(engine, index, gid, key, games[gid]["slots"][key])
                    frames.append(frame)
                    counts[frame["turn"]]+=1
                    seen.add(gid)
                    if len(frames)==384:
                        flush()
        flush()
        if sum(counts.values())!=size or len(seen)!=matches:
            raise ValueError((split,counts,len(seen)))
        manifest=dict(split=split,turns=dict(sorted(counts.items())),frames=size,games=sorted(seen),
                      digest=library_digest(target),fidelity="board_only_eot_approx",fingerprint=fp,
                      input_receipts_sha256=sha(args.receipts))
        write_json(target/"manifest.json",manifest)
        manifests[split]=manifest
    if set(manifests["train"]["games"]) & set(manifests["hold"]["games"]):
        raise ValueError("split overlap")
    root.rename(args.out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("dir","data","receipts","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    build(parser.parse_args())
