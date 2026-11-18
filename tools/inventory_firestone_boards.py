"""Count exact-turn Firestone observations and probe them in the frozen native engine."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.extract_firestone_boards import FLAGS, split_game
from workflows.parse_firestone_replay_xml import KW_BITS


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data)


@lru_cache(None)
def cards():
    rows = read(ROOT / "engine/crates/hsbg-engine/data/cards.json")["cards"]
    return {r["id"]: (i, r) for i, r in enumerate(rows, 1)}


def visible(raw):
    p = raw.get("player") or {}
    units = raw.get("board")
    if not units or len(units) > 7:
        return "empty_or_size"
    if not isinstance(p.get("hpLeft"), int) or p["hpLeft"] <= 0:
        return "dead_or_missing_hero_hp"
    tier = p.get("tavernTier")
    if type(tier) is not int or not 1 <= tier <= 6:
        return "tier_outside_1_6"
    result = []
    for unit in units:
        key = unit.get("cardId")
        if key not in cards():
            return "unknown_id"
        cid, card = cards()[key]
        if card["kind"] != "minion" or card["name"] == "Expert Aviator":
            return "non_minion_or_removed"
        if not card["reachable"] and not cards().get(card["normal"], (0, {}))[1].get("reachable"):
            return "reachability_unconfirmed"
        attack, hp = unit.get("attack"), unit.get("health")
        if (unit.get("definitelyDead") or type(attack) is not int or type(hp) is not int
                or attack < 0 or hp <= 0 or attack + hp > 100000):
            return "stats_or_dead_unit"
        printed = 0
        for key in card.get("keywords", []):
            printed |= KW_BITS.get(key, 0)
        kw = printed
        for flag, name in FLAGS.items():
            if flag in unit:
                bit = KW_BITS[name]
                kw = kw | bit if unit[flag] else kw & ~bit
        result.append([cid, attack, hp, kw, kw & ~printed, bool(card["normal"]), 1])
    if sum(u[1] + u[2] for u in result) > 250000:
        return "board_stat_limit"
    return [tier, result]


def unpack(task):
    path, meta = task
    slots = {}
    for battle in read(path).get("battleResultHistory") or []:
        info = battle.get("battleInfo") or {}
        turn = (info.get("gameState") or {}).get("currentTurn")
        if (battle.get("actualResult") is None or type(turn) is not int or not 1 <= turn <= 100
                or any(info.get(k) is not None for k in ("playerTeammateBoard", "opponentTeammateBoard"))):
            continue
        for side, field in enumerate(("playerBoard", "opponentBoard")):
            slot = 2 * turn + side
            val = visible(info.get(field) or {})
            slots[slot] = val if slots.get(slot, val) == val else "conflict"
    return [meta, slots]


def tasks(catalog, folder, tag, index=None):
    aliases = {}
    for file in [catalog] if index is None else [index, catalog]:
        for row in read(file):
            review = row["reviewId"]
            gid = row.get("originalReviewId") or review
            meta = [gid, review, (row.get("originalCreationDate") or row["creationDate"])[:10],
                    row.get("buildNumber"), tag]
            aliases[gid] = aliases[review] = meta
    for path in sorted(folder.glob("*.json")):
        key = path.name.split(".")[0]
        if key not in aliases:
            raise ValueError(f"Missing catalog entry {key}")
        yield path, aliases[key]


def local(args):
    jobs = []
    for folder in sorted(args.data.glob("*/pms")):
        index = folder.parent / "index.json"
        if index.exists():
            jobs.extend(tasks(index, folder, "downloaded"))
    jobs.extend(tasks(args.dir / "feed.json", args.dir / "latest-pms", "new"))
    with gzip.open(args.dir / "local.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as out:
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(unpack, jobs, chunksize=10):
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
    print("packed", len(jobs), flush=True)


@lru_cache(None)
def template(turn):
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn
    if E.game_version() != "37.0.2.256589" or E.server_rules_version() != "37.0.2":
        raise ValueError("wrong native engine")
    if turn == 0:
        for key, (i, _) in cards().items():
            if E.card_info(i)[0] != key:
                raise ValueError("native card index disagreement")
        state = json.loads(_drive_to_turn(E.Game(seed=730000), 1).state_json())
        for player in state["players"]:
            player["hero"] = player["deity"] = None
            for key in ("powers", "pending", "trinkets", "quests", "rewards", "secrets", "effects"):
                player[key] = []
        return state
    return json.dumps(dict(template(0), turn=turn))


FIELDS = ["card", "atk", "hp", "kw", "bonus_kw", "golden", "shield_max"]


def probe_game(item):
    import hsbg_engine as E
    from workflows.compare_board_libraries import install_board
    gid, slots = item
    verdict = {}
    for turn in sorted({int(s) // 2 for s in slots}):
        receiver = E.Game.from_json(template(turn))
        ready = []
        for side in (0, 1):
            key = str(2 * turn + side)
            if key not in slots:
                continue
            val = slots[key]
            if isinstance(val, str):
                verdict[key] = val
                continue
            tier, units = val
            sandbox = E.Game.from_json(template(turn))
            board = [dict(zip(FIELDS, u)) for u in units]
            install_board(sandbox, side, {"tier": tier, "board_detail": board})
            player = json.loads(sandbox.player_json(side))
            for field in ("hand", "shop", "hand_queue", "hand_queue_marks"):
                player[field] = []
            player.update(ended_turn=True, gold=0)
            rep = receiver.apply_turn_snapshots([[side, json.dumps(player)]], False)
            actual = json.loads(receiver.player_json(side))
            if (rep["applied"] != 1 or rep["rejected"] or not rep["pool_balance_ok"]
                    or not rep["pool_species_ok"] or receiver.invariant_violations()
                    or actual["tier"] != tier or not actual["ended_turn"]
                    or [[m[k] for k in FIELDS] for m in actual["board"]] != units):
                verdict[key] = "native_rejected"
                continue
            ready.append(key)
        if ready:
            result = json.loads(receiver.debug('{"cmd":"combat_preview","a":0,"b":1,"n":1,"details":true}'))
            trial, = result["trials"]
            for key in ready:
                verdict[key] = "combat_truncated" if trial["truncated"] else "pass"
    return gid, verdict


def bucket(day):
    if day < "2026-11-03":
        return "before"
    if day == "2026-11-03":
        return "nov03"
    if day < "2026-11-17":
        return "main"
    return "nov17" if day == "2026-11-17" else "after"


def assess(args):
    feed = read(args.dir / "feed.json")
    latest = {r.get("originalReviewId") or r["reviewId"] for r in feed}
    games, aliases = {}, {}
    def add(record):
        (gid, review, day, build, tag), rows = record
        aliases[review] = aliases[gid] = gid
        g = games.setdefault(gid, dict(days=set(), builds=set(), tags=set(), slots={}))
        g["days"].add(day)
        g["tags"].add(tag)
        g["builds"].add(build)
        for s, v in rows.items():
            s = str(s)
            g["slots"][s] = v if g["slots"].get(s, v) == v else "conflict"
    with gzip.open(args.dir / "local.jsonl.gz", "rt", encoding="utf8") as inp:
        for line in inp:
            add(json.loads(line))
    with ProcessPoolExecutor(args.workers) as pool:
        for name, catalog in (("refresh-20261112T031450Z", "bgs-perfect-games.json"),
                              ("refresh-20261116", "feed-fresh.json")):
            root = args.data / "firestone_replays" / name
            for row in pool.map(unpack, tasks(root/catalog, root/"pms", "downloaded", root/"index.json"), chunksize=8):
                add(row)
    joint = {}
    hold = set()
    historic_train = set()
    for part in ("train", "eval"):
        ids, kinds, extra = set(), Counter(), Counter()
        with (args.data/"real_library/firestone-37.0.2-joint"/(part+"-joint.jsonl")).open() as file:
            for line in file:
                row = json.loads(line)
                gid = aliases.get(row["game_id"], row["game_id"])
                ids.add(gid)
                kinds[row.get("source")] += 1
                side = {"self": 0, "opponent": 1}.get(row.get("perspective"))
                if side is None or str(2*row["turn"]+side) not in games.get(gid, {}).get("slots", {}):
                    extra[row.get("source")] += 1
        joint[part] = dict(games=len(ids), sources=kinds, rows_without_pms_slot=extra,
                           games_without_pms=len(ids-games.keys()))
        (hold if part == "eval" else historic_train).update(ids)
    if not latest <= games.keys():
        raise ValueError("latest replay coverage incomplete")
    stats = defaultdict(lambda: dict(games=0, games_with_pass=0, statuses=Counter(), turns=Counter()))
    def count(label, verdict):
        stat = stats[label]
        stat["games"] += 1
        stat["games_with_pass"] += "pass" in verdict.values()
        stat["statuses"].update(verdict.values())
        stat["turns"].update(int(s)//2 for s, status in verdict.items() if status=="pass")
    print("native games",len(games),flush=True)
    with gzip.open(args.out/"receipts.jsonl.gz","xt", encoding="utf8") as out, ProcessPoolExecutor(args.workers) as pool:
        for i, (gid, verdict) in enumerate(pool.map(probe_game, ((g,d["slots"]) for g,d in games.items()), chunksize=4)):
            g = games[gid]
            patch = bucket(min(g["days"])) if len(g["days"]) == len(g["builds"]) == 1 else "metadata_conflict"
            if gid in hold:
                split = "historic_eval"
            elif gid in historic_train:
                split = "historic_train"
            else:
                split = split_game(gid)
            role = "trainable" if split in ("train","historic_train") else "reserved"
            labels = ["all", patch, patch+"/"+split, patch+"/"+role, "role/"+role]
            for origin, yes in [("old","downloaded" in g["tags"]), ("latest",gid in latest),
                                ("increment",g["tags"]=={"new"})]:
                if yes:
                    labels.extend([origin,origin+"/"+patch,origin+"/"+patch+"/"+role])
            for label in labels:
                count(label,verdict)
            out.write(json.dumps([gid,sorted(g["days"]),sorted(g["builds"]),split,verdict])+"\n")
            if i%1000==0:
                print("checked",i,flush=True)
    from hsbg.az.train.mirror_snapshots import engine_stamp
    from hsbg.az.train.personal import fingerprint
    report = dict(engine=engine_stamp(), fingerprint=fingerprint(), feed_sha256=hashlib.sha256((args.dir/"feed.json").read_bytes()).hexdigest(),
                  joint=joint, joint_split_overlap=len(hold & historic_train), stats=stats)
    (args.out/"assessment.json").write_text(json.dumps(report,indent=2), encoding="utf8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["local","assess"])
    parser.add_argument("--dir",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    (local if args.mode=="local" else assess)(args)
