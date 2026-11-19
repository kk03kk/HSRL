"""Measure the independent combat gain available from one positional move."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool, capture_eot
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json
from hsbg.runtime.provenance import source_identity
from tools.inventory_firestone_boards import template
from tools.summarize_peer_finetune import estimate

TRACE = Path("artifacts/runs/nus-peer-standard-20261119-1")
LIB = Path("artifacts/libraries/human-turn-3702-20261118/train")
SALT = 45161119


def board_orders(board):
    size = len(board)
    base = tuple(range(size))
    # One native remove/insert action, promoting one unit to the front or back.
    orders = [base]
    for i in range(size):
        for end in (0, size-1):
            alt = list(base)
            alt.insert(end, alt.pop(i))
            if tuple(alt) not in orders:
                orders.append(tuple(alt))
    return orders


def score(own, order, opponent, turn, seed, trials):
    import hsbg_engine
    state = json.loads(template(turn))
    state["seed"] = seed
    game = hsbg_engine.Game.from_json(json.dumps(state))
    player = dict(own, board=[own["board"][i] for i in order])
    rep = game.apply_turn_snapshots([[0, json.dumps(player)], [1, opponent.player_json]], False)
    if rep["applied"] != 2 or rep["rejected"] or not rep["pool_balance_ok"] or not rep["pool_species_ok"]:
        raise ValueError("invalid hypothetical combat input")
    result = json.loads(game.debug(json.dumps(dict(cmd="combat_preview", a=0, b=1, n=trials, details=True))))
    rows = result["trials"]
    if len(rows) != trials or any(row["truncated"] for row in rows):
        raise ValueError("incomplete combat")
    return sum([1, 0, .5][r["winner"]] for r in rows)/trials


def measure(pool, own, turn, seed):
    rng = random.Random(SALT + seed)
    panels, used = [], set()
    for count in (8, 24):
        opponents = []
        while len(opponents) < count:
            frame = pool.get(turn, 0, rng=rng)
            if frame.source_game_seed in used:
                continue
            used.add(frame.source_game_seed)
            opponents.append((frame, rng.randrange(1, 2**32)))
        panels.append(opponents)
    orders = board_orders(own["board"])
    def mean(order, opponents):
        return sum(score(own, order, frame, turn, salt, 4) for frame,salt in opponents)/len(opponents)
    train = [mean(order, panels[0]) for order in orders]
    best = max(range(len(orders)), key=train.__getitem__)
    validation = [mean(order, panels[1]) for order in [orders[0], orders[best]]] if best else [None, None]
    return dict(seed=seed, turn=turn, size=len(own["board"]), orders=len(orders), pick=best,
                development=train[best]-train[0], test=0 if not best else validation[1]-validation[0])


def shard(index):
    import hsbg_engine
    pool = MirrorSnapshotPool(str(LIB), sample="random")
    records = json.loads((TRACE/f"shard-{index:02}.json").read_bytes())["rows"][:16]
    goals = {r["seed"]:9+j%4 for j,r in enumerate(records)}
    rows, ended = [], set()
    with gzip.open(TRACE/f"shard-{index:02}.decisions.jsonl.gz", "rt", encoding="utf8") as stream:
        for line in stream:
            if line.startswith('{"event":"game_end"'):
                ended.add(json.loads(line)["seed"])
                if ended == goals.keys():
                    break
            if not line.startswith('{"event":"action"') or '"action":0,' not in line:
                continue
            r = json.loads(line)
            if r["action"] or r["state_before"]["turn"] != goals[r["seed"]]:
                continue
            game = hsbg_engine.Game.from_json(json.dumps(r["state_before"]))
            frame = capture_eot(game, r["seat"])
            if frame is None:
                raise ValueError("missing EOT board")
            rows.append(measure(pool, frame["player"], game.turn, r["seed"]))
    if ended != goals.keys() or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("trace coverage failed")
    return rows


def run(out):
    source = source_identity(ROOT)
    digest = library_digest(LIB)
    if source["tracked_dirty"] or digest != "3b18d4d5eceaf458990c7031098df76e9dea042c083dc1d95d306d379b721e81":
        raise ValueError("wrong release or board pool")
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json", dict(source=source, library=digest, trace=str(TRACE), salt=SALT,
        shards=[0,4,8,12], per_shard=16, turns="9 + within-shard index modulo 4",
        combat="win + half tie, eight source games for choice, 24 other games for scoring, four simulations per board",
        gate="lower paired 95% CI > 0 across source boards; combat headroom only"))
    with ProcessPoolExecutor(4) as executor:
        rows = sum(executor.map(shard, [0,4,8,12]), [])
    atomic_json(out/"boards.json", rows)
    atomic_json(out/"complete.json", dict(eligible=len(rows), assigned=64, changed=sum(bool(r["pick"]) for r in rows),
        development=estimate([r["development"] for r in rows]), test=estimate([r["test"] for r in rows])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
