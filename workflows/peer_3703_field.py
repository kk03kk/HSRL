"""Generate disjoint current-rule EOT libraries and a simultaneous four-on-four diagnostic."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity

PARENT = Path("artifacts/runs/nus-peer-heads-20261119-1/initial/model.pt")
R1 = Path("artifacts/runs/nus-peer-rounds-production-20261120-1/round-01/train-1170497/fit/train/snapshots/final_000647/model.pt")
PARENT_SHA = "55f6665c0b357860eefd6b6916620a0118e1dc2c5d5e77bbea9136f64f48c5a9"
R1_SHA = "1bcb12c6038f0d07223d925893c58a4be244531769f32e9820eeb2e9f10e134a"
TRAINER_SHA = "d2e1b926d8cae3afd17f8b37431285fb760f662f8408b76f6e19ab595bdab949"
LIB = Path("artifacts/libraries/peer-3703-mixed-20261120-02")
ENGINE = "c1d91053387aeac3"
START = 511730000
# Sensitivity test using the historical round duration divided by 1.6s,
# rounded down. The final two entries extend the last recorded turn.
SECONDS = [45,30,45,60,70,95,90,100,125,115,115,115]
TRAIN_GAMES, HOLD_GAMES = 128, 32
ACTORS = None
PLAYER150 = os.environ.get("HSAI_ACTION_PROFILE") == "human-turn-3703-player150"
if PLAYER150:
    ENGINE = "8bd1f4785018c474"
    LIB = Path("artifacts/libraries/player150-mixed-20261120-01")
    START = 541730000

def initialize():
    global ACTORS
    import torch
    import hsbg_engine as engine
    from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
    torch.set_num_threads(1)
    if not engine.build_is_current(ROOT/"engine") or engine.build_info()["source_hash"] != ENGINE:
        raise ValueError("wrong native engine")
    if engine.server_rules_version() != "37.0.3" or os.environ["HSAI_ACTION_PROFILE"] not in ("human-turn-3702-economy", "human-turn-3703-player150"):
        raise ValueError("wrong server or action profile")
    proto = EvalProtocol()
    proto.assert_training_contract()
    assert (proto.max_actions_per_turn, proto.move_cap, proto.freeze_cap) == ((150, 3, 2) if PLAYER150 else (300, 10, 3))
    ACTORS = []
    for path, sha in ((PARENT, PARENT_SHA),(R1,R1_SHA)):
        if file_identity(path)["sha256"] != sha:
            raise ValueError("checkpoint identity")
        ACTORS.append(make_eval_policy(str(path),protocol=proto,device="cpu")["policy"])

def simulate(index):
    import hsbg_engine as engine
    from hsbg.az.train.engine_quality import require_clean_engine
    from hsbg.az.train.mirror_snapshots import capture_eot, engine_stamp, held_pool_parts, write_snapshot_file
    timed = index >= TRAIN_GAMES + HOLD_GAMES
    i = index - HOLD_GAMES if timed else index
    seed = START+i
    split = "stress" if timed else ("train" if i < TRAIN_GAMES else "hold")
    dest = LIB / split
    snap, receipt = dest/f"snap_{seed}.json.gz", dest/f"game-{seed}.json"
    if receipt.exists() or snap.exists():
        raise FileExistsError(seed)
    from hsbg.az.train.lobby_filter import make_filtered_game
    game, actual_seed = make_filtered_game(engine, seed)
    cfg = json.loads(game.state_json())["cfg"]
    assert actual_seed == seed and cfg["freeze_presses_per_turn"] == (2 if PLAYER150 else 3)
    assert bool(cfg.get("player_view_150")) == PLAYER150
    roles = [(seat+i)%2 for seat in range(8)]
    agents = []
    for role in roles:
        a = ACTORS[role]
        b = deepcopy(a, {id(a.evaluator):a.evaluator})
        b.reset()
        agents.append(b)
    frames, bodies = [], [{} for _ in agents]
    forced = 0
    actions = 0
    lobby = json.loads(game.view(0))["lobby_races"]
    while not game.is_over():
        turn = int(game.turn)
        order = list(range(8))
        random.Random(seed*1000+turn).shuffle(order)
        counts = Counter()
        while not game.is_over() and game.turn == turn:
            before = actions
            for seat in order:
                if game.is_over() or game.turn != turn:
                    break
                if not game.needs_action(seat):
                    continue
                mask = game.legal_mask(seat)
                action = int(agents[seat].decide(game,seat).action)
                budget = int(SECONDS[min(max(0,turn-1),len(SECONDS)-1)]/1.6)
                if timed and turn > 0 and counts[seat] >= budget-1 and mask[0] and action != 0:
                    forced += 1
                    action = 0
                if not mask[action] or actions > 60000:
                    raise ValueError((seed, turn, seat, action))
                if turn and action == 0:
                    f = capture_eot(game,seat)
                    if f is None:
                        raise ValueError(("missing EOT",seed,turn,seat))
                    bodies[seat][turn] = sum(m["atk"]+m["hp"] for m in f["player"]["board"])
                    if not timed:
                        f.update(source="native_3703_peer",source_turn=turn)
                        # An external combat board does not purchase from the recipient's shop.
                        # Keep the native board and its enchantments. Omit private zones on every frame.
                        player = f["player"]
                        for zone in ("hand","shop","hand_queue","hand_queue_marks"):
                            player[zone] = []
                        player["gold"] = 0
                        payload = json.dumps(player, ensure_ascii=False)
                        f["held_pool_parts"] = held_pool_parts(player)
                        f["frame_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
                        probe = game.copy()
                        rep = probe.apply_turn_snapshots([[seat,payload]],False)
                        if rep["applied"] != 1 or rep["rejected"] or not rep["pool_balance_ok"] or not rep["pool_species_ok"]:
                            raise ValueError(("source board is not injectable",seed,seat,turn,rep))
                        copied = json.loads(probe.player_json(seat))
                        fields = ("card","atk","hp","kw","bonus_kw","golden","shield_max")
                        if [[m[k] for k in fields] for m in copied["board"]] != [[m[k] for k in fields] for m in player["board"]]:
                            raise ValueError(("board did not round-trip",seed,seat,turn))
                        frames.append(f)
                game.step(seat,action)
                counts[seat] += 1
                actions += 1
            if actions == before and game.turn == turn and not game.is_over():
                raise ValueError(("stalled",seed,turn))
    diag = require_clean_engine(game)
    ranks = list(game.placements())
    if sorted(ranks) != list(range(1,9)) or len(set(lobby)) != 5:
        raise ValueError(("incomplete game",seed))
    if not timed:
        write_snapshot_file(str(snap),frames,seed=seed,engine=engine_stamp(),meta={"roles":roles})
    atomic_json(receipt, dict(seed=seed,roles=roles,ranks=ranks,boards=bodies,forced=forced,
        decisions=actions,lobby=lobby,quality=diag,turns=dict(Counter(f["turn"] for f in frames)),
        file=file_identity(snap) if not timed else None))
    return seed

def seal():
    from hsbg.az.train.personal import fingerprint, library_digest
    from hsbg.az.train.eval_policy import protocol_sha256_of
    fp = fingerprint()
    assert fp["engine"]["source_hash"] == ENGINE
    summaries = {}
    for split, lo, hi in (("train",0,TRAIN_GAMES),("hold",TRAIN_GAMES,TRAIN_GAMES+HOLD_GAMES),
                          ("stress",TRAIN_GAMES,TRAIN_GAMES+HOLD_GAMES)):
        if split == "stress" and PLAYER150:
            continue
        dest = LIB/split
        rows = [json.loads((dest/f"game-{START+i}.json").read_bytes()) for i in range(lo,hi)]
        assert all(r["seed"] == START+i for i,r in enumerate(rows,lo))
        field = dict(engine=ENGINE,source_models=[PARENT_SHA,R1_SHA],games=[r["seed"] for r in rows],
                     desired_seconds=SECONDS if split=="stress" else [],action_profile=os.environ["HSAI_ACTION_PROFILE"])
        ranks = [mean(r["ranks"][s] for s in range(8) if r["roles"][s])-
                 mean(r["ranks"][s] for s in range(8) if not r["roles"][s]) for r in rows]
        def growth(role):
            x=[math.log1p(b["12"])-math.log1p(b["9"]) for r in rows for s,b in enumerate(r["boards"])
               if r["roles"][s]==role and "9" in b and "12" in b]
            return dict(mean=mean(x) if x else None,n=len(x))
        from tools.summarize_peer_finetune import estimate
        summaries[split] = dict(rank=estimate(ranks),growth_parent=growth(0),growth_r1=growth(1),
                                forced=sum(r["forced"] for r in rows),field_sha=protocol_sha256_of(field))
        if split != "stress":
            counts=Counter()
            for r in rows:
                assert file_identity(dest/f"snap_{r['seed']}.json.gz") == r["file"]
                counts.update({int(k):v for k,v in r["turns"].items()})
            if sorted(counts)!=list(range(1,max(counts)+1)) or max(counts)<13:
                raise ValueError("incomplete library coverage")
            atomic_json(dest/"manifest.json", dict(split=split, fingerprint=fp,digest=library_digest(dest),
                games=field["games"],turns=dict(counts),frames=sum(counts.values()),field=field))
    atomic_json(LIB/"complete.json",summaries)
    print(json.dumps(summaries),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("generate","seal"))
    p.add_argument("--shard",type=int,default=0)
    args=p.parse_args()
    os.chdir(ROOT)
    if args.stage == "seal":
        seal()
    else:
        workers=int(os.environ["SLURM_CPUS_PER_TASK"])
        assert workers==8 and 0<=args.shard<(20 if PLAYER150 else 24)
        with ProcessPoolExecutor(workers,initializer=initialize) as pool:
            print(list(pool.map(simulate,range(args.shard*8,args.shard*8+8))),flush=True)
