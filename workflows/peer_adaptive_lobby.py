"""Feasibility of an eight-policy native lobby with a shared tavern pool."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.lobby_filter import make_filtered_game
from hsbg.az.train.mirror_snapshots import capture_eot
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol, read
from workflows.peer_sale_study import RetainMaterial
from workflows import quilboar_study as native


def play(weights, index, change, start=343410000, *, opening=None, on_turn=None, on_focus=None, after_focus=None):
    import hsbg_engine
    seed = start + index
    focus = index%8
    if opening is None:
        game, actual = make_filtered_game(hsbg_engine, seed, learner_seat=focus)
        assert actual == seed
    else:
        if opening["seed"] != seed or opening["focus"] != focus:
            raise ValueError("adaptive opening belongs to another seed or focal seat")
        game = hsbg_engine.Game.from_json(opening["state"])
        if game.state_json() != opening["state"] or game.turn != opening["turn"]:
            raise ValueError("native round-boundary state did not round-trip")
    agents = []
    for seat in range(8):
        # Opponent roles are identical across arms and rotate with the focal seat.
        if seat == focus and "intent" in weights:
            agent = weights["intent" if change else "material"]
        else:
            p = weights["native" if (seat-focus)%8 in (2,4,6) else "peer"]
            agent = deepcopy(p,{id(p.evaluator):p.evaluator})
            if seat==focus and change:
                agent = RetainMaterial(agent)
        agent.reset()
        agents.append(agent)
    old_turn = -1
    prefix = bytearray.fromhex(opening["prefix"]) if opening is not None else bytearray()
    actions = len(prefix)//3
    trace = sha256(prefix)
    boards = {int(k):v for k,v in opening["boards"].items()} if opening is not None else {}
    while not game.is_over():
        turn = int(game.turn)
        if turn!=old_turn:
            if on_turn is not None:
                on_turn(game,focus,prefix,boards)
            seats = list(range(8))
            random.Random(seed*1000+turn).shuffle(seats)
            old_turn = turn
        progress = False
        for seat in seats:
            if game.is_over() or int(game.turn)!=turn:
                break
            if not game.needs_action(seat):
                continue
            action = int(agents[seat].decide(game,seat).action)
            if seat==focus and on_focus is not None:
                replacement = on_focus(game, action, trace)
                if replacement is not None:
                    game = replacement
            if not game.legal_mask(seat)[action] or actions >= 50000:
                raise ValueError("invalid action or unbounded lobby")
            if turn>0 and seat==focus and action==0:
                frame = capture_eot(game,seat)
                if frame is not None:
                    boards[turn] = sum(u["atk"]+u["hp"] for u in frame["player"]["board"])
            step = bytes([turn,seat,action])
            trace.update(step)
            if on_turn is not None:
                prefix.extend(step)
            game.step(seat,action)
            if seat==focus and after_focus is not None:
                after_focus(game,action)
            actions += 1
            progress = True
        if not progress and not game.is_over():
            raise ValueError("stalled native lobby")
    quality = require_clean_engine(game)
    ranks = list(game.placements())
    if sorted(ranks)!=list(range(1,9)):
        raise ValueError("invalid final placements")
    return dict(seed=seed, focus=focus, rank=ranks[focus], placements=ranks, decisions=actions,
        turn=game.turn, boards=boards, quality=quality, trace=trace.hexdigest(),
        overrides=agents[focus].overrides if change else [])


def shard(out, index, pairs=8, workers=8, start=343410000, bundle=None):
    import torch
    torch.set_num_threads(1)
    proto,_ = holdout_protocol()
    weights = {k:make_eval_policy(str(p),protocol=proto,device="cpu")["policy"]
               for k,p in (("peer",INITIAL),("native",native.BASE))}
    if bundle is not None:
        weights["intent"] = make_eval_policy("research-policy:"+str(bundle),protocol=proto,device="cpu")["policy"]
        parent = bundle/read(bundle/"policy.json")["parent"]["path"]
        weights["material"] = make_eval_policy(str(parent),protocol=proto,device="cpu")["policy"]
    for j in range(index, pairs, workers):
        raw,kept = play(weights,j,False,start),play(weights,j,True,start)
        if not kept["overrides"] and raw!=kept:
            raise ValueError("unaffected lobby not replayed exactly")
        atomic_json(out/f"pair-{j}.json",dict(parent=raw,candidate=kept))


def run(out, pairs=8, workers=8, start=343410000, bundle=None):
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    if source_identity(ROOT)["tracked_dirty"] or file_identity(INITIAL)["sha256"]!=INITIAL_SHA or file_identity(
            native.BASE)["sha256"]!=native.PLAN["base_sha"]:
        raise ValueError("release or starting policies differ")
    out.mkdir(exist_ok=False)
    if pairs <= 0 or pairs % 8 or not 1 <= workers <= min(pairs,int(os.environ["SLURM_CPUS_PER_TASK"])):
        raise ValueError("require complete seat rotations and allocated CPU workers")
    spec=dict(source=source_identity(ROOT),seed=start,pairs=pairs,opponents="4 public actors, 3 native 3265",
        focus="equal seat rotation",order="seed*1000+turn shuffle, one action per seat per sweep",
        parent=INITIAL_SHA,native=native.PLAN["base_sha"],rules="human-turn-3702-standard",
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),live_pin=file_identity(ROOT/"config/live/model.json"),
        job=os.environ["SLURM_JOB_ID"],scope="adaptive simulated field; estimate incremental utility and common T12 strength")
    if bundle is not None:
        spec.update(peer_opponent=INITIAL_SHA,parent=read(bundle/"policy.json")["parent"]["identity"]["sha256"],
                    candidate=file_identity(bundle/"policy.json")["sha256"])
    spec["protocol_sha256"]=sha256(json.dumps({k:v for k,v in spec.items() if k not in ("job","source")},
        sort_keys=True).encode()).hexdigest()
    atomic_json(out/"plan.json",spec)
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(out),"--worker",str(i),"--pairs",str(pairs),
                            "--workers",str(workers),"--seed",str(start)]+(["--bundle",str(bundle)] if bundle else []),check=True,
                           stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(launch,range(workers)))
    from math import log1p
    from tools.summarize_peer_finetune import UTILITY, estimate, p50
    rows=[read(out/f"pair-{i}.json") for i in range(pairs)]
    for i,row in enumerate(rows):
        if any(g["seed"] != start+i or g["focus"] != i%8 for g in row.values()):
            raise ValueError("seed or seat coverage differs")
    boards = {arm:[r[arm]["boards"]["12"] for r in rows if "12" in r[arm]["boards"]] for arm in rows[0]}
    common = [(r["parent"],r["candidate"]) for r in rows if all("12" in g["boards"] for g in r.values())]
    atomic_json(out/"complete.json",dict(pairs=pairs, affected=sum(bool(r["candidate"]["overrides"]) for r in rows),
        utility=estimate([UTILITY[r["candidate"]["rank"]-1]-UTILITY[r["parent"]["rank"]-1] for r in rows]),
        t12={arm:dict(n=len(vals),p50=p50(vals)) for arm,vals in boards.items()},
        log12=estimate([log1p(b["boards"]["12"])-log1p(a["boards"]["12"]) for a,b in common]),
        ranks=[[r[k]["rank"] for k in ("parent","candidate")] for r in rows],
        actions=sum(g["decisions"] for r in rows for g in r.values())))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--pairs",type=int,default=8)
    parser.add_argument("--seed",type=int,default=343410000)
    parser.add_argument("--bundle",type=Path)
    args=parser.parse_args()
    if args.worker is None:
        run(args.out,args.pairs,args.workers,args.seed,args.bundle)
    else:
        shard(args.out,args.worker,args.pairs,args.workers,args.seed,args.bundle)
