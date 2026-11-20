"""Paired action-budget comparison on new complete games and archived high-economy seeds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.az.train.strict_mirror import BOUNDED_ACTION_PROFILES
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol, read
from tools.summarize_peer_finetune import estimate

SEED = 398410000
GAMES = 256
SPECIAL = Path("artifacts/runs/nus-peer-cap-audit-20261119-1/rounds.json")


class Trace:
    def begin_game(self,game,index,seed,seat):
        self.turns = {}
        self.hash = sha256()

    def before(self,game,seat):
        return int(game.turn)

    def after(self,game,turn,decision,action,forced):
        counts = self.turns.setdefault(turn,Counter())
        counts["n"] += 1
        counts["moves"] += int(action>=196)
        counts["freezes"] += int(action==2)
        counts["cap"] += int(decision.info.get("forced_reason")=="action_cap")
        self.hash.update(bytes((turn,action)))

    def end_game(self,game,rank,quality):
        pass


def trial(seed):
    from hsbg.az.train.evaluate import evaluate_policy, EvalConfig
    trace = Trace()
    r = evaluate_policy(ACTOR,EvalConfig(games=1,seed=seed,mirror_pool=POOL,mirror_strict=True,
                                         collect_economy=True,decision_recorder=trace))
    mirror = r["mirror"]
    if (r["games_finished"]!=1 or r["seed_sequence"]!=[seed]
            or any(r[k] for k in ("illegal_actions","truncated_games","truncated_combats"))
            or any(mirror["mirror_"+k] for k in ("errors","missing","fallback","rejected","pool_unbalanced","pool_species_unbalanced"))
            or not mirror["mirror_applied"] or mirror["mirror_applied"]!=mirror["mirror_attempt"]):
        raise ValueError(f"incomplete or invalid game {seed}")
    boards={}
    for t in r["end_turn_economy"]:
        if not t["eot_available"]:
            raise ValueError("missing precombat board")
        boards[t["turn"]]=t["eot_attack"]+t["eot_board_hp"]
    if any(t["moves"]>PROTO.move_cap or t["freezes"]>PROTO.freeze_cap for t in trace.turns.values()):
        raise ValueError("actual action exceeded selected profile")
    return dict(seed=seed,rank=r["rank_sequence"][0],steps=r["decisions"],trace=trace.hash.hexdigest(),
                turns=trace.turns,boards=boards,tail=r["mirror_strict_counts"].get("human_tail_rank_upper",0))


def evaluate(out,profile):
    global POOL, ACTOR, PROTO
    import torch
    from hsbg.az.train.eval_policy import make_eval_policy
    torch.set_num_threads(1)
    PROTO,POOL = holdout_protocol()
    if file_identity(INITIAL)["sha256"]!=INITIAL_SHA:
        raise ValueError("unverified starting policy")
    ACTOR=make_eval_policy(str(INITIAL),protocol=PROTO,device="cpu")["policy"]
    seeds=read(out/"plan.json")["seeds"]
    with ProcessPoolExecutor(int(os.environ["SLURM_CPUS_PER_TASK"]),mp_context=mp.get_context("fork")) as pool:
        rows=list(pool.map(trial,seeds))
    atomic_json(out/(profile+".json"),dict(protocol=PROTO.as_dict(),sha=PROTO.sha256(),games=rows))


def paired(a,b):
    if [x["seed"] for x in a]!=[x["seed"] for x in b]:
        raise ValueError("unmatched seeds")
    common=[(x,y) for x,y in zip(a,b) if all("12" in g["boards"] for g in (x,y))]
    def growth(g):
        return math.log1p(g["boards"]["12"])-math.log1p(g["boards"]["9"])
    return dict(rank=estimate([x["rank"]-y["rank"] for x,y in zip(a,b)]),
                growth=estimate([growth(y)-growth(x) for x,y in common]),
                common=len(common),steps_ratio=sum(y["steps"] for y in b)/sum(x["steps"] for x in a),
                reach_delta=sum("12" in y["boards"] for y in b)-sum("12" in x["boards"] for x in a),
                trace_changed=sum(x["trace"]!=y["trace"] for x,y in zip(a,b)))


def describe(rows):
    turns=[v for g in rows for v in g["turns"].values()]
    boards=[g["boards"]["12"] for g in rows if "12" in g["boards"]]
    return dict(n=len(rows),rank=mean(g["rank"] for g in rows),wins=sum(g["rank"]==1 for g in rows),
                top4=sum(g["rank"]<=4 for g in rows),t12=dict(n=len(boards),p50=median(boards) if boards else None),
                steps=sum(g["steps"] for g in rows),moves=sum(t["moves"] for t in turns),
                capped=sum(t["cap"] for t in turns),over200=sum(t["n"]>201 for t in turns))


def main(out):
    source=source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("release must be frozen")
    selected=sorted({r["seed"] for r in read(SPECIAL)})
    seeds=list(range(SEED,SEED+GAMES))+selected
    if len(selected)!=26 or len(set(seeds))!=len(seeds):
        raise ValueError("unexpected coverage")
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],historical=file_identity(SPECIAL),
        initial=file_identity(INITIAL),engine=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),seeds=seeds,profiles=BOUNDED_ACTION_PROFILES,
        cohort_sizes=[GAMES,len(selected)]))
    results={}
    for profile in BOUNDED_ACTION_PROFILES:
        subprocess.run([sys.executable,__file__,str(out),"--profile",profile],check=True,
                       env=os.environ|{"HSAI_ACTION_PROFILE":profile,"HSAI_HUMAN_TAIL":"last-observed-v1"})
        results[profile]=read(out/(profile+".json"))["games"]
    base,extended,compact=BOUNDED_ACTION_PROFILES
    fresh={k:v[:GAMES] for k,v in results.items()}
    rare={k:v[GAMES:] for k,v in results.items()}
    econ=paired(fresh[base],fresh[extended])
    movement=paired(fresh[extended],fresh[compact])
    supported=(econ["rank"]["ci95"][0]>=-.10 and econ["growth"]["ci95"][0]>=-.10 and econ["steps_ratio"]<=1.03
               and sum(any(t["n"]>202 for t in g["turns"].values()) for g in rare[extended])>=6)
    cheaper=(movement["steps_ratio"]<=.97 and movement["rank"]["ci95"][0]>=-.10 and
             movement["growth"]["ci95"][0]>=-.08 and movement["reach_delta"]>=-5)
    atomic_json(out/"complete.json",dict(job=os.environ["SLURM_JOB_ID"],files={k:file_identity(out/(k+".json")) for k in results},
        fresh={k:describe(g) for k,g in fresh.items()},targeted={k:describe(g) for k,g in rare.items()},
        economy=econ,movement=movement,high_economy=paired(rare[base],rare[extended]),
        selected=compact if supported and cheaper else extended if supported else base))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--profile",choices=BOUNDED_ACTION_PROFILES)
    args=parser.parse_args()
    evaluate(args.out,args.profile) if args.profile else main(args.out)
