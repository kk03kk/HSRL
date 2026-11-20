"""Store exact pre-recruitment states and the natural terminal outcome of all eight actors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows.adaptive_decision_study import OLD,POLICY,REGISTRY,lobby,read,weights


def collect(i):
    seed = START+i
    opening = None
    def capture(game,seat,prefix,boards):
        nonlocal opening
        if int(game.turn)==8:
            opening = dict(seed=seed,focus=seat,turn=8,state=game.state_json(),prefix=prefix.hex(),boards=dict(boards))
    natural = lobby.play(ACTORS,i,True,START,on_turn=capture)
    if opening is None:
        raise ValueError("game ended before the capture boundary")
    atomic_json(OUT/f"opening-{i}.json",opening)
    target = dict(natural=natural,opening=file_identity(OUT/f"opening-{i}.json"))
    if i < 8:
        restored = lobby.play(ACTORS,i,True,START,opening=opening)
        if any(restored[k]!=v for k,v in natural.items() if k!="overrides"):
            raise ValueError(f"neutral restoration diverged in seat {i}")
        target["neutral_replay"] = restored["trace"]
    atomic_json(OUT/f"base-{i}.json",target)
    return natural["decisions"]


def main(out,start,n,workers):
    global OUT,START,ACTORS
    OUT,START = out,start
    source = source_identity(ROOT)
    if source["tracked_dirty"] or n%8 or n<8 or not 0<workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        raise ValueError("bad release or unsupported allocation")
    ACTORS = weights()
    out.mkdir(exist_ok=False)
    plan = read(OLD/"plan.json")
    atomic_json(out/"plan.json",dict(source=source,n=n,seed=start,policy=POLICY,registry=file_identity(REGISTRY/"policy.json"),
        field=plan,job=os.environ["SLURM_JOB_ID"],engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),method="Natural complete games; rotate all 8 focal seats. Store full native T8 boundary, RNG and chronological three-byte all-seat action prefix. Recheck all eight first seat restorations."))
    with ProcessPoolExecutor(workers,mp_context=mp.get_context("fork")) as pool:
        steps = list(pool.map(collect,range(n)))
    atomic_json(out/"complete.json",dict(n=n,steps=sum(steps),neutral_verified=8,
        outcomes=[file_identity(out/f"base-{i}.json") for i in range(n)]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--games",type=int,required=True)
    p.add_argument("--workers",type=int,required=True)
    a=p.parse_args()
    main(a.out,a.seed,a.games,a.workers)
