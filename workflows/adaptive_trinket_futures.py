"""Compare greater-trinket alternatives at the same exact decision under new paired futures."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows import adaptive_counterfactual as branch
from workflows.adaptive_decision_study import POLICY,RUNS,lobby,read,weights

BANK = RUNS/"nus-adaptive-openings-20261120-1"
PREVIOUS = RUNS/"nus-adaptive-branch-20261120-1"
OUT = RUNS/"nus-adaptive-trinket-futures-20261120-1"
RULE = "greater_runner"


class Boundary(Exception):
    pass


def capture(i):
    original = read(BANK/f"opening-{i}.json")
    def on_turn(game,seat,prefix,boards):
        if game.turn==9:
            atomic_json(OUT/f"t9-{i}.json",dict(seed=original["seed"],focus=seat,turn=9,
                state=game.state_json(),prefix=prefix.hex(),boards=dict(boards)))
            raise Boundary()
    try:
        lobby.play(branch.ACTORS,i,True,branch.START,opening=original,on_turn=on_turn)
    except Boundary:
        return
    raise ValueError("selected state failed to reach T9")


def trial(task):
    import hsbg_engine
    i,rep = task
    opening = read(OUT/f"t9-{i}.json")
    old = read(PREVIOUS/f"game-{i}.json")
    event = old["alternatives"][RULE]["trigger"]
    tag = f"hsai-adaptive-future-v1/{opening['seed']}/{rep}"
    donor_seed = int.from_bytes(sha256(tag.encode()).digest()[:8],"little")
    # Use a valid state of the same native engine RNG from an independent donor.
    rng = json.loads(hsbg_engine.Game(seed=donor_seed).state_json())["rng"]
    result = {}
    for label,apply in (("parent",False),("candidate",True)):
        def reset_rng(game,events):
            if events[RULE] != event:
                raise ValueError("original menu, neural scores, RNG or eight-seat prefix differ")
            raw = game.state_json()
            state = json.loads(raw)
            if rep>=0:
                if state["rng"]==rng:
                    raise ValueError("distinct future expected")
                state["rng"] = rng
            restored = hsbg_engine.Game.from_json(json.dumps(state,separators=(",",":")))
            if json.loads(restored.state_json()) != state:
                raise ValueError("engine rejected exact restoration")
            return restored
        game,applied,_ = branch.arm(i,[RULE],apply,opening,at_branch=reset_rng,capture_turns=False)
        if applied.get(RULE)!=event:
            raise ValueError("expected exactly one matched selection")
        if rep<0:
            expected = old["alternatives"][RULE]["result"] if apply else old["baseline"]
            if any(json.loads(json.dumps(game[k]))!=v for k,v in expected.items() if k!="overrides"):
                raise ValueError("native neutral RNG round-trip altered the archived game")
        result[label] = game
    atomic_json(OUT/f"pair-{i}-{rep}.json",dict(index=i,rep=rep,donor=donor_seed if rep>=0 else None,
        rng=rng if rep>=0 else None,event=event,**result))


def main():
    source = source_identity(ROOT)
    plan=read(BANK/"plan.json")
    seal=read(RUNS/"nus-adaptive-seal-20261120-1/seal.json")
    if source["tracked_dirty"] or plan["policy"]!=POLICY:
        raise ValueError("unexpected release or policy")
    cards=read(ROOT/"engine/crates/hsbg-engine/data/cards.json")["cards"]
    eligible=[]
    for i in range(plan["n"]):
        path=PREVIOUS/f"game-{i}.json"
        if file_identity(path)!=seal["shards"][str(i)]:
            raise ValueError("changed original paired evidence")
        e=read(path)["alternatives"][RULE]["trigger"]
        if e and all(cards[c]["kind"]=="trinket_greater" for c in e["cards"]):
            eligible.append(i)
    def key(i):
        return sha256(f"hsai-adaptive-state-1/{plan['seed']+i}".encode()).digest()
    selected=[i for seat in range(8) for i in sorted((i for i in eligible if i%8==seat),key=key)[:4]]
    if len(selected)!=32:
        raise ValueError("insufficient greater-trinket cases for four states per seat")
    branch.START=plan["seed"]
    branch.ACTORS=weights()
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],indices=selected,field=plan,
        protocol="adaptive-choice-future/v1",source_seal=file_identity(RUNS/"nus-adaptive-seal-20261120-1/seal.json"),
        trials=4,selection="Hash order within seat; no ranks used. Reps 0,1 for conditional action, 2,3 for measurement."))
    with ProcessPoolExecutor(32,mp_context=mp.get_context("fork")) as pool:
        list(pool.map(capture,selected))
        # Exercise identity restoration before issuing stochastic continuations.
        identity=selected[::4]
        list(pool.map(trial,[(i,-1) for i in identity]))
        list(pool.map(trial,[(i,r) for r in range(4) for i in selected]))
    files=[OUT/f"pair-{i}-{r}.json" for i in selected for r in range(4)]
    if len({read(p)["donor"] for p in files})!=len(files):
        raise ValueError("future RNG donors are not distinct across states and replications")
    atomic_json(OUT/"complete.json",dict(n=32,neutral_pairs=len(identity),replicated_pairs=len(files),
        interventions={p.name:file_identity(p) for p in files}))


if __name__=="__main__":
    main()
