"""Read-only checkpoint inference from protected T9 openings through T18."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.probe_four_growth_curves import ABSORB,debug,make,own,step,card_ids,item_card
from workflows.quilboar_study import PLAN,BASE,ROOT,config,policy,engine_info
from workflows.quilboar_gem_returns import ReturnRecorder

OUT=Path("artifacts/runs/nus-frozen-late-growth-20261119-1")
MODELS={
    "base":(BASE,PLAN["base_sha"]),
    "imitation":("artifacts/models/human-turn-3702-quilboar-choice-20261119-1/model.pt",
                 "d92f7e709fd8f6897bec1521e500c54c9312b70c00cce465695efd1a8c8bc7bc"),
    "investment":("artifacts/models/human-turn-3702-quilboar-investment-20261119-1/model.pt",
                  "bd85b7ae526094e1194dca991a8e49869584d90df1a6e192b7015934e1f9ead7")}
CASES=("elem_budget","elem_offer","quill_ready","quill_offer","mech_double","dragon_poet")
if os.environ.get("HSAI_QUILL_COLD_PROBE")=="1":
    OUT=Path("artifacts/runs/nus-frozen-late-growth-cold-20261119-1")
    CASES=("quill_cold_ready","quill_cold_search","quill_lone_source")
NORMAL_HEALTH=os.environ.get("HSAI_GROWTH_NORMAL_HEALTH")=="1"
if NORMAL_HEALTH:
    OUT=Path("artifacts/runs/nus-frozen-late-growth-30hp-20261119-1")
    CASES=("elem_budget","elem_offer","quill_ready","quill_offer","quill_cold_ready",
           "quill_cold_search","quill_lone_source","mech_double","dragon_poet")
SEEDS=list(range(520100,520124))


def absorption(p,a):
    if not 30<=a<46 or not p["pending"]:
        return None
    target=p["pending"][-1].get("Target",{})
    hand=target.get("cont",{}).get("HandSpell")
    if hand is None:
        return None
    spell=next((h for h in p["hand"] if h["hid"]==hand["hid"]),None)
    if spell is None or item_card(spell)!=card_ids()[ABSORB]:
        return None
    eid=target["options"][a-30]
    m=next(m for m in p["board"] if m["eid"]==eid)
    return eid,m["atk"]+m["hp"],p["stats"]["tavern_elemental_atk"]


def play(seed,case,agent,*,opening=None,normal_health=None):
    g=make(seed,case) if opening is None else opening
    normal_health=NORMAL_HEALTH if normal_health is None else normal_health
    agent.reset()
    rec=ReturnRecorder()
    rec.begin_game(g,0,seed,0)
    turns=[]
    for t in range(9,19):
        assert g.turn==t and not g.is_over()
        if normal_health:
            for seat in range(8):
                debug(g,"set_player",seat,health=30,armor=0)
        events=[]
        actions=0
        gem_start=len(rec.measured)
        while not own(g)["ended_turn"]:
            p=own(g)
            decision=agent.decide(g,0)
            a=int(decision.action)
            before=rec.before(g,0)
            target=absorption(p,a)
            step(g,a)
            rec.after(g,before,decision,a,False)
            if target is not None:
                eid,old,shop=target
                m=next(m for m in own(g)["board"] if m["eid"]==eid)
                events.append(dict(gain=m["atk"]+m["hp"]-old,shop=shop))
            actions+=1
            if actions>125:
                raise RuntimeError("action budget guard did not terminate")
        p=own(g)
        s=p["stats"]
        turns.append(dict(turn=t,strength=sum(m["hp"]+m["atk"] for m in p["board"]),
                     board=[m["card"] for m in p["board"]],shop=s["tavern_elemental_atk"],
                     gem=2+s["gem_atk"]+s["gem_hp"],absorptions=events,
                     choices=rec.measured[gem_start:],actions=actions))
        if normal_health:
            # The learner sees ordinary health and may spend it during shopping.
            # Refill after its last decision to protect only the combat horizon.
            debug(g,"set_player",0,health=30,armor=0)
        for seat in range(1,8):
            step(g,0,seat)
        if g.truncated_combats:
            raise ValueError("truncated combat")
    return turns


def shard(i):
    torch.set_num_threads(1)
    proto,_=config("hold")
    models={k:policy(path,proto) for k,(path,_) in MODELS.items()}
    rows=[]
    for s in SEEDS[i::8]:
        for case in CASES:
            for k,p in models.items():
                if k!="base" and not case.startswith("quill"):
                    continue
                rows.append(dict(seed=s,case=case,model=k,turns=play(s,case,p)))
    atomic_json(OUT/f"{i}.json",rows)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    assert all(file_identity(path)["sha256"]==sha for path,sha in MODELS.values())
    engine=engine_info()
    src=source_identity(ROOT)
    assert engine["source_hash"]==PLAN["engine"] and not src["tracked_dirty"]
    atomic_json(OUT/"plan.json",dict(seeds=SEEDS,models=MODELS,cases=CASES,engine=engine,
        source=src,job=os.environ["SLURM_JOB_ID"],end=18,actions="human-turn-3702",
        protected_health=30 if NORMAL_HEALTH else 1000,
        diagnostic="T13-T18: native payout on 3 distinct turns including T16 or later; report T18 body and measured payouts separately. Natural gold and shops after initial T9 fixture."))
    def run_shard(i):
        with (OUT/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run_shard,range(8)))
    rows=sum([json.loads((OUT/f"{i}.json").read_text()) for i in range(8)],[])
    keys={(r["seed"],r["case"],r["model"]) for r in rows}
    assert keys=={(s,c,k) for s in SEEDS for c in CASES for k in MODELS if k=="base" or c.startswith("quill")}
    assert len(keys)==len(rows) and all([v["turn"] for v in r["turns"]]==list(range(9,19)) for r in rows)
    atomic_json(OUT/"complete.json",dict(runs=len(rows),shards=[file_identity(OUT/f"{i}.json") for i in range(8)]))


if __name__=="__main__":
    run() if len(sys.argv)==1 else shard(int(sys.argv[1]))
