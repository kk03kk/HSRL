"""Launch one sealed PPO fork and assess the initial, intermediate and final policies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows import joint_short_ppo as J
from workflows import azerite_course as C
from workflows.human_turn_preflight import verified_generation
from workflows.quilboar_gem_returns import ReturnRecorder
from tools.probe_four_growth_curves import DRONE,ROBOT,HOPE,POET,RIDER,SURF,AZ
from tools.assess_quilboar_study import interval
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity

SEG = J.SERIES/"segment-01"
EVAL = Path("artifacts/runs/nus-joint-growth-assess-20261119-1")
MID = Path("artifacts/runs/nus-joint-growth-monitor-20261119-1")
NAMES=("base3265","joint","ppo")


def receipt():
    sealed=J.read(J.OUT/"complete.json")
    recipe=J.read(J.OUT/"ppo-recipe.json")
    model=J.MODEL/"model.pt"
    if (file_identity(J.OUT/"ppo-recipe.json")!=sealed["recipe"] or file_identity(model)!=sealed["model"]
        or str(model)!=recipe["environment"]["HSAI_ARENA_CKPT"] or recipe["checkpoint_sha256"]!=sealed["model"]["sha256"]
        or recipe["seed"]!=J.PPO_SEED or recipe["segments"]!=1 or recipe["series"]!=str(J.SERIES)
        or sealed["model"]["sha256"] in (C.quill.PLAN["base_sha"],J.FIRST_SHA)):
        raise ValueError("training receipt does not identify the new joint student")
    return recipe


def train():
    recipe=receipt()
    J.SERIES.mkdir(parents=True,exist_ok=False)
    panel=J.read(ROOT/recipe["evaluation"])
    panel.update(seed=171600000,games=512,models=[dict(name=n,checkpoint=str(p)) for n,p in
                 zip(NAMES,(C.quill.BASE,J.MODEL/"model.pt",J.SERIES/"latest-complete/model.pt"))])
    atomic_json(J.SERIES/"comparison-plan.json",panel)
    recipe.update(evaluation=str(J.SERIES/"comparison-plan.json"),endpoint_evaluation=str(J.SERIES/"comparison-plan.json"))
    atomic_json(J.SERIES/"recipe.json",recipe)
    print(json.dumps(dict(confirmed_initial_sha256=recipe["checkpoint_sha256"],
                     start=recipe["environment"]["HSAI_ARENA_CKPT"],one_ppo_job=os.environ["SLURM_JOB_ID"])),flush=True)
    os.execv(sys.executable,[sys.executable,"workflows/human_turn_preflight.py","--segment","1",
                            "--train","--recipe",str(J.SERIES/"recipe.json")])


class Mechanisms(ReturnRecorder):
    def begin_game(self,game,index,seed,seat):
        super().begin_game(game,index,seed,seat)
        self.absorptions=[]
        self.unmeasured_absorptions=[]
        self.mechanics=[]
        self.offers=set()
        self.buys=set()
        self.hope_growth=set()
        self.previous_hope={}

    def before(self,game,seat):
        b=super().before(game,seat)
        if game.turn<6:
            return b,None
        p=C.own(game,seat)
        ids=C.card_ids()
        if self.lobby&4:
            for m in p["board"]:
                if m["card"] in (ids[HOPE],ids.get(HOPE+"_G")):
                    old=self.previous_hope.get(m["eid"])
                    if old is not None and m["counter"]>old:
                        self.hope_growth.add(int(game.turn))
                    self.previous_hope[m["eid"]]=m["counter"]
        if not p["pending"]:
            mask=game.legal_mask(seat)
            for i,s in enumerate(p["shop"]):
                card=C.item_card(s)
                if mask[8+i] and card in (ids[AZ],ids[RIDER]):
                    self.offers.add("element" if card==ids[AZ] else "pig")
        return b,p

    def after(self,game,before,decision,a,forced):
        b,p=before
        super().after(game,b,decision,a,forced)
        if p is None:
            return
        ids=C.card_ids()
        target=C.probe.absorption(p,a)
        if target is not None:
            eid,old,shop=target
            new=next((m for m in C.own(game)["board"] if m["eid"]==eid),None)
            if new is None:
                # A targeted action can replace the entity, for example when a triple forms.
                # Its individual change is unobserved, which is distinct from a zero gain.
                self.unmeasured_absorptions.append(dict(turn=int(game.turn), action=a, eid=eid))
            else:
                self.absorptions.append([int(game.turn),new["atk"]+new["hp"]-old,shop])
        if 8<=a<16:
            card=C.item_card(p["shop"][a-8])
            if card in (ids[AZ],ids[RIDER]):
                self.buys.add("element" if card==ids[AZ] else "pig")
        if 126<=a<196:
            self.mechanics.append(int(game.turn))


def pair(old,new):
    return interval([b-a for a,b in zip(old,new)]) if old else dict(n=0)


def p50(a):
    return sorted(a)[round((len(a)-1)*.5)] if a else None


def stats(arm):
    rr=arm["rank"]
    events=arm["absorptions"]
    boards={int(t):set(c.removesuffix("_G") for c in r[1]) for t,r in arm["eot"].items()}
    def consecutive(times):
        return int(any(t+1 in times for t in times))
    return dict(rank=rr,utility=float(C.quill.U[rr-1]),top4=int(rr<=4),top2=int(rr<=2),win=int(rr==1),
                repeat_element=int(len({t for t,g,s in events if s>0 and g>0})>=2),
                elemental_return=sum(g for t,g,s in events if s>0 and g>0),
                repeat_pig=int(len({e["turn"] for e in arm["gems"]})>=2),
                gem_return=sum(e["immediate_stats"] for e in arm["gems"]),
                mech=consecutive({t for t,b in boards.items() if {ROBOT,DRONE}<=b
                                  and len({u for u in arm["magnet"] if u<=t})>=2}),
                dragon=consecutive({t for t,b in boards.items() if {HOPE,POET}<=b
                                     and any(u<=t for u in arm["hope_growth"])}),
                element_offered=int("element" in arm["offers"]),element_bought=int("element" in arm["buys"]),
                pig_offered=int("pig" in arm["offers"]),pig_bought=int("pig" in arm["buys"]))


def natural_trial(agent,pool,seed):
    from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
    rec=Mechanisms()
    r=evaluate_policy(agent,EvalConfig(games=1,seed=seed,mirror_pool=pool,mirror_strict=True,
                                       decision_recorder=rec,collect_economy=True))
    mirror=r["mirror"]
    if (r["games_finished"]!=1 or r["seed_sequence"]!=[seed] or any(r[k] for k in
            ("illegal_actions","truncated_games","truncated_combats")) or
            any(mirror["mirror_"+k] for k in ("fallback","errors","rejected","missing","pool_unbalanced","pool_species_unbalanced"))
            or not mirror["mirror_applied"] or mirror["mirror_applied"]!=mirror["mirror_attempt"]):
        raise RuntimeError(f"invalid natural game {seed}")
    eot={t["turn"]:[t["eot_attack"]+t["eot_board_hp"],[m["card_id"] for m in t["eot_board"]]]
         for t in r["end_turn_economy"] if t["turn"]>=1}
    return dict(seed=seed,lobby=rec.lobby,rank=r["rank_sequence"][0],eot=eot,decisions=r["decisions"],
                mirror=mirror["mirror_applied"],absorptions=rec.absorptions,gems=rec.measured,magnet=rec.mechanics,
                hope_growth=sorted(rec.hope_growth),offers=sorted(rec.offers),buys=sorted(rec.buys))


def shard(stage,i):
    out=EVAL if stage=="final" else MID
    plan=J.read(out/"plan.json")
    device="cuda:0" if stage=="final" else "cpu"
    proto,pool=C.quill.config("hold" if stage=="final" else "train")
    models={k:make_eval_policy(str(p["path"]),protocol=proto,device=device)["policy"] for k,p in plan["models"].items()}
    natural,protected=[],[]
    for seed in range(plan["natural_seed"]+i,plan["natural_seed"]+plan["natural_n"],8):
        trial={}
        for k,agent in models.items():
            trial[k]=natural_trial(agent,pool,seed)
        natural.append(trial)
    for seed in range(plan["class_seed"]+i,plan["class_seed"]+plan["class_n"],8):
        for case in C.EXAM_CASES:
            trial={}
            for k,agent in models.items():
                turns=C.protected(seed,case,agent,C.opening(seed,case))
                trial[k]=dict(**C.measure(turns,case.startswith("elem")),
                             absorption_return=sum(e["gain"] for t in turns for e in t["absorptions"] if e["shop"]>0),
                             gem_return=sum(e["immediate_stats"] for t in turns for e in t["choices"]))
            protected.append(dict(seed=seed,case=case,arms=trial))
    atomic_json(out/f"{i}.json",dict(natural=natural,protected=protected))


def summarize(out,plan):
    shards=[J.read(out/f"{i}.json") for i in range(8)]
    natural=[row for sh in shards for row in sh["natural"]]
    protected=[row for sh in shards for row in sh["protected"]]
    keys=list(plan["models"])
    for row in natural:
        assert set(row)==set(keys) and len({(r["seed"],r["lobby"]) for r in row.values()})==1
    assert sorted(row[keys[0]]["seed"] for row in natural)==list(range(plan["natural_seed"],plan["natural_seed"]+plan["natural_n"]))
    totals={}
    for k in keys:
        scores=[stats(r[k]) for r in natural]
        totals[k]={s:float(np.mean([r[s] for r in scores])) for s in scores[0]}
        turns={}
        for t in range(9,19):
            vals=[r[k]["eot"][str(t)][0] for r in natural if str(t) in r[k]["eot"]]
            turns[t]=dict(n=len(vals),p50=p50(vals),reach=len(vals)/len(natural))
        totals[k]["turns"]=turns
    paired={}
    for k in keys:
        if k=="ppo":
            continue
        left=[stats(r[k]) for r in natural]
        right=[stats(r["ppo"]) for r in natural]
        paired[k]={s:pair([r[s] for r in left],[r[s] for r in right]) for s in left[0]}
        paired[k]["paired_log_strength"]={}
        for t in range(9,19):
            both=[r for r in natural if all(str(t) in r[m]["eot"] for m in (k,"ppo"))]
            paired[k]["paired_log_strength"][t]=pair(*[[np.log1p(r[m]["eot"][str(t)][0]) for r in both] for m in (k,"ppo")])
    classes={}
    for case in C.EXAM_CASES:
        rows=[r for r in protected if r["case"]==case]
        assert sorted(r["seed"] for r in rows)==list(range(plan["class_seed"],plan["class_seed"]+plan["class_n"]))
        classes[case]={k:dict(sustained=sum(r["arms"][k]["sustained"] for r in rows),
                  turns=[p50([r["arms"][k]["body"][t] for r in rows]) for t in range(10)],
                  absorption=p50([r["arms"][k]["absorption_return"] for r in rows]),
                  gems=p50([r["arms"][k]["gem_return"] for r in rows])) for k in keys}
    cohorts={}
    for name,bit in (("element",8),("pig",128),("mech",16),("dragon",4)):
        games=[r for r in natural if r[keys[0]]["lobby"]&bit]
        cohort={}
        for k in keys:
            scores=[stats(r[k]) for r in games]
            cohort[k]={s:float(np.mean([r[s] for r in scores])) for s in scores[0]} if scores else {}
        cohorts[name]=dict(n=len(games),models=cohort)
    atomic_json(out/"complete.json",dict(plan=file_identity(out/"plan.json"),natural_seeds=len(natural),
                protected_starts=len(protected),natural=totals,paired=paired,classes=classes,
                eligible_lobbies=cohorts,
                shards=[file_identity(out/f"{i}.json") for i in range(8)]))


def pin_middle(out):
    # Keep the chosen rolling snapshot alive across the trainer's retention pass.
    for _ in range(20):
        live=J.read(SEG/"train/latest.json")
        if live["seed"]!=J.PPO_SEED or live["cum_decisions"]<=0 or live["version"]<1:
            raise ValueError("wrong rolling checkpoint")
        kind="final" if live.get("final") else "step"
        generation=(SEG/"train/snapshots"/f"{kind}_{live['version']:06d}").resolve()
        checkpoint=Path(live["model"])
        if checkpoint.resolve()!=generation/"model.pt" or Path(live["trainer"]).resolve()!=generation/"trainer.pt":
            raise ValueError("rolling receipt points outside its generation")
        path=out/"model.pt"
        try:
            if kind=="step":
                record=J.read(generation/"record.json")
                if record["ppo"]["trainer_updates"]!=live["version"] or record["ppo"]["updates"]<1:
                    raise ValueError("snapshot does not match its completed update")
            os.link(checkpoint,path)
        except FileNotFoundError:
            time.sleep(.5)
            continue
        identity=file_identity(path)
        if kind=="final" and identity!=live["files"]["model.pt"]:
            raise ValueError("rolling checkpoint identity mismatch")
        checked=C.quill.engine_info()
        if checked["source_hash"]!=C.quill.PLAN["engine"]:
            raise ValueError("monitor engine changed")
        from hsbg.az.train.personal import inspect_checkpoint
        verified=inspect_checkpoint(str(path),"cpu")
        if not verified["forward_ok"] or verified["sha256"]!=identity["sha256"]:
            raise ValueError("pinned policy failed load or forward verification")
        return path,live
    raise RuntimeError("no rolling checkpoint available")


def assess(stage):
    out=EVAL if stage=="final" else MID
    out.mkdir(parents=True,exist_ok=False)
    recipe=receipt()
    split="hold" if stage=="final" else "train"
    assert library_digest(Path("artifacts/libraries/human-turn-3702-20261118")/split)==recipe["library_digests"][split]
    if stage=="final":
        seal=J.read(SEG/"complete.json")
        latest=verified_generation(SEG,J.PPO_SEED)
        if seal["checkpoint"]!=latest or seal["identity"]["recipe_sha256"]!=file_identity(J.SERIES/"recipe.json")["sha256"]:
            raise ValueError("invalid final receipt")
        final=Path(latest["model"])
    else:
        final,latest=pin_middle(out)
    paths=dict(joint=J.MODEL/"model.pt",ppo=final)
    if stage=="final":
        paths=dict(base3265=Path(C.quill.BASE),**paths)
        assert file_identity(paths["base3265"])["sha256"]==C.quill.PLAN["base_sha"]
    proto,_=C.quill.config(split)
    plan=dict(models={k:dict(path=str(p),**file_identity(p)) for k,p in paths.items()},
        version=latest["version"],decisions=latest["cum_decisions"],source=source_identity(ROOT),
        natural_seed=171600000 if stage=="final" else 781000,natural_n=512 if stage=="final" else 32,
        class_seed=770000 if stage=="final" else 780000,class_n=48 if stage=="final" else 8,
        cases=C.EXAM_CASES,job=os.environ["SLURM_JOB_ID"],protocol=proto.sha256(),library=recipe["library_digests"][split])
    atomic_json(out/"plan.json",plan)
    def worker(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,stage,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(worker,range(8)))
    summarize(out,plan)


if __name__=="__main__":
    torch.set_num_threads(1)
    stage=sys.argv[1]
    if stage=="train":
        train()
    elif stage=="smoke":
        proto,pool=C.quill.config("train")
        player=make_eval_policy(C.quill.BASE,protocol=proto,device="cpu")["policy"]
        result=natural_trial(player,pool,797001)
        print(json.dumps(dict(seed=result["seed"],metrics=stats(result),mirror=result["mirror"])))
    elif stage in ("final","middle"):
        assess(stage) if len(sys.argv)==2 else shard(stage,int(sys.argv[2]))
    else:
        raise ValueError(stage)
