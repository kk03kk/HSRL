"""Learn a low-gold purchase prior from disjoint human XML demonstrations."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import estimate
from tools.summarize_peer_distillation import body
from workflows.parse_firestone_replay_xml import CardDb,XmlReplay
from workflows.peer_finetune_holdout import read
from workflows.peer_spell_choice import INPUTS,RUNS

CARDS=Path("engine/crates/hsbg-engine/data/cards.json")
ARCHIVE=Path("data/firestone_replays/refresh-20261112T031450Z")
OUT=RUNS/"nus-peer-human-spell-prior-20261120-1"


class Offers(XmlReplay):
    def _on_block_start(self,el):
        super()._on_block_start(el)
        a=self.block_stack[-1]["action"]
        if not a or a["turn"]<7 or a["action_type"] not in ("reroll","buy_spell","buy_minion"):
            return
        slots=dict(zip(a["shop_pos"],a["shop_cards"],strict=True))
        if a["action_type"] in ("buy_spell","buy_minion"):
            self.buys+=1
            self.mapped+=int(a["shop_slot"] is not None and slots.get(a["shop_slot"])==a["card_id"])
        if a["gold"] not in (1,2) or a["action_type"]=="buy_minion":
            return
        offers=[cid for cid in slots.values() if self.cdb.kind(cid) in ("spell","tavern_spell")
                and 0 < self.costs.get(cid,0) <= a["gold"]]
        if len(offers)!=1 or a["turn"] in self.turns:
            return
        cid=offers[0]
        if a["action_type"]=="buy_spell" and (cid!=a["card_id"] or slots.get(a["shop_slot"])!=cid):
            return
        self.turns.add(a["turn"])
        self.choices.append(dict(turn=a["turn"],gold=a["gold"],card=cid,buy=a["action_type"]=="buy_spell"))


def parse_one(task):
    original,path=task
    cdb=CardDb(str(CARDS))
    policy=Offers(cdb)
    policy.buys=policy.mapped=0
    policy.turns=set()
    policy.choices=[]
    policy.costs={c["id"]:int(c.get("cost") or 0) for c in cdb.cards}
    policy.parse(str(path))
    return dict(original=original,file=file_identity(path),buys=policy.buys,mapped=policy.mapped,choices=policy.choices)


def fit(rows):
    totals=defaultdict(Counter)
    support=defaultdict(set)
    for r in rows:
        votes=r["choices"]
        for v in votes:
            gold=str(v["gold"])
            key=gold+"/"+v["card"]
            support[key].add(r["original"])
            for part in (gold,key):
                totals[part]["n"]+=1/len(votes)
                totals[part]["buy"]+=int(v["buy"])/len(votes)
    by_gold={g:(totals[g]["buy"]+1)/(totals[g]["n"]+2) for g in ("1","2")}
    learned={}
    for key,ids in sorted(support.items()):
        pooled=by_gold[key.split("/")[0]]
        p=(totals[key]["buy"]+12*pooled)/(totals[key]["n"]+12)
        learned[key]=dict(games=len(ids),rate=p,buy=len(ids)>=16 and p>=max(.5,pooled+.10))
    return dict(rates=by_gold,choices=learned)


def run():
    source=source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("dirty release")
    train,hold=[set(read(Path("artifacts/libraries/human-turn-3702-20261118")/s/"manifest.json")["games"]) for s in ("train","hold")]
    transfer={p.stem for p in Path("data/firestone_replays/transfer-20261119-01/raw-01").glob("*.json")}
    if train&hold or train&transfer:
        raise ValueError("sources overlap")
    feed=ARCHIVE/"bgs-perfect-games.json"
    available={}
    for v in read(feed):
        k=v.get("originalReviewId")
        path=ARCHIVE/"xmlz"/(v["reviewId"]+".xml.zip")
        if k in train and "2026-11-04"<=v["originalCreationDate"][:10]<"2026-11-17" and path.is_file():
            available.setdefault(k,path)
    selected=sorted(available,key=lambda k:sha256(("hsai-human-spell-train-v1:"+k).encode()).digest())[:384]
    if len(selected)!=384:
        raise ValueError("insufficient disjoint human originals")
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source,feed=file_identity(feed),card_table=file_identity(CARDS),
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),live_pin=file_identity(ROOT/"config/live/model.json"),
        job=os.environ["SLURM_JOB_ID"],originals=selected,simulator_evidence=[p[0] for p in INPUTS],
        method="One first eligible reroll/buy per turn. One visible spell with positive listed cost within spendable gold; equal total weight per contributing human game."))
    with ProcessPoolExecutor(8) as ex:
        rows=list(ex.map(parse_one,[(k,available[k]) for k in selected]))
    mapped=sum(r["mapped"] for r in rows)
    purchases=sum(r["buys"] for r in rows)
    atomic_json(OUT/"observations.json",rows)
    if not purchases or mapped/purchases<.95:
        raise ValueError("physical shop-slot agreement below 95 percent")
    model=fit(rows)
    atomic_json(OUT/"policy.json",model)
    from hsbg.az.train.economic_tasks import card_ids
    names={v:k for k,v in card_ids().items()}
    gain,late,growth=[],[],[]
    chosen=0
    per_input={}
    for name,start,n in INPUTS:
        pairs,part=[],[]
        for i in range(8):
            run=RUNS/name
            for a,c in zip(read(run/f"panel-{i}.json"),read(run/f"cases-{i}.json"),strict=True):
                base=a["parent"];case=c["case"]
                assert base["seed"]==c["seed"]==a["candidate"]["seed"]
                pairs.append(c["seed"])
                allowed=case and case["original"]==1 and model["choices"].get(str(case["gold"])+"/"+names[case["card"]],{}).get("buy")
                chosen+=int(bool(allowed))
                candidate=a["candidate"] if allowed else base
                part.append(candidate["metrics"]["utility"]-base["metrics"]["utility"])
                if body(base,12) is not None and body(candidate,12) is not None:
                    diff=math.log1p(body(candidate,12))-math.log1p(body(base,12))
                    late.append(diff)
                    growth.append(diff-math.log1p(body(candidate,9))+math.log1p(body(base,9)))
        assert sorted(pairs)==list(range(start,start+n))
        gain.extend(part)
        per_input[name]=estimate(part)
    utility,log12=estimate(gain),estimate(late)
    atomic_json(OUT/"complete.json",dict(policy=file_identity(OUT/"policy.json"),originals=len(rows),
        human_events=sum(len(r["choices"]) for r in rows),purchases=purchases,mapped=mapped,
        trained_on_simulation_returns=False,chosen=chosen,utility=utility,log12=log12,growth=estimate(growth),
        per_input=per_input,passed=utility["ci95"][0]>0 and log12["ci95"][0]>=-.12))


if __name__=="__main__":
    run()
