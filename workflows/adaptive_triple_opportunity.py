"""Find actual purchases and refreshes at public, affordable three-copy completions."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows.adaptive_decision_study import POLICY,RUNS,lobby,read,weights

BANK=RUNS/"nus-adaptive-openings-20261120-1"
OUT=RUNS/"nus-adaptive-triple-opportunity-20261120-1"


def offers(game,seat):
    p=json.loads(game.player_json(seat))
    held=Counter(m["card"] for m in p["board"] if not m["golden"] and not m.get("combat_only"))
    held.update(h["item"]["Minion"]["card"] for h in p["hand"] if h["locked"]==0
        and "Minion" in h["item"] and not h["item"]["Minion"]["golden"])
    legal=game.legal_mask(seat)
    possible=[]
    for j,s in enumerate(p["shop"]):
        m=s["item"].get("Minion")
        if j<8 and m and not m["golden"] and held[m["card"]]==2 and legal[8+j]:
            possible.append((j,m["card"]))
    if not possible:
        return []
    shop=json.loads(game.view(seat))["me"]["shop"]
    return [(8+j,c) for j,c in possible if shop[j].get("triple_need")==3 and shop[j].get("owned")==2
            and not shop[j].get("costs_health") and shop[j].get("price") is not None]


def check(i):
    opening=read(BANK/f"opening-{i}.json")
    rows=[]
    def log(game,action,trace):
        if 8<=game.turn<=12 and (action==1 or 8<=action<16):
            eligible=offers(game,i%8)
            if eligible:
                rows.append(dict(turn=game.turn,action=action,available=eligible,
                    label="roll" if action==1 else "triple_buy" if action in {a for a,c in eligible} else "other_buy",
                    prefix=trace.hexdigest()))
    result=lobby.play(ACTORS,i,True,opening["seed"]-i,opening=opening,on_focus=log)
    expected=read(BANK/f"base-{i}.json")["natural"]
    if result["trace"]!=expected["trace"] or result["rank"]!=expected["rank"]:
        raise ValueError("instrumented trace diverged")
    atomic_json(OUT/f"game-{i}.json",dict(seed=opening["seed"],events=rows,rank=result["rank"],trace=result["trace"]))
    return rows


if __name__=="__main__":
    source=source_identity(ROOT);plan=read(BANK/"plan.json")
    if source["tracked_dirty"] or plan["policy"]!=POLICY:
        raise ValueError("wrong release or baseline")
    ACTORS=weights()
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source,bank=file_identity(BANK/"plan.json"),job=os.environ["SLURM_JOB_ID"],
        scope="Ordinary unmodified three-copy triple; two normal copies in hand/board; legal gold purchase; no wild or health purchases. Count only actual buys or irreversible refreshes, T8-12."))
    with ProcessPoolExecutor(24,mp_context=mp.get_context("fork")) as pool:
        games=list(pool.map(check,range(plan["n"])))
    stats={}
    for label in ("roll","triple_buy","other_buy"):
        by_game=[[r for r in g if r["label"]==label] for g in games]
        stats[label]=dict(games=sum(bool(g) for g in by_game),events=sum(map(len,by_game)),
            turns=sum(len({r["turn"] for r in g}) for g in by_game))
    atomic_json(OUT/"complete.json",dict(n=plan["n"],statistics=stats,followup=stats["roll"]["games"]>=12))
