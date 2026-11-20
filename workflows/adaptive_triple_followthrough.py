"""Replay triggered paired purchases and measure whether the focal agent uses the golden card."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows import adaptive_counterfactual as branch
from workflows.adaptive_decision_study import RUNS, read, weights

BANK = RUNS / "nus-adaptive-triple-openings-20261120-1"
RUN = RUNS / "nus-adaptive-triple-branch-20261120-1"
SEAL = RUNS / "nus-adaptive-triple-seal-20261120-1" / "seal.json"
OUT = RUNS / "nus-adaptive-triple-followthrough-20261120-1"
RULE = "complete_triple"


def replay(i):
    pair = read(RUN / f"game-{i}.json")
    expected = pair["alternatives"][RULE]
    e = expected["trigger"]
    opening = read(BANK / f"opening-{i}.json")
    parts = {}
    for arm, apply in (("baseline",False),("candidate",True)):
        row = dict(target_plays=0)
        def observe(game,action,pending):
            if game.turn!=e["turn"]:
                return
            if RULE in pending:
                if pending[RULE]["state"]!=e["state"] or pending[RULE]["prefix"]!=e["prefix"]:
                    raise ValueError("different purchase decision")
                pl=json.loads(game.player_json(i%8))
                row.update(source_board_copies=sum(m["card"]==e["triple_card"] and not m["golden"] for m in pl["board"]),
                           golden_before=branch.count_golden(pl,e["gold_card"]),start_gold=pl["gold"])
            if "start_gold" not in row:
                return
            if 46<=action<126:
                pl=json.loads(game.player_json(i%8))
                unit=pl["hand"][(action-46)//8]["item"].get("Minion")
                if unit and unit["card"]==e["gold_card"] and unit["golden"]:
                    row["target_plays"]+=1
            if action==0:
                pl=json.loads(game.player_json(i%8))
                row.update(end_gold=pl["gold"],end_size=len(pl["board"]),
                           end_target=sum(m["card"]==e["gold_card"] and m["golden"] for m in pl["board"]))
        game, events, _ = branch.arm(i,[RULE],apply,opening,capture_turns=False,on_focus_action=observe)
        stored = expected["result"] if apply else pair["baseline"]
        clean = json.loads(json.dumps(game))
        if (any(clean[k]!=v for k,v in stored.items() if k!="overrides") or "end_gold" not in row
                or json.loads(json.dumps(events[RULE]))!=e):
            raise ValueError("instrumented full game differs from the sealed result")
        row["end_body"]=stored["boards"][str(e["turn"])]
        parts[arm]=row
    return dict(index=i,turn=e["turn"],card=e["triple_card"],**parts)


def main():
    source=source_identity(ROOT)
    seal=read(SEAL)
    plan=read(BANK/"plan.json")
    if source["tracked_dirty"] or file_identity(RUN/"complete.json")!=seal["report"]:
        raise ValueError("unverified input")
    ids=[]
    for i in range(plan["n"]):
        path=RUN/f"game-{i}.json"
        if file_identity(path)!=seal["shards"][str(i)]:
            raise ValueError("paired evidence changed")
        if read(path)["alternatives"][RULE]["trigger"] is not None:
            ids.append(i)
    if not ids:
        raise ValueError("no eligible decisions")
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source,original_seal=file_identity(SEAL),ids=ids,
                                    job=os.environ["SLURM_JOB_ID"]))
    branch.START=plan["seed"]
    branch.ACTORS=weights()
    with ProcessPoolExecutor(min(len(ids),int(os.environ["SLURM_CPUS_PER_TASK"])),mp_context=mp.get_context("fork")) as pool:
        rows=list(pool.map(replay,ids))
    atomic_json(OUT/"rows.json",rows)
    unique=[r for r in rows if r["candidate"]["golden_before"]==0]
    summaries={}
    for arm in ("baseline","candidate"):
        group=[r[arm] for r in rows]
        summaries[arm]=dict(n=len(group),mean_end_size=mean(g["end_size"] for g in group),
            eot_p50=median(g["end_body"] for g in group),mean_end_gold=mean(g["end_gold"] for g in group),
            same_turn_target_played=sum(r[arm]["target_plays"]>0 for r in unique),
            same_turn_target_on_board=sum(r[arm]["end_target"]>0 for r in unique))
    atomic_json(OUT/"complete.json",dict(n=len(rows),without_prior_golden=len(unique),summary=summaries,
        source_already_on_board=sum(r["candidate"]["source_board_copies"]>0 for r in rows),
        eot_candidate_stronger=sum(r["candidate"]["end_body"]>r["baseline"]["end_body"] for r in rows),
        eot_candidate_weaker=sum(r["candidate"]["end_body"]<r["baseline"]["end_body"] for r in rows),
        rows=file_identity(OUT/"rows.json")))


if __name__=="__main__":
    main()
