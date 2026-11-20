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
OUT = RUNS / "nus-adaptive-triple-followthrough-20261120-2"
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
        differences=[k for k,v in stored.items() if k!="overrides" and clean[k]!=v]
        observed=json.loads(json.dumps(events[RULE]))
        event_differences=[k for k in e if observed.get(k)!=e[k]]
        if differences or event_differences:
            atomic_json(OUT/f"mismatch-{i}-{arm}.json",dict(index=i,arm=arm,fields=differences,events=event_differences,
                original={k:stored[k] for k in differences},observed={k:clean[k] for k in differences}))
            raise ValueError(f"game {i} {arm} differs: terminal={differences}, event={event_differences}")
        if "start_gold" not in row:
            raise ValueError(f"game {i} {arm} missed decision")
        has_end=str(e["turn"]) in stored["boards"]
        if has_end!=("end_gold" in row):
            raise ValueError(f"game {i} {arm}: EndTurn log does not match sealed observation ({has_end})")
        if has_end:
            row["end_body"]=stored["boards"][str(e["turn"])]
        parts[arm]=row
    atomic_json(OUT/f"game-{i}.json",parts)
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
        ends=[g for g in group if "end_gold" in g]
        summaries[arm]=dict(n=len(group),end_observed=len(ends),mean_end_size=mean(g["end_size"] for g in ends),
            eot_p50=median(g["end_body"] for g in ends),mean_end_gold=mean(g["end_gold"] for g in ends),
            same_turn_target_played=sum(r[arm]["target_plays"]>0 for r in unique),
            same_turn_target_on_board=sum(r[arm].get("end_target",0)>0 for r in unique))
    both=[r for r in rows if all("end_body" in r[k] for k in ("baseline","candidate"))]
    atomic_json(OUT/"complete.json",dict(n=len(rows),without_prior_golden=len(unique),summary=summaries,
        source_already_on_board=sum(r["candidate"]["source_board_copies"]>0 for r in rows),
        common_end=len(both),
        eot_candidate_stronger=sum(r["candidate"]["end_body"]>r["baseline"]["end_body"] for r in both),
        eot_candidate_weaker=sum(r["candidate"]["end_body"]<r["baseline"]["end_body"] for r in both),
        rows=file_identity(OUT/"rows.json")))


if __name__=="__main__":
    main()
