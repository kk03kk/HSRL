"""Verify sealed rows and aggregate by source game, including all non-triggers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

UTILITY = [1,.35,-1,-1.15,-1.30,-1.45,-1.60,-1.75]


def distribution(values):
    return dict(n=len(values), p50=median(values) if values else None)


def baseline(rows):
    denom = {"pirate":6, "boar":7, "dragon":2, "mech":4, "element":3}
    low = [r for r in rows if r["low_turn"] is not None]
    opp = [r for r in rows if r["first"]]
    target = [r for r in rows if r["target_rows"]]
    finals = [r for r in rows if r["final_tech_rows"]]
    return dict(games=len(rows), rank=mean(r["rank"] for r in rows),
        opportunities=len(opp), low_opportunities=sum(r["first"]["hp"]<=15 for r in opp),
        responded=sum(r["invested"] for r in opp), low_games=len(low), low_top4=sum(r["rank"]<=4 for r in low),
        formations={k:dict(n=sum(bool(r["lobby"] & (1<<idx)) for r in rows),
            formed=sum(r["formed"][k] for r in rows if r["lobby"] & (1<<idx))) for k,idx in denom.items()},
        pirate_payoff_games=sum(r["events"].get("castaway_pick_with_hooktusk",0)>0 for r in rows),
        pirate_payoff_events=sum(r["events"].get("castaway_pick_with_hooktusk",0) for r in rows),
        magnet_payoff_games=sum(r["events"].get("direct_target_attachment",0)>0 for r in rows),
        direct_target_games=len(target), direct_target_rows=sum(r["target_rows"] for r in rows),
        direct_target_actions=sum(r["target_actions"] for r in rows),
        final_tech_opportunity_games=len(finals),final_tech_chosen_games=sum(r["final_tech_choices"]>0 for r in finals),
        turn_strength={t:distribution([r["strength"][str(t)] for r in rows if str(t) in r["strength"]]) for t in range(1,21)})


def interval(values):
    if not values:
        return None
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(1176707)
    samples=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return dict(n=len(x),mean=mean(values),ci95=np.quantile(samples,[.025,.975]).tolist())


def score(outcome, turn):
    rank=outcome["rank"]
    return dict(rank=rank,utility=UTILITY[rank-1],first=int(rank==1),top4=int(rank<=4),bottom2=int(rank>=7),
        formed=int(outcome["formed"]),payout=int(bool(outcome["events"])),
        early_payout=int(any(t < turn+2 for t,_,_ in outcome["events"])),reach12=int(outcome["last"]>=12),
        got=int(outcome["acquisition"]),return_alive=int(outcome.get("return_steps",0)>0))


def groups(rows):
    opp=[r for r in rows if r["opportunity"]]
    return dict(all=rows,opportunities=opp,low=[r for r in opp if r["opportunity"]["hp"]<=15],
        higher=[r for r in opp if r["opportunity"]["hp"]>15],
        castaway=[r for r in opp if "BG36_342" in r["opportunity"]["offered"].values()],
        hooktusk=[r for r in opp if "BG36_344" in r["opportunity"]["offered"].values()])


def compare(rows, window):
    values={}; totals={}; skipped=Counter(); naturally={}
    for row in rows:
        opportunity=row["opportunity"]
        turn=opportunity["turn"] if opportunity else 0
        if opportunity:
            pairs=[(f["0"], f[str(window)]) for f in row["futures"]]
            if len(pairs)!=4: raise ValueError("incomplete futures")
        else:
            pairs=[(row["natural"],row["natural"])]
        control=[score(a,turn) for a,_ in pairs]
        test=[score(b,turn) for _,b in pairs]
        for key in control[0]:
            values.setdefault(key,[]).append(mean(s[key]-c[key] for c,s in zip(control,test)))
            totals.setdefault(key,[]).append([mean(c[key] for c in control),mean(s[key] for s in test)])
            naturally.setdefault(key,[]).append(test[0][key]-control[0][key])
        if opportunity:
            for _,b in pairs:
                skipped["branches"]+=1
                skipped["no_return"]+=b["return_steps"]==0
                skipped["no_payout"]+=not bool(b["events"])
                skipped["no_formation"]+=not b["formed"]
                skipped["tail_seat_frames"]+=b["tail"]
                skipped["instant_core"]+=str(b["first"]) in opportunity["offered"]
    return dict(n=len(rows), effects={k:interval(x) for k,x in values.items()},
        rates={k:np.mean(v,axis=0).tolist() for k,v in totals.items()},counts=dict(skipped),
        original_future={k:interval(x) for k,x in naturally.items()})


def main(root):
    rows={}; protocols=set();seen=set(); receipts={}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():continue
        plan=json.loads((folder/"plan.json").read_bytes())
        done=json.loads((folder/"complete.json").read_bytes())
        data=(folder/"rows.jsonl").read_bytes()
        if done["evidence"]!={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}:raise ValueError(folder)
        batch=[json.loads(l) for l in data.splitlines()]
        if len(batch)!=done["rows"]:raise ValueError(folder)
        stage=plan["stage"]
        for r in batch:
            ident=(stage,r["panel"],r["seed"],r.get("model"))
            if ident in seen:raise ValueError(ident)
            seen.add(ident)
        rows.setdefault(stage,[]).extend(batch)
        protocols.add(plan["protocol_sha256"])
        receipts[folder.name]=done
    if len(protocols)!=1 or len(rows["baseline"])!=576:raise ValueError("incomplete/mixed baseline")
    out=dict(protocol_sha256=protocols.pop(),receipts=receipts,baseline={},p1={})
    for panel,start in (("hold",691210000),("human",692210000)):
        data=rows[panel]
        if {r["seed"] for r in data}!=set(range(start,start+160)):raise ValueError(panel)
        out["baseline"][panel]={}
        for m in ("0750","r1","3286"):
            sample=[r for r in rows["baseline"] if r["panel"]==panel and r["model"]==m]
            if len(sample)!=96:raise ValueError((panel,m))
            out["baseline"][panel][m]=baseline(sample)
        opp=[r for r in data if r["opportunity"]]
        out["p1"][panel]=dict(games=len(data),exposed=len(opp),
           low=sum(r["opportunity"]["hp"]<=15 for r in opp),
           turns=dict(Counter(r["opportunity"]["turn"] for r in opp)),
           no_opportunity_rank=dict(Counter(r["natural"]["rank"] for r in data if not r["opportunity"])),
           windows={k:{str(d):compare(g,d) for d in (1,2,100)} for k,g in groups(data).items() if g})
    (root/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    main(parser.parse_args().root)
