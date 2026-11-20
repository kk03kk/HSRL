"""Evaluate one focal economic decision and finish the lobby with all eight actors."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import UTILITY,estimate
from workflows.adaptive_decision_study import POLICY,RANGES,family,lobby,read,weights


def public_options(game,seat,p):
    pl = json.loads(game.player_json(seat))
    best = {}
    for name,(lo,hi) in RANGES.items():
        a = lo+int(np.argmax(p[lo:hi]))
        if p[a] > 0:
            best[name] = (a,float(p[a]))
    return pl,best


def select(name,turn,original,player,best,p,game,seat,features):
    from hsbg.env.obs_std import GLOBAL_FIELD_INDEX
    if name in ("play_first","level_first"):
        family,ratio = ("play",.5) if name=="play_first" else ("level",.2)
        if (original!=1 or family not in best or features[GLOBAL_FIELD_INDEX["refresh_price"]]<=0
                or best[family][1]<ratio*best["roll"][1]):
            return None
        action = best[family][0]
        detail = {"upgrade_cost":player["upgrade_cost"]}
        if family=="play":
            item = player["hand"][(action-46)//8]["item"]
            kind,data = next(iter(item.items()))
            detail.update(card=data["card"],kind=kind)
        return action,detail
    elif name=="greater_runner":
        if turn!=9 or not 30<=original<46 or not player["pending"]:
            return None
        menu = player["pending"][-1].get("Trinket")
        if not menu or menu["free"] or menu["replace"] is not None:
            return None
        choices = [30+j for j in range(len(menu["options"])) if p[30+j]>0]
        if original not in choices or len(choices)<2:
            return None
        alternate = max((j for j in choices if j!=original),key=lambda j:p[j])
        options = json.loads(game.view(seat))["me"]["pending"]["options"]
        return alternate,dict(cards=menu["options"],costs=[x["cost"] for x in options],available=choices)
    else:
        raise ValueError(f"unknown decision rule {name}")
    return None


class BranchPolicy:
    def __init__(self,actor,rules,apply):
        self.actor = actor
        self.model = actor.model
        self.rules = rules
        self.apply = apply

    @property
    def overrides(self):
        return self.actor.overrides

    def reset(self):
        self.actor.reset()
        self.events = {}

    def decide(self,game,seat):
        self.pending = {}
        self.game = game
        decision = self.actor.decide(game,seat)
        if decision.info.get("forced"):
            self.pending = {}
        for rule,event in self.pending.items():
            expected = event["alternative"] if self.apply else event["original"]
            if expected != decision.action:
                raise ValueError("proposed and realized focal action differ")
            self.events[rule] = event
        return decision

    def __call__(self,f,c,mask,seat):
        p,v = self.actor(f,c,mask,seat)
        original = int(p.argmax())
        if (self.actor.proposal is not None or not 8<=self.game.turn<=12
                or len(self.events)==len(self.rules) or (original!=1 and not 30<=original<46)):
            return p,v
        player,best = public_options(self.game,seat,p)
        for name in self.rules:
            if name in self.events:
                continue
            proposal = select(name,self.game.turn,original,player,best,p,self.game,seat,f)
            if proposal is None:
                continue
            alternative,detail = proposal
            if alternative==original or not mask[alternative] or p[alternative]<=0:
                raise ValueError("branch must choose a different available public model action")
            self.pending[name] = dict(turn=self.game.turn,original=original,alternative=alternative,
                gold=player["gold"],hp=player["health"]+player["armor"],tier=player["tier"],
                size=len(player["board"]),confidence=[float(p[original]),float(p[alternative])],**detail)
            if self.apply:
                replay = np.zeros_like(p)
                replay[alternative] = 1
                return replay,v
        return p,v


def arm(i,specs,apply,opening,*,at_branch=None,capture_turns=True):
    actor = ACTORS["intent"]
    policy = BranchPolicy(actor,specs,apply)
    capture = {}
    def observe_turn(game,seat,tape,boards):
        if 8<=game.turn<=12:
            capture[game.turn] = dict(seed=START+i,focus=seat,turn=game.turn,boards=dict(boards),
                                      state=game.state_json(),prefix=tape.hex())
    def observe_focus(game,action,trace):
        for e in policy.pending.values():
            e["state"] = sha256(game.state_json().encode()).hexdigest()
            e["prefix"] = trace.hexdigest()
        if at_branch is not None and policy.pending:
            return at_branch(game,policy.pending)
    original = actor.parent.evaluator
    actor.parent.evaluator = policy
    try:
        result = lobby.play({**ACTORS,"intent":policy},i,True,START,opening=opening,
            on_turn=observe_turn if capture_turns and not apply else None,on_focus=observe_focus)
    finally:
        actor.parent.evaluator = original
    return result,policy.events,capture


def trial(i):
    parent = read(BANK/f"base-{i}.json")
    opening_file = BANK/f"opening-{i}.json"
    if parent["opening"]!=file_identity(opening_file):
        raise ValueError("opening differs from sealed baseline")
    baseline,events,turns = arm(i,RULES,False,read(opening_file))
    expected = json.loads(json.dumps(baseline))
    if any(expected[k]!=v for k,v in parent["natural"].items() if k!="overrides"):
        raise ValueError("neutral full-lobby continuation differs from the original natural game")
    arms = {}
    for name in RULES:
        if name not in events:
            arms[name] = dict(trigger=None,result=baseline)
            continue
        event = events[name]
        candidate,applied,_ = arm(i,[name],True,turns[event["turn"]])
        if applied.get(name)!=event:
            raise ValueError("counterfactual failed complete state and all-seat prefix equality")
        arms[name] = dict(trigger=event,result=candidate)
    atomic_json(OUT/f"game-{i}.json",dict(seed=START+i,baseline=baseline,alternatives=arms))


def summarize(out,n,rules):
    rows = [read(out/f"game-{i}.json") for i in range(n)]
    results = {}
    for rule in rules:
        ranks,utility,growth,body,reach = [],[],[],[],[]
        turns = {"parent":{},"candidate":{}}
        affected = 0
        changed = Counter()
        for row in rows:
            a,b = row["baseline"],row["alternatives"][rule]["result"]
            affected += row["alternatives"][rule]["trigger"] is not None
            ranks.append(a["rank"]-b["rank"])
            changed["better" if ranks[-1]>0 else "worse" if ranks[-1]<0 else "tie"] += 1
            utility.append(UTILITY[b["rank"]-1]-UTILITY[a["rank"]-1])
            reach.append(int("12" in b["boards"])-int("12" in a["boards"]))
            if "12" in a["boards"] and "12" in b["boards"]:
                body.append(math.log1p(b["boards"]["12"])-math.log1p(a["boards"]["12"]))
                growth.append(body[-1]-math.log1p(b["boards"]["9"])+math.log1p(a["boards"]["9"]))
            for name,r in (("parent",a),("candidate",b)):
                for t,x in r["boards"].items():
                    turns[name].setdefault(t,[]).append(x)
        curves = {k:{t:dict(n=len(v),p50=median(v)) for t,v in d.items()} for k,d in turns.items()}
        results[rule] = dict(n=n,affected=affected,outcomes=changed,rank_gain=estimate(ranks),utility=estimate(utility),
                             common_growth=estimate(growth),common_log12=estimate(body),reach12_delta=estimate(reach),curves=curves)
    return results


def main(out,bank,rules,workers):
    global OUT,BANK,ACTORS,START,RULES
    source = source_identity(ROOT)
    plan = read(bank/"plan.json")
    complete = read(bank/"complete.json")
    if (source["tracked_dirty"] or plan["policy"]!=POLICY or complete["n"]!=plan["n"]
            or not 0<workers<=int(os.environ["SLURM_CPUS_PER_TASK"])):
        raise ValueError("invalid opening bank or release")
    OUT,BANK,START,RULES = out,bank,plan["seed"],rules
    ACTORS = weights()
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,bank_plan=file_identity(bank/"plan.json"),
        bank_seal=file_identity(bank/"complete.json"),field=plan,policy=POLICY,rules=rules,job=os.environ["SLURM_JOB_ID"]))
    with ProcessPoolExecutor(workers,mp_context=mp.get_context("fork")) as pool:
        list(pool.map(trial,range(plan["n"])))
    atomic_json(out/"complete.json",summarize(out,plan["n"],rules))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("bank",type=Path)
    p.add_argument("--rules",nargs="+",required=True)
    p.add_argument("--workers",type=int,required=True)
    args=p.parse_args()
    main(args.out,args.bank,args.rules,args.workers)
