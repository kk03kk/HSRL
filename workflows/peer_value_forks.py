"""Evaluate two legal spending choices with independent planning and outcome futures."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hsbg_engine
import numpy as np
import torch
from hsbg.az.train.common import mask_from_bytes
from hsbg.az.train.economic_tasks import card_ids
from hsbg.az.train.mirror_snapshots import capture_eot
from hsbg.runtime.artifacts import atomic_json
from workflows.d142_crossover import ResumedPolicy, digest
from workflows.peer_distillation import opening, alternate, rollout
from workflows.peer_value_study import OUT, PLAN, SHARDS, field, parallel
from tools.probe_four_growth_curves import RAT,RIDER,AZ,SURF,BRANN,KELP,BAL,HOPE,POET,ROBOT,DRONE


def options(game, growth):
    p = json.loads(game.player_json(0))
    mask = mask_from_bytes(game.legal_mask(0))
    bodies, components = [], []
    for slot,offer in enumerate(p["shop"]):
        m = offer["item"].get("Minion")
        if not m or not mask[8+slot]:
            continue
        entry = (m["atk"]+m["hp"],8+slot,m["card"])
        (components if m["card"] in growth else bodies).append(entry)
    if not bodies:
        return None
    body,tempo,tc = max(bodies)
    small = [v for v in components if v[0] < body]
    if small:
        _,invest,ic = max(small)
        kind = "growth_purchase"
    elif mask[3]:
        invest,ic,kind = 3,None,"tier"
    else:
        return None
    return dict(invest=invest,tempo=tempo,kind=kind,invest_card=ic,tempo_card=tc,
                hp=p["health"]+p["armor"],tier=p["tier"],gold=p["gold"],board=len(p["board"]),
                prepared=any(m["card"] in growth for m in p["board"]))


def find_fork(policy, scorer, pool, seed, turn, growth):
    scorer.begin(seed)
    fork = opening(policy,pool,seed,turn)
    if fork.state is None:
        return None,None,"ended"
    game = hsbg_engine.Game.from_json(json.dumps(fork.state))
    for step in range(12):
        if game.is_over() or int(game.turn)!=turn or not game.needs_action(0):
            break
        pair = options(game,growth)
        if pair:
            fork.state = json.loads(game.state_json())
            fork.opening = digest(fork.state)
            pair["step_in_turn"] = step
            return fork,pair,"eligible"
        action = int(fork.policy.decide(game,0).action)
        if action==0:
            break
        game.step(0,action)
        fork.strict.after_learner_step(game)
    return None,None,"no_pair"


class RecruitFinished(Exception):
    pass


class Branch(ResumedPolicy):
    def __init__(self, fork, scorer, action, planning):
        super().__init__(fork,scorer)
        self.action = action
        self.planning = planning
        self.count = 0
        self.body = None

    def decide(self, game, seat):
        if self.planning and int(game.turn)!=self.fork.turn:
            raise RecruitFinished()
        if self.count==0:
            self.evaluator.force = self.action
        elif self.count==1 and self.planning:
            self.evaluator.capture_next = True
        result = super().decide(game,seat)
        if self.count==0 and int(result.action)!=self.action:
            raise ValueError("specified first action did not execute")
        self.count += 1
        if self.planning and int(result.action)==0:
            end = capture_eot(game,seat)
            if end:
                self.body = sum(m["atk"]+m["hp"] for m in end["player"]["board"])
            raise RecruitFinished()
        return result


def shard(i):
    pool,policy,scorer = field("hold")
    ids = card_ids()
    growth = {ids[v] for k in (RAT,RIDER,"BG36_331",AZ,SURF,BRANN,KELP,BAL,HOPE,POET,ROBOT,DRONE,"BG33_825")
              for v in (k,k+"_G") if v in ids}
    start,n = PLAN["fork_test"]
    rows, vectors = [], []
    for j in range(i,n,SHARDS):
        seed, turn = start+j, 8+j%5
        fork,pair,status = find_fork(policy,scorer,pool,seed,turn,growth)
        row = dict(seed=seed, turn=turn, status=status)
        rows.append(row)
        if fork is None:
            continue
        row.update(pair, opening=fork.opening, lobby=fork.state["lobby_races"])
        records = []
        bodies = []
        for rep in range(PLAN["planning_futures"]):
            future = PLAN["fork_planning_seed"] + 10*j + rep
            state,strict = alternate(fork,future)
            one,body = [],[]
            for action in (pair["invest"],pair["tempo"]):
                scorer.begin(future)
                branch = Branch(fork,scorer,action,True)
                try:
                    rollout(branch,pool,future,state=state,context=strict)
                except RecruitFinished:
                    pass
                one.append(scorer.last)
                body.append(branch.body)
            records.append(one)
            bodies.append(body)
        if any(v is None for rep in records+bodies for v in rep):
            row["status"] = "missing_planning_state"
            continue
        vectors.append(records)
        row.update(vector=len(vectors)-1,planning_bodies=bodies,ranks=[],bounds=[])
        for rep in range(PLAN["outcome_futures"]):
            future = PLAN["fork_label_seed"] + 10*j + rep
            state,strict = alternate(fork,future)
            rr,bb = [],[]
            for action in (pair["invest"],pair["tempo"]):
                scorer.begin(future)
                bound = []
                rank = rollout(Branch(fork,scorer,action,False),pool,future,state=state,context=strict,bounds=bound)
                rr.append(rank)
                bb.append(bound[0])
            row["ranks"].append(rr)
            row["bounds"].append(bb)
    dest = OUT/"forks"
    with (dest/f"features-{i}.npz").open("xb") as stream:
        np.savez_compressed(stream,x=np.asarray(vectors,np.float32))
    atomic_json(dest/f"cases-{i}.json",rows)


if __name__=="__main__":
    torch.set_num_threads(1)
    parallel("forks",__file__) if len(sys.argv)==1 else shard(int(sys.argv[1]))
