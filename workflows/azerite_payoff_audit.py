"""Matched counterfactuals for a taught student's low T18 payoff."""
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows import azerite_course as course
from workflows.azerite_course import AZ,BAL,BRANN,KELP,SURF,WILD,card_ids,own
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity

SOURCE = Path("artifacts/runs/nus-azerite-course-20261119-2")
SHA = "0d1c7aadbb73e4d2f2983170000f090766a6c6c6bdbeb034b2749d97cd78acc9"
ARMS = ("parent","teacher","student","protect","shop","protect_shop")


class Intervention:
    def __init__(self, agent, game, arm):
        self.agent = agent
        self.game = game
        self.arm = arm
        self.turn = None
        self.interventions = Counter()
        self.tutor = course.Examples(agent,np.random.default_rng(0),game=game,role="element") if arm=="teacher" else None
        self.model = agent.model

    def __call__(self,f,c,mask,seat):
        if self.tutor:
            return self.tutor(f,c,mask,seat)
        prior,value = self.agent(f,c,mask,seat)
        if self.arm in ("parent","student"):
            return prior,value
        p = own(self.game)
        if p["pending"]:
            return prior,value
        if self.turn != self.game.turn:
            self.turn = self.game.turn
            self.rolls = 0
        target = prior.copy()
        ids = card_ids()
        if self.arm in ("protect","protect_shop"):
            # Restrict sales only for this diagnostic and only while the engine exists.
            core = {ids[k] for k in (AZ,SURF,KELP,BRANN,BAL)}
            if any(m["card"]==ids[AZ] for m in p["board"]):
                biggest = max((m["atk"]+m["hp"] for m in p["board"] if m["races"] & 8),default=0)
                for i,m in enumerate(p["board"]):
                    if m["card"] in core or (m["races"] & 8 and m["atk"]+m["hp"]==biggest):
                        target[16+i]=0
                if target.sum() == 0:
                    target = prior.copy()
                self.interventions["protect"] += int(target.argmax()!=prior.argmax())
        if self.arm in ("shop","protect_shop") and self.rolls<3 and mask[1]:
            # When the student intends to absorb from an unscaled shop, test paid refreshes.
            a = int(target.argmax())
            if (46<=a<126 and course.item_card(p["hand"][(a-46)//8])==ids[course.ABSORB]
                    and not any(s["item"].get("Minion",{}).get("races",0)&8 for s in p["shop"])
                    and any(m["card"]==ids[AZ] for m in p["board"])):
                self.rolls += 1
                self.interventions["roll"] += 1
                target = np.zeros_like(prior)
                target[1]=1
        return target/target.sum(),value


class Trace:
    def __init__(self, policy):
        self.policy = policy
        self.sales = []

    def reset(self):
        self.policy.reset()

    def decide(self,g,seat):
        decision = self.policy.decide(g,seat)
        if 16<=decision.action<23:
            m = own(g)["board"][decision.action-16]
            self.sales.append(dict(turn=g.turn,card=m["card"],size=m["atk"]+m["hp"],races=m["races"]))
        return decision


def worker(out,i):
    proto,_ = course.quill.config("hold")
    agents = {k:course.quill.policy(p,proto) for k,p in (("parent",course.quill.BASE),
              ("student",SOURCE/"candidate"/"model.pt"))}
    rows = []
    for seed in range(630000+i,630024,8):
        for arm in ARMS:
            g = course.opening(seed,"elem_budget")
            agent = agents["parent" if arm in ("parent","teacher") else "student"]
            intervention = Intervention(agent.evaluator,g,arm)
            agent.evaluator = intervention
            traced = Trace(agent)
            try:
                turns = course.protected(seed,"elem_budget",traced,g)
            finally:
                agent.evaluator = intervention.agent
            rows.append(dict(seed=seed,arm=arm,sales=traced.sales,interventions=intervention.interventions,
                             turns=turns,**course.measure(turns,True)))
    atomic_json(out/f"{i}.json",rows)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    assert file_identity(SOURCE/"candidate"/"model.pt")["sha256"]==SHA
    assert file_identity(course.quill.BASE)["sha256"]==course.quill.PLAN["base_sha"]
    assert course.quill.engine_info()["source_hash"]==course.quill.PLAN["engine"]
    src = source_identity(course.ROOT)
    assert not src["tracked_dirty"]
    atomic_json(out/"plan.json",dict(source=src,job=os.environ["SLURM_JOB_ID"],arms=ARMS,sha=SHA,
                                    seeds=[630000,630023],parent=course.quill.PLAN["base_sha"]))
    def shard(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(out),str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(shard,range(8)))
    all_rows = [r for i in range(8) for r in json.loads((out/f"{i}.json").read_bytes())]
    summary = {}
    for arm in ARMS:
        rows = [r for r in all_rows if r["arm"]==arm]
        assert sorted(r["seed"] for r in rows)==list(range(630000,630024))
        metrics = []
        for row in rows:
            turns=row["turns"]
            events=[e for t in turns for e in t["absorptions"]]
            metrics.append(dict(body=turns[-1]["strength"],shop=turns[-1]["shop"],n_abs=len(events),
                gross_gain=sum(e["gain"] for e in events),sold_element=sum(s["size"] for s in row["sales"] if s["races"]&8),
                retain=sum(all(card_ids()[k] in t["board"] for k in (AZ,KELP,SURF)) for t in turns),
                protect=row["interventions"].get("protect",0),roll=row["interventions"].get("roll",0)))
        summary[arm]=dict(sustained=sum(r["sustained"] for r in rows),
                          medians={k:sorted(m[k] for m in metrics)[12] for k in metrics[0]})
    atomic_json(out/"complete.json",dict(summary=summary,trajectories=len(all_rows)))


if __name__=="__main__":
    torch.set_num_threads(1)
    out=Path(sys.argv[1])
    main(out) if len(sys.argv)==2 else worker(out,int(sys.argv[2]))
