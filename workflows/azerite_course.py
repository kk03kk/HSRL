"""D-146 classroom distillation into a standard, PPO-loadable entity network."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hsbg_engine
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.probe_four_growth_curves import (
    ABSORB, AZ, BAL, BRANN, KELP, SURF, WILD, card_ids, item_card, make, own,
)
from workflows import probe_frozen_late_growth as probe
from workflows import quilboar_study as quill

TRAIN_CASES = ("elem_budget", "elem_offer", "elem_no_double", "elem_no_brann",
               "quill_ready", "quill_cold_search", "natural")
EXAM_CASES = ("elem_budget", "elem_offer", "elem_budget_lean", "quill_ready",
              "quill_cold_search", "mech_double", "dragon_poet")
WORKERS = 8
INVEST, INVEST_SHA = probe.MODELS["investment"]


def opening(seed, case):
    # Permute public board slots to defeat demonstrations based on positions.
    state = json.loads(make(seed, case).state_json())
    np.random.default_rng(seed + 137).shuffle(state["players"][0]["board"])
    return hsbg_engine.Game.from_json(json.dumps(state))


def actionable(game, mask):
    """Equivalent legal choices for one mechanism decision, or no advice."""
    ids = card_ids()
    p = own(game)
    board = p["board"]
    present = {m["card"] for m in board}
    missing_az = ids[AZ] not in present
    hand = [item_card(h) for h in p["hand"]]
    if p["pending"]:
        target = p["pending"][-1].get("Target", {})
        if not target:
            return []
        cont = target["cont"]
        options = target["options"]
        by_eid = {m["eid"]: m for m in board}
        if "Activate" in cont and by_eid.get(cont["Activate"]["eid"], {}).get("card") == ids[KELP]:
            return [30+i for i,e in enumerate(options) if by_eid.get(e, {}).get("card") == ids[SURF] and mask[30+i]]
        if "HandSpell" in cont:
            spell = next((h for h in p["hand"] if h["hid"] == cont["HandSpell"]["hid"]), None)
            if spell and item_card(spell) == ids[ABSORB]:
                legal = [30+i for i,e in enumerate(options) if e in by_eid and mask[30+i]]
                wildfire = [a for a in legal if by_eid[options[a-30]]["card"] == ids[WILD]]
                return wildfire or legal
        return []
    if not {ids[KELP], ids[SURF]}.issubset(present):
        return []
    if missing_az:
        if ids[AZ] in hand:
            plays = [a for a in range(46,126) if mask[a] and hand[(a-46)//8] == ids[AZ]]
            if plays:
                return plays
            core = {ids[k] for k in (KELP,SURF,BAL,BRANN,WILD)}
            return [16+i for i,m in enumerate(board) if m["card"] not in core and mask[16+i]]
        return [8+i for i,s in enumerate(p["shop"]) if mask[8+i] and item_card(s) == ids[AZ]]
    casts = [a for a in range(46,126) if mask[a] and hand[(a-46)//8] == ids[ABSORB]]
    if casts and any("Minion" in s["item"] for s in p["shop"]):
        return casts
    return [23+i for i,m in enumerate(board) if m["card"] == ids[KELP] and mask[23+i]]


class Examples:
    def __init__(self, evaluator, rng, *, game=None, role):
        self.original = evaluator
        self.model = evaluator.model
        self.game = game
        self.role = role
        self.rng = rng
        self.rows = []

    def __call__(self, f, c, mask, seat):
        prior, value = self.original(f, c, mask, seat)
        target = prior
        group = self.role
        if group == "element":
            choices = actionable(self.game, mask)
            if choices:
                target = np.zeros_like(prior)
                target[choices] = 1 / len(choices)
                # For interchangeable placements, demonstrate the base's preferred one.
                chosen = choices[int(prior[choices].argmax())]
                behavior = np.eye(1, len(prior), chosen, dtype=np.float32)[0]
            else:
                p = own(self.game)
                keep = {card_ids()[k] for k in (AZ,SURF,KELP,BRANN,BAL,WILD)}
                target = prior.copy()
                if not p["pending"]:
                    for i,m in enumerate(p["board"]):
                        if m["card"] in keep:
                            target[16+i] = 0
                target = target / target.sum() if target.sum() else prior
                behavior = target
                group = "retain"
        else:
            behavior = target
        if mask.sum() > 1 and (group == "element" or self.rng.random() < 0.40):
            self.rows.append((f.copy(), c.copy(), mask.copy(), target.copy(), group))
        return behavior, value


def protected(seed, case, agent, game):
    return probe.play(seed, case, agent, opening=game, normal_health=True)


def measure(turns, element):
    active = []
    for t in turns:
        if element:
            success = any(e["shop"] > 0 and e["gain"] > 0 for e in t["absorptions"])
        else:
            success = bool(t["choices"])
        if success and t["turn"] >= 13:
            active.append(t["turn"])
    return dict(body=[t["strength"] for t in turns], active=active,
                sustained=len(active) >= 3 and max(active, default=0) >= 16)


def generate(out, worker):
    proto, pool = quill.config("train")
    parent = quill.policy(quill.BASE, proto)
    pig = quill.policy(INVEST, proto)
    rows, scores = [], []
    rng = np.random.default_rng(810000 + worker)
    for seed in range(610000 + worker, 610032, WORKERS):
        for case in TRAIN_CASES:
            role = "natural" if case == "natural" else ("pig" if case.startswith("quill") else "element")
            agent = pig if role == "pig" else parent
            game = None if role == "natural" else opening(seed,case)
            recorder = Examples(agent.evaluator, rng, game=game, role=role)
            agent.evaluator = recorder
            try:
                if role == "natural":
                    quill.play(agent, pool, seed)
                else:
                    scores.append(dict(seed=seed,case=case,**measure(
                        protected(seed,case,agent,game),role=="element")))
            finally:
                agent.evaluator = recorder.original
            rows.extend(recorder.rows)
    f,c,m,q,k = zip(*rows)
    np.savez_compressed(out/f"examples-{worker}.npz",f=np.asarray(f,np.float32),
                        c=np.asarray(c,np.int32),m=np.asarray(m,bool),q=np.asarray(q,np.float32),k=k)
    atomic_json(out/f"teacher-{worker}.json",scores)


def fit(out):
    torch.manual_seed(7723)
    rng = np.random.default_rng(7723)
    parts = []
    for i in range(WORKERS):
        with np.load(out/f"examples-{i}.npz") as archive:
            parts.append({k:archive[k] for k in archive.files})
    data = {k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
    groups = {k:np.flatnonzero(data["k"] == k) for k in ("element","retain","pig","natural")}
    assert all(len(v) for v in groups.values())
    net, meta = load_checkpoint(quill.BASE, "cuda:0")
    assert meta["arch"] in ("entity","entity18m") and meta.get("norm_mean") is None and meta.get("norm_std") is None
    net.eval()
    teacher_sha = file_identity(INVEST)["sha256"]
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.00003, weight_decay=0.0)
    losses = []
    for _ in range(480):
        indexes = np.concatenate([rng.choice(groups[k], n) for k,n in
                                  (("element",24),("retain",16),("pig",12),("natural",12))])
        def batch(key, dtype):
            return torch.as_tensor(data[key][indexes],dtype=dtype,device="cuda:0")
        mask = batch("m", torch.bool)
        logits = net(batch("f",torch.float32),batch("c",torch.long),mask=mask)[0]
        log_probs = torch.log_softmax(logits.masked_fill(~mask,-1e8),dim=-1)
        loss = -(batch("q",torch.float32)*log_probs).sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite distillation loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    ckpt = out/"candidate"/"model.pt"
    save_checkpoint(net,str(ckpt),meta=dict(meta,course="azerite-1",parent_sha=quill.PLAN["base_sha"],
                                            quill_teacher_sha=teacher_sha))
    reload_net,_ = load_checkpoint(str(ckpt),"cuda:0")
    with torch.no_grad():
        inputs = (batch("f",torch.float32),batch("c",torch.long))
        assert torch.equal(net(*inputs,mask=mask)[0],reload_net.eval()(*inputs,mask=mask)[0])
    atomic_json(out/"fit.json",dict(steps=len(losses),groups={k:len(v) for k,v in groups.items()},
                                    start=losses[:5],end=losses[-5:],model=file_identity(ckpt)))


def exam(out, worker):
    proto,_ = quill.config("hold")
    parent = quill.policy(quill.BASE,proto)
    student = quill.policy(out/"candidate"/"model.pt",proto)
    pig = quill.policy(INVEST,proto)
    rows = []
    for seed in range(620000+worker,620024,WORKERS):
        for case in EXAM_CASES:
            for label,agent in (("parent",parent),("student",student),("pig",pig)):
                if label == "pig" and not case.startswith("quill"):
                    continue
                turns = protected(seed,case,agent,opening(seed,case))
                rows.append(dict(seed=seed,case=case,model=label,**measure(turns,case.startswith("elem"))))
    atomic_json(out/f"exam-{worker}.json",rows)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    src = source_identity(ROOT)
    assert not src["tracked_dirty"] and quill.engine_info()["source_hash"] == quill.PLAN["engine"]
    assert file_identity(quill.BASE)["sha256"] == quill.PLAN["base_sha"]
    assert file_identity(INVEST)["sha256"] == INVEST_SHA
    for field in ("train","hold"):
        assert library_digest(Path("artifacts/libraries/human-turn-3702-20261118")/field) == quill.PLAN["libraries"][field]
    proto,_ = quill.config("hold")
    atomic_json(out/"plan.json",dict(source=src,engine=quill.engine_info(),protocol=proto.sha256(),
                job=os.environ["SLURM_JOB_ID"],training_seed=[610000,610031],exam_seed=[620000,620023],
                teacher=INVEST_SHA,parent=quill.PLAN["base_sha"],steps=480,exam_cases=EXAM_CASES,
                gate="full >=12/24 sustained, higher T18 median than parent; offer or lean >=6/24 sustained; quill warm >=12/24. Classroom pass only."))
    def stage(name):
        def worker(i):
            with (out/f"{name}-{i}.log").open("x") as log:
                subprocess.run([sys.executable,__file__,str(out),name,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(worker,range(WORKERS)))
    stage("generate")
    fit(out)
    stage("exam")
    rows = [row for i in range(WORKERS) for row in json.loads((out/f"exam-{i}.json").read_text())]
    summary = {}
    for case in EXAM_CASES:
        summary[case] = {}
        for label in (("parent","student","pig") if case.startswith("quill") else ("parent","student")):
            subset = [r for r in rows if r["case"]==case and r["model"]==label]
            assert sorted(r["seed"] for r in subset)==list(range(620000,620024))
            summary[case][label] = dict(sustained=sum(r["sustained"] for r in subset),
                p50=np.sort([r["body"] for r in subset],axis=0)[12].tolist())
    full = summary["elem_budget"]
    passed = (full["student"]["sustained"]>=12 and full["student"]["p50"][-1]>full["parent"]["p50"][-1]
        and max(summary[c]["student"]["sustained"] for c in ("elem_offer","elem_budget_lean"))>=6
        and summary["quill_ready"]["student"]["sustained"]>=12)
    atomic_json(out/"complete.json",dict(classroom_pass=passed,summary=summary,trajectories=len(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    parser.add_argument("stage",nargs="?",default="run",choices=("run","generate","exam"))
    parser.add_argument("worker",nargs="?",type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=="run":
        run(args.out)
    elif args.stage=="generate":
        generate(args.out,args.worker)
    else:
        exam(args.out,args.worker)
