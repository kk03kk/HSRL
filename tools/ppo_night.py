"""Restartable Slurm controller for bounded, staged PPO and sealed evaluation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import taverngym

UTILITY = np.array([1, .3, 0, -.05, -.1, -.15, -.2, -.25])
RECIPES = {"hybrid": 7.5e-6, "stronger": 7.5e-6, "human-late": 7.5e-6,
           "script": 7.5e-6, "safe": 3.75e-6}
TRACK = {
    "mixed": (240000000, 512), "human": (240010000, 768), "bank": (240020000, 512),
}
FINAL = {
    "original": (250000000, 2048), "mixed": (250010000, 2048),
    "human": (250020000, 2048), "fresh": (250030000, 2048),
    "bank": (250030000, 2048),
}
BASE_SHA = "e3d8fa7bcd1484de5b4116ce7be369ac3a52423a8bdf4daa9a1b0af5dcd90220"
BC_ENGINE = "9432249ae4888d0f"
ENGINE = "b37403fe50d5b72b"


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


class Night:
    def __init__(self, root, assets, bc, parent):
        self.root, self.assets, self.bc, self.parent = map(Path, (root, assets, bc, parent))
        self.init = self.bc / "admission-1/ppo-init.pt"
        self.parity = self.assets / "parity-coverage.json"
        self.commit = (self.root / "source_commit").read_text().strip()
        self.workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "10"))
        self.path = self.root / "state.json"

    def event(self, kind, **details):
        row = {"epoch": time.time(), "job": os.environ.get("SLURM_JOB_ID"), "event": kind, **details}
        with (self.root / "events.jsonl").open("a") as f:
            print(json.dumps(row, separators=(",", ":")), file=f)
        print(json.dumps(row), flush=True)

    def verify(self):
        from taverngym.warband import Fields
        check(taverngym.build_info()["source_hash"] == ENGINE, "Invalid native engine")
        parity = read(self.parity)
        check(parity["engine"] == ENGINE and parity["identical_trajectory"], "Invalid parity proof")
        check(digest(self.parent) == BASE_SHA, "Invalid frozen parent SHA-256")
        initial = read(self.init.with_name("ppo-admission.json"))
        check(initial["approved"] and initial["sha256"] == digest(self.init)
              and initial["native_source"] == BC_ENGINE, "Invalid BC initialization")
        info = torch.load(self.parent, weights_only=False, map_location="cpu")
        check(info["iteration"] == 288 and info["meta"]["source"] == BC_ENGINE
              and info["meta"]["initial_sha256"] == initial["sha256"], "Unexpected parent identity")
        check(np.array_equal(info["meta"]["placement_utility"],UTILITY), "Wrong reward")
        opt = info["optimizer"]
        steps = [float(v["step"]) for v in opt["state"].values()]
        check(len(steps)>100 and min(steps)>0 and len(info["rollout_history"])==288,
              "The original Adam state or training history is missing")
        receipt = {"parent": {"iteration":288,"sha256":BASE_SHA,"optimizer_states":len(steps),
                              "optimizer_step_min":min(steps)}, "engine":ENGINE, "commit":self.commit,
                   "native_parity":parity, "plans":{}}
        for name in ("hybrid","human-late","stronger","script","safe","development"):
            fields = Fields(self.root/f"plans/{name}.json")
            receipt["plans"][name] = fields.identity
        atomic(self.root/"preflight-receipt.json", receipt)

    def call(self, entry, args, log=None, timeout=None):
        command = [sys.executable, "-u", "tools/run_frozen_python.py", entry, *map(str,args)]
        if log is None:
            return subprocess.run(command, timeout=timeout).returncode
        with log.open("a") as stream:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode

    def identify(self, path):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        return {"path":str(path), "sha256":digest(path), "engine":ck["meta"]["source"],
                "iteration":ck["iteration"]}

    def branch(self, state, recipe, anchor=None):
        number = state["branches"]
        state["branches"] += 1
        state["active"] = {"name":f"{number:03d}-{recipe}", "recipe":recipe,
                           "anchor":anchor or state["accepted"]["checkpoint"],
                           "first":15400000 + number*4000000, "seed":421997+number}
        state["tested"] = sorted(set(state["tested"]+[recipe]))
        state["flat"] = 0
        atomic(self.path,state)
        self.event("branch_open", **state["active"])

    def train(self, branch, target, budget, *, games=1024, cap=2048, snapshot=16):
        out = self.root / "branches" / branch["name"]
        anchor = branch["anchor"]
        count = self.current(branch)
        if count >= target:
            return 0
        out.mkdir(parents=True,exist_ok=True)
        args = ["train","--out",out,"--init",self.init,"--admission",self.init.with_name("ppo-admission.json"),
                "--parent",anchor["path"],"--parent-sha256",anchor["sha256"],"--parent-engine",anchor["engine"],
                "--bc-engine",BC_ENGINE,"--parent-parity",self.parity,"--code-commit",self.commit,
                "--arena-plan",self.root/f"plans/{branch['recipe']}.json","--first-seed",branch["first"],
                "--seed",branch["seed"],"--games",games,"--concurrent",games,"--workers",self.workers,
                "--minibatch",2048,"--microbatch",1024,"--policy-batch-size",512,"--resident-rollout",
                "--expected-gpu", "NVIDIA H100 NVL" if games==1024 else "NVIDIA H100 NVL MIG 3g.47gb",
                "--bc-kl-scale",0,"--kl-check-every",4,"--target-kl",.02,"--rollback-kl",.05,
                "--continuation-lr",RECIPES[branch["recipe"]],"--iterations",cap,
                "--stop-after",target,"--snapshot-every",snapshot,"--time-budget-seconds",int(budget)]
        if (out/"latest.pt").exists():
            args.append("--resume")
        self.event("launch_train", branch=branch["name"], start=count, target=target, budget=int(budget), games=games)
        return self.call("python/taverngym/ppo_train.py",args, out/"trainer.log", budget+100)

    def current(self, branch):
        p = self.root/"branches"/branch["name"]/"latest.pt"
        return torch.load(p,map_location="cpu",weights_only=False)["iteration"] if p.exists() else 0

    def freeze(self, branch, iteration):
        out = self.root/"branches"/branch["name"]
        path = out/f"candidate-{iteration:04d}.pt"
        if not path.exists():
            os.link(out/"latest.pt", path)
        return self.identify(path)

    def evaluation(self, name, checkpoint, field, final=False):
        panels, scope = (FINAL,"final") if final else (TRACK,"dev")
        first,n = panels[field]
        dest = self.root/scope/f"{name}-{field}.jsonl"
        receipt = dest.with_suffix(".json")
        if receipt.is_file() and dest.is_file():
            data = read(receipt)
            if data["checkpoint_sha256"] == checkpoint["sha256"] and data["first_seed"] == first:
                return dest
            raise RuntimeError("A result exists for a different checkpoint or panel")
        spec = read(self.root/f"plans/{'independent' if final else 'development'}.json")
        plan_file = self.root/"plans"/f"{scope}-{field}.json"
        atomic(plan_file, {field:spec[field]})
        args=[self.root,"evaluate","holdout" if final else "development",field,plan_file,
              checkpoint["path"],n,first,dest]
        code = self.call("tools/warband_research.py",args,timeout=420)
        if code:
            raise RuntimeError(f"Evaluation failed for {name}/{field}")
        return dest

    def screen(self, name, checkpoint):
        return {f:str(self.evaluation(name,checkpoint,f)) for f in TRACK}

    def refs(self):
        return {"checkpoint":{"path":str(self.parent),"sha256":BASE_SHA,"engine":BC_ENGINE,"iteration":288},
                "name":"parent","recipe":"hybrid"}

    def run_segment(self, number, seconds):
        check((self.root/"preflight-receipt.json").exists(),"CPU preflight has not succeeded")
        check((self.root/"gpu-preflight.json").exists(),"GPU preflight has not succeeded")
        with (self.root/"controller.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                self.segment(number,seconds)
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

    def segment(self, number, seconds):
        self.event("segment_start",segment=number)
        deadline = time.time()+seconds
        if self.path.exists():
            state = read(self.path)
        else:
            state={"started":time.time(),"accepted":self.refs(),"branches":0,
                   "tested":[],"flat":0,"failures":0,"streak":0,"reviewed":{}}
            self.branch(state,"hybrid")
        deadline = min(deadline,state["started"]+9*3600)
        if state.get("stop"):
            self.event("segment_skipped",reason=state["stop"])
            return
        parent_rows = self.screen("parent",self.refs()["checkpoint"])
        while deadline-time.time() >= 250:
            branch = state["active"]
            key = branch["name"]
            reviewed = state["reviewed"].get(key,0)
            current = self.current(branch)
            target = (reviewed//32+1)*32
            budget = deadline-time.time()-165
            if current < target:
                code = None
                try:
                    code = self.train(branch,target,budget)
                except subprocess.TimeoutExpired:
                    self.event("training_process_timeout", branch=key)
                current = self.current(branch)
                if code is None or code != 0:
                    out=self.root/"branches"/key
                    tail = tail_text(out/"trainer.log")
                    stale = (out/f"failure-{current+1}.json").is_file()
                    deterministic = stale or any(x in tail for x in
                        ("Iteration rejected", "Same-weight batch replay", "Nonfinite"))
                    self.event("training_error",branch=key,iteration=current,code=code,
                               deterministic=deterministic)
                    if not deterministic:
                        # Later scheduled segments resume from the last atomic checkpoint.
                        return
                    state["failures"]+=1
                    if branch["recipe"]=="safe" or state["failures"]>=3:
                        state["stop"]="repeated_engine_or_numeric_failure"
                        atomic(self.path,state)
                        break
                    self.branch(state,"safe")
                    continue
            if deadline-time.time() < 130 or (current < target and (number != 3 or current-reviewed < 8)):
                break
            name = f"{key}-{current:04d}"
            checkpoint = self.freeze(branch,current)
            try:
                rows = self.screen(name,checkpoint)
                reference = self.screen(state["accepted"]["name"],state["accepted"]["checkpoint"])
                comparison = {f: diff(rows[f], reference[f]) for f in TRACK}
                origin = {f: diff(rows[f], parent_rows[f]) for f in TRACK}
            except (subprocess.TimeoutExpired,RuntimeError) as exc:
                # An unavailable or invalid development panel never promotes a candidate.
                state["reviewed"][key] = current
                state["monitor_failures"] = state.get("monitor_failures",0)+1
                if state["monitor_failures"]>=2 and branch["recipe"]=="safe":
                    state["stop"]="no_valid_development_panel"
                self.event("monitor_failed",branch=key,iteration=current,error=str(exc))
                atomic(self.path,state)
                if state.get("stop"):
                    break
                if branch["recipe"]!="safe":
                    self.branch(state,"safe")
                continue
            state["monitor_failures"]=0
            decision, severe, score = assess(comparison,origin)
            state["reviewed"][key]=current
            state["streak"] = state["streak"]+1 if (origin["human"]["utility"]["lo"]>0
                               and origin["mixed"]["rank"]["hi"]<=.15
                               and origin["mixed"]["top4"]["lo"]>=-.03) else 0
            if decision:
                state["accepted"]={"checkpoint":checkpoint,"name":name,"recipe":branch["recipe"]}
                state["flat"]=0
            else:
                state["flat"]+=1
            self.event("development_gate",branch=key,iteration=current,accept=decision,
                       severe=severe,score=score,versus_kept=comparison,versus_288=origin)
            atomic(self.path,state)
            if severe:
                state["failures"]+=1
                if branch["recipe"]=="safe" and state["failures"]>=3:
                    state["stop"]="sustained_clear_regression"
                    atomic(self.path,state)
                    break
                self.branch(state,"safe")
            elif branch["recipe"]=="hybrid" and state["streak"]>=2 and "stronger" not in state["tested"] and replica_support(self.root):
                self.branch(state,"stronger")
            elif state["flat"]>=2 and branch["recipe"] in ("hybrid","stronger") and "human-late" not in state["tested"]:
                self.branch(state,"human-late")
            elif state["flat"]>=2 and branch["recipe"] in ("human-late","stronger"):
                self.branch(state,state["accepted"]["recipe"]
                            if state["accepted"]["recipe"]!=branch["recipe"] else "hybrid")
            elif state["flat"]>=3 and branch["recipe"]=="safe" and state["accepted"]["recipe"]!="safe":
                self.branch(state,state["accepted"]["recipe"])
            if current < target:
                break
        self.event("segment_end", segment=number, accepted=state["accepted"], stop=state.get("stop"))

    def smoke(self):
        check((self.root/"preflight-receipt.json").exists(), "CPU preflight has not succeeded")
        branch={"name":"gpu-preflight","recipe":"hybrid","anchor":self.refs()["checkpoint"],
                "first":184500000,"seed":413909}
        check(not self.train(branch,1,360,cap=2,snapshot=1), "GPU smoke first iteration failed")
        check(not self.train(branch,2,360,cap=2,snapshot=1), "GPU smoke resume failed")
        out=self.root/"branches"/branch["name"]
        r=read(out/"iteration-002.json")
        score=r["optimization"]["before"]["max_abs_log_ratio"]
        check(r["sampling"]["terminal_events"]==1024 and score<1e-4
              and r["sampling"]["fields"]["d"]["warband"]["frames"]>0
              and read(out/"summary.json")["from_iteration"]==1, "GPU smoke parity mismatch")
        first=torch.load(out/"snapshot-001.pt",map_location="cpu",weights_only=False)
        second=torch.load(out/"latest.pt",map_location="cpu",weights_only=False)
        info=read(self.root/"preflight-receipt.json")["parent"]
        minimum=min(float(v["step"]) for v in first["optimizer"]["state"].values())
        check(minimum>info["optimizer_step_min"]
              and second["next_game_seed"]-first["next_game_seed"]==1024,
              "Adam or game seed did not resume")
        checkpoint=self.identify(out/"latest.pt")
        hop={"name":"gpu-preflight-hop","recipe":"safe","anchor":checkpoint,
             "first":184600000,"seed":413910}
        check(not self.train(hop,1,360,cap=1,snapshot=1), "GPU smoke recipe transition failed")
        switched=torch.load(self.root/"branches/gpu-preflight-hop/latest.pt",
                            map_location="cpu",weights_only=False)
        check(switched["meta"]["branch"]["parent_sha256"]==checkpoint["sha256"]
              and min(float(v["step"]) for v in switched["optimizer"]["state"].values()) >
              min(float(v["step"]) for v in second["optimizer"]["state"].values()),
              "New parent or Adam state lost in recipe transition")
        atomic(self.root/"gpu-preflight.json",{"gpu":torch.cuda.get_device_name(),"iteration":2,
               "engine":ENGINE,"probability_replay_error":score,"adam_steps_min":minimum,
               "frames":r["sampling"]["fields"]["d"]["warband"]["frames"],
               "confirmed_resume":True,"confirmed_recipe_switch":True})

    def control(self, lane):
        recipe="script" if lane==0 else "hybrid"
        branch={"name":f"replica-{recipe}", "recipe":recipe,
                "anchor":self.refs()["checkpoint"],"first":185000000,"seed":623619}
        try:
            code=self.train(branch,32,5700,games=512,cap=32,snapshot=8)
        except subprocess.TimeoutExpired:
            return
        if code or self.current(branch)<32:
            self.event("replica_incomplete",lane=lane, code=code)
            return
        chk=self.freeze(branch,self.current(branch))
        rows=self.screen(f"replica-{recipe}",chk)
        atomic(self.root/f"replica-{recipe}.json",{"checkpoint":chk, "panels":rows})

    def seal(self):
        state=read(self.path) if self.path.exists() else {}
        candidates={"parent":self.refs()["checkpoint"],
                    "selected":state.get("accepted",self.refs())["checkpoint"]}
        for recipe in ("script","hybrid"):
            p=self.root/f"replica-{recipe}.json"
            if p.exists():
                candidates[f"replica-{recipe}"]=read(p)["checkpoint"]
        for name, checkpoint in candidates.items():
            check(digest(checkpoint["path"])==checkpoint["sha256"],f"Digest mismatch {name}")
        fresh=(self.root/"fresh-bank-receipt.json").is_file()
        fields=["original","mixed","human","bank"]+(["fresh"] if fresh else [])
        receipt={"candidates":candidates,"fields":fields,
                 "state":state, "created":time.time(), "engine":ENGINE}
        atomic(self.root/"final-selection.json",receipt)
        self.event("final_selection", checkpoints=candidates, fresh_bank=fresh)

    def final_field(self, slot):
        receipt = read(self.root/"final-selection.json")
        if slot >= len(receipt["fields"]):
            return
        field=receipt["fields"][slot]
        for name,checkpoint in receipt["candidates"].items():
            for attempt in range(2):
                try:
                    self.evaluation(name,checkpoint,field,final=True)
                    break
                except (RuntimeError,subprocess.TimeoutExpired) as exc:
                    self.event("final_field_error",field=field,model=name,attempt=attempt,error=str(exc))


def tail_text(path):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(path.stat().st_size-3000,0))
        return f.read().decode(errors="replace")


def load_games(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def diff(path_a,path_b, bootstrap=False):
    a,b=load_games(path_a),load_games(path_b)
    check(len(a)==len(b) and all((x["seed"],x["seat"],x["lobby"])==(y["seed"],y["seat"],y["lobby"])
          for x,y in zip(a,b,strict=True)), "Panel pairing differs")
    x,y=(np.array([g["rank"] for g in rows]) for rows in (a,b))
    result={}
    rng=np.random.default_rng(814139)
    differences = {
        "rank":x-y,"top4":(x<=4).astype(float)-(y<=4),"top2":(x<=2).astype(float)-(y<=2),
        "first":(x==1).astype(float)-(y==1),"utility":UTILITY[x-1]-UTILITY[y-1],
        "formation":np.array([float(bool(g["formations"])) for g in a]) - np.array([bool(g["formations"]) for g in b]),
        "freeze":np.array([g["freeze_adjacency"] for g in a])-np.array([g["freeze_adjacency"] for g in b]),
    }
    for key,ds in differences.items():
        mean=float(ds.mean())
        if bootstrap:
            samples=np.concatenate([ds[rng.integers(len(ds),size=(64,len(ds)))].mean(axis=1)
                                    for _ in range(96)])
            lo,hi=map(float,np.quantile(samples,[.025,.975]))
        else:
            spread=1.96*float(ds.std(ddof=1))/math.sqrt(len(ds))
            lo,hi=mean-spread,mean+spread
        result[key]={"delta":mean,"lo":lo,"hi":hi}
    return result


def assess(comparison, origin):
    s,h,b=(comparison[f] for f in ("mixed","human","bank"))
    g=origin["mixed"]
    score=.75*h["utility"]["delta"]+.25*b["utility"]["delta"]
    accepted=(score>=.003 and h["utility"]["delta"]>=-.004 and
              g["rank"]["delta"]<=.075 and g["top4"]["delta"]>=-.02 and
              h["freeze"]["delta"]<=.5)
    severe=(s["rank"]["lo"]>.15 or s["top4"]["hi"]<-.03 or h["utility"]["hi"]<-.02)
    return accepted and not severe, severe, score


def replica_support(root):
    paths=[root/f"replica-{r}.json" for r in ("script","hybrid")]
    if not all(p.exists() for p in paths):
        return False
    control,hybrid=(read(p)["panels"] for p in paths)
    return diff(hybrid["human"],control["human"])["utility"]["delta"]>0


def report(root):
    sealed=read(root/"final-selection.json")
    checks={}
    for name,checkpoint in sealed["candidates"].items():
        fields={}
        for field in sealed["fields"]:
            path=root/"final"/f"{name}-{field}.jsonl"
            base=root/"final"/f"parent-{field}.jsonl"
            ref=root/"final"/f"replica-script-{field}.jsonl"
            if not path.is_file() or not path.with_suffix(".json").is_file():
                fields[field]={"unavailable":True}
                continue
            receipt=read(path.with_suffix(".json"))
            check(receipt["checkpoint_sha256"]==checkpoint["sha256"],"Final checkpoint digest differs")
            fields[field]={"stats":receipt["stats"],
                           "versus_288":diff(path,base,True) if base.exists() else None,
                           "versus_script":diff(path,ref,True) if ref.exists() else None}
        checks[name]=fields
    branches={}
    for path in (root/"branches").iterdir():
        iterations=[read(f) for f in sorted(path.glob("iteration-*.json"))]
        if iterations:
            branches[path.name]={"iterations":len(iterations),
                                 "lobbies":sum(x["sampling"]["terminal_events"] for x in iterations),
                                 "decisions":sum(x["sampling"]["decisions"] for x in iterations),
                                 "last":iterations[-1]["iteration"],
                                 "max_kl":max(x["optimization"]["after"]["kl"] for x in iterations),
                                 "seconds":sum(x["sampling"]["rollout_seconds"]+x["sampling"]["update_seconds"]+
                                      x["sampling"]["checkpoint_seconds"] for x in iterations)}
    chosen=checks.get("selected",{})
    h=chosen.get("human",{}).get("versus_288")
    c=chosen.get("human",{}).get("versus_script")
    s=chosen.get("mixed",{}).get("versus_288")
    complete=all(not f.get("unavailable") for panel in checks.values() for f in panel.values())
    passes=bool(complete and h and c and s and h["utility"]["lo"]>0 and c["utility"]["lo"]>0
                and s["rank"]["hi"]<=.15 and s["top4"]["lo"]>=-.03)
    result={"selection":sealed,"checks":checks,"training":branches,"complete":complete,"promotion":passes}
    atomic(root/"RESULTS.json",result)
    lines=["# 37.0.3 战团 PPO 九小时训练", "",
           f"独立验收数据齐全。{complete}。晋级结果。{'通过' if passes else '未通过'}。","",
           f"开发选择的检查点为 `{sealed['candidates']['selected']['path']}`。正式基线为第 288 轮。","",
           "## 训练", "",
           "| 分支 | 落盘轮数 | 有效局 | 决策 | 核心训练小时 | KL 峰值 |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,r in branches.items():
        lines.append(f"| {name} | {r['iterations']} | {r['lobbies']:,} | {r['decisions']:,} | {r['seconds']/3600:.2f} | {r['max_kl']:.5f} |")
    lines += ["","## 留出评测","",
              "| 模型 | 大厅 | 局数 | 名次 | 前四 | 前二 | 吃鸡 | 终盘齐备 | 效用 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,fields in checks.items():
        for field,row in fields.items():
            if row.get("unavailable"):
                lines.append(f"| {name} | {field} | — | — | — | — | — | — | — |")
                continue
            a=row["stats"]
            lines.append(f"| {name} | {field} | {a['games']:,} | {a['rank']:.3f} | {a['top4']:.2%} | {a['top2']:.2%} | {a['first']:.2%} | {a['formation_rate']:.2%} | {a['utility']:+.4f} |")
    lines += ["","按种子配对的置信区间、开发选择、检查点摘要与每回合数据见 RESULTS.json。"
              "真人使用封存旧来源的全新对局种子。fresh 为新生成且未参与训练的脚本供体。"]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    for key in ("root","assets","bc","parent"):
        p.add_argument(key,type=Path)
    p.add_argument("mode",choices=("verify","smoke","segment","control","seal","evaluate","report"))
    p.add_argument("number",type=int,nargs="?",default=0)
    p.add_argument("seconds",type=int,nargs="?",default=10680)
    args=p.parse_args()
    torch.set_num_threads(2)
    n=Night(args.root,args.assets,args.bc,args.parent)
    {"verify":n.verify,"smoke":n.smoke,"segment":lambda:n.run_segment(args.number,args.seconds),
     "control":lambda:n.control(args.number),"seal":n.seal,"evaluate":lambda:n.final_field(args.number),
     "report":lambda:report(args.root)}[args.mode]()

if __name__=="__main__":
    main()
