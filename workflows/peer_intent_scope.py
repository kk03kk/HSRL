"""Direct, fresh-seed confirmation of conditional versus universal material protection."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare
from workflows.peer_ppo_panel import trial
from workflows.peer_sale_intent import SaleIntent, panel, read, RUNS

REGISTRY = Path("artifacts/policies/peer-sale-intent-v1-20261120")
POLICY = "91de333c0382a3a5a48b0d26598065467beeeed763f1ad0cae1656e7b9833114"
SEEDS = {"hold":383410000, "market":386410000}


def root(split):
    return RUNS/f"nus-peer-intent-scope-{split}-20261120-1"


def shard(split, i):
    import torch
    torch.set_num_threads(1)
    proto, library = panel(split)
    out = root(split)
    plan = read(out/"plan.json")
    if (proto.sha256(),library) != (plan["protocol"],plan["library"]):
        raise ValueError("field differed from frozen plan")
    pool = MirrorSnapshotPool(proto.mirror_snaps,sample=proto.mirror_sample)
    path = REGISTRY/read(REGISTRY/"policy.json")["parent"]["path"]
    fit = read(REGISTRY/"classifier.json")
    conditional = make_eval_policy("research-policy:"+str(REGISTRY),protocol=proto,device="cuda:0")["policy"]
    always = SaleIntent(make_eval_policy(str(path),protocol=proto,device="cuda:0")["policy"],
                        {**fit,"bias":30.,"coef":[0.]*len(fit["coef"])}, True)
    pairs, audit = [], []
    for seed in range(plan["seed"]+i,plan["seed"]+plan["n"],8):
        a = trial(conditional,pool,seed,event_turns=True)
        b = trial(always,pool,seed,event_turns=True)
        decisions = [(e["turn"],e["original"],e["executed"]) for e in always.events if e["changed"]]
        if a["lobby"] != b["lobby"] or (conditional.overrides == decisions and a != b):
            raise ValueError("unexplained paired divergence")
        pairs.append(dict(parent=a,candidate=b))
        audit.append(dict(seed=seed,conditional=conditional.overrides,unconditional=decisions,divergent=a!=b))
    atomic_json(out/f"panel-{i}.json",pairs)
    atomic_json(out/f"actions-{i}.json",audit)


def main(split):
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(REGISTRY/"policy.json")["sha256"] != POLICY:
        raise ValueError("unexpected release or incumbent")
    proto,library = panel(split)
    out = root(split)
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(source=source,policy=POLICY,engine=proto.engine,
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),live_pin=file_identity(ROOT/"config/live/model.json"),
        parent=read(REGISTRY/"policy.json")["parent"],classifier=file_identity(REGISTRY/"classifier.json"),
        protocol=proto.sha256(),library=library,n=512,seed=SEEDS[split],job=os.environ["SLURM_JOB_ID"],
        intervention="same first eligible sale, no intent threshold; direct against frozen registered V1"))
    def launch(i):
        with (out/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,split,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(launch, range(8)))
    result = compare(out,SEEDS[split],512)
    rows = [r for i in range(8) for r in read(out/f"actions-{i}.json")]
    result.update(divergent=sum(r["divergent"] for r in rows),
        passed=result["delta"]["utility"]["ci95"][0]>0 and result["delta"]["growth9to12"]["ci95"][0]>=0)
    atomic_json(out/"complete.json",result)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        shard(sys.argv[1],int(sys.argv[2]))
    else:
        main(sys.argv[1])
