"""Isolate the incremental value of classifying intent versus protecting every eligible sale."""
from concurrent.futures import ThreadPoolExecutor
import json
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
from workflows.peer_sale_intent import BASE, BASE_SHA, DEVELOPMENT, FIT, RUNS, SaleIntent, panel, read

OUT = RUNS/"nus-peer-intent-ablation-20261120-1"


def shard(index):
    import torch
    torch.set_num_threads(1)
    proto, library = panel("development")
    plan = read(DEVELOPMENT/"plan.json")
    if proto.sha256() != plan["protocol"] or library != plan["library"]:
        raise ValueError("development environment changed")
    model = read(FIT/"complete.json")["model"]
    constant = {**model, "coef":[0.]*len(model["coef"]), "bias":30.}
    actor = SaleIntent(make_eval_policy(str(BASE/"model.pt"), protocol=proto, device="cuda:0")["policy"], constant, True)
    pool = MirrorSnapshotPool(proto.mirror_snaps,sample=proto.mirror_sample)
    pairs = []
    audit = []
    for record, observation in zip(read(DEVELOPMENT/f"panel-{index}.json"),
                                   read(DEVELOPMENT/f"events-{index}.json"), strict=True):
        parent, learned = record["parent"], record["candidate"]
        seed = parent["seed"]
        if learned["seed"] != seed or observation["seed"] != seed:
            raise ValueError("inconsistent paired input")
        actual = trial(actor,pool,seed,event_turns=True)
        actual = json.loads(json.dumps(actual, allow_nan=False))
        changes = [e for e in actor.events if e["changed"]]
        if actual["lobby"] != parent["lobby"] or bool(changes) != bool(observation["events"]):
            raise ValueError("unmatched field or intervention")
        if not changes and actual != parent:
            raise ValueError("unexpected divergence on a game with no eligible sale")
        pairs.append(dict(parent=actual, candidate=learned))
        audit.append(dict(seed=seed, unconditional_changes=len(changes), learned_differs=actual!=learned))
    atomic_json(OUT/f"panel-{index}.json",pairs)
    atomic_json(OUT/f"audit-{index}.json",audit)


def main():
    source = source_identity(ROOT)
    plan = read(DEVELOPMENT/"plan.json")
    if source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"] != BASE_SHA or plan["model"] != BASE_SHA:
        raise ValueError("wrong model or release")
    OUT.mkdir(exist_ok=False)
    inputs = [file_identity(DEVELOPMENT/f"{stem}-{i}.json") for stem in ("panel","events") for i in range(8)]
    atomic_json(OUT/"plan.json",dict(source=source,original_plan=file_identity(DEVELOPMENT/"plan.json"),
        inputs=inputs,field=plan,fit=file_identity(FIT/"complete.json"),job=os.environ["SLURM_JOB_ID"]))
    def launch(i):
        with (OUT/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(launch, range(8)))
    result = compare(OUT,plan["seed"],plan["n"])
    audits = [r for i in range(8) for r in read(OUT/f"audit-{i}.json")]
    result.update(affected=sum(r["unconditional_changes"]>0 for r in audits),
                  divergent=sum(r["learned_differs"] for r in audits))
    atomic_json(OUT/"complete.json",result)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        shard(int(sys.argv[1]))
    else:
        main()
