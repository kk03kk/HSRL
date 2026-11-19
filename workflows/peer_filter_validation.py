"""Test the frozen demonstration selector on a third unused future per opening."""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows import peer_distillation as D


def validate(i,out):
    _,pool,policies=D.sources("train")
    records=D.read(D.OUT/f"returns-{i}.json")
    rows=[]
    for r in records:
        if r.get("missed"):
            continue
        j=r["j"]
        fork=D.opening(policies[r["prefix"]],pool,D.TRAIN_SEED+j,r["turn"])
        if fork.opening!=r["opening"]:
            raise ValueError(f"opening changed {j}")
        seed=214180000+j
        state,context=D.alternate(fork,seed)
        ranks=[D.rollout(policies[k],pool,seed,state=state,context=context) for k in ("base","peer")]
        rows.append(dict(j=j,chosen=r["chosen"],ranks=ranks))
    D.atomic_json(out/f"blind-{i}.json",rows)


if __name__=="__main__":
    torch.set_num_threads(1)
    out=Path(os.environ["HSAI_FILTER_TEST"])
    if len(sys.argv)>1:
        validate(int(sys.argv[1]),out)
    else:
        out.mkdir(parents=True,exist_ok=False)
        def run(i):
            with (out/f"worker-{i}.log").open("x") as log:
                subprocess.run([sys.executable,__file__,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(D.SHARDS) as e:
            list(e.map(run,range(D.SHARDS)))
        D.atomic_json(out/"complete.json",dict(seed=214180000,job=os.environ["SLURM_JOB_ID"]))
