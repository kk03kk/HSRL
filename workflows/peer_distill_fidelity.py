"""Measure teacher imitation on unseen teacher- and base-visited decisions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from workflows import peer_distillation as D


@torch.inference_mode()
def measure(i,out):
    _,pool,policies=D.sources("train")
    base=policies["base"].evaluator
    peer=policies["peer"].evaluator
    students={k:D.load_checkpoint(str(D.TRAINED/k/"model.pt"),"cuda:0")[0].eval() for k in D.ARMS}
    rng=np.random.default_rng(146571+i)
    rows=[]
    for idx in range(i,64,8):
        for source,policy in policies.items():
            sampler=D.Samples(policy)
            seed=212860000+idx
            D.rollout(sampler,pool,seed)
            for label,lo,hi in (("early",1,5),("prep",6,8),("middle",9,11),("late",12,99)):
                available=[r for r in sampler.rows if lo<=r[4]<=hi]
                if not available:
                    continue
                sample=[available[n] for n in rng.choice(len(available),min(32,len(available)),replace=False)]
                f,c,m,_,_=zip(*sample)
                t=lambda a:torch.as_tensor(np.array(a),device="cuda:0")
                f,c,m=t(f),t(c).long(),t(m)
                teacher=peer.net(f,c,m)[0].masked_fill(~m,-1e8).log_softmax(-1)
                if base.mean is not None:
                    f=f-t(base.mean)
                if base.std is not None:
                    f=f/t(base.std)
                data={}
                for k,net in {"base":base.model,**students}.items():
                    pred=net(f,c,mask=m)[0].masked_fill(~m,-1e8).log_softmax(-1)
                    kl=(teacher.exp()*(teacher-pred)).sum(-1)
                    data[k]=dict(matches=int((teacher.argmax(-1)==pred.argmax(-1)).sum()),kl=float(kl.sum()))
                rows.append(dict(seed=seed,source=source,phase=label,n=len(sample),scores=data))
    D.atomic_json(out/f"fidelity-{i}.json",rows)


if __name__=="__main__":
    torch.set_num_threads(1)
    out=Path(os.environ["HSAI_FIDELITY_OUT"])
    if len(sys.argv)>1:
        measure(int(sys.argv[1]),out)
    else:
        out.mkdir(parents=True,exist_ok=False)
        def worker(i):
            with (out/f"worker-{i}.log").open("x") as log:
                subprocess.run([sys.executable,__file__,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(8) as e:
            list(e.map(worker,range(8)))
        D.atomic_json(out/"complete.json",dict(seed=212860000,n=64))
