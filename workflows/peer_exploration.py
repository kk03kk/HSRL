"""Measure stochastic exploration of the frozen peer policy and calibrate a critic."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from workflows import peer_ppo_panel as P
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.references import reference_directory
from tools.summarize_peer_distillation import estimate,p50

SEED=218510000
N=192
TEMPS=(0.,.55,1.)
OUT=Path(os.environ.get("HSAI_PEER_PROBE","artifacts/runs/nus-peer-explore-20261119-1"))


def load_peer():
    if file_identity(P.WEIGHT)["sha256"]!=P.PEER_SHA:
        raise ValueError("foreign weights changed")
    sys.path.insert(0,str(reference_directory("hsbg-ppo")))
    import hsbg_net
    net,ck,r=hsbg_net.load(P.WEIGHT,device="cuda:0",strict=True)
    assert ck["net_config"]["z_vocab"][-1]=="none" and not r.missing_keys and not r.unexpected_keys
    return net.eval(),ck


class Sample(P.Scores):
    @torch.inference_mode()
    def __call__(self,f,c,mask,seat):
        assert not self.blocked
        ts=[torch.as_tensor(x,device=self.device)[None] for x in (f,c,mask)]
        logits,value=self.net(*ts)
        best=int(logits[0].argmax())
        if self.temp:
            probs=(logits[0]/self.temp).softmax(0).cpu().numpy().astype(np.float64)
            a=int(self.rng.choice(len(mask),p=probs/probs.sum()))
        else:
            a=best
        if not mask[a]:
            raise RuntimeError("invalid peer action")
        self.different+=int(a!=best)
        self.total+=1
        v=float(value[0])
        if self.total%7==1:
            self.rows.append(np.r_[f[:68],v].astype(np.float32))
        pi=np.zeros(len(mask),np.float32)
        pi[a]=1.
        return pi,v


def worker(i):
    proto,pool=P.Q.config("train")
    net,_=load_peer()
    ev=Sample(net,"cuda:0")
    policy=P.Peer(ev,proto,False)
    rows=[]
    inputs=[]
    targets=[]
    groups=[]
    for j in range(i,N,8):
        for h,temp in enumerate(TEMPS):
            ev.temp=temp
            ev.rng=np.random.default_rng(28410000+10*j+h)
            ev.different=ev.total=0
            ev.rows=[]
            r=P.trial(policy,pool,SEED+j,event_turns=True)
            inputs.extend(ev.rows)
            targets.extend([r["metrics"]["rank"]-1]*len(ev.rows))
            groups.extend([j]*len(ev.rows))
            rows.append(dict(j=j,h=h,rank=r["metrics"]["rank"],utility=r["metrics"]["utility"],
                turns={t:v["atk"]+v["hp"] for t,v in r["turns"].items()},different=ev.different,total=ev.total))
    np.savez_compressed(OUT/f"critic-{i}.npz",x=inputs,y=targets,g=groups)
    atomic_json(OUT/f"shard-{i}.json",rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUT/"plan.json",dict(seed=SEED,n=N,temps=TEMPS,sha=P.PEER_SHA,
        selection="highest temperature with paired utility lower CI >= -0.20 and T12 paired log body mean >= -0.20; otherwise no long training"))
    def launch(i):
        with (OUT/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as e:
        list(e.map(launch,range(8)))
    rows=[r for i in range(8) for r in json.loads((OUT/f"shard-{i}.json").read_bytes())]
    arms=[sorted([r for r in rows if r["h"]==i],key=lambda r:r["j"]) for i in range(len(TEMPS))]
    assert all([r["j"] for r in arm]==list(range(N)) for arm in arms)
    result=[]
    for arm,temp in zip(arms,TEMPS):
        pairs=[(a["turns"]["12"],b["turns"]["12"]) for a,b in zip(arms[0],arm) if "12" in a["turns"] and "12" in b["turns"]]
        vals=[r["turns"]["12"] for r in arm if "12" in r["turns"]]
        result.append(dict(temp=temp,rank=np.mean([r["rank"] for r in arm]),top2=sum(r["rank"]<=2 for r in arm),
            utility=estimate([b["utility"]-a["utility"] for a,b in zip(arms[0],arm)]),
            log12=estimate([np.log1p(b)-np.log1p(a) for a,b in pairs]),n12=len(vals),p50=p50(vals),
            changed=sum(r["different"] for r in arm)/sum(r["total"] for r in arm)))
    viable=[r["temp"] for r in result if r["temp"] and r["utility"]["ci95"][0]>=-.2 and r["log12"]["mean"]>=-.2]
    atomic_json(OUT/"complete.json",dict(arms=result,temperature=max(viable,default=0)))


if __name__=="__main__":
    torch.set_num_threads(1)
    main() if len(sys.argv)==1 else worker(int(sys.argv[1]))
