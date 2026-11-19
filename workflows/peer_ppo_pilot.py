"""Prepare, test and train a head-only PPO fork of the published peer actor."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
import json
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from workflows import peer_exploration as X
from workflows import peer_ppo_panel as P
from workflows.human_turn_preflight import verified_generation,allocation_end
from hsbg.az.train.peer_actor import PeerActor
from hsbg.az.train.model import load_checkpoint,save_checkpoint
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_distillation import estimate,p50

OUT=Path(os.environ.get("HSAI_PEER_PILOT","artifacts/runs/nus-peer-heads-20261119-1"))
SEED=232700000


def read(path):
    return json.loads(Path(path).read_bytes())


def workers(stage):
    def launch(i):
        with (OUT/f"{stage}-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,stage,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as e:
        list(e.map(launch,range(8)))


def prepare():
    evidence=read(X.OUT/"complete.json")
    temp=evidence["temperature"]
    if temp<=0:
        raise RuntimeError("exploration screen did not pass")
    OUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source_identity(ROOT),peer=P.PEER_SHA,temperature=temp,
        calibration=file_identity(X.OUT/"complete.json"),pilot_seed=SEED,iterations=60,games=256,
        eval_seed=189430000,eval_n=384,trainable="native action heads plus separate rank predictor",
        gate="on 384 new paired games candidate utility positive, lower confidence >= -0.05; paired T12 log strength lower CI >= -0.12; no material survival or specialized regression; require independent confirmation before overnight"))
    actor,ck=X.load_peer()
    torch.manual_seed(66135)
    model=PeerActor(actor,ck["net_config"],temp).cuda().eval()
    pieces=[]
    for i in range(8):
        with np.load(X.OUT/f"critic-{i}.npz") as d:
            pieces.append({k:d[k] for k in d})
    x,y,g=[np.concatenate([d[k] for d in pieces]) for k in ("x","y","g")]
    assert set(g)==set(range(X.N))
    x=torch.from_numpy(x).cuda()
    y=torch.as_tensor(y,dtype=torch.long,device="cuda")
    train=np.flatnonzero(g%6!=0)
    val=np.flatnonzero(g%6==0)
    rng=np.random.default_rng(66135)
    opt=torch.optim.AdamW(model.rank_head.parameters(),lr=.001)
    with torch.no_grad():
        prior=(y[train].bincount(minlength=8).float()+1).log().log_softmax(0)
        reference=float(-prior[y[val]].mean())
    for _ in range(400):
        ix=rng.choice(train,256)
        loss=torch.nn.functional.cross_entropy(model.rank_head(x[ix]),y[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ce=float(torch.nn.functional.cross_entropy(model.rank_head(x[val]),y[val]))
    if not np.isfinite(ce) or ce>=reference:
        raise RuntimeError(f"critic failed out-of-game calibration: {ce} vs {reference}")
    path=OUT/"initial/model.pt"
    save_checkpoint(model,str(path),meta={"origin_license":"CC BY-NC 4.0","author":"PrentendYan / jimmypretend"})
    reloaded,_=load_checkpoint(str(path),"cuda")
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    atomic_json(path.parent/"manifest.json",dict(model=file_identity(path),origin=P.PEER_SHA,temp=temp,
        critic_ce=ce,prior_ce=reference,train_rows=len(train),val_rows=len(val)))
    atomic_json(path.parent/"checkpoint.json",dict(model=file_identity(path),kind="calibrated fork"))
    (path.parent/"MODEL_CARD.md").write_text("# Peer PPO fork\nPrentendYan / jimmypretend. CC BY-NC 4.0. Original actor; calibrated independent HSAI rank predictor.\n",encoding="ascii")
    workers("identity")


def identity(i):
    proto,pool=P.Q.config("train")
    actor,_=X.load_peer()
    old=P.Peer(P.Scores(actor,"cuda:0"),proto,False)
    new=P.Q.policy(OUT/"initial/model.pt",proto)
    for seed in range(219740000+i,219740032,8):
        a=P.trial(old,pool,seed,event_turns=True)
        b=P.trial(new,pool,seed,event_turns=True)
        if a!=b:
            raise ValueError(f"greedy reproduction differs {seed} ranks {a['metrics']['rank']},{b['metrics']['rank']}")
    atomic_json(OUT/f"identity-{i}.json",dict(games=4))


def environment():
    env=read("config/research/human-turn-3702-ppo.json")["environment"].copy()
    env.update(HSAI_HUMAN_TAIL="last-observed-v1",HSAI_ARENA_GAMES="256",
        HSAI_ARENA_LR="1.5e-5",HSAI_ARENA_ANCHOR="0.12",HSAI_ARENA_ANCHOR_TAIL="0.08",
        HSAI_ARENA_TARGET_KL="0.025",HSAI_ARENA_ENTROPY="0.001")
    return env


def pilot():
    prepare()
    assert len(list(OUT.glob("identity-*.json")))==8
    env=os.environ.copy()
    env.update(environment(),HSRL_ROOT=str(ROOT),HSAI_ARENA_OUT=str(OUT/"pilot"),
        HSAI_ARENA_SEED=str(SEED),HSAI_ARENA_ITERATIONS="60",
        HSAI_ARENA_CKPT=str(OUT/"initial/model.pt"),HSAI_ARENA_STOP_EPOCH=str(allocation_end()-150))
    subprocess.run(["bash","deploy/slurm/stage_arena.sbatch"],env=env,check=True)
    manifest=verified_generation(OUT/"pilot",SEED)
    if manifest["version"]!=60:
        raise ValueError("incomplete pilot budget")
    atomic_json(OUT/"trained.json",manifest)


def panel(i):
    proto,pool=P.Q.config("hold",allow_tail_variant=True)
    seal=read(OUT/"trained.json")
    assert file_identity(seal["model"])==seal["files"]["model.pt"]
    paths=dict(parent=OUT/"initial/model.pt",candidate=seal["model"])
    players={k:P.Q.policy(p,proto) for k,p in paths.items()}
    rec=[]
    for seed in range(189430000+i,189430384,8):
        rec.append({k:P.trial(p,pool,seed,event_turns=True) for k,p in players.items()})
    atomic_json(OUT/f"panel-{i}.json",rec)


def summarize():
    rows=[r for i in range(8) for r in read(OUT/f"panel-{i}.json")]
    assert sorted(r["parent"]["seed"] for r in rows)==list(range(189430000,189430384))
    assert all(r["parent"]["seed"]==r["candidate"]["seed"] and r["parent"]["lobby"]==r["candidate"]["lobby"] for r in rows)
    result={}
    for k in rows[0]:
        arms=[r[k] for r in rows]
        turns={}
        for t in range(8,19):
            vals=[v["atk"]+v["hp"] for a in arms if (v:=a["turns"].get(str(t))) is not None]
            turns[t]=dict(n=len(vals),p50=p50(vals))
        result[k]=dict(means={m:np.mean([a["metrics"][m] for a in arms]) for m in arms[0]["metrics"]},turns=turns)
    d={m:estimate([r["candidate"]["metrics"][m]-r["parent"]["metrics"][m] for r in rows]) for m in rows[0]["parent"]["metrics"]}
    for t in (12,14,15):
        pair=[tuple(v["turns"][str(t)]["atk"]+v["turns"][str(t)]["hp"] for v in (r["parent"],r["candidate"]))
              for r in rows if all(str(t) in a["turns"] for a in r.values())]
        d[f"log{t}"]=estimate([np.log1p(b)-np.log1p(a) for a,b in pair])
    atomic_json(OUT/"assessment.json",dict(models=result,delta=d,n=len(rows)))


if __name__=="__main__":
    torch.set_num_threads(1)
    stage=sys.argv[1]
    if len(sys.argv)>2:
        {"identity":identity,"panel":panel}[stage](int(sys.argv[2]))
    elif stage=="pilot":
        pilot()
    elif stage=="assess":
        workers("panel")
        summarize()
