"""Equal-update peer imitation and outcome-selected imitation from HSAI 3265."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import hsbg_engine
from workflows import peer_ppo_panel as P
from workflows.d142_crossover import Fork, PrefixComplete, digest
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy, _mix_seed
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.runtime.references import reference_directory

OUT=Path(os.environ["HSAI_DISTILL_OUT"])
N=256
EVAL_N=512
MATCH_N=384
STEPS=1600
WIDTH=96
TRAIN_SEED=211340000
FUTURE_SEED=213640000
TEST_SEED=186730000
MATCH_SEED=187810000
SHARDS=8
ARMS=("ordinary","filtered")


def read(path):
    return json.loads(Path(path).read_bytes())


def sources(split):
    proto,pool=P.Q.config(split)
    if (file_identity(P.WEIGHT)["sha256"]!=P.PEER_SHA or
            file_identity(P.Q.BASE)["sha256"]!=P.Q.PLAN["base_sha"]):
        raise ValueError("source weights changed")
    sys.path.insert(0,str(reference_directory("hsbg-ppo")))
    import hsbg_net
    net,ck,r=hsbg_net.load(P.WEIGHT,device="cuda:0",strict=True)
    assert not r.missing_keys and not r.unexpected_keys and ck["net_config"]["z_vocab"][45]=="none"
    net.eval()
    return proto,pool,dict(base=P.Q.policy(P.Q.BASE,proto),peer=P.Peer(Soft(net,"cuda:0"),proto,False))


class Soft(P.Scores):
    @torch.inference_mode()
    def __call__(self,f,c,mask,seat):
        assert not self.blocked
        tensors=[torch.as_tensor(v,device=self.device)[None] for v in (f,c,mask)]
        logits,value=self.net(*tensors)
        prob=logits[0].softmax(0)
        if not torch.isfinite(prob).all() or int(prob.argmax())!=int(logits[0].argmax()):
            raise RuntimeError("invalid peer distribution")
        return prob.cpu().numpy(),float(value[0])


class Samples:
    def __init__(self,policy):
        self.policy=policy
        self.original=policy.evaluator
        self.rows=[]
        self.model=getattr(self.original,"model",None)

    def reset(self):
        self.policy.reset()

    def __call__(self,f,c,m,seat):
        q,v=self.original(f,c,m,seat)
        self.last=(f,c,m,q,self.turn) if m.sum()>1 else None
        return q,v

    def decide(self,game,seat):
        self.turn=int(game.turn)
        self.last=None
        try:
            self.policy.evaluator=self
            if isinstance(self.original,Soft):
                self.original.blocked=[]
            dec=self.policy.decide(game,seat)
        finally:
            self.policy.evaluator=self.original
        if self.last is not None and not dec.info.get("forced"):
            self.rows.append(self.last)
        return dec


def rollout(policy,pool,seed,*,fork=None,state=None,context=None):
    result=evaluate_policy(policy,EvalConfig(games=1,seed=seed,mirror_pool=pool,mirror_strict=True,
        research_capture_strict=fork,research_initial_state=state,research_strict_context=context))
    m=result["mirror"]
    if (result["games_finished"]!=1 or result["seed_sequence"]!=[seed] or any(result[k] for k in
        ("illegal_actions","truncated_games","truncated_combats")) or any(m["mirror_"+k] for k in
        ("errors","missing","rejected","fallback","pool_unbalanced","pool_species_unbalanced"))
        or not m["mirror_applied"] or m["mirror_applied"]!=m["mirror_attempt"]):
        raise ValueError(f"invalid rollout {seed}")
    return result["rank_sequence"][0]


def opening(policy,pool,seed,turn):
    fork=Fork(stop=True)
    fork.turn=turn
    try:
        rollout(policy,pool,seed,fork=fork)
    except PrefixComplete:
        pass
    return fork


def alternate(fork,seed):
    # Keep the prepared shop and seven opponents. Only unrevealed futures change.
    rng=json.loads(hsbg_engine.Game(seed=seed).state_json())["rng"]
    state=dict(fork.state,seed=seed,rng=rng)
    strict=deepcopy(fork.strict,{id(fork.strict.library):fork.strict.library})
    strict.frame_rng=lambda t,s:random.Random(_mix_seed(seed,t,s))
    return state,strict


def collect(i):
    proto,pool,policies=sources("train")
    rng=np.random.default_rng(75931+i)
    examples=[]
    records=[]
    for j in range(i,N,SHARDS):
        seed=TRAIN_SEED+j
        turn=(1,6,8,10)[j%4]
        prefix=("base","peer")[(j//4)%2]
        fork=opening(policies[prefix],pool,seed,turn)
        if fork.state is None:
            records.append(dict(j=j,missed=True))
            continue
        ranks=[]
        for rep in range(2):
            future=FUTURE_SEED+2*j+rep
            state,strict=alternate(fork,future)
            ranks.append([rollout(policies[k],pool,future,state=state,context=strict) for k in ("base","peer")])
        u=P.Q.U[np.array(ranks)-1]
        gain=float(np.mean(u[:,1]-u[:,0]))
        for arm,k in enumerate(("base","peer")):
            sample=Samples(policies[k])
            rollout(sample,pool,seed,state=fork.state,context=fork.strict)
            if not sample.rows:
                raise ValueError("empty demonstration")
            # Equal cap per source and arm; even long APM games cannot dominate.
            idx=rng.choice(len(sample.rows),min(144,len(sample.rows)),replace=False)
            examples.extend((*sample.rows[n],j,arm) for n in idx)
        records.append(dict(j=j,prefix=prefix,turn=turn,opening=fork.opening,ranks=ranks,
                            gain=gain,chosen=int(gain>=0)))
    f,c,m,q,t,j,a=zip(*examples)
    np.savez_compressed(OUT/f"rows-{i}.npz",f=np.array(f,np.float32),c=np.array(c,np.int16),
        m=np.array(m,bool),q=np.array(q,np.float32),turn=t,source=j,arm=a)
    atomic_json(OUT/f"returns-{i}.json",records)


def fit():
    data=[]
    outcomes=[]
    for i in range(SHARDS):
        with np.load(OUT/f"rows-{i}.npz") as d:
            data.append({k:d[k] for k in d.files})
        outcomes.extend(read(OUT/f"returns-{i}.json"))
    assert sorted(r["j"] for r in outcomes)==list(range(N))
    data={k:np.concatenate([v[k] for v in data]) for k in data[0]}
    valid=sorted(r["j"] for r in outcomes if not r.get("missed"))
    labels={r["j"]:r["chosen"] for r in outcomes if not r.get("missed")}
    cells={(j,a):np.flatnonzero((data["source"]==j)&(data["arm"]==a)) for j in valid for a in (0,1)}
    ref,meta=load_checkpoint(P.Q.BASE,"cuda:0")
    ref.eval()
    for arm in ARMS:
        rng=np.random.default_rng(87914)
        torch.manual_seed(87914)
        net,meta=load_checkpoint(P.Q.BASE,"cuda:0")
        net.eval()
        optimizer=torch.optim.AdamW(net.parameters(),lr=2e-5,weight_decay=0.)
        losses=[]
        start=time.monotonic()
        for step in range(STEPS):
            js=rng.choice(valid,WIDTH)
            fraction=rng.random(WIDTH)
            # An identical 1/4 reference batch protects old behaviors in both arms.
            ix=[]
            for n,j in enumerate(js):
                teacher=0 if n<WIDTH//4 else (1 if arm=="ordinary" else labels[j])
                cell=cells[j,teacher]
                ix.append(cell[int(fraction[n]*len(cell))])
            def batch(k,dtype):
                return torch.as_tensor(data[k][ix],dtype=dtype,device="cuda:0")
            f,c,m=batch("f",torch.float32),batch("c",torch.long),batch("m",torch.bool)
            for key,op in (("norm_mean",torch.sub),("norm_std",torch.div)):
                if meta.get(key) is not None:
                    f=op(f,torch.as_tensor(meta[key],device=f.device,dtype=f.dtype))
            logits,rank,*_=net(f,c,mask=m)
            with torch.no_grad():
                rank_target=ref(f,c,mask=m)[1].softmax(-1)
            loss=-(batch("q",torch.float32)*logits.masked_fill(~m,-1e8).log_softmax(-1)).sum(-1).mean()
            loss+=.35*torch.nn.functional.kl_div(rank.log_softmax(-1),rank_target,reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step%100==0:
                losses.append(float(loss))
        path=OUT/arm/"model.pt"
        save_checkpoint(net,str(path),meta=dict(meta,distill_arm=arm,distill_source=P.PEER_SHA,
            parent=P.Q.PLAN["base_sha"],steps=STEPS))
        load_checkpoint(str(path),"cpu")
        atomic_json(path.parent/"manifest.json",dict(arm=arm,checkpoint=file_identity(path),parent=P.Q.PLAN["base_sha"],
            teacher=P.PEER_SHA,steps=STEPS,examples=STEPS*WIDTH,losses=losses,seconds=time.monotonic()-start))
        atomic_json(path.parent/"checkpoint.json",dict(weights=file_identity(path),kind="distilled weights"))
        (path.parent/"MODEL_CARD.md").write_text(f"# {arm} peer student\nHSAI 3265 initialization. Research use. Teacher: PrentendYan / jimmypretend, CC BY-NC 4.0.\n",encoding="ascii")
        print(arm,file_identity(path),flush=True)


def check(i):
    proto,pool,policies=sources("hold")
    for arm in ARMS:
        path=OUT/arm/"model.pt"
        assert file_identity(path)==read(path.parent/"manifest.json")["checkpoint"]
        policies[arm]=P.Q.policy(path,proto)
    dest=OUT/f"eval-{i}"
    dest.mkdir(exist_ok=False)
    for j in range(i,EVAL_N,SHARDS):
        seed=TEST_SEED+j
        rows={k:P.trial(p,pool,seed,event_turns=True) for k,p in policies.items()}
        assert len({r["lobby"] for r in rows.values()})==1
        atomic_json(dest/f"{j}.json",rows)
    rows=[]
    from hsbg.az.train.economic_tasks import card_ids
    ids=card_ids()
    for j in range(i,MATCH_N,SHARDS):
        turn=(9,10,11,12)[j%4]
        seed=MATCH_SEED+j
        fork=opening(policies["base"],pool,seed,turn)
        if fork.state is None:
            continue
        game=hsbg_engine.Game.from_json(json.dumps(fork.state))
        p=json.loads(game.player_json(0))
        cards={m["card"] for m in p["board"]}
        groups=[]
        if p["health"]+p["armor"]<=15:
            groups.append("low")
        for name,keys in (("mech",("BG36_366","BG26_152")),("dragon",("BG36_364","BG29_813"))):
            if any(ids.get(key) in cards for raw in keys for key in (raw,raw+"_G")):
                groups.append(name)
        if groups:
            ranks={k:rollout(pol,pool,seed,state=fork.state,context=fork.strict) for k,pol in policies.items()}
            rows.append(dict(j=j,groups=groups,hp=p["health"]+p["armor"],opening=fork.opening,ranks=ranks))
    atomic_json(OUT/f"matched-{i}.json",rows)


def workers(stage):
    def run(i):
        with (OUT/f"{stage}-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,stage,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(SHARDS) as e:
        list(e.map(run,range(SHARDS)))


def main():
    stage=sys.argv[1]
    if len(sys.argv)>2:
        return {"collect":collect,"check":check}[stage](int(sys.argv[2]))
    if stage=="study":
        OUT.mkdir(parents=True,exist_ok=False)
        recipe=read("config/research/human-turn-3702-ppo.json")
        assert all(library_digest(Path("artifacts/libraries/human-turn-3702-20261118")/s)==v for s,v in recipe["library_digests"].items())
        source=source_identity(ROOT)
        assert not source["tracked_dirty"] and P.Q.engine_info()["source_hash"]==recipe["engine_source_hash"]
        atomic_json(OUT/"plan.json",dict(source=source,engine=recipe["engine_source_hash"],libraries=recipe["library_digests"],
            base=P.Q.PLAN["base_sha"],teacher=P.PEER_SHA,protocol=P.Q.PLAN["protocol"],train=[TRAIN_SEED,N],
            test=[TEST_SEED,EVAL_N],matched=[MATCH_SEED,MATCH_N],future=[FUTURE_SEED,2*N],steps=STEPS,width=WIDTH,
            protected="25 percent base on base trajectories; train source weighted uniformly",
            filter="both arms share all branches; filtered uses base if mean paired terminal utility difference is negative on 2 independent futures; demonstrations use original future",
            gate="consider short PPO only with positive utility point estimate and positive paired T12 log growth, no material regression (>0.05 utility or 3 points reach12 in aggregate); matched low HP and old-core regressions reviewed with uncertainty"))
        workers("collect")
        fit()
        atomic_json(OUT/"trained.json",{k:file_identity(OUT/k/"model.pt") for k in ARMS})
    elif stage=="assessment":
        assert all(file_identity(OUT/k/"model.pt")==v for k,v in read(OUT/"trained.json").items())
        workers("check")
        atomic_json(OUT/"evaluated.json",dict(n=EVAL_N,matched_candidates=MATCH_N,job=os.environ["SLURM_JOB_ID"]))
    else:
        raise ValueError(stage)


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
