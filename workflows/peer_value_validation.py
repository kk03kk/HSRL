"""Fit rank heads on frozen policy data and evaluate predeclared out-of-sample gates."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from hsbg.az.train.model import load_checkpoint
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.peer_value_study import INITIAL, INITIAL_SHA, OUT, PLAN, PLAN_PATH, SHARDS
from tools.summarize_peer_finetune import UTILITY, estimate

U = np.asarray(UTILITY)
KEYS = ("historical","compact","rich")


def read(path):
    return json.loads(path.read_bytes())


def natural_data(split):
    values = []
    for i in range(SHARDS):
        with np.load(OUT/"natural"/f"{split}-{i}.npz") as archive:
            values.append({k:archive[k] for k in archive.files})
    x,y,seed,t = [np.concatenate([a[k] for a in values]) for k in ("x","y","seed","turn")]
    start,n = PLAN[split]
    if (not np.array_equal(np.unique(seed),np.arange(start,start+n)) or not np.isfinite(x).all()
            or x.shape[1]!=69+3*352 or not np.all((0<=y)&(y<8))):
        raise ValueError("incomplete or invalid observations")
    return torch.from_numpy(x),torch.from_numpy(y),seed,t


def projection(x):
    result = []
    for part in x[:,69:].split(352,dim=1):
        mean = part.mean(0)
        centered = part-mean
        eig,basis = torch.linalg.eigh(centered.T @ centered/(len(x)-1))
        axes = basis[:,-24:] / eig[-24:].clamp_min(.01).sqrt()
        result.append((mean,axes))
    return result


def transform(x,proj):
    return torch.cat([x[:,:69]]+[(part-mean)@axes for part,(mean,axes) in zip(x[:,69:].split(352,dim=1),proj)],dim=1)


def head(width):
    return nn.Sequential(nn.Linear(width,128),nn.Tanh(),nn.Linear(128,8))


def historical():
    if file_identity(INITIAL)["sha256"]!=INITIAL_SHA:
        raise ValueError("wrong historical head")
    model,_=load_checkpoint(str(INITIAL),"cpu")
    return model.rank_head.eval()


def by_seed(values, ids):
    _,inverse=np.unique(ids,return_inverse=True)
    return np.bincount(inverse,weights=values)/np.bincount(inverse)


def fit():
    if not (OUT/"natural/complete.json").is_file():
        raise ValueError("natural data incomplete")
    dest=OUT/"heads"
    dest.mkdir(exist_ok=False)
    x,y,seeds,_=natural_data("train")
    train=seeds%8!=0
    proj=projection(x[train])
    rich=transform(x,proj)
    controls=historical()
    rng=np.random.default_rng(388159)
    groups=np.unique(seeds[train])
    members=[np.flatnonzero(seeds==s) for s in groups]
    # Identical groups and decision positions for both models. Longer games get
    # the same expected weight per optimizer step as short games.
    batches=[]
    for _ in range(1000):
        gs=rng.integers(len(groups),size=256)
        frac=rng.random(256)
        batches.append([members[g][int(f*len(members[g]))] for g,f in zip(gs,frac)])
    heads={}
    selection={}
    for name,features in (("compact",x[:,:69]),("rich",rich)):
        net=head(features.shape[1])
        with torch.no_grad():
            net[0].weight.zero_()
            net[0].weight[:,:69].copy_(controls[0].weight)
            net[0].bias.copy_(controls[0].bias)
            net[2].load_state_dict(controls[2].state_dict())
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.01)
        best=float("inf")
        for step,ix in enumerate(batches,1):
            loss=nn.functional.cross_entropy(net(features[ix]),y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step%100==0:
                with torch.no_grad():
                    losses=nn.functional.cross_entropy(net(features[~train]),y[~train],reduction="none").numpy()
                score=by_seed(losses,seeds[~train]).mean()
                if score<best:
                    best=score
                    heads[name]={k:v.detach().clone() for k,v in net.state_dict().items()}
                    selection[name]=dict(step=step,dev_nll=score)
    with (dest/"heads.pt").open("xb") as stream:
        torch.save(dict(projection=proj,heads=heads),stream)
    atomic_json(dest/"sealed.json",dict(weights=file_identity(dest/"heads.pt"),selection=selection,
        train_games=len(groups),validation_games=len(set(seeds[~train])),examples=len(seeds),plan=file_identity(PLAN_PATH)))


def models():
    path=OUT/"heads/heads.pt"
    if file_identity(path)!=read(OUT/"heads/sealed.json")["weights"]:
        raise ValueError("unsealed value heads")
    data=torch.load(path,map_location="cpu",weights_only=True)
    nets=dict(historical=historical())
    for k,state in data["heads"].items():
        nets[k]=head(state["0.weight"].shape[1])
        nets[k].load_state_dict(state,strict=True)
    return nets,data["projection"]


@torch.inference_mode()
def predict(x,nets,proj):
    rich=transform(x,proj)
    return {k:net(rich if k=="rich" else x[:,:69]).softmax(-1).numpy() for k,net in nets.items()}


def rank_test():
    x,y,ids,t=natural_data("rank_test")
    y=y.numpy()
    probs=predict(x,*models())
    loss={}
    for k,p in probs.items():
        loss[k]=dict(nll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-20)),
            utility_mse=(p@U-U[y])**2,top2_brier=(p[:,:2].sum(1)-(y<2))**2,
            mean_rank_error=np.abs(p@np.arange(1,9)-(y+1)))
    def summary(mask):
        metrics={k:{m:by_seed(v[mask],ids[mask]) for m,v in d.items()} for k,d in loss.items()}
        return dict(games=len(np.unique(ids[mask])),rows=int(mask.sum()),
            means={k:{m:float(a.mean()) for m,a in d.items()} for k,d in metrics.items()},
            reduction={k:{m:estimate(a-metrics["rich"][m]) for m,a in metrics[k].items()} for k in ("historical","compact")})
    all_rows=summary(np.ones(len(ids),bool))
    delta=all_rows["reduction"]
    passed=all(delta[k]["nll"]["ci95"][0]>0 for k in delta) and delta["compact"]["utility_mse"]["mean"]>=0
    atomic_json(OUT/"rank-result.json",dict(all=all_rows,midgame=summary((t>=8)&(t<=12)),passed=bool(passed)))


def fork_test(dest=None):
    if not (OUT/"forks/complete.json").is_file():
        raise ValueError("continuations incomplete")
    nets,proj=models()
    rows=[]
    screened=[]
    for i in range(SHARDS):
        cases=read(OUT/"forks"/f"cases-{i}.json")
        screened.extend(cases)
        with np.load(OUT/"forks"/f"features-{i}.npz") as archive:
            values=archive["x"]
        for r in cases:
            if r["status"]!="eligible":
                continue
            v=values[r["vector"]]
            pp=predict(torch.from_numpy(v.reshape(-1,v.shape[-1])),nets,proj)
            q={k:(p@U).reshape(PLAN["planning_futures"],2).mean(0) for k,p in pp.items()}
            ranks=np.array(r["ranks"])
            if ranks.shape!=(PLAN["outcome_futures"],2) or np.any((ranks<1)|(ranks>8)):
                raise ValueError("invalid outcomes")
            outcomes=U[ranks-1].mean(0)
            bodies=np.mean(r["planning_bodies"],axis=0)
            rows.append(dict(seed=r["seed"],t=r["turn"],kind=r["kind"],low=r["hp"]<=15,prepared=r["prepared"],
                body_delta=float(bodies[1]-bodies[0]),returns=outcomes.tolist(),rank=r["ranks"],bounds=r["bounds"],
                prediction={k:val.tolist() for k,val in q.items()},choice={k:int(val[1]>val[0]) for k,val in q.items()}))
    start,n=PLAN["fork_test"]
    if sorted(r["seed"] for r in screened)!=list(range(start,start+n)):
        raise ValueError("incomplete source coverage")
    def summary(rs):
        if not rs:
            return dict(n=0)
        realized=np.asarray([r["returns"] for r in rs])
        # Utilities are exact multiples of 1/20. Rank permutations with the
        # same utility must remain ties regardless of floating-point sum order.
        ticks=np.rint(U*20).astype(np.int64)
        replicas=ticks[np.asarray([r["rank"] for r in rs])-1]
        truth=np.diff(replicas.sum(1),axis=1)[:,0]/(20*replicas.shape[1])
        half=replicas.shape[1]//2
        split1=np.diff(replicas[:,:half].sum(1),axis=1)[:,0]
        split2=np.diff(replicas[:,half:].sum(1),axis=1)[:,0]
        comparable=(split1!=0)&(split2!=0)
        choices={k:np.array([r["choice"][k] for r in rs]) for k in KEYS}
        values={k:realized[np.arange(len(rs)),choice] for k,choice in choices.items()}
        return dict(n=len(rs),invest_better=int(np.sum(truth<0)),tempo_better=int(np.sum(truth>0)),ties=int(np.sum(truth==0)),
            accuracy={k:float(np.mean((v[truth!=0]==(truth[truth!=0]>0)))) if np.any(truth!=0) else None for k,v in choices.items()},
            tempo_choices={k:int(a.sum()) for k,a in choices.items()},mean_utility={k:v.mean() for k,v in values.items()},
            disagreements={k:int(np.sum(choices[k]!=choices["rich"])) for k in KEYS[:-1]},
            gain={k:estimate(values["rich"]-values[k]) for k in KEYS[:-1]},
            constant_utility=dict(invest=realized[:,0].mean(),tempo=realized[:,1].mean()),
            rich_vs_constant=dict(invest=estimate(values["rich"]-realized[:,0]),tempo=estimate(values["rich"]-realized[:,1])),
            future_half_agreement=dict(n=int(comparable.sum()),agree=int(np.sum(comparable&(np.sign(split1)==np.sign(split2))))),
            mean_effect_of_tempo=estimate(truth))
    verified=[r for r in rows if r["body_delta"]>0]
    result=summary(verified)
    passed=(result["n"]>=60 and result["disagreements"]["compact"]>=15 and
        result["gain"]["compact"]["ci95"][0]>0 and result["gain"]["historical"]["mean"]>0)
    if dest is None:
        dest=OUT
    else:
        dest.mkdir(parents=True,exist_ok=False)
    atomic_json(dest/"fork-decisions.json",rows)
    atomic_json(dest/"fork-result.json",dict(screened=n,eligible=summary(rows),verified=result,
        verified_groups={k:summary([r for r in verified if fn(r)]) for k,fn in (
            ("low",lambda r:r["low"]),("prepared",lambda r:r["prepared"]),("tier",lambda r:r["kind"]=="tier"),
            ("component",lambda r:r["kind"]=="growth_purchase"))},
        status={k:sum(r["status"]==k for r in screened) for k in sorted({r["status"] for r in screened})},
        turn_counts={t:sum(r["t"]==t for r in verified) for t in range(8,13)},
        tail_cases=sum(bool(np.any(r["bounds"])) for r in rows),passed=bool(passed)))


if __name__=="__main__":
    torch.set_num_threads(3)
    if sys.argv[1]=="fit":
        fit()
        rank_test()
    elif sys.argv[1]=="fork":
        fork_test(Path(sys.argv[2]) if len(sys.argv)>2 else None)
    else:
        raise ValueError("unknown stage")
