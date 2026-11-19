"""Paired four-turn classroom, standard student checkpoints, and a single PPO fork."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows import azerite_course as C
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.az.train.personal import inspect_checkpoint
from hsbg.az.train.model import load_checkpoint,save_checkpoint

FIRST = Path("artifacts/models/human-turn-3702-azerite-course-20261119-2/model.pt")
FIRST_SHA = "0d1c7aadbb73e4d2f2983170000f090766a6c6c6bdbeb034b2749d97cd78acc9"
OUT = Path("artifacts/runs/nus-joint-growth-class-20261119-1")
SERIES = Path("artifacts/runs/nus-joint-growth-ppo-20261119-1")
MODEL = Path("artifacts/models/human-turn-3702-joint-growth-20261119-1")
CASES = ("elem_budget","elem_offer","quill_ready","quill_cold_search","quill_lone_source")
LABELS = ("early","late")
PPO_SEED = 197200000


def read(path):
    return json.loads(Path(path).read_bytes())


def workers(stage, out, count=8):
    def run(i):
        with (out/f"{stage}-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,stage,str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(count) as pool:
        list(pool.map(run,range(count)))


def trial(game,agent,role,seed,end,rng):
    game=C.hsbg_engine.Game.from_json(game.state_json())
    rec=C.Examples(agent.evaluator,rng,game=game,role=role)
    agent.evaluator=rec
    try:
        turns=C.probe.play(seed,"joint",agent,opening=game,normal_health=True,last_turn=end)
    finally:
        agent.evaluator=rec.original
    power=np.log1p([t["strength"] for t in turns])
    score=float(.75*power[-1]+.25*power.mean())
    return score,rec.rows


def collect(i):
    proto,pool=C.quill.config("train")
    parent=C.quill.policy(C.quill.BASE,proto)
    first=C.quill.policy(FIRST,proto)
    boar=C.quill.policy(C.INVEST,proto)
    rng=np.random.default_rng(740100+i)
    examples,comparisons=[],[]
    for seed in range(740000+i,740032,8):
        for case in CASES:
            g=C.opening(seed,case)
            # Sample both the intact opportunity and a state visited by the
            # current student. Each branch receives exactly the same native RNG.
            for offset in (0,3):
                if offset:
                    C.probe.play(seed,case,first,opening=g,normal_health=True,last_turn=11)
                end=int(g.turn)+3
                element=case.startswith("elem")
                options=[("student",first,"prior"),("parent",parent,"prior"),
                         ("expert",parent if element else boar,"element" if element else "pig")]
                branches=[trial(g,p,r,seed,end,rng) for _,p,r in options]
                scores=[s for s,_ in branches]
                best=int(np.argmax(scores))
                # Near-ties retain the previous student; only material end-board
                # improvements qualify a different continuation as a teacher.
                if scores[best]-scores[0]<.08:
                    best=0
                chosen=options[best][0]
                comparisons.append(dict(seed=seed,case=case,turn=int(g.turn),scores=scores,chosen=chosen))
                kind="element" if element else "pig"
                for f,c,m,q,_ in branches[best][1]:
                    examples.append((f,c,m,q,kind if chosen=="expert" else "alternative"))
        for k in range(2):
            rec=C.Examples(parent.evaluator,rng,role="natural")
            parent.evaluator=rec
            try:
                C.quill.play(parent,pool,751000+2*(seed-740000)+k)
            finally:
                parent.evaluator=rec.original
            examples.extend(rec.rows)
    f,c,m,q,k=zip(*examples)
    np.savez_compressed(OUT/f"examples-{i}.npz",f=np.asarray(f,np.float32),c=np.asarray(c,np.int32),
                        m=np.asarray(m,bool),q=np.asarray(q,np.float32),k=k)
    atomic_json(OUT/f"branches-{i}.json",comparisons)


def fit():
    rng=np.random.default_rng(66113)
    torch.manual_seed(66113)
    pieces=[]
    for i in range(8):
        with np.load(OUT/f"examples-{i}.npz") as f:
            pieces.append({k:f[k] for k in f.files})
    data={k:np.concatenate([d[k] for d in pieces]) for k in pieces[0]}
    groups={k:np.flatnonzero(data["k"]==k) for k in ("element","pig","alternative","natural")}
    assert all(len(v) for v in groups.values())
    net,meta=load_checkpoint(str(FIRST),"cuda:0")
    base,_=load_checkpoint(C.quill.BASE,"cuda:0")
    net.eval()
    base.eval()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1.4e-5,weight_decay=0.)
    losses=[]
    for step in range(1,451):
        ix=np.concatenate([rng.choice(groups[k],n) for k,n in
                          (("element",16),("pig",16),("alternative",12),("natural",20))])
        def batch(k,dtype):
            return torch.as_tensor(data[k][ix],dtype=dtype,device="cuda:0")
        x,c,m=batch("f",torch.float32),batch("c",torch.long),batch("m",torch.bool)
        logits,rank,*_=net(x,c,mask=m)
        with torch.no_grad():
            parent_rank=base(x[-20:],c[-20:],mask=m[-20:])[1].softmax(-1)
        action_loss=-(batch("q",torch.float32)*logits.masked_fill(~m,-1e8).log_softmax(-1)).sum(-1).mean()
        value_loss=torch.nn.functional.kl_div(rank[-20:].log_softmax(-1),parent_rank,reduction="batchmean")
        loss=action_loss+.5*value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite classroom loss")
        losses.append(float(loss.detach()))
        if step in (150,450):
            label="early" if step==150 else "late"
            save_checkpoint(net,str(OUT/label/"model.pt"),meta=dict(meta,course="four-turn-growth",
                            classroom_parent=FIRST_SHA,classroom_steps=step))
    atomic_json(OUT/"fit.json",dict(steps=450,groups={k:len(v) for k,v in groups.items()},
                start=losses[:8],end=losses[-8:],models={k:file_identity(OUT/k/"model.pt") for k in LABELS}))


def develop(i):
    proto,pool=C.quill.config("train")
    paths={k:OUT/k/"model.pt" for k in LABELS}
    paths.update(first=FIRST,parent=C.quill.BASE)
    agents={k:C.quill.policy(p,proto) for k,p in paths.items()}
    protected,natural=[],[]
    for seed in range(760000+i,760016,8):
        for case in C.EXAM_CASES:
            for k,agent in agents.items():
                rows=C.protected(seed,case,agent,C.opening(seed,case))
                protected.append(dict(seed=seed,case=case,model=k,**C.measure(rows,case.startswith("elem"))))
    for seed in range(761000+i,761048,8):
        for k,agent in agents.items():
            row=C.quill.play(agent,pool,seed)
            natural.append(dict(seed=seed,model=k,rank=row["rank"],reach12=12 in row["eot"]))
    atomic_json(OUT/f"dev-{i}.json",dict(protected=protected,natural=natural))


def select():
    parts=[read(OUT/f"dev-{i}.json") for i in range(8)]
    scores={}
    for k in (*LABELS,"first","parent"):
        grouped={}
        for case in C.EXAM_CASES:
            cells=[r for p in parts for r in p["protected"] if (r["case"],r["model"])==(case,k)]
            assert sorted(r["seed"] for r in cells)==list(range(760000,760016))
            grouped[case]=dict(logbody=np.mean([np.log1p(r["body"][-1]) for r in cells]),
                sustained=sum(r["sustained"] for r in cells),p50=sorted(r["body"][-1] for r in cells)[8])
        games=[r for p in parts for r in p["natural"] if r["model"]==k]
        assert sorted(r["seed"] for r in games)==list(range(761000,761048))
        utility=float(np.mean([C.quill.U[r["rank"]-1] for r in games]))
        score=np.mean([v["logbody"] for v in grouped.values()])+.35*utility
        scores[k]=dict(cases=grouped,natural_utility=utility,score=score,
                       top2=sum(r["rank"]<=2 for r in games),reach12=sum(r["reach12"] for r in games))
    eligible=[k for k in LABELS if scores[k]["cases"]["elem_budget"]["sustained"]>=2
              and scores[k]["cases"]["quill_ready"]["sustained"]>=2]
    if not eligible:
        raise RuntimeError("neither new candidate demonstrated both skills")
    chosen=max(eligible,key=lambda k:scores[k]["score"])
    path=OUT/chosen/"model.pt"
    ident=file_identity(path)
    if ident!=read(OUT/"fit.json")["models"][chosen] or ident["sha256"] in (FIRST_SHA,C.quill.PLAN["base_sha"]):
        raise ValueError("invalid candidate identity")
    verified=inspect_checkpoint(str(path),"cpu")
    assert verified["sha256"]==ident["sha256"] and verified["forward_ok"]
    partial=MODEL.with_name(MODEL.name+".partial")
    if MODEL.exists():
        raise FileExistsError(MODEL)
    partial.mkdir(exist_ok=False)
    os.link(path,partial/"model.pt")
    atomic_json(partial/"manifest.json",dict(schema="hsai.model/v1.1",status="PPO research starting point",
              arch=verified["meta"]["arch"],files={"model.pt":ident},teacher_source=FIRST_SHA,
              base=C.quill.PLAN["base_sha"],engine=C.quill.engine_info(),protocol=C.quill.PLAN["protocol"],
              libraries=C.quill.PLAN["libraries"],source=source_identity(ROOT)))
    atomic_json(partial/"checkpoint.json",dict(model=ident,fit=file_identity(OUT/"fit.json"),selected=chosen,
                examples="Paired four-turn continuations and ordinary natural-game policy replay"))
    (partial/"MODEL_CARD.md").write_text("# Joint growth student\n\nSelected on separate development seeds. "
        "Azerite and Quilboar classroom, ordinary entity policy. Natural strength is under evaluation.\n",
        encoding="ascii")
    partial.rename(MODEL)
    recipe=read(ROOT/"config/research/human-turn-3702-ppo.json")
    recipe.update(checkpoint_sha256=ident["sha256"],model_manifest_sha256=file_identity(MODEL/"manifest.json")["sha256"],
                  series=str(SERIES),segments=1,seed=PPO_SEED,endpoint="Final sealed generation of the only PPO job")
    recipe["environment"]["HSAI_ARENA_CKPT"]=str(MODEL/"model.pt")
    atomic_json(OUT/"ppo-recipe.json",recipe)
    atomic_json(OUT/"complete.json",dict(selected=chosen,model=ident,development=scores,
                recipe=file_identity(OUT/"ppo-recipe.json"),parent=FIRST_SHA))


def start():
    OUT.mkdir(parents=True,exist_ok=False)
    source=source_identity(ROOT)
    assert not source["tracked_dirty"] and C.quill.engine_info()["source_hash"]==C.quill.PLAN["engine"]
    for path,sha in ((FIRST,FIRST_SHA),(C.quill.BASE,C.quill.PLAN["base_sha"]),(C.INVEST,C.INVEST_SHA)):
        assert file_identity(path)["sha256"]==sha
    atomic_json(OUT/"plan.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],base=FIRST_SHA,
                branch_seeds=[740000,740031],holdout_class=[760000,760015],holdout_dev=[761000,761047],
                independent_natural=[171600000,171600511],independent_class=[770000,770047],
                selection="Among two learned students with both skills, mean of 7 case mean log T18 + .35 natural utility",
                horizon=4,steps=[150,450],lr=1.4e-5,ppo_seed=PPO_SEED))
    workers("collect",OUT)
    fit()
    workers("develop",OUT)
    select()


if __name__=="__main__":
    torch.set_num_threads(1)
    if len(sys.argv)==1:
        start()
    else:
        {"collect":collect,"develop":develop}[sys.argv[1]](int(sys.argv[2]))
