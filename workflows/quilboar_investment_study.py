"""Follow-up training focused on gem size, distribution and conversion cost."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ["HSAI_QUILL_PLAN"]="config/research/quilboar-investment-study.json"
import numpy as np
import torch
from hsbg.az.train.quilboar_investment import QuilboarInvestment
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows.quilboar_study import (
    ROOT,OUT,PLAN,BASE,U,read,launch,load_checkpoint,save_checkpoint,engine_info)
from tools.assess_quilboar_study import cohort,summary


def make(tag,theta,prior):
    base,meta=load_checkpoint(BASE,"cpu")
    gate,size,distribution=theta
    weights=np.r_[prior,gate*prior,size,distribution]
    model=QuilboarInvestment(base,meta,weights)
    path=OUT/"models"/tag/"model.pt"
    if path.exists():
        raise FileExistsError(path)
    save_checkpoint(model,str(path))
    atomic_json(path.parent/"manifest.json",dict(name=tag,base=PLAN["base_sha"],
                theta=list(theta),model=file_identity(path),terms=model.terms,
                engine=PLAN["engine"],protocol=PLAN["protocol"]))
    return str(path)


def score(g):
    # A one-log-unit improvement at T10-T13 is worth 0.45 rank-utility units.
    # Missing turns are below a weak surviving board; no survivor filtering.
    refs = (609,922,1432,1938)
    relative = [float(np.clip(np.log1p(g["eot"][str(t)][0])-np.log1p(ref),-1.5,2))
                if str(t) in g["eot"] else -1.5 for t,ref in enumerate(refs,10)]
    return U[g["rank"]-1]+.45*np.mean(relative)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=source_identity(ROOT)
    old=Path(PLAN["demonstration"])
    fit=old/"fit.json"
    original=read(old/"selection.json")
    if (source["tracked_dirty"] or file_identity(BASE)["sha256"]!=PLAN["base_sha"] or
            file_identity(original["path"])!=original["model"] or engine_info()["source_hash"]!=PLAN["engine"]):
        raise ValueError("source, engine or checkpoint changed")
    for field in ("train","hold"):
        if library_digest(Path("artifacts/libraries/human-turn-3702-20261118")/field)!=PLAN["libraries"][field]:
            raise ValueError("field fingerprint mismatch")
    atomic_json(OUT/"plan.json",dict(plan=PLAN,source=source,engine=engine_info(),
                job=os.environ.get("SLURM_JOB_ID"),fit=file_identity(fit)))
    torch.set_num_threads(2)
    prior=np.array(read(fit)["weights"])
    # Paired outcome search. The settings include isolating size from delivery.
    population=np.array([[.5,4,5],[.9,8,8],[.9,10,3],[.9,3,11],[.1,8,8],[.9,0,0]],float)
    arms={str(i):make("train-"+str(i),v,prior) for i,v in enumerate(population)}
    train=launch("investment_fit",arms)
    values=[np.mean([score(g[k]) for g in train]) for k in arms]
    best=np.argsort(-np.array(values),kind="stable")[:2]
    theta=population[best].mean(axis=0)
    model=make("investment",theta,prior)
    atomic_json(OUT/"optimization.json",dict(population=population.tolist(),score=values,
                                             elite=best.tolist(),theta=theta.tolist()))
    dev=launch("develop",dict(parent=BASE,student=model,previous=original["path"]))
    metrics=cohort([(summary(r["parent"]),summary(r["student"])) for r in dev])
    growth=metrics["growth_10_12"]["mean"]
    advance=bool(growth>=-.025 and metrics["utility"]["paired"]["mean"]>=-.10 and
                 metrics["repeat"]["paired"]["mean"]>0)
    atomic_json(OUT/"selection.json",dict(candidate="investment",path=model,model=file_identity(model),
                advance=advance,development=dict(growth=growth,utility=metrics["utility"],
                                                  repeat=metrics["repeat"],reach12=metrics["reach12"])))
    if not advance:
        atomic_json(OUT/"development-stop.json",dict(selected=file_identity(model),valid=True))
        return
    for stage in ("natural","special"):
        launch(stage,dict(parent=BASE,student=model))
    atomic_json(OUT/"complete.json",dict(valid=True,candidate="investment",selected=file_identity(model),
                panels={s:file_identity(OUT/s/"complete.json") for s in ("natural","special")}))


if __name__=="__main__":
    main()
