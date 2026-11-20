"""Qualify and train one 37.0.3 round from the sealed R1 generation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.peer_3703_field import ENGINE, LIB, PARENT, PARENT_SHA, R1, R1_SHA, TRAINER_SHA
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

SEED = 452410000
MONITOR_SEED = 523730000
PROFILE = "human-turn-3702-economy"
MODELS = []
POOL = None
OLD = Path("artifacts/runs/nus-peer-rounds-production-20261120-1/round-01")

def read(path):
    return json.loads(path.read_bytes())

def verify_asset(path, sha):
    if file_identity(path)["sha256"] != sha:
        raise ValueError("unexpected identity: " + str(path))

def old_generation():
    from workflows.human_turn_preflight import verified_generation
    old = read(OLD/"train.json")
    parent = verified_generation(Path(old["fit"]), SEED)
    if parent != old["checkpoint"] or parent["files"]["model.pt"]["sha256"] != R1_SHA or parent["version"] != 647:
        raise ValueError("unsealed R1 parent")
    verify_asset(Path(parent["trainer"]),TRAINER_SHA)
    verify_asset(PARENT,PARENT_SHA)
    return parent

def field(split):
    from hsbg.az.train.personal import checked_library
    from hsbg.az.train.common import engine_info
    from hsbg.az.train.eval_policy import EvalProtocol, protocol_sha256_of
    man=checked_library(LIB/split)
    if man["split"] != split or man["fingerprint"]["engine"]["source_hash"] != ENGINE:
        raise ValueError("invalid field")
    proto=EvalProtocol(mirror_snaps=str(LIB/split),mirror_sample="random",mirror_strict=1,
                      engine=engine_info(),human_tail="last-observed-v1",tp_forbid_cancel=True,keep_start_board=False)
    proto.assert_training_contract()
    spec=dict(protocol=proto.as_dict(),library_sha256=man["digest"], tail="last-observed-v1",
              format="paired single learner, seven strict EOT opponents")
    return proto,man,spec,protocol_sha256_of(spec)

def game_pair(seed):
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    rows = []
    for policy in MODELS:
        r=evaluate_policy(policy,EvalConfig(games=1,seed=seed,mirror_pool=POOL,mirror_strict=True,collect_economy=True))
        m=r["mirror"]
        if (r["games_finished"]!=1 or r["seed_sequence"]!=[seed] or any(r[k] for k in
            ("illegal_actions","truncated_games","truncated_combats")) or not m["mirror_applied"]
            or m["mirror_applied"]!=m["mirror_attempt"] or any(m["mirror_"+k] for k in
            ("errors","rejected","missing","fallback","pool_unbalanced","pool_species_unbalanced"))):
            raise ValueError(("invalid paired match",seed,m))
        boards={int(t["turn"]):t["eot_attack"]+t["eot_board_hp"] for t in r["end_turn_economy"]}
        rows.append(dict(rank=r["rank_sequence"][0],boards=boards,injections=m["mirror_applied"],
                         tail=r["mirror_strict_counts"].get("human_last_observed_live_seats",0)))
    return dict(seed=seed,base=rows[0],candidate=rows[1])

def panel(split,models,start,n,out):
    global MODELS,POOL
    import torch
    from hsbg.az.train.eval_policy import make_eval_policy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from tools.summarize_peer_distillation import estimate
    torch.set_num_threads(1)
    proto,man,spec,sha=field(split)
    POOL=MirrorSnapshotPool(str(LIB/split),sample="random")
    assert POOL.frames==man["frames"] and not POOL.bad_files
    MODELS=[make_eval_policy(str(p),protocol=proto,device="cpu")["policy"] for p in models]
    # Process-local actors and read-only snapshot banks.
    with ProcessPoolExecutor(min(n,int(os.environ["SLURM_CPUS_PER_TASK"])),
                             mp_context=multiprocessing.get_context("fork")) as pool:
        rows=list(pool.map(game_pair,range(start,start+n)))
    assert [r["seed"] for r in rows]==list(range(start,start+n))
    atomic_json(out/"pairs.json",rows)
    def growth(b):
        return math.log1p(b[12])-math.log1p(b[9])
    common=[r for r in rows if all(9 in r[k]["boards"] and 12 in r[k]["boards"] for k in ("base","candidate"))]
    metrics={}
    for k in ("base","candidate"):
        ranks=[r[k]["rank"] for r in rows]
        curves={}
        for turn in range(1,max(max(r[k]["boards"]) for r in rows)+1):
            xs=[r[k]["boards"][turn] for r in rows if turn in r[k]["boards"]]
            curves[turn]=dict(n=len(xs),p50=median(xs) if xs else None)
        metrics[k]=dict(mean_rank=sum(ranks)/n,top4=sum(i<=4 for i in ranks)/n,win=ranks.count(1)/n,
                        curves=curves,injections=sum(r[k]["injections"] for r in rows),tail=sum(r[k]["tail"] for r in rows))
    result=dict(n=n,start=start,models=[file_identity(p) for p in models],protocol_sha256=sha,
                field=spec,source=source_identity(ROOT),metrics=metrics,
                rank=estimate([r["candidate"]["rank"]-r["base"]["rank"] for r in rows]),
                growth=estimate([growth(r["candidate"]["boards"])-growth(r["base"]["boards"]) for r in common]))
    atomic_json(out/"complete.json",result)
    print(json.dumps({k:result[k] for k in ("n","rank","growth","protocol_sha256")}),flush=True)
    return result

def production(args,dest,parent):
    from workflows.human_turn_preflight import allocation_end, verified_generation
    from workflows.peer_open_ppo import inspect_trainer
    baseline_path=args.out/"baseline/complete.json"
    verify_asset(baseline_path,args.baseline_sha)
    smoke=read(args.out/"train-test/complete.json")
    hold=read(baseline_path)
    _,train,train_spec,train_sha=field("train")
    _,held,_,hold_sha=field("hold")
    if (set(train["games"])&set(held["games"]) or hold["protocol_sha256"]!=hold_sha
        or smoke["protocol_sha256"]!=train_sha or hold["n"]!=64 or smoke["n"]!=8
        or [i["sha256"] for i in hold["models"]] != [PARENT_SHA,R1_SHA]):
        raise ValueError("misaligned new fields or missing qualification")
    if os.environ["SLURM_JOB_PARTITION"]!="gpu":
        raise ValueError("unexpected partition")
    source=source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("dirty release")
    config=read(ROOT/"config/research/peer-gpu-rounds.json")
    stop=allocation_end()-240
    if stop-time.time()<600:
        raise ValueError("insufficient allocation")
    env=dict(config["environment"],HSAI_ARENA_SNAPS=str(LIB/"train"),HSAI_ACTION_PROFILE=PROFILE,
             HSRL_ROOT=str(ROOT),HSAI_ARENA_SEED=str(SEED),HSAI_ARENA_CKPT=str(R1.resolve()),
             HSAI_ARENA_PARENT_TRAINER=parent["trainer"],HSAI_ARENA_STOP_EPOCH=str(stop),
             HSAI_ARENA_MINUTES="175",HSAI_ARENA_ITERATIONS="0",HSAI_ARENA_OUT=str(dest/"fit"))
    state=inspect_trainer(parent,initial_model=PARENT)
    manifest=dict(parent=parent,trainer=state,source=source,env=env,train_field=train_spec,
                  protocol_sha256=hold_sha,baseline=file_identity(baseline_path),
                  library_seal=file_identity(LIB/"complete.json"), engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
                  live_pin=read(ROOT/"config/live/model.json")["sha256"],generation_record=file_identity(R1.parent/"record.json"),
                  job=os.environ["SLURM_JOB_ID"])
    atomic_json(dest/"plan.json",manifest)
    subprocess.run(["bash","deploy/slurm/stage_arena.sbatch"],env=os.environ|env,check=True)
    seal=verified_generation(dest/"fit",SEED)
    if seal["version"]<=647 or seal["cum_decisions"]<=parent["cum_decisions"]:
        raise ValueError("training did not advance")
    atomic_json(dest/"complete.json",dict(checkpoint=seal,trainer=inspect_trainer(seal,initial_model=PARENT),
                                         train_field_sha256=train_sha,plan=file_identity(dest/"plan.json")))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("baseline","train-test","train","monitor"))
    p.add_argument("out",type=Path)
    p.add_argument("--baseline-sha")
    args=p.parse_args()
    os.chdir(ROOT)
    args.out=args.out.resolve()
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_","HSAI_RECRUIT_START_","HSAI_QUILL_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE=PROFILE,HSAI_HUMAN_TAIL="last-observed-v1")
    dest=args.out/args.stage
    dest.mkdir(exist_ok=False)
    parent=old_generation()
    if args.stage in ("baseline","train-test"):
        train=args.stage=="train-test"
        panel("train" if train else "hold",[PARENT,R1],526730000 if train else MONITOR_SEED,8 if train else 64,dest)
    elif args.stage=="train":
        production(args,dest,parent)
    else:
        from workflows.human_turn_preflight import verified_generation
        report=read(args.out/"train/complete.json")
        seal=verified_generation(args.out/"train/fit",SEED)
        if seal!=report["checkpoint"]:
            raise ValueError("endpoint identity mismatch")
        baseline=read(args.out/"baseline/complete.json")
        _,_,_,sha=field("hold")
        if sha!=baseline["protocol_sha256"]:
            raise ValueError("hold field changed")
        panel("hold",[R1,Path(seal["model"])],MONITOR_SEED,256,dest)
