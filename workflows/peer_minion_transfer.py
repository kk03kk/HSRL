"""Evaluate both frozen checkpoints on the three-minion sensitivity build, using paired CPU seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.az.train.common import engine_info
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.lobby_filter import engine_rule_overrides
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare, estimate
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, read
from workflows.peer_minion_rules import RULES, CARDS
from workflows.peer_ppo_panel import trial
from workflows.peer_stop_learning import BASE_SHA

RUNS = Path("artifacts/runs")
OLD = RUNS/"nus-peer-sale-shift-20261119-1"
MODEL = Path("artifacts/models/peer-material-v1-20261120/model.pt")
OLD_PROTOCOL = "ebb6ba8f94e34567240033086f32b23eb834417e2f83f485b367253d11382970"
LIB = "8a769ef0f43fc8edf2a718514898b1ae4e60831aa8bf37355e11c4d2627c4be0"
START, N = 318410000, 512


def protocol(build):
    import hsbg_engine
    info = read(build/"complete.json")
    identity = info["identity"]
    if (not Path(hsbg_engine.__file__).resolve().is_relative_to((build/"site").resolve())
            or file_identity(build/"checkout"/CARDS) != identity["card_table"]
            or file_identity(build/"checkout/research-engine-lock.json") != identity["research_lock"]
            or identity["source_hash"] != "77b1f008c9d046b2"):
        raise ValueError("unverified sensitivity installation")
    previous = read(OLD/"plan.json")
    original = previous["field"]
    engine = engine_info()
    expected = dict(original["engine"],source_hash=identity["source_hash"],server_rules_version=RULES)
    if dict((k,engine[k]) for k in expected) != expected:
        raise ValueError("runtime binary or layout changed")
    proto = EvalProtocol(**original["contract"], **original["field"], engine=expected,
        tp_forbid_cancel=original["tp_forbid_cancel"],keep_start_board=original["keep_start_board"])
    canonical = replace(proto,engine=original["engine"])
    if (canonical.sha256()!=previous["protocol"] or previous["protocol"]!=OLD_PROTOCOL
            or proto.engine_rules!=engine_rule_overrides() or proto.sha256()==OLD_PROTOCOL
            or library_digest(proto.mirror_snaps)!=LIB):
        raise ValueError("protocol or game library differs beyond the three minion changes")
    proto.assert_training_contract()
    return proto


def shard(task):
    build,out,start,n,idx,shards = task
    import torch
    torch.set_num_threads(1)
    proto = protocol(build)
    pool = MirrorSnapshotPool(proto.mirror_snaps,sample=proto.mirror_sample)
    models = [make_eval_policy(str(p),protocol=proto,device="cpu")["policy"] for p in (INITIAL,MODEL)]
    rows = []
    for seed in range(start+idx,start+n,shards):
        a,b = [trial(m,pool,seed,event_turns=True) for m in models]
        if a["lobby"]!=b["lobby"]:
            raise ValueError("unpaired lobbies")
        rows.append(dict(parent=a,candidate=b))
    atomic_json(out/f"chunk-{idx}.json",rows)


def run(build,out,smoke):
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(INITIAL)["sha256"]!=INITIAL_SHA or file_identity(MODEL)["sha256"]!=BASE_SHA:
        raise ValueError("research inputs differ")
    proto=protocol(build)
    n,shards = (8,2) if smoke else (N,24)
    out.mkdir(exist_ok=False)
    atomic_json(out/"plan.json",dict(job=os.environ["SLURM_JOB_ID"],source=source,canonical_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"),manifest=file_identity(MODEL.parent/"manifest.json"),sensitivity=file_identity(build/"complete.json"),
        parent=INITIAL_SHA,candidate=BASE_SHA,protocol=proto.sha256(),field=proto.as_dict(),library=LIB,device="cpu",seed=START,games=n))
    with ProcessPoolExecutor(shards) as pool:
        list(pool.map(shard,[(build,out,START,n,i,shards) for i in range(shards)]))
    rows=sorted((r for i in range(shards) for r in read(out/f"chunk-{i}.json")),key=lambda r:r["parent"]["seed"])
    for i in range(8):
        atomic_json(out/f"panel-{i}.json",rows[i::8])
    result=compare(out,START,n)
    old={r["parent"]["seed"]:r for i in range(8) for r in read(OLD/f"panel-{i}.json")}
    if any(r["parent"]["lobby"]!=old[r["parent"]["seed"]]["parent"]["lobby"] for r in rows):
        raise ValueError("same seed produced different starting tribe pools")
    def margin(r):
        return r["candidate"]["metrics"]["utility"]-r["parent"]["metrics"]["utility"]
    result.update(change_in_advantage=estimate([margin(r)-margin(old[r["parent"]["seed"]]) for r in rows]),
        old_panels=[file_identity(OLD/f"panel-{i}.json") for i in range(8)],smoke=smoke,
        passed=not smoke and result["delta"]["utility"]["ci95"][0]>0 and result["delta"]["growth9to12"]["ci95"][0]>0)
    atomic_json(out/"complete.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.build,args.out,args.smoke)
