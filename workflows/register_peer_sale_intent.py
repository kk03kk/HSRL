"""Register a composite research policy only after matched field gates and action parity."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.peer_minion_rules import CARD_SHA
from workflows.peer_ppo_panel import trial
from workflows.peer_sale_intent import DEVELOPMENT, FIT, RUNS, SaleIntent, panel, read
from workflows.register_peer_material import MODEL, copy_verified


class Trace:
    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        self.actor.reset()
        self.trace = sha256()
        self.count = 0

    def decide(self, game, seat):
        choice = self.actor.decide(game, seat)
        self.trace.update(bytes((game.turn, seat, choice.action)))
        self.count += 1
        return choice


def parity(args):
    import torch
    torch.set_num_threads(1)
    directory, seeds = args
    proto, _ = panel("development")
    pool = MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)
    parent = directory / read(directory/"policy.json")["parent"]["path"]
    reference = SaleIntent(make_eval_policy(str(parent), protocol=proto)["policy"], read(FIT/"complete.json")["model"], True)
    packed = make_eval_policy("research-policy:"+str(directory), protocol=proto)["policy"]
    left, right = Trace(reference), Trace(packed)
    result = []
    for seed in seeds:
        first = trial(left, pool, seed, event_turns=True)
        second = trial(right, pool, seed, event_turns=True)
        wanted = [(e["turn"], e["original"], e["executed"]) for e in reference.events if e["changed"]]
        if first != second or left.trace.digest() != right.trace.digest() or wanted != packed.overrides:
            raise ValueError(f"packed policy changed a natural decision in {seed}")
        result.append(dict(seed=seed, actions=left.count, interventions=len(wanted), trace=left.trace.hexdigest()))
    return result


def main():
    source = source_identity(ROOT)
    evidence = RUNS/"nus-peer-sale-intent-evidence-20261120-1"
    seal = read(evidence/"seal.json")
    if (source["tracked_dirty"] or not seal["passed"] or set(seal["fields"]) != {"effect","hold","market"}
            or seal["classifier"] != file_identity(FIT/"complete.json")
            or file_identity(evidence/"audit.py") != seal["script"]):
        raise ValueError("independent gates or fit identity differ")
    registry = Path("artifacts/policies")
    registry.mkdir(exist_ok=True)
    final = registry/"peer-sale-intent-v1-20261120"
    if final.exists():
        raise FileExistsError(final)
    stage = Path(mkdtemp(prefix=".pending-sale-intent-",dir=registry))
    parent_ref = "../../models/peer-material-v1-20261120/model.pt"
    parent = stage/parent_ref
    if file_identity(parent)["sha256"] != MODEL:
        raise ValueError("registered parent identity differs")
    for name in ("LICENSE",):
        copy_verified(parent.parent/name, stage/name)
    copy_verified(evidence/"audit.py",stage/"audit.py")
    copy_verified(evidence/"seal.json",stage/"seal.json")
    for name, field in seal["fields"].items():
        root = RUNS/f"nus-peer-sale-intent-{name}-20261120-1"
        target = stage/"reports"/name
        target.mkdir(parents=True)
        if not field["passed"] or field["model"] != MODEL:
            raise ValueError("unexpected field result")
        for key, filename in (("plan","plan.json"),("report","complete.json")):
            if copy_verified(root/filename,target/filename) != field[key]:
                raise ValueError("paired evidence identity differs")
        if any(file_identity(root/path) != identity for path,identity in field["shards"].items()):
            raise ValueError("paired raw evidence differs")
    copy_verified(FIT/"plan.json",stage/"fit-plan.json")
    copy_verified(FIT/"complete.json",stage/"fit-report.json")
    atomic_json(stage/"classifier.json",read(FIT/"complete.json")["model"])
    atomic_json(stage/"policy.json",dict(schema="hsai.sale_intent/v1", threshold=.8,
        classifier=file_identity(stage/"classifier.json"),parent=dict(path=parent_ref,identity=file_identity(parent)),
        feature_cards_sha256=CARD_SHA,action_profile="human-turn-3702-standard",
        engine_sources=["c4c71014e3d41760","1b04acb00da7044c"]))
    events = sorted((r for i in range(8) for r in read(DEVELOPMENT/f"events-{i}.json")),key=lambda r:r["seed"])
    seeds = [r["seed"] for r in events if any(e["selected"] for e in r["events"])][:8]
    seeds += [r["seed"] for r in events if not any(e["selected"] for e in r["events"])][:8]
    if len(set(seeds)) != 16:
        raise ValueError("insufficient deterministic parity fixtures")
    workers = min(4,int(os.environ["SLURM_CPUS_PER_TASK"]))
    with ProcessPoolExecutor(workers) as pool:
        groups = pool.map(parity,[(stage, seeds[i::workers]) for i in range(workers)])
        records = [r for group in groups for r in group]
    if sum(r["interventions"] > 0 for r in records) < 4:
        raise ValueError("insufficient exercised sale overrides during portable parity")
    atomic_json(stage/"parity.json",dict(records=records, selection="first 8 selected and 8 neutral development seeds; CPU"))
    (stage/"MODEL_CARD.md").write_text(
        "# Peer sale-intent v1\n\nComposite single-seat greedy research policy. Restore using "
        "make_eval_policy('research-policy:<directory>', protocol=..., device=...). "
        "Keep the parent checkpoint at the relative location in policy.json. "
        "The classifier predicts a subsequent refresh from public pre-sale information. "
        "Evidence covers the specific two held simulated fields in seal.json. Live MMR is unmeasured.\n",
        encoding="ascii")
    atomic_json(stage/"manifest.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],
        policy=file_identity(stage/"policy.json"),seal=file_identity(stage/"seal.json"),
        parity=file_identity(stage/"parity.json"),license=file_identity(stage/"LICENSE"),
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),live_pin=file_identity(ROOT/"config/live/model.json")))
    stage.rename(final)
    print(final, file_identity(final/"policy.json")["sha256"], flush=True)


if __name__ == "__main__":
    main()
