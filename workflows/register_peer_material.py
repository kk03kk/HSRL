"""Archive the confirmed D-161 research checkpoint and its independent field evidence."""
import json
import os
from pathlib import Path
import shutil
from statistics import median
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

RUNS = Path("artifacts/runs")
PACKAGE = RUNS / "nus-peer-sale-package-20261119-1"
MODEL = "9ecf26967e850d96e73ef8bd05b93388c30c2086d2bce1e04b4ce53403e102d9"
FIELDS = (
    ("dev", "train", 310410000, 512, "ee216399c6892c6ce2935368899fbea767f90dbb55bdbaf932051858620390c1"),
    ("hold", "hold", 314410000, 1024, "1d45f262cdb8323820beb2f258e62cdc71d6201f097ed5d175899fbad215a350"),
    ("shift", "transfer", 318410000, 512, "ebb6ba8f94e34567240033086f32b23eb834417e2f83f485b367253d11382970"),
)


def read(path):
    return json.loads(path.read_bytes())


def copy_verified(src, dst):
    identity = file_identity(src)
    with src.open("rb") as reader, dst.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if file_identity(dst) != identity:
        raise ValueError(f"archived bytes differ for {src}")
    return identity


def archive_field(target, field, manifest):
    name, split, seed, n, protocol = field
    run = RUNS / f"nus-peer-sale-{name}-20261119-1"
    cdf = RUNS / f"nus-peer-cdf-{name}-20261120-1" / "seal.json"
    plan, report, audit = (read(p) for p in (run/"plan.json", run/"complete.json", cdf))
    if (plan["split"], plan["games"], plan["seed"], plan["protocol"], plan["parent"]) != (
            split, n, seed, protocol, manifest["initial"]):
        raise ValueError(f"{name} field identity differs")
    if (report["n"] != n or audit["n"] != n or audit["seed"] != seed or not report["passed"]
            or report["delta"]["utility"]["ci95"][0] <= 0 or report["delta"]["log12"]["ci95"][0] < -.12
            or abs(report["delta"]["utility"]["mean"] - audit["utility"]["mean"]) > 1e-10):
        raise ValueError(f"{name} results failed confirmation")
    dst = target/"reports"/name
    dst.mkdir(parents=True)
    evidence = {p.name:copy_verified(p, dst/p.name) for p in (run/"plan.json", run/"complete.json", cdf)}
    by_arm = {"parent":{}, "candidate":{}}
    coverage = []
    shards = {}
    for i in range(8):
        p = run/f"panel-{i}.json"
        shards[p.name] = file_identity(p)
        for pair in read(p):
            if pair["parent"]["seed"] != pair["candidate"]["seed"] or pair["parent"]["lobby"] != pair["candidate"]["lobby"]:
                raise ValueError("unpaired episode")
            coverage.append(pair["parent"]["seed"])
            for arm, game in pair.items():
                for turn, board in game["turns"].items():
                    by_arm[arm].setdefault(turn, []).append(board["atk"]+board["hp"])
    if sorted(coverage) != list(range(seed, seed+n)):
        raise ValueError("missing or duplicate episode")
    curves = {arm:{t:dict(n=len(v),p50=median(v)) for t,v in turns.items()} for arm,turns in by_arm.items()}
    atomic_json(dst/"medians.json",curves)
    return dict(run=str(run),protocol=protocol,library=plan["library"],reports=evidence,
                panels=shards,medians=file_identity(dst/"medians.json"))


def main():
    source = source_identity(ROOT)
    manifest, verified = read(PACKAGE/"manifest.json"), read(PACKAGE/"verified.json")
    if (source["tracked_dirty"] or file_identity(PACKAGE/"model.pt")["sha256"] != MODEL
            or manifest["model"] != file_identity(PACKAGE/"model.pt")
            or (verified["games"],verified["actions"],verified["affected"]) != (64,29732,16)):
        raise ValueError("checkpoint or parity evidence differs")
    final = Path("artifacts/models/peer-material-v1-20261120")
    if final.exists():
        raise FileExistsError(final)
    staging = Path(mkdtemp(prefix=".pending-peer-material-", dir=final.parent))
    members = {name:copy_verified(PACKAGE/name,staging/name) for name in
               ("model.pt","checkpoint.json","manifest.json","MODEL_CARD.md","verified.json")}
    ref = manifest["reference"]
    members["LICENSE"] = copy_verified(Path(ref["directory"])/"LICENSE",staging/"LICENSE")
    if members["LICENSE"]["sha256"] != ref["files"]["LICENSE"]:
        raise ValueError("unexpected upstream license")
    fields = {f[0]:archive_field(staging,f,manifest) for f in FIELDS}
    atomic_json(staging/"research.json",dict(source=source,job=os.environ["SLURM_JOB_ID"],
        scope="Research reference for human-turn simulated fields",members=members,fields=fields))
    staging.rename(final)
    print(final, MODEL, flush=True)


if __name__ == "__main__":
    main()
