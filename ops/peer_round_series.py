"""Submit or preview the frozen peer round series from a login node using only git and sbatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

RECIPE = ROOT / "config/research/peer-gpu-rounds.json"
PANELS = ("monitor", "blind", "adaptive")


def read(path):
    return json.loads(Path(path).read_bytes())


def settings(smoke):
    cfg = read(RECIPE)
    if smoke:
        trial = cfg["smoke"]
        cfg.update(rounds=trial["rounds"], train_seed=trial["train_seed"])
        for name in PANELS:
            cfg[name] = dict(seed=trial[name + "_seed"], games=trial["games"])
    panels = [range(cfg[k]["seed"], cfg[k]["seed"] + cfg[k]["games"]) for k in PANELS]
    if max(p.stop for p in panels) >= cfg["train_seed"] or len(set().union(*panels)) != sum(map(len, panels)):
        raise ValueError("overlapping panel or training seeds")
    return cfg


def stages(rounds):
    for index in range(1, rounds + 1):
        yield index, "train"
        yield index, "monitor"
    yield rounds, "blind"
    yield rounds, "adaptive"


def job_args(stage, smoke):
    exclusions = ["--exclude=xgpi11,xgpi15,xgpi17"]
    if stage == "adaptive":
        return ["-p", "normal", "-c", "32", "--mem=48G", "-t", "01:50:00"]
    if stage == "train":
        return ["-p", "gpu", "--gres=gpu:h100-47:2", "-c", "32", "--mem=96G",
                "-t", "00:15:00" if smoke else "03:00:00", *exclusions]
    return ["-p", "gpu", "--gres=gpu:h100-47:1", "-c", "16", "--mem=48G",
            "-t", "00:15:00" if smoke else "00:55:00", *exclusions]


def submit(out, cfg, smoke, dry_run):
    source = source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("use a clean, frozen release")
    if not dry_run:
        if ROOT.parent.name != "releases":
            raise ValueError("submit from an immutable server release")
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "series.json", dict(source=source, recipe=file_identity(RECIPE), smoke=smoke))
    parent = None
    for index, stage in stages(cfg["rounds"]):
        cmd = ["sbatch", "--parsable", "-A", "allusers", "-q", "normal", "--propagate=NONE",
               "--kill-on-invalid-dep=yes", "-J", f"peer_r{index}_{stage}",
               "--output=artifacts/runs/slurm/%x-%j.log", *job_args(stage, smoke)]
        if parent:
            cmd.append("--dependency=afterok:" + parent)
        cmd.extend(["deploy/slurm/peer_round_stage.sh", stage, str(out), str(index)])
        if smoke:
            cmd.append("--smoke")
        if dry_run:
            print(" ".join(cmd))
            parent = f"<{stage}-{index}>"
        else:
            parent = subprocess.check_output(cmd, text=True).strip().split(";")[0]
            if not parent.isdigit():
                raise ValueError("unexpected scheduler response")
            atomic_json(out / f"submit-{index}-{stage}.json", dict(job=parent, command=cmd))
            print(index, stage, parent, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "submit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    submit(args.out.resolve(), settings(args.smoke), args.smoke, args.action == "preview")
