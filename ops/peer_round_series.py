"""Submit or preview the frozen peer round series from a login node using only git and sbatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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


def round_one_ready(scores, gate):
    d = scores["delta"]
    return (d["rank"]["mean"] < 0 and d["growth9to12"]["mean"] > 0
            and d["utility"]["mean"] >= 0
            and d["rank"]["ci95"][1] <= gate["maximum_rank_delta_ci95_upper"]
            and d["growth9to12"]["ci95"][0] >= gate["minimum_growth_delta_ci95_lower"]
            and d["reach12"]["mean"] >= gate["minimum_reach12_delta"])


def validate_continuation(out, cfg, source, smoke):
    series = read(out / "series.json")
    if (series["source"]["commit"] != source["commit"] or series["recipe"] != file_identity(RECIPE)
            or series["smoke"] != smoke):
        raise ValueError("use the original series release and recipe")
    monitor = read(out / "round-01/monitor.json")
    training = read(out / "round-01/train.json")
    if (monitor["checkpoint"] != training["checkpoint"]["files"]["model.pt"]
            or monitor["seed"] != cfg["monitor"]["seed"] or monitor["n"] != cfg["monitor"]["games"]
            or monitor["protocol"] != cfg["protocol_sha256"] or file_identity(monitor["report"]) != monitor["report_identity"]):
        raise ValueError("invalid first-round assessment")
    for attempt in range(6):
        state = subprocess.check_output(
            ["sacct", "-X", "-n", "-P", "-j", monitor["job"], "--format=JobID,State"], text=True)
        if [monitor["job"], "COMPLETED"] in [row.strip().split("|")[:2] for row in state.splitlines()]:
            break
        if attempt < 5:
            time.sleep(3)
    else:
        raise ValueError("first-round assessment has not finished successfully")
    if not smoke and not round_one_ready(read(monitor["report"]), cfg["round_one_budget_gate"]):
        decision = out / "budget-not-released.json"
        if not decision.exists():
            atomic_json(decision, dict(first_monitor=monitor["report_identity"], gate=cfg["round_one_budget_gate"]))
        print("First round complete. Remaining allocations not submitted.")
        return False
    atomic_json(out / "continuation.json", dict(first_monitor=monitor["report_identity"], source=source))
    return True


def submit(out, cfg, smoke, dry_run, continuation=False):
    source = source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("use a clean, frozen release")
    if not dry_run:
        if ROOT.parent.name != "releases":
            raise ValueError("submit from an immutable server release")
        if continuation:
            if not validate_continuation(out, cfg, source, smoke):
                return
        else:
            out.mkdir(parents=True, exist_ok=False)
            atomic_json(out / "series.json", dict(source=source, recipe=file_identity(RECIPE), smoke=smoke))
    parent = None
    for index, stage in stages(cfg["rounds"]):
        if continuation and index == 1:
            continue
        if not dry_run and not continuation and index > 1:
            break
        if dry_run and index == 2 and stage == "train":
            print("# automatic CPU budget decision after round 1; remaining jobs submitted only on release")
            parent = None
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
    if not continuation and not dry_run:
        cmd = ["sbatch", "--parsable", "-A", "allusers", "-q", "normal", "-p", "normal", "-c", "1",
               "--mem=1G", "-t", "00:05:00", "-J", "peer_r1_budget", "--output=artifacts/runs/slurm/%x-%j.log",
               "--dependency=afterok:" + parent, "--kill-on-invalid-dep=yes", "deploy/slurm/peer_round_budget.sh", str(out)]
        if smoke:
            cmd.append("--smoke")
        job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("unexpected scheduler response for budget decision")
        atomic_json(out / "submit-1-budget.json", dict(job=job, command=cmd))
        print(1, "budget", job, flush=True)


def progress(out):
    """Print the learning curve from completed monitoring panels."""
    cfg = settings(read(out / "series.json")["smoke"])
    print("round decisions rank top4% win% T8_P50/n T10_P50/n T12_P50/n T14_P50/n rank_delta[95%] growth_9_12_delta[95%]/n")
    for index in range(1, cfg["rounds"] + 1):
        seal = out / f"round-{index:02d}/monitor.json"
        if not seal.exists():
            break
        panel = read(seal)
        if (panel["protocol"] != cfg["protocol_sha256"] or panel["seed"] != cfg["monitor"]["seed"]
                or file_identity(panel["report"]) != panel["report_identity"]):
            raise ValueError("unexpected monitoring evidence")
        scores = read(panel["report"])
        for arm in ("parent", "candidate"):
            model = scores["models"][arm]
            m = model["means"]
            bodies = [model["turns"].get(str(t), {"p50": None, "n": 0}) for t in (8, 10, 12, 14)]
            bounds = []
            if arm == "candidate":
                for key in ("rank", "growth9to12"):
                    d = scores["delta"][key]
                    if d["n"]:
                        lo, hi = d["ci95"]
                        bounds.append(f'{d["mean"]:+.3f}[{lo:+.3f},{hi:+.3f}]/{d["n"]}')
                    else:
                        bounds.append("NA/0")
            print(f'{index}-{arm}', panel["cumulative_decisions"] if arm == "candidate" else 0,
                  f'{m["rank"]:.3f}', f'{100*m["top4"]:.1f}', f'{100*m["win"]:.1f}',
                  *(f'{b["p50"]}/{b["n"]}' for b in bodies), *bounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "submit", "continue", "progress"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "progress":
        progress(args.out.resolve())
    else:
        submit(args.out.resolve(), settings(args.smoke), args.smoke, args.action == "preview", args.action == "continue")
