"""Preview or submit the five qualified rounds from the frozen NUS release."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity


def schedule(rounds):
    for index in range(1, rounds+1):
        yield index, "train"
        yield index, "monitor"
    yield rounds, "external"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", required=True, type=Path)
    p.add_argument("--qualification", required=True, type=Path)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    out, check = args.series.resolve(), args.qualification.resolve()
    recipe = ROOT / "config/research/player150-five-rounds.json"
    config = json.loads(recipe.read_bytes())
    if args.submit:
        evidence = [json.loads((check / k / "qualified.json").read_bytes()) for k in ("baseline", "smoke")]
        if (evidence[0] != evidence[1] or evidence[0]["source"] != source_identity(ROOT)
            or evidence[0]["source"]["tracked_dirty"] or evidence[0]["recipe"] != file_identity(recipe)):
            raise ValueError("release has not passed both qualification stages")
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "series.json", dict(identity=evidence[0], qualification=str(check), rounds=config["rounds"]))
    previous = None
    for index, stage in schedule(config["rounds"]):
        gpu = stage == "train"
        resources = (["-p","gpu","--gres=gpu:h100-47:2","-c","32","--mem=96G","-t","03:00:00",
                      "--exclude=xgpi11,xgpi15,xgpi17"] if gpu else
                     ["-p","normal","-c","24","--mem=40G","-t","02:30:00"])
        command = [str(ROOT/".venv/bin/python"), "workflows/player150_rounds.py", stage, "--round",str(index),
                   "--series", str(out), "--qualification", str(check)]
        args_s = ["sbatch","--parsable","--account=allusers","--qos=normal","--propagate=NONE",
                  "--kill-on-invalid-dep=yes","--chdir="+str(ROOT),"-J",f"p150-{stage}{index}",
                  "-o",str(out / f"{stage}-{index}-%j.log"),*resources]
        if previous:
            args_s.append("--dependency=afterok:"+previous)
        args_s += ["--wrap", "exec "+shlex.join(command)]
        if args.submit:
            previous = subprocess.check_output(args_s,text=True).strip().split(";")[0]
            if not previous.isdecimal():
                raise ValueError("unrecognized Slurm job ID")
            atomic_json(out / f"submit-{index}-{stage}.json", dict(job=previous, args=args_s))
            print(index,stage,previous,flush=True)
        else:
            print(shlex.join(args_s))
            previous = f"JOB_{index}_{stage}"


if __name__ == "__main__":
    main()
