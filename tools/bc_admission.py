"""Freeze the development winner and check a complete pair of untouched BC fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

FIELDS = ("original", "mixed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(source, target):
    pending = target.with_suffix(".tmp")
    shutil.copyfile(source, pending)
    pending.replace(target)


def selection_score(scores):
    fields = [scores[name] for name in FIELDS]
    worst = max(
        max(f["paired"]["rank"]["ci95"][1] / 0.15, -f["paired"]["top4"]["ci95"][0] / 0.03)
        for f in fields
    )
    return worst, sum(f["bc"]["rank"] for f in fields), -sum(f["bc"]["top4"] for f in fields)


def select(args):
    by_directory = {p.name: read(p / "selected.json") for p in args.screens}
    if len(by_directory) != len(args.screens):
        raise ValueError("Duplicate development screen names")
    selected = min(args.screens, key=lambda p: selection_score(by_directory[p.name]["scores"]))
    args.out.mkdir(exist_ok=False)
    copy(selected / "selected.pt", args.out / "selected.pt")
    result = {
        "winner": selected.name,
        "candidate": by_directory[selected.name]["candidate"],
        "checkpoint_sha256": digest(args.out / "selected.pt"),
        "development": by_directory,
    }
    write(args.out / "selection.json", result)
    print(json.dumps({"winner": result["winner"], "candidate": result["candidate"],
                      "checkpoint_sha256": result["checkpoint_sha256"]}))


def evaluate(args):
    frozen = read(args.out / "selection.json")
    prepared = read(args.run / "preparation.json")
    source = prepared["engine_fingerprint"]
    weight = args.out / "selected.pt"
    sha = digest(weight)
    if sha != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen candidate changed")
    fields = {}
    test = []
    for name, path in zip(FIELDS, args.reports, strict=True):
        report = read(path / "report.json")
        if (
            report["source"] != source or report["checkpoint_sha256"] != sha
            or report["placement_utility"] != prepared["placement_utility"]
            or set(report["fields"]) != {name} or list(path.glob("failure-*.json"))
            or len(report["test"]) != 1
        ):
            raise ValueError(f"Inconsistent field report in {path}")
        manifest = report["test"][0]["manifest"]
        if manifest["games"] != args.games or manifest["field"] != name or manifest["rows"] != 0:
            raise ValueError(f"Wrong independent split in {path}")
        seen = 0
        with (path / f"pairs-{name}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                pair = json.loads(line)
                model, teacher = pair["bc"], pair["teacher"]
                if (
                    model["seed"] != manifest["first_seed"] + seen
                    or model["seat"] != model["seed"] % 8 or model["lobby"].bit_count() != 5
                    or any(model[key] != teacher[key] for key in ("seed", "seat", "lobby"))
                    or any(x["rank"] not in range(1, 9) for x in (model, teacher))
                ):
                    raise ValueError(f"{name}: invalid independent pair {seen}")
                seen += 1
        if seen != args.games or report["fields"][name]["games"] != seen:
            raise ValueError(f"Incomplete independent field {name}")
        test.append(report["test"][0])
        fields[name] = report["fields"][name]
    result = {
        "source": source, "checkpoint_sha256": sha,
        "placement_utility": prepared["placement_utility"],
        "test": test, "fields": fields,
        "ppo_gate": all((
            field["paired"]["rank"]["ci95"][1] <= 0.15
            and field["paired"]["top4"]["ci95"][0] >= -0.03
        ) for field in fields.values()),
    }
    dest = args.out / "evaluation"
    dest.mkdir(exist_ok=False)
    for name, path in zip(FIELDS, args.reports, strict=True):
        leaf = f"pairs-{name}.jsonl"
        (dest / leaf).symlink_to(os.path.relpath(path / leaf, dest))
    write(dest / "report.json", result)
    print(json.dumps({"ppo_gate": result["ppo_gate"], "test_pairs": args.games * 2}))


def admit(args):
    report = read(args.out / "evaluation" / "report.json")
    check = read(args.out / "selected-check.json")
    path = args.out / "selected.pt"
    if (
        not report["ppo_gate"]
        or check["checkpoint_sha256"] != report["checkpoint_sha256"]
        or digest(path) != report["checkpoint_sha256"]
        or not check["same_weights_ratio"] or not check["finite_log_probabilities_value_and_gradients"]
        or check["optimizer_updates"] != 0 or check["valid_samples"] < 192
        or set(check["stochastic"]) != set(FIELDS)
        or any(check["stochastic"][name]["games"] < 1024 for name in FIELDS)
    ):
        raise ValueError("The candidate has not passed both BC admission checks")
    copy(path, args.out / "ppo-init.pt")
    write(args.out / "ppo-admission.json", {
        "approved": True, "checkpoint": "ppo-init.pt", "sha256": report["checkpoint_sha256"],
        "native_source": report["source"], "model": check["model"],
        "placement_utility": report["placement_utility"],
        "test_pairs": sum(x["games"] for x in report["fields"].values()),
        "evaluation": "evaluation/report.json", "policy_handoff": "selected-check.json",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    selection = stages.add_parser("select")
    selection.add_argument("out", type=Path)
    selection.add_argument("screens", type=Path, nargs="+")
    evaluation = stages.add_parser("assemble")
    evaluation.add_argument("run", type=Path)
    evaluation.add_argument("out", type=Path)
    evaluation.add_argument("reports", type=Path, nargs=2)
    evaluation.add_argument("--games", type=int, default=8192)
    handoff = stages.add_parser("admit")
    handoff.add_argument("out", type=Path)
    args = parser.parse_args()
    {"select": select, "assemble": evaluate, "admit": admit}[args.stage](args)


if __name__ == "__main__":
    main()
