"""Preserve a fully evaluated research student without touching model pins."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive(root, branches, target, report):
    selected = load(root / "selection.json")
    plan = load(root / "plan.json")
    source = Path(selected["path"])
    weights = file_identity(source)
    if not (load(root / "complete.json")["selected"] == weights == selected["model"]):
        raise ValueError("selected weights and completed evaluation disagree")
    if load(report)["selection"] != selected or not load(branches)["valid"]:
        raise ValueError("missing verified analysis")
    manifest = load(source.with_name("manifest.json"))
    if manifest["model"] != weights:
        raise ValueError("model manifest identity mismatch")
    target.mkdir(parents=True, exist_ok=False)
    os.link(source, target / "model.pt")
    if file_identity(target / "model.pt") != weights:
        raise ValueError("archived checkpoint identity mismatch")
    atomic_json(target / "manifest.json", dict(**manifest, source_run=str(root),
                libraries=plan["plan"]["libraries"], source_identity=plan["source"],
                natural_and_special_report={"path": str(report), **file_identity(report)},
                same_state_report={"path": str(branches), **file_identity(branches)}))
    atomic_json(target / "checkpoint.json", dict(format="quilboar_choice_v1",
                model=weights, parent=plan["plan"]["base"], parent_sha256=manifest["base"],
                selected_arm=selected["candidate"], fit=str(root / "fit.json"),
                policy_search=[str(root / stage / "update.json") for stage in ("search1", "search2")]))
    data = load(report)
    all_games = data["natural"]["cohorts"]["all"]
    pigs = data["special"]["cohorts"]["all"]
    with (target / "MODEL_CARD.md").open("x", encoding="utf-8") as card:
        card.write(
            f"# Choice Quilboar research policy\n\n"
            f"Selected arm: {selected['candidate']}. Self-contained weights: {weights['sha256']}.\n\n"
            "The frozen v3265 policy supplies the prior. A learned 19-coefficient residual uses "
            "publicly visible card roles on eligible boards and shops. All decisions use normal "
            "legal masks. The training examples came from a scripted expert in natural games. "
            "Development compared imitation with two generations of whole-game policy search.\n\n"
            f"Blind natural field: {all_games['rank']['paired']['n']} paired seeds, average placement "
            f"{all_games['rank']['parent']:.3f} to {all_games['rank']['student']:.3f}. "
            f"Separate eligible field: {pigs['rank']['paired']['n']} paired seeds, "
            f"repeated native activation {pigs['repeat']['parent']:.1%} to "
            f"{pigs['repeat']['student']:.1%}. "
            "See the reports in manifest.json for uncertainty, reach and legacy outcomes. "
            "The original v3265 and the live pin are stored separately.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "branches", "target", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    archive(args.root, args.branches, args.target, args.report)
