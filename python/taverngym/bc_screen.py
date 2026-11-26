"""Compare frozen BC candidates on a shared development panel."""

import argparse
from pathlib import Path

import torch

from taverngym import build_info

from taverngym.bc_train import Demonstrations, configure, emit, json_file, paired, play, save
from taverngym.entity_policy import EntityPolicy


def run(args):
    if not 0 <= args.part < args.parts:
        raise ValueError("Invalid candidate partition")
    configure(0)
    panel = Demonstrations(args.panel, load_rows=False)
    groups = {
        field: [s for s, (f, _) in sorted(panel.games.items()) if f == field]
        for field in ("original", "mixed")
    }
    if not all(groups.values()):
        raise ValueError("Development comparison requires both opponent fields")
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoints = {}
    for path in args.checkpoints:
        name = path.stem if path.stem not in checkpoints else f"{path.parent.name}-{path.stem}"
        if name in checkpoints:
            raise ValueError(f"Duplicate checkpoint name {name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["meta"]
        if meta["source"] != build_info()["source_hash"]:
            raise ValueError(f"Different engine for {path}")
        if any(
            s in panel.games
            for key in ("train", "dev", "screen", "upstream_splits")
            for split in meta.get(key, [])
            for s in range(split["manifest"]["first_seed"],
                           split["manifest"]["first_seed"] + split["manifest"]["games"])
        ):
            raise ValueError(f"Development panel overlaps a prior split for {path}")
        if "optimizer" in checkpoint:
            if checkpoint["block"]:
                raise ValueError("Screening requires a completed training epoch")
            checkpoint["best"] = {"epoch": checkpoint["epoch"], "snapshot": "latest"}
        checkpoints[name] = checkpoint
    if args.blend:
        endpoints = list(checkpoints)[:2]
        if len(endpoints) != 2:
            raise ValueError("Interpolation requires two checkpoint endpoints")
        first, last = (checkpoints[name] for name in endpoints)
        for key in ("model", "source"):
            if first["meta"][key] != last["meta"][key]:
                raise ValueError(f"Interpolation requires matching checkpoint {key}")
        if first["meta"].get("placement_utility") != last["meta"].get("placement_utility"):
            raise ValueError("Interpolation requires matching placement utility")
        for key in ("static_cards", "rule_cards"):
            if not torch.equal(first["model"][key], last["model"][key]):
                raise ValueError(f"Interpolation requires matching model buffer {key}")
        metadata = {
            **first["meta"],
            "upstream_splits": [
                split
                for c in (first, last)
                for key in ("train", "dev", "screen", "upstream_splits")
                for split in c["meta"].get(key, [])
            ],
        }
        for weight in (0.25, 0.5, 0.75):
            blend = {
                k: a.lerp(last["model"][k], weight) if a.is_floating_point() else a
                for k, a in first["model"].items()
            }
            checkpoints[f"blend-{weight}"] = {
                "meta": metadata,
                "model": blend,
                "best": {"interpolation": weight, "from": endpoints},
            }
    metrics = {}
    candidates = list(checkpoints.items())[args.part :: args.parts]
    if not candidates:
        raise ValueError("No candidates in this partition")
    for name, checkpoint in candidates:
        model = EntityPolicy(**checkpoint["meta"]["model"]).to("cuda")
        model.load_state_dict(checkpoint["model"])
        metrics[name] = {}
        for field, seeds in groups.items():
            rows = play(model, seeds, field, args.arena_batch, args.workers, args.out / f"failure-{name}-{field}.json")
            metrics[name][field] = paired(
                rows, panel.games, checkpoint["meta"].get("placement_utility")
            )
        json_file(args.out / "scores.json", metrics)
        emit(
            {
                "candidate": name,
                "fields": {
                    k: {m: v["bc"][m] for m in ("rank", "first", "top4")}
                    for k, v in metrics[name].items()
                },
            }
        )

    def criterion(name):
        result = list(metrics[name].values())
        rank = sum(row["bc"]["rank"] for row in result)
        top4 = sum(row["bc"]["top4"] for row in result)
        if not args.gate:
            return rank, -top4
        worst = max(
            max(row["paired"]["rank"]["ci95"][1] / 0.15, -row["paired"]["top4"]["ci95"][0] / 0.03)
            for row in result
        )
        return worst, rank, -top4

    choice = min(metrics, key=criterion)
    selected = checkpoints[choice]
    prior_panels = selected["meta"].get("screen", [])
    selected["meta"] = {
        **selected["meta"],
        "screen": prior_panels if prior_panels == panel.identity else prior_panels + panel.identity,
    }
    save(
        args.out / "selected.pt",
        {k: selected[k] for k in ("model", "meta", "best")},
    )
    json_file(args.out / "selected.json", {"candidate": choice, "scores": metrics[choice]})
    emit({"selected": choice})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--panel", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--arena-batch", type=int, default=384)
    parser.add_argument("--blend", action="store_true")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument(
        "--gate", action="store_true", help="Rank candidates by worst normalized gate bound"
    )
    run(parser.parse_args())
