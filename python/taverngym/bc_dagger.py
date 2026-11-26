"""Label states visited by the learner with the frozen HSRL teacher, then export BC shards."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from taverngym import N_ACTIONS, OBS_CARDS, OBS_FLOATS, build_info, server_rules_version
from taverngym._core import bc_teacher_name
from taverngym.bc_train import ROW, Demonstrations, configure, emit, json_file, play
from taverngym.entity_policy import EntityPolicy


class Recorder:
    def __init__(self, binary, index, beta):
        self.binary = binary
        self.index = index
        self.beta = beta
        self.offset = 0
        self.corrections = 0
        self.expert_actions = 0
        self.games = 0

    def start(self, group):
        self.rollin = np.random.default_rng([29177, group[0]])
        self.examples = [bytearray() for _ in group]

    def step(self, arena, slots, features, cards, mask, actions, choice):
        selection = np.flatnonzero(choice)
        targets = np.frombuffer(arena.teacher_targets(), dtype="<u2").reshape(len(actions), 3)
        if not np.array_equal((targets[:, 2] & 4) != 0, choice):
            raise RuntimeError("Teacher and arena disagree about forced actions")
        labels = targets[:, 0]
        if not mask[np.arange(len(actions)), labels].all():
            raise RuntimeError("Teacher returned a masked action")
        disagree = labels != actions
        self.corrections += int(disagree.sum())
        rows = np.empty(len(selection), dtype=ROW)
        rows["features"] = np.frombuffer(features, dtype="<f4").reshape(-1, OBS_FLOATS)[selection]
        rows["cards"] = np.frombuffer(cards, dtype="<i4").reshape(-1, OBS_CARDS)[selection]
        rows["mask"] = mask[selection]
        rows["action"] = labels[selection]
        rows["turn"] = targets[selection, 1]
        rows["flags"] = targets[selection, 2] | (disagree[selection].astype("u1") << 3)
        rows["rank"] = 0
        players = np.frombuffer(slots, dtype="<u4")[selection]
        payload = memoryview(rows).cast("B")
        for i, player in enumerate(players):
            self.examples[player].extend(payload[i * ROW.itemsize : (i + 1) * ROW.itemsize])
        use_expert = self.rollin.random(len(actions)) < self.beta
        self.expert_actions += int((use_expert & disagree).sum())
        return np.where(use_expert, labels, actions)

    def finish(self, finished):
        for game, example in zip(finished, self.examples, strict=True):
            rows = np.frombuffer(example, dtype=ROW)
            rows["rank"] = game["rank"]
            self.index.write(
                json.dumps({"game": game, "offset": self.offset, "rows": len(rows)}) + "\n"
            )
            self.binary.write(example)
            self.offset += len(rows)
            self.games += 1
        self.binary.flush()
        self.index.flush()
        emit({"games": self.games, "rows": self.offset, "corrections": self.corrections})


def collect(args):
    configure(0)
    if not 0 <= args.beta <= 1 or args.games < 1:
        raise ValueError("Invalid demonstration count or expert intervention probability")
    with args.checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["meta"]["source"] != build_info()["source_hash"]:
        raise ValueError("Student checkpoint was trained with a different simulator")
    model = EntityPolicy(**checkpoint["meta"]["model"]).to("cuda")
    for key in ("static_cards", "rule_cards"):
        if not torch.equal(checkpoint["model"][key], model.state_dict()[key].cpu()):
            raise ValueError(f"Teacher data has a different card table: {key}")
    model.load_state_dict(checkpoint["model"])
    seeds = list(range(args.first, args.first + args.games))
    args.out.mkdir()
    with (
        (args.out / "rows-000.bin").open("wb") as binary,
        (args.out / "games-000.jsonl").open("w", encoding="utf-8") as index,
    ):
        recorder = Recorder(binary, index, args.beta)
        play(
            model, seeds, args.field, args.batch, args.workers, args.out / "failure.json", recorder
        )
        if args.verify:
            if args.beta:
                raise ValueError("Neutrality verification requires beta zero")
            unlabelled = play(model, seeds, args.field, args.batch, args.workers)
            labelled = [
                json.loads(s)["game"]
                for s in (args.out / "games-000.jsonl").read_text().splitlines()
            ]
            if unlabelled != labelled:
                raise RuntimeError("Label collection perturbed the played games")
            emit({"label_rng_isolated": True, "games": len(unlabelled)})
    manifest = {
        "format": 2,
        "teacher": bc_teacher_name(),
        "allow_move": False,
        "allow_cancel": False,
        "rules": server_rules_version(),
        "first_seed": args.first,
        "games": recorder.games,
        "rows": recorder.offset,
        "field": args.field,
        "row_bytes": ROW.itemsize,
        "floats": OBS_FLOATS,
        "cards": OBS_CARDS,
        "actions": N_ACTIONS,
        "turn_action_cap": 100,
        "lobby": "five random of ten",
        "method": "dagger",
        "beta": args.beta,
        "student_sha256": digest,
        "source_hash": build_info()["source_hash"],
        "corrections": recorder.corrections,
        "expert_actions": recorder.expert_actions,
    }
    binary_path = args.out / "rows-000.bin"
    index_path = args.out / "games-000.jsonl"
    with binary_path.open("rb") as binary_data:
        manifest["shards"] = {
            binary_path.name: {
                "sha256": hashlib.file_digest(binary_data, "sha256").hexdigest(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "bytes": binary_path.stat().st_size,
            }
        }
    json_file(args.out / "manifest.json", manifest)
    Demonstrations([args.out], load_rows=True)
    emit(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--field", choices=["original", "mixed"], required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--verify", action="store_true")
    collect(parser.parse_args())
