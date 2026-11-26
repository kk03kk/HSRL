"""Behavior cloning of the frozen HSRL teacher, with resumable training and paired complete games."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from taverngym import N_ACTIONS, OBS_CARDS, OBS_FLOATS, build_info, layout, server_rules_version
from taverngym._core import BcArena, bc_teacher_name
from taverngym.entity_policy import EntityPolicy

ROW = np.dtype(
    [
        ("features", "<f4", OBS_FLOATS),
        ("cards", "<i4", OBS_CARDS),
        ("mask", "u1", N_ACTIONS),
        ("action", "<u2"),
        ("turn", "<u2"),
        ("rank", "u1"),
        ("flags", "u1"),
    ]
)
DEVICE = "cuda"


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def json_file(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def append_event(path, event):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    emit(event)


class Demonstrations:
    def __init__(self, directories, load_rows):
        self.games = {}
        self.tables = []
        self.game_spans = []
        self.identity = []
        for directory in directories:
            manifest = json.loads((directory / "manifest.json").read_text())
            expected = {
                "format": 2,
                "teacher": bc_teacher_name(),
                "source_hash": build_info()["source_hash"],
                "allow_move": False,
                "allow_cancel": False,
                "rules": server_rules_version(),
                "row_bytes": ROW.itemsize,
                "floats": OBS_FLOATS,
                "cards": OBS_CARDS,
                "actions": N_ACTIONS,
                "turn_action_cap": 100,
            }
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Incompatible demonstration manifest {directory}")
            before = len(self.games)
            rows = 0
            first = manifest["first_seed"]
            files = []
            for path in sorted(directory.glob("games-*.jsonl")):
                offset = 0
                spans = []
                content = path.read_bytes()
                for line in content.splitlines():
                    row = json.loads(line)
                    game = row["game"]
                    seed = game["seed"]
                    if not first <= seed < first + manifest["games"] or seed in self.games:
                        raise ValueError(f"Unexpected or repeated seed {seed} in {path}")
                    if row["offset"] != offset or game["rank"] not in range(1, 9):
                        raise ValueError(f"Invalid index at seed {seed} in {path}")
                    end = offset + row["rows"]
                    spans.append((seed, offset, end))
                    offset = end
                    self.games[seed] = (manifest["field"], game)
                data = directory / path.name.replace("games-", "rows-").replace(".jsonl", ".bin")
                if data.stat().st_size != offset * ROW.itemsize:
                    raise ValueError(f"Incomplete demonstration shard {data}")
                if load_rows and offset:
                    self.tables.append(np.memmap(data, dtype=ROW, mode="r", shape=(offset,)))
                    self.game_spans.append(spans)
                expected_shard = manifest["shards"].get(data.name)
                if (
                    not expected_shard
                    or expected_shard["index_sha256"] != hashlib.sha256(content).hexdigest()
                    or expected_shard["bytes"] != data.stat().st_size
                ):
                    raise ValueError(f"Unsealed demonstration shard {data}")
                files.append([data.name, offset, expected_shard])
                rows += offset
            if (len(self.games) - before, rows) != (manifest["games"], manifest["rows"]):
                raise ValueError(f"Incomplete demonstration set {directory}")
            self.identity.append({"manifest": manifest, "shards": files})

    def blocks(self, size):
        return [
            (shard, left, min(left + size, len(rows)))
            for shard, rows in enumerate(self.tables)
            for left in range(0, len(rows), size)
        ]

    def balanced_rows(self, seed, epoch):
        """Use equal numbers of whole games. Rotate the excess field across epochs."""
        recorded = {s for spans in self.game_spans for s, left, right in spans if right > left}
        groups = [
            sorted(s for s, (tag, _) in self.games.items() if tag == field and s in recorded)
            for field in ("original", "mixed")
        ]
        per_field = min(map(len, groups))
        if not per_field:
            raise ValueError("Field balancing needs training games from both fields")
        selected = set()
        for field, group in enumerate(groups):
            order = np.random.default_rng([seed, field, 71_113]).permutation(group)
            indexes = (epoch * per_field + np.arange(per_field)) % len(group)
            selected.update(order[indexes].tolist())
        masks = []
        for table, spans in zip(self.tables, self.game_spans, strict=True):
            mask = np.zeros(len(table), dtype=bool)
            for game_seed, left, right in spans:
                if game_seed in selected:
                    mask[left:right] = True
            masks.append(mask)
        return masks, {"original": per_field, "mixed": per_field}


def configure(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BC jobs need a CUDA GPU with BF16 support")


def tensors(features, cards, mask):
    return tuple(
        torch.from_numpy(np.array(x, dtype=dtype, copy=True, order="C")).to(DEVICE)
        for x, dtype in zip((features, cards, mask), (np.float32, np.int32, np.bool_), strict=True)
    )


def forward(model, rows):
    labels = rows["action"].astype(np.int64)
    legal = rows["mask"]
    if (
        not np.all(labels < N_ACTIONS)
        or not np.all(legal[np.arange(len(rows)), labels] == 1)
        or not np.all(np.isfinite(rows["features"]))
        or not np.all((rows["rank"] >= 1) & (rows["rank"] <= 8))
    ):
        raise ValueError("Invalid demonstration row")
    target = torch.from_numpy(labels).to(DEVICE)
    rank = torch.from_numpy(rows["rank"].astype(np.int64) - 1).to(DEVICE)
    output = model(*tensors(rows["features"], rows["cards"], legal))
    return output, target, rank


def value_target(rank, utility):
    return 1 - 2 * rank.float() / 7 if utility is None else utility[rank]


def score(output, target, rank, utility=None):
    nll = F.cross_entropy(output.logits.float(), target, reduction="none")
    placement = F.cross_entropy(output.placement_logits.float(), rank)
    value = F.mse_loss(output.value.float(), value_target(rank, utility))
    return nll, placement, value


@torch.inference_mode()
def offline(model, data, batch_size, utility=None):
    model.eval()
    total = np.zeros((5, 4), dtype=np.float64)
    boundaries = dict(layout())
    family = np.searchsorted(
        [
            boundaries[k]
            for k in (
                "A_BUY",
                "A_SELL",
                "A_ACTIVATE",
                "A_CHOOSE",
                "A_PLAY",
                "A_MAGNETIZE",
                "A_MOVE",
            )
        ],
        np.arange(N_ACTIONS),
        side="right",
    )
    decisions = np.zeros((8, 2), dtype=np.int64)
    value_stats = np.zeros(4, dtype=np.float64)
    for table in data.tables:
        for start in range(0, len(table), batch_size):
            rows = table[start : start + batch_size]
            rows = rows[(rows["flags"] & 4) != 0]
            if not len(rows):
                continue
            with torch.autocast(DEVICE, torch.bfloat16):
                out, target, rank = forward(model, rows)
                losses, _, value_loss = score(out, target, rank, utility)
                values = torch.stack(
                    (
                        value_loss * len(rows),
                        out.value.float().sum(),
                        value_target(rank, utility).sum(),
                    )
                )
            value_stats[:3] += values.cpu().numpy()
            value_stats[3] += len(rows)
            guess = out.logits.argmax(dim=1).cpu().numpy()
            losses = losses.cpu().numpy()
            correct = guess == rows["action"]
            top3 = (out.logits.topk(3).indices == target.unsqueeze(1)).any(dim=1).cpu().numpy()
            for i, mask in enumerate(
                (
                    np.ones(len(rows), dtype=bool),
                    (rows["flags"] & 1) != 0,
                    (rows["flags"] & 2) != 0,
                    rows["turn"] <= 5,
                    rows["turn"] >= 9,
                )
            ):
                total[i] += [mask.sum(), losses[mask].sum(), correct[mask].sum(), top3[mask].sum()]
            indexes = family[rows["action"]]
            np.add.at(decisions[:, 0], indexes, 1)
            np.add.at(decisions[:, 1], indexes, family[guess] == indexes)
    result = {}
    for tag, (count, loss, acc, top3) in zip(
        ("all", "hero", "menu", "early", "late"), total, strict=True
    ):
        if count:
            result[tag] = {
                "count": int(count),
                "nll": loss / count,
                "top1": acc / count,
                "top3": top3 / count,
            }
    result["families"] = {
        name: {"count": int(n), "accuracy": int(hit) / n if n else None}
        for name, (n, hit) in zip(
            ("scalar", "buy", "sell", "activate", "choose", "play", "magnetize", "move"),
            decisions,
            strict=True,
        )
    }
    if value_stats[3]:
        result["value"] = dict(
            zip(
                ("mse", "mean_prediction", "mean_target"),
                value_stats[:3] / value_stats[3],
                strict=True,
            )
        )
    return result


def state(model, optimizer, metadata, epoch, block, updates, best):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": metadata,
        "epoch": epoch,
        "block": block,
        "updates": updates,
        "best": best,
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def train(args):
    configure(args.seed)
    train_data = Demonstrations(args.train, load_rows=True)
    dev_data = Demonstrations(args.dev, load_rows=True)
    if (
        not train_data.tables
        or not dev_data.tables
        or train_data.games.keys() & dev_data.games.keys()
    ):
        raise ValueError("Training and development require nonempty, disjoint games")
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    if latest.exists() != args.resume:
        raise ValueError("Existing latest.pt requires --resume, or choose a new run directory")
    dimensions = {
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "feedforward": args.feedforward if args.feedforward is not None else 4 * args.width,
    }
    metadata = {
        "model": dimensions,
        "seed": args.seed,
        "batch": args.batch,
        "block_rows": args.block_rows,
        "learning_rate": args.lr,
        "precision": "bf16",
        "source": build_info()["source_hash"],
        "torch": torch.__version__,
        "train": train_data.identity,
        "dev": dev_data.identity,
        "dev_games": args.dev_games,
    }
    utility = None
    if args.placement_utility is not None:
        if not all(np.isfinite(args.placement_utility)):
            raise ValueError("Placement utility must contain eight finite numbers")
        metadata["placement_utility"] = args.placement_utility
        utility = torch.tensor(args.placement_utility, dtype=torch.float32, device=DEVICE)
    if args.balance_fields:
        metadata["field_sampling"] = "equal_games_rotating"
    panel_data = Demonstrations(args.panel, load_rows=False) if args.panel else dev_data
    if train_data.games.keys() & panel_data.games.keys():
        raise ValueError("Selection panel overlaps training games")
    if args.panel:
        metadata["screen"] = panel_data.identity
    if args.initialize:
        with args.initialize.open("rb") as f:
            metadata["initialize"] = hashlib.file_digest(f, "sha256").hexdigest()
        initial = torch.load(args.initialize, map_location="cpu", weights_only=False)
        metadata["upstream_splits"] = [
            split
            for key in ("train", "dev", "screen", "upstream_splits")
            for split in initial["meta"].get(key, [])
        ]
    if args.correction_weight or args.late_weight:
        metadata["decision_weights"] = [args.correction_weight, args.late_weight]
    model = EntityPolicy(**dimensions).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    start_epoch = start_block = updates = 0
    best = None
    if args.resume:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["meta"] != metadata:
            raise ValueError(
                "Checkpoint and current model, engine, data or training configuration differ"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.random.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        start_epoch, start_block, updates, best = (
            checkpoint[k] for k in ("epoch", "block", "updates", "best")
        )
    elif args.initialize:
        if initial["meta"]["model"] != dimensions:
            raise ValueError("Warm start requires matching model architecture")
        for key in ("static_cards", "rule_cards"):
            if not torch.equal(initial["model"][key], model.state_dict()[key].cpu()):
                raise ValueError(f"Warm start card buffer differs: {key}")
        model.load_state_dict(initial["model"])
    blocks = train_data.blocks(args.block_rows)
    emit(
        {
            "event": "train_start",
            "epoch": start_epoch,
            "block": start_block,
            "updates": updates,
            "games": len(train_data.games),
            "blocks": len(blocks),
            "gpu": torch.cuda.get_device_name(),
        }
    )
    consumed = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        offset = start_block if epoch == start_epoch else 0
        order = np.random.default_rng([args.seed, epoch]).permutation(len(blocks))
        selected, field_games = (
            train_data.balanced_rows(args.seed, epoch) if args.balance_fields else (None, None)
        )
        if field_games is not None:
            emit({"event": "epoch_games", "epoch": epoch + 1, "fields": field_games})
        for position, block_index in enumerate(order[offset:], start=offset):
            shard, left, right = blocks[block_index]
            table = train_data.tables[shard][left:right]
            eligible = (table["flags"] & 4) != 0
            if selected is not None:
                eligible &= selected[shard][left:right]
            indexes = np.flatnonzero(eligible)
            np.random.default_rng([args.seed, epoch, int(block_index)]).shuffle(indexes)
            last_loss = None
            for start in range(0, len(indexes), args.batch):
                rows = table[indexes[start : start + args.batch]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(DEVICE, torch.bfloat16):
                    out, target, rank = forward(model, rows)
                    nll, placement, value = score(out, target, rank, utility)
                    if args.correction_weight or args.late_weight:
                        weights = 1 + args.correction_weight * ((rows["flags"] & 8) != 0)
                        weights += args.late_weight * (rows["turn"] >= 8)
                        importance = torch.from_numpy(weights.astype(np.float32)).to(DEVICE)
                        policy_loss = (nll * importance).sum() / importance.sum()
                    else:
                        policy_loss = nll.mean()
                    loss = policy_loss + 0.05 * placement + 0.05 * value
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                last_loss = loss.detach()
            consumed += 1
            pause = args.stop_after_blocks == consumed
            if pause or (position + 1) % args.save_every == 0:
                save(latest, state(model, optimizer, metadata, epoch, position + 1, updates, best))
                append_event(
                    args.out / "progress.jsonl",
                    {
                        "event": "checkpoint",
                        "epoch": epoch + 1,
                        "block": position + 1,
                        "blocks": len(blocks),
                        "updates": updates,
                        "loss": float(last_loss) if last_loss is not None else None,
                    },
                )
            if pause:
                return
        dev = offline(model, dev_data, args.batch, utility)
        if args.dev_games:
            panel = {}
            for field in ("original", "mixed"):
                seeds = [s for s, (f, _) in sorted(panel_data.games.items()) if f == field][
                    : args.dev_games
                ]
                if seeds:
                    panel[field] = paired(
                        play(model, seeds, field, args.arena_batch, args.workers),
                        panel_data.games,
                        args.placement_utility,
                    )
            criterion = [
                sum(p["bc"]["rank"] for p in panel.values()) / len(panel),
                dev["all"]["nll"],
            ]
        else:
            panel = {}
            criterion = [dev["all"]["nll"]]
        candidate = {"epoch": epoch + 1, "criterion": criterion}
        weights = {"model": model.state_dict(), "meta": metadata, "best": candidate}
        if args.save_epochs:
            save(args.out / f"epoch-{epoch + 1:02}.pt", weights)
        if best is None or criterion < best["criterion"]:
            best = candidate
            save(args.out / "best.pt", weights)
        append_event(
            args.out / "progress.jsonl",
            {
                "event": "epoch",
                "epoch": epoch + 1,
                "updates": updates,
                "dev": dev,
                "dev_play": panel,
                "best": best,
            },
        )
        save(latest, state(model, optimizer, metadata, epoch + 1, 0, updates, best))
    json_file(
        args.out / "complete.json",
        {"epochs": args.epochs, "updates": updates, "best": best, "meta": metadata},
    )


@torch.inference_mode()
def play(model, seeds, field, batch_size, workers, failures=None, recorder=None, *, sample=False):
    model.eval()
    games = []
    for left in range(0, len(seeds), batch_size):
        group = seeds[left : left + batch_size]
        arena = BcArena(group, field == "mixed", workers)
        if recorder:
            recorder.start(group)
        try:
            while True:
                slots, f, c, m = arena.observe()
                n = len(slots) // 4
                if not n:
                    break
                mask = np.frombuffer(m, dtype=np.uint8).reshape(n, N_ACTIONS)
                if np.any(~mask.any(axis=1)):
                    raise RuntimeError("Active arena position has no legal moves")
                actions = mask.argmax(axis=1)
                choice = mask.sum(axis=1) > 1
                if choice.any():
                    features = np.frombuffer(f, dtype="<f4").reshape(n, OBS_FLOATS)
                    cards = np.frombuffer(c, dtype="<i4").reshape(n, OBS_CARDS)
                    with torch.autocast(DEVICE, torch.bfloat16):
                        out = model(*tensors(features[choice], cards[choice], mask[choice]))
                    if sample:
                        distribution = torch.distributions.Categorical(logits=out.logits)
                        picked = distribution.sample()
                        log_prob = distribution.log_prob(picked)
                        if not torch.isfinite(log_prob).all() or not torch.isfinite(out.value).all():
                            raise RuntimeError("Nonfinite sampled log probability or value")
                    else:
                        picked = out.logits.argmax(dim=1)
                    actions[choice] = picked.cpu().numpy()
                if recorder:
                    actions = recorder.step(arena, slots, f, c, mask, actions, choice)
                arena.step(actions.tolist())
            finished = json.loads(arena.results())
            if recorder:
                recorder.finish(finished)
            games.extend(finished)
        except RuntimeError as exc:
            seed = re.search(r"seed (\d+)", str(exc))
            if failures is not None:
                details = {"error": str(exc), "seeds": group}
                if seed and int(seed[1]) in group:
                    details["state"] = json.loads(arena.state_json(group.index(int(seed[1]))))
                json_file(failures, details)
            raise
    return games


def ci(diff, rng):
    replicates = np.concatenate(
        [diff[rng.integers(len(diff), size=(128, len(diff)))].mean(axis=1) for _ in range(80)]
    )
    return [float(x) for x in np.quantile(replicates, [0.025, 0.975])]


def paired(games, teachers, utility=None):
    baseline = [teachers[g["seed"]][1] for g in games]
    if any(
        (b["seat"], b["lobby"]) != (g["seat"], g["lobby"])
        for b, g in zip(baseline, games, strict=True)
    ):
        raise ValueError("Paired levels must share lobby and learner seat")
    rank = np.array([g["rank"] for g in games])
    teacher_rank = np.array([g["rank"] for g in baseline])

    def summarize(rows, ranks):
        summary = {
            "rank": float(ranks.mean()),
            "first": float((ranks == 1).mean()),
            "top4": float((ranks <= 4).mean()),
            "top2": float((ranks <= 2).mean()),
            "rank_counts": np.bincount(ranks, minlength=9)[1:].tolist(),
            "actions": sum(g["actions"] for g in rows) / len(rows),
            "decision_rows": sum(g["decisions"] for g in rows),
            "budget_rate": sum(g["budget_turns"] for g in rows)
            / max(1, sum(len(g["rounds"]) for g in rows)),
            "turns": {t: round_summary(rows, t) for t in (5, 8, 10, 12)},
        }
        if utility is not None:
            summary["utility"] = float(np.asarray(utility)[ranks - 1].mean())
        return summary

    rng = np.random.default_rng(91_660)
    intervals = {}
    for name, diff in (
        ("rank", rank - teacher_rank),
        ("top4", (rank <= 4).astype(int) - (teacher_rank <= 4)),
        ("first", (rank == 1).astype(int) - (teacher_rank == 1)),
        ("top2", (rank <= 2).astype(int) - (teacher_rank <= 2)),
    ):
        intervals[name] = {"delta": float(diff.mean()), "ci95": ci(diff, rng)}
    if utility is not None:
        values = np.asarray(utility)
        diff = values[rank - 1] - values[teacher_rank - 1]
        intervals["utility"] = {"delta": float(diff.mean()), "ci95": ci(diff, rng)}
    return {
        "games": len(games),
        "bc": summarize(games, rank),
        "teacher": summarize(baseline, teacher_rank),
        "paired": intervals,
        "noninferior": (
            intervals["rank"]["ci95"][1] <= 0.15 and intervals["top4"]["ci95"][0] >= -0.03
        ),
    }


def round_summary(games, turn):
    reached = [r for g in games for r in g["rounds"] if r["turn"] == turn]
    means = (
        {k: float(np.mean([r[k] for r in reached])) for k in ("tier", "attack", "health")}
        if reached
        else {}
    )
    return dict(fraction=len(reached) / len(games), **means)


def evaluate(args):
    configure(0)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    if meta["source"] != build_info()["source_hash"]:
        raise ValueError("Evaluation simulator differs from training simulator")
    data = Demonstrations(args.test, load_rows=True)
    for split in (
        split
        for key in ("train", "dev", "screen", "upstream_splits")
        for split in meta.get(key, [])
    ):
        m = split["manifest"]
        if any(s in data.games for s in range(m["first_seed"], m["first_seed"] + m["games"])):
            raise ValueError("Final evaluation contains training or development seeds")
    model = EntityPolicy(**meta["model"]).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    lookup = meta.get("placement_utility")
    utility = torch.tensor(lookup, dtype=torch.float32, device=DEVICE) if lookup else None
    if any(args.out.glob("*.json")):
        raise ValueError("Evaluation already exists")
    with args.checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    metrics = {
        "source": meta["source"],
        "checkpoint_sha256": digest,
        "selected": checkpoint["best"],
        "placement_utility": lookup,
        "test": data.identity,
        "offline": offline(model, data, args.arena_batch, utility) if data.tables else {},
        "fields": {},
    }
    for field in ("original", "mixed"):
        seeds = [s for s, (f, _) in sorted(data.games.items()) if f == field]
        if not seeds:
            continue
        games = play(
            model, seeds, field, args.arena_batch, args.workers, args.out / f"failure-{field}.json"
        )
        with (args.out / f"pairs-{field}.jsonl").open("w", encoding="utf-8") as f:
            for game in games:
                f.write(json.dumps({"bc": game, "teacher": data.games[game["seed"]][1]}) + "\n")
        metrics["fields"][field] = paired(games, data.games, lookup)
        emit(dict(field=field, **metrics["fields"][field]))
    metrics["ppo_gate"] = set(metrics["fields"]) == {"original", "mixed"} and all(
        p["noninferior"] for p in metrics["fields"].values()
    )
    json_file(args.out / "report.json", metrics)


def compare(args):
    a, b = (torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints)

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(equal(i, j) for i, j in zip(x, y, strict=True))
        return x == y

    if not equal(a, b):
        raise RuntimeError(
            f"Resumed checkpoint differs in {[k for k in a if not equal(a[k], b[k])]}"
        )
    result = {
        "resume_equal": True,
        "model_tensors": len(a["model"]),
        "optimizer_tensors": len(a["optimizer"]["state"]),
        "epoch": a["epoch"],
        "block": a["block"],
        "updates": a["updates"],
    }
    json_file(args.out, result)
    emit(result)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    training = commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    for command in (training, evaluation):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--arena-batch", type=int, default=192)
        command.add_argument("--workers", type=int, default=8)
    training.add_argument("--train", type=Path, required=True, nargs="+")
    training.add_argument("--dev", type=Path, required=True, nargs="+")
    training.add_argument("--panel", type=Path, nargs="+")
    training.add_argument("--initialize", type=Path)
    training.add_argument("--correction-weight", type=float, default=0.0)
    training.add_argument("--late-weight", type=float, default=0.0)
    training.add_argument("--save-epochs", action="store_true")
    training.add_argument("--seed", type=int, default=71427)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--batch", type=int, default=768)
    training.add_argument("--block-rows", type=int, default=4096)
    training.add_argument("--save-every", type=int, default=12)
    training.add_argument("--dev-games", type=int, default=64, help="Per field per epoch")
    training.add_argument("--stop-after-blocks", type=int)
    training.add_argument("--width", type=int, default=384)
    training.add_argument("--layers", type=int, default=8)
    training.add_argument("--heads", type=int, default=8)
    training.add_argument("--feedforward", type=int, default=1664)
    training.add_argument("--placement-utility", type=float, nargs=8, default=[1.0, 0.3, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25])
    training.add_argument("--balance-fields", action="store_true")
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--resume", action="store_true")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--test", type=Path, nargs="+", required=True)
    comparison = commands.add_parser("compare-resume")
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("checkpoints", type=Path, nargs=2)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    {"train": train, "evaluate": evaluate, "compare-resume": compare}[arguments.mode](arguments)
