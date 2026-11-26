"""On-policy PPO for complete TavernGym games against fixed scripted opponents."""

import argparse
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from taverngym import N_ACTIONS, OBS_CARDS, OBS_FLOATS, build_info
from taverngym._core import BcArena
from taverngym.bc_train import Demonstrations, configure, emit, json_file, paired, play, save
from taverngym.entity_policy import EntityPolicy, PolicyOutput

FIELDS = ("original", "mixed")
UTILITY = (1.0, 0.3, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25)
MODEL = {"width": 384, "layers": 8, "heads": 8, "feedforward": 1664}
RECIPE = {
    "discount": 1.0,
    "gae_lambda": 0.995,
    "clip": 0.2,
    "target_kl": 0.02,
    "rollback_kl": 0.05,
    "value": 0.5,
    "placement": 0.05,
    "entropy": 0.002,
    "grad_norm": 1.0,
    "lr_start": 6e-6,
    "lr_end": 5e-6,
    "schedule_iterations": 60,
    "epochs_per_rollout": 1,
    "temperature": 1.0,
}


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def snapshot_checkpoint(latest, iteration, every):
    if every and iteration and iteration % every == 0:
        path = latest.with_name(f"snapshot-{iteration:03}.pt")
        if not path.exists():
            os.link(latest, path)


def load_reference(path, receipt_path):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = sha256(path)
    if not receipt["approved"] or receipt["sha256"] != digest:
        raise ValueError("Starting checkpoint has not passed the recorded BC admission")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = saved["meta"]
    if meta["source"] != build_info()["source_hash"] or receipt["native_source"] != meta["source"]:
        raise ValueError("The starting checkpoint requires its frozen native engine")
    if (
        meta["model"] != MODEL
        or receipt["model"] != MODEL
        or tuple(meta["placement_utility"]) != UTILITY
        or tuple(receipt["placement_utility"]) != UTILITY
    ):
        raise ValueError("Expected the admitted 18M architecture and placement utility")
    return saved, digest


class FixedBatchPolicy(EntityPolicy):
    """Use the same BF16 kernel shape for rollout, replay, and policy gradients."""

    def __init__(self, width_per_forward):
        super().__init__(**MODEL)
        self.width_per_forward = width_per_forward

    def forward(self, features, cards, action_mask):
        width = self.width_per_forward
        if len(features) == width:
            return super().forward(features, cards, action_mask)
        outputs = []
        for start in range(0, len(features), width):
            batch = (features[start : start + width], cards[start : start + width],
                     action_mask[start : start + width])
            size = len(batch[0])
            if size < width:
                batch = tuple(F.pad(t, (0, 0, 0, width - size)) for t in batch)
            out = super().forward(*batch)
            outputs.append(PolicyOutput(out.logits[:size], out.value[:size], out.placement_logits[:size]))
        return PolicyOutput(*(torch.cat(rows) for rows in zip(*outputs, strict=True)))


def make_model(weights, policy_batch_size=0):
    model = FixedBatchPolicy(policy_batch_size) if policy_batch_size else EntityPolicy(**MODEL)
    model.to("cuda").eval()
    model.load_state_dict(weights, strict=True)
    return model


def model_output(model, features, cards, mask, precision):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        return model(features, cards, mask)


def inputs(features, cards, mask):
    return tuple(
        torch.from_numpy(np.ascontiguousarray(x)).to("cuda") for x in (features, cards, mask)
    )


def gae(values, terminal_reward, discount=1.0, decay=0.995):
    """One complete seat trajectory, with the reward on its final decision."""
    advantage = np.empty_like(values, dtype=np.float32)
    carry = 0.0
    next_value = 0.0
    for i in range(len(values) - 1, -1, -1):
        delta = (
            (terminal_reward if i == len(values) - 1 else 0.0) + discount * next_value - values[i]
        )
        carry = delta + discount * decay * carry
        advantage[i] = carry
        next_value = values[i]
    return advantage, advantage + values


def game_metrics(games):
    ranks = np.array([g["rank"] for g in games])
    return {
        "games": len(games),
        "rank": float(ranks.mean()),
        "top2": float((ranks <= 2).mean()),
        "top4": float((ranks <= 4).mean()),
        "first": float((ranks == 1).mean()),
        "eighth": float((ranks == 8).mean()),
        "utility": float(np.asarray(UTILITY)[ranks - 1].mean()),
        "budget_rate": sum(g["budget_turns"] for g in games) / sum(len(g["rounds"]) for g in games),
    }


@torch.inference_mode()
def sample(
    model, first_seed, n_games, concurrent, workers, precision, failure_path, fields=FIELDS,
    *, seed_groups=None, generator=None, defer_batch_finalization=False,
):
    """Generate fresh data under one frozen iteration policy. Opponents are native bots."""
    columns = {k: [] for k in ("features", "cards", "mask", "actions", "old_logp", "old_value")}
    trajectories = []
    records = {field: [] for field in fields}
    count = 0
    profile = dict.fromkeys(
        ("observe", "inference", "prepare_inputs", "forward_readback", "simulator_step"), 0.0
    )
    games_per_field = n_games // len(fields)
    concurrent_per_field = concurrent // len(fields)
    seeds_by_field = seed_groups or [
        range(start, start + games_per_field)
        for start in range(first_seed, first_seed + n_games, games_per_field)
    ]
    for offset in range(0, games_per_field, concurrent_per_field):
        cohorts = [list(seeds[offset : offset + concurrent_per_field]) for seeds in seeds_by_field]
        arenas = [
            BcArena(seeds, field == "mixed", workers)
            for field, seeds in zip(fields, cohorts, strict=True)
        ]
        history = [[[] for _ in seeds] for seeds in cohorts]
        try:
            while True:
                tick = time.perf_counter()
                pending = []
                for arena in arenas:
                    slots, f, c, m = arena.observe()
                    ids = np.frombuffer(slots, dtype="<u4")
                    masks = np.frombuffer(m, dtype="u1").reshape(len(ids), N_ACTIONS)
                    legal_count = masks.sum(axis=1)
                    if np.any(legal_count == 0):
                        raise RuntimeError("Active arena position has no legal moves")
                    choice = legal_count > 1
                    feats = np.frombuffer(f, dtype="<f4").reshape(len(ids), OBS_FLOATS)[choice]
                    cards = np.frombuffer(c, dtype="<i4").reshape(len(ids), OBS_CARDS)[choice]
                    pending.append(
                        (ids[choice], feats, cards, masks[choice], masks.argmax(axis=1), choice)
                    )
                profile["observe"] += time.perf_counter() - tick
                if not any(len(p[4]) for p in pending):
                    break
                tick = time.perf_counter()
                n_choices = sum(len(p[0]) for p in pending)
                if n_choices:
                    features, cards, masks = (
                        np.concatenate([p[k] for p in pending], axis=0) for k in (1, 2, 3)
                    )
                    device_inputs = inputs(features, cards, masks)
                    forward_start = time.perf_counter()
                    profile["prepare_inputs"] += forward_start - tick
                    result = model_output(model, *device_inputs, precision)
                    dist = torch.distributions.Categorical(logits=result.logits.float())
                    picked = (
                        dist.sample() if generator is None else
                        torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
                    )
                    selected = picked.cpu().numpy()
                    logp = dist.log_prob(picked).cpu().numpy()
                    value = result.value.float().cpu().numpy()
                    profile["forward_readback"] += time.perf_counter() - forward_start
                    if not np.all(np.isfinite(logp)) or not np.all(np.isfinite(value)):
                        raise RuntimeError("Nonfinite sampled log probability or value")
                    for column, data in zip(
                        columns.values(),
                        (features, cards, masks, selected, logp, value),
                        strict=True,
                    ):
                        column.append(data)
                    cursor = 0
                    for p, paths in zip(pending, history, strict=True):
                        ids, _, _, _, actions, choice = p
                        end = cursor + len(ids)
                        actions[choice] = selected[cursor:end]
                        for i, seat in enumerate(ids):
                            paths[seat].append(count + cursor + i)
                        cursor = end
                    count += n_choices
                profile["inference"] += time.perf_counter() - tick
                tick = time.perf_counter()
                for arena, p in zip(arenas, pending, strict=True):
                    if len(p[4]):
                        arena.step(p[4].tolist())
                profile["simulator_step"] += time.perf_counter() - tick
            for field, arena, paths in zip(fields, arenas, history, strict=True):
                result = json.loads(arena.results())
                for game, path in zip(result, paths, strict=True):
                    if game["decisions"] != len(path) or not path:
                        raise RuntimeError(f"Unexpected decision count for seed {game['seed']}")
                    trajectories.append((path, game["rank"]))
                records[field].extend(result)
        except (RuntimeError, ValueError) as exc:
            data = {"error": str(exc), "cohorts": cohorts}
            seed = re.search(r"seed (\d+)", str(exc))
            if seed:
                for arena, cohort in zip(arenas, cohorts, strict=True):
                    if int(seed[1]) in cohort:
                        data["game"] = json.loads(arena.state_json(cohort.index(int(seed[1]))))
            json_file(failure_path, data)
            raise

    batch = {k: np.concatenate(rows) for k, rows in columns.items()}
    advantage, returns = (np.empty(count, dtype=np.float32) for _ in range(2))
    placement = np.empty(count, dtype=np.int64)
    terminal = np.zeros(count, dtype=np.float32)
    for path, rank in trajectories:
        reward = UTILITY[rank - 1]
        advantage[path], returns[path] = gae(
            batch["old_value"][path], reward, RECIPE["discount"], RECIPE["gae_lambda"]
        )
        terminal[path[-1]] = reward
        placement[path] = rank - 1
    batch["advantage"] = advantage
    batch["returns"] = returns
    batch["placement"] = placement
    if defer_batch_finalization:
        return batch, records, profile
    return finish_batch(batch, records, profile, first_seed, n_games, float(terminal.sum(dtype=np.float64)))


def finish_batch(batch, records, profile, first_seed, n_games, reward_sum):
    if sum(map(len, records.values())) != n_games:
        raise RuntimeError("Incomplete on-policy batch")
    advantage = batch["advantage"]
    batch["advantage"] = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
    digest = hashlib.sha256()
    for array in batch.values():
        digest.update(memoryview(array))
    return batch, {
        "seed_range": [first_seed, first_seed + n_games],
        "rollout_sha256": digest.hexdigest(),
        "decisions": len(advantage),
        "forced_actions": sum(
            g["actions"] - g["decisions"] for rows in records.values() for g in rows
        ),
        "terminal_events": n_games,
        "reward_sum": reward_sum,
        "fields": {field: game_metrics(rows) for field, rows in records.items()},
        "profile_seconds": profile,
    }


def sample_pair(model, secondary, first, n_games, concurrent, workers, precision, failure, seed):
    """Split both opponent fields across the two actors and merge before updating."""
    per_field = n_games // 2

    def actor(device, policy):
        left = device * (per_field // 2)
        groups = [
            range(first + field * per_field + left, first + field * per_field + left + per_field // 2)
            for field in range(2)
        ]
        suffix = failure.with_name(f"{failure.stem}-actor{device}.json")
        with torch.cuda.device(device):
            generator = torch.Generator(device=f"cuda:{device}").manual_seed(seed + device)
            started = time.perf_counter()
            result = sample(
                policy, first, n_games // 2, concurrent // 2, workers, precision, suffix,
                seed_groups=groups, generator=generator, defer_batch_finalization=True,
            )
            return (*result, time.perf_counter() - started)

    started = time.perf_counter()
    # Take a synchronous copy from the current learner before exposing either actor.
    secondary.load_state_dict(model.state_dict(), strict=True)
    torch.cuda.synchronize(1)
    broadcast_seconds = time.perf_counter() - started
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(actor, device, policy) for device, policy in enumerate((model, secondary))]
        shards = [job.result() for job in jobs]
    merge_start = time.perf_counter()
    batch = {key: np.concatenate([shard[0][key] for shard in shards]) for key in shards[0][0]}
    records = {field: [game for shard in shards for game in shard[1][field]] for field in FIELDS}
    for index, field in enumerate(FIELDS):
        expected = list(range(first + index * per_field, first + (index + 1) * per_field))
        if [game["seed"] for game in records[field]] != expected:
            raise RuntimeError(f"Actor shards do not cover the exact {field} seed range")
    profile = {key: sum(shard[2][key] for shard in shards) for key in shards[0][2]}
    reward = sum(UTILITY[game["rank"] - 1] for games in records.values() for game in games)
    batch, metrics = finish_batch(batch, records, profile, first, n_games, reward)
    metrics.update(
        actors=2, actor_seconds=[shard[3] for shard in shards],
        broadcast_seconds=broadcast_seconds, merge_seconds=time.perf_counter() - merge_start,
    )
    return batch, metrics


def distribution(logits, mask):
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.masked_fill(~mask, 0), logp.exp().masked_fill(~mask, 0)


def approximation(log_ratio):
    return log_ratio.expm1() - log_ratio


def index_tensors(batch, rows):
    if isinstance(rows, torch.Tensor):
        return (
            tuple(batch[key].index_select(0, rows) for key in ("features", "cards", "mask")),
            {key: batch[key].index_select(0, rows) for key in (
                "actions", "old_logp", "old_value", "advantage", "returns", "placement"
            )},
        )
    x = inputs(batch["features"][rows], batch["cards"][rows], batch["mask"][rows].astype(bool))
    return x, {
        k: torch.from_numpy(batch[k][rows]).to("cuda")
        for k in ("actions", "old_logp", "old_value", "advantage", "returns", "placement")
    }


@torch.no_grad()
def policy_kl(model, batch, rows, microbatch, precision, *, defer_metrics=False):
    if defer_metrics:
        statistics = torch.zeros(3, device="cuda", dtype=torch.float64)
        finite = torch.ones((), device="cuda", dtype=torch.bool)
    else:
        total = square = largest = 0.0
    for left in range(0, len(rows), microbatch):
        x, y = index_tensors(batch, rows[left : left + microbatch])
        output = model_output(model, *x, precision)
        logp, _ = distribution(output.logits, x[2])
        delta = logp.gather(1, y["actions"][:, None]).squeeze(1) - y["old_logp"]
        if defer_metrics:
            finite &= torch.isfinite(delta).all()
            statistics[0].add_(approximation(delta).sum())
            statistics[1].add_(delta.square().sum())
            statistics[2].copy_(torch.maximum(statistics[2], delta.abs().max()))
        else:
            if not torch.isfinite(delta).all():
                raise RuntimeError("Nonfinite policy log ratio")
            total += float(approximation(delta).sum())
            square += float(delta.square().sum())
            largest = max(largest, float(delta.abs().max()))
    if defer_metrics:
        if not bool(finite):
            raise RuntimeError("Nonfinite policy log ratio")
        total, square, largest = statistics.tolist()
    return {
        "kl": total / len(rows),
        "log_ratio_rms": (square / len(rows)) ** 0.5,
        "max_abs_log_ratio": largest,
    }


def update(model, reference, optimizer, batch, args, iteration):
    count = len(batch["actions"])
    order = np.random.default_rng([args.seed, iteration]).permutation(count)
    started = time.perf_counter()
    if args.resident_rollout:
        batch = {key: torch.from_numpy(values).to("cuda") for key, values in batch.items()}
        batch["mask"] = batch["mask"].bool()
        order = torch.from_numpy(order).to("cuda")
        torch.cuda.synchronize()
    staging_seconds = time.perf_counter() - started
    interval = min(4096, count)
    started = time.perf_counter()
    kl_options = {"defer_metrics": args.defer_metrics}
    before = policy_kl(model, batch, order[:interval], args.microbatch, args.precision, **kl_options)
    initial_kl_seconds = time.perf_counter() - started
    if before["log_ratio_rms"] > 0.03 or before["max_abs_log_ratio"] > 0.15:
        raise RuntimeError(f"Same-weight batch replay differs from rollout: {before}")
    progress = min(iteration / (RECIPE["schedule_iterations"] - 1), 1.0)
    lr = args.lr_scale * (RECIPE["lr_start"] + progress * (RECIPE["lr_end"] - RECIPE["lr_start"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    anchor = args.bc_kl_scale * 0.10 * min(1.0, max(0.0, (39 - iteration) / 20))
    names = ("policy", "value", "placement", "entropy", "anchor_kl", "clip_fraction", "loss")
    totals = (
        torch.zeros(len(names), device="cuda", dtype=torch.float64)
        if args.defer_metrics else np.zeros(len(names))
    )
    steps = trained = 0
    max_grad = 0.0
    stopped = "all_minibatches"
    probe_seconds = 0.0
    for left in range(0, count, args.minibatch):
        rows = order[left : left + args.minibatch]
        optimizer.zero_grad(set_to_none=True)
        if left and steps % args.kl_check_every == 0:
            tick = time.perf_counter()
            probe = policy_kl(model, batch, order[:interval], args.microbatch, args.precision, **kl_options)
            probe_seconds += time.perf_counter() - tick
            if probe["kl"] > RECIPE["target_kl"]:
                stopped = "target_kl"
                break
        for start in range(0, len(rows), args.microbatch):
            x, y = index_tensors(batch, rows[start : start + args.microbatch])
            out = model_output(model, *x, args.precision)
            logp, prob = distribution(out.logits, x[2])
            log_ratio = logp.gather(1, y["actions"][:, None]).squeeze(1) - y["old_logp"]
            ratio = log_ratio.exp()
            eps = RECIPE["clip"]
            adv = y["advantage"]
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
            vf = F.mse_loss(out.value.float(), y["returns"])
            aux = F.cross_entropy(out.placement_logits.float(), y["placement"])
            entropy = -(prob * logp).sum(-1).mean()
            anchor_kl = logp.new_zeros(())
            if anchor:
                with torch.no_grad():
                    ref = model_output(reference, *x, args.precision)
                    ref_logp, ref_prob = distribution(ref.logits, x[2])
                anchor_kl = (ref_prob * (ref_logp - logp)).sum(-1).mean()
            loss = (
                pg
                + RECIPE["value"] * vf
                + RECIPE["placement"] * aux
                - RECIPE["entropy"] * entropy
                + anchor * anchor_kl
            )
            n = len(adv)
            (loss * (n / len(rows))).backward()
            fraction = ((ratio - 1).abs() > eps).float().mean()
            values = torch.stack((pg, vf, aux, entropy, anchor_kl, fraction, loss)).detach()
            if args.defer_metrics:
                totals.add_(n * values)
            else:
                totals += n * values.cpu().numpy()
            trained += n
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), RECIPE["grad_norm"], error_if_nonfinite=True
        )
        max_grad = max(max_grad, float(norm))
        optimizer.step()
        steps += 1
    tick = time.perf_counter()
    after = policy_kl(model, batch, order, args.microbatch, args.precision, **kl_options)
    if not steps or after["kl"] > RECIPE["rollback_kl"]:
        raise RuntimeError(
            f"Iteration rejected before checkpoint publication: steps={steps}, {after}"
        )
    if args.defer_metrics:
        totals = totals.cpu().numpy()
    return dict(
        zip(names, map(float, totals / trained), strict=True),
        lr=lr,
        anchor=anchor,
        before=before,
        after=after,
        max_grad_norm=max_grad,
        optimizer_steps=steps,
        updated_decisions=trained,
        stop=stopped,
        profile_seconds={
            "staging": staging_seconds,
            "initial_kl": initial_kl_seconds,
            "minibatch_kl": probe_seconds,
            "terminal_kl": time.perf_counter() - tick,
        },
    )


def train(args):
    started = time.monotonic()
    fields = ("original",) if args.original_only else FIELDS
    if not (
        0 < args.games
        and args.games % len(fields) == 0
        and 0 < args.concurrent
        and args.concurrent % len(fields) == 0
        and 0 < args.microbatch <= args.minibatch
        and args.iterations > 0
        and (args.stop_after is None or 0 < args.stop_after <= args.iterations)
        and math.isfinite(args.lr_scale)
        and args.lr_scale > 0
        and math.isfinite(args.bc_kl_scale)
        and args.bc_kl_scale >= 0
        and args.kl_check_every > 0
        and args.snapshot_every >= 0
        and (args.policy_batch_size == 0 or args.microbatch % args.policy_batch_size == 0)
        and args.policy_batch_size >= 0
        and (args.time_budget_seconds is None or args.time_budget_seconds > 0)
        and not (
            args.sample_only and (
                args.resume or args.stop_after is not None or args.snapshot_every
                or args.time_budget_seconds is not None
            )
        )
    ):
        raise ValueError("Invalid rollout or minibatch settings")
    if args.actors == 2 and (
        torch.cuda.device_count() != 2 or args.games % 4 or args.concurrent % 4
        or fields != FIELDS
    ):
        raise ValueError("Two actors need two visible GPUs and equally divisible mixed-field games")
    if args.expected_gpu:
        for device in range(args.actors):
            if torch.cuda.get_device_name(device) != args.expected_gpu:
                raise RuntimeError(
                    f"GPU {device}: expected {args.expected_gpu}, got {torch.cuda.get_device_name(device)}"
                )
    configure(args.seed)
    torch.use_deterministic_algorithms(True)
    initial, fingerprint = load_reference(args.init, args.admission)
    meta = {
        "schema": 1,
        "source": build_info()["source_hash"],
        "code_commit": args.code_commit,
        "initial_sha256": fingerprint,
        "model": MODEL,
        "placement_utility": list(UTILITY),
        "torch": str(torch.__version__),
        "gpu": [torch.cuda.get_device_name(device) for device in range(args.actors)],
        "recipe": RECIPE,
        "fields": list(fields),
        **{
            key: getattr(args, key)
            for key in (
                "seed",
                "first_seed",
                "iterations",
                "games",
                "concurrent",
                "minibatch",
                "microbatch",
                "workers",
                "precision",
                "lr_scale",
                "bc_kl_scale",
                "kl_check_every",
                "sample_only",
                "actors",
                "resident_rollout",
                "defer_metrics",
            )
        },
    }
    if args.snapshot_every:
        meta["snapshot_every"] = args.snapshot_every
    if args.policy_batch_size:
        meta["policy_batch_size"] = args.policy_batch_size
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    if latest.exists() != args.resume:
        raise ValueError("Use a new output directory, or pass --resume for an existing checkpoint")
    model = make_model(initial["model"], args.policy_batch_size)
    secondary = None
    if args.actors == 2:
        with torch.cuda.device(1):
            secondary = make_model(initial["model"], args.policy_batch_size)

    def collect(iteration):
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        first = args.first_seed + iteration * args.games
        failure = args.out / f"failure-{iteration + 1}.json"
        if secondary is None:
            batch, metrics = sample(
                model, first, args.games, args.concurrent, args.workers, args.precision, failure, fields,
            )
        else:
            batch, metrics = sample_pair(
                model, secondary, first, args.games, args.concurrent, args.workers, args.precision,
                failure, args.seed + iteration * 2,
            )
        metrics["rollout_seconds"] = time.perf_counter() - tick
        return batch, metrics

    if args.sample_only:
        for iteration in range(args.iterations):
            batch, metrics = collect(iteration)
            metrics["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            del batch
            report = {"iteration": iteration + 1, "sampling": metrics}
            json_file(args.out / f"iteration-{iteration + 1:03}.json", report)
            emit(report)
        json_file(args.out / "summary.json", {"iterations": args.iterations, "config": meta})
        return
    reference = make_model(initial["model"], args.policy_batch_size)
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["lr_start"], weight_decay=0, fused=True
    )
    start, history = 0, []
    if args.resume:
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["meta"] != meta:
            raise ValueError("Resume configuration differs from the saved experiment")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        start, history = saved["iteration"], saved["rollout_history"]
        snapshot_checkpoint(latest, start, args.snapshot_every)
    finish = args.stop_after or args.iterations
    if finish <= start:
        raise ValueError("No additional iterations requested")

    def checkpoint(iteration):
        save(
            latest,
            {
                "meta": meta,
                "iteration": iteration,
                "rollout_history": history,
                "next_game_seed": args.first_seed + iteration * args.games,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.random.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
        )
        snapshot_checkpoint(latest, iteration, args.snapshot_every)

    if not args.resume:
        checkpoint(0)
    emit(
        {
            "event": "train_start",
            "from_iteration": start,
            "through_iteration": finish,
            "time_budget_seconds": args.time_budget_seconds,
            "config": meta,
        }
    )
    completed = start
    reason = "target_iterations"
    for iteration in range(start, finish):
        if args.time_budget_seconds is not None and time.monotonic() - started >= args.time_budget_seconds:
            reason = "segment_time_budget"
            break
        batch, metrics = collect(iteration)
        tick = time.perf_counter()
        diagnostics = update(model, reference, optimizer, batch, args, iteration)
        metrics["update_seconds"] = time.perf_counter() - tick
        metrics["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        del batch
        history.append(
            {
                "rollout_sha256": metrics["rollout_sha256"],
                "optimizer_steps": diagnostics["optimizer_steps"],
            }
        )
        tick = time.perf_counter()
        checkpoint(iteration + 1)
        metrics["checkpoint_seconds"] = time.perf_counter() - tick
        report = {"iteration": iteration + 1, "sampling": metrics, "optimization": diagnostics}
        json_file(args.out / f"iteration-{iteration + 1:03}.json", report)
        emit(report)
        completed = iteration + 1
    segment = {
        "event": "segment_end", "from_iteration": start, "iterations": completed,
        "target_iteration": finish, "reason": reason,
        "elapsed_seconds": time.monotonic() - started,
    }
    json_file(
        args.out / "summary.json",
        {**segment, "config": meta, "rollout_history": history},
    )
    emit(segment)


def compare(args):
    first, second = (
        torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoints
    )

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    if args.compare_residency:
        for checkpoint in (first, second):
            checkpoint["meta"] = {k: v for k, v in checkpoint["meta"].items() if k != "resident_rollout"}
    different = [k for k in first if k not in second or not equal(first[k], second[k])]
    if different or first.keys() != second.keys():
        raise RuntimeError(f"Resumed run differs in {different}")
    report = {
        "exact_resume": True,
        "iterations": first["iteration"],
        "model_tensors": len(first["model"]),
        "optimizer_parameter_states": len(first["optimizer"]["state"]),
        "next_game_seed": first["next_game_seed"],
        "matching_rollouts": len(first["rollout_history"]),
    }
    json_file(args.out, report)
    emit(report)


def eval_compare(games, opponents):
    stats = paired(games, opponents, UTILITY)
    stats["ppo"] = stats.pop("bc")
    stats["baseline"] = stats.pop("teacher")
    stats.pop("noninferior")
    return stats


def evaluate(args):
    configure(0)
    initial, fingerprint = load_reference(args.init, args.admission)
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = trained["meta"]
    if meta["initial_sha256"] != fingerprint or meta["source"] != build_info()["source_hash"]:
        raise ValueError("Checkpoint and frozen reference do not match")
    data = Demonstrations(args.test, load_rows=False)
    first = meta["first_seed"]
    end = trained["next_game_seed"]
    if any(first <= seed < end for seed in data.games):
        raise ValueError("PPO evaluation overlaps rollout seeds")
    for key in ("train", "dev", "screen", "upstream_splits"):
        for split in initial["meta"].get(key, []):
            m = split["manifest"]
            if any(m["first_seed"] <= seed < m["first_seed"] + m["games"] for seed in data.games):
                raise ValueError("PPO evaluation overlaps a BC data or selection split")
    args.out.mkdir(parents=True, exist_ok=False)
    candidate_model = make_model(trained["model"], meta.get("policy_batch_size", 0))
    reference_model = None
    cached = args.reference_pairs
    if cached is None:
        reference_model = make_model(initial["model"])
    else:
        report = json.loads((cached / "report.json").read_text(encoding="utf-8"))
        if (
            report["checkpoint_sha256"] != fingerprint or report["source"] != meta["source"]
            or report["gpu"] != torch.cuda.get_device_name()
            or report["arena_batch"] != args.arena_batch or report["test"] != data.identity
        ):
            raise ValueError("The cached BC reference uses different weights, panel, or runtime")
    fields = {}
    for field in FIELDS:
        seeds = sorted(seed for seed, (name, _) in data.games.items() if name == field)
        if not seeds:
            raise ValueError(f"No evaluation seeds for {field}")
        if cached is not None:
            with (cached / f"pairs-{field}.jsonl").open(encoding="utf-8") as stream:
                reference = [json.loads(line)["bc"] for line in stream]
            if [game["seed"] for game in reference] != seeds:
                raise ValueError(f"Cached BC panel differs for {field}")
        else:
            reference = play(
                reference_model, seeds, field, args.arena_batch, args.workers,
                args.out / f"failure-{field}-reference.json",
            )
        candidate = play(
            candidate_model, seeds, field, args.arena_batch, args.workers,
            args.out / f"failure-{field}-ppo.json",
        )
        with (args.out / f"pairs-{field}.jsonl").open("w", encoding="utf-8") as f:
            for bc, ppo in zip(reference, candidate, strict=True):
                f.write(
                    json.dumps({"ppo": ppo, "bc18": bc, "v2": data.games[ppo["seed"]][1]}) + "\n"
                )
        fields[field] = {
            "versus_bc18": eval_compare(
                candidate, {row["seed"]: (field, row) for row in reference}
            ),
            "versus_v2": eval_compare(candidate, data.games),
        }
    report = {
        "evaluation": f"{args.panel}_greedy_pairs",
        "promotion_decision": None,
        "arena_batch": args.arena_batch,
        "evaluation_gpu": torch.cuda.get_device_name(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "initial_sha256": fingerprint,
        "iteration": trained["iteration"],
        "test": data.identity,
        "fields": fields,
    }
    json_file(args.out / "report.json", report)
    for field, scores in fields.items():
        emit({"field": field, **scores})


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training, evaluation, comparison = (
        sub.add_parser(mode) for mode in ("train", "evaluate", "compare")
    )
    for command in (training, evaluation, comparison):
        command.add_argument("--out", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--init", type=Path, required=True)
        command.add_argument("--admission", type=Path, required=True)
        command.add_argument("--workers", type=int, default=8)
    training.add_argument("--code-commit", required=True)
    training.add_argument("--seed", type=int, default=20261126)
    training.add_argument("--first-seed", type=int, required=True)
    training.add_argument("--games", type=int, default=128)
    training.add_argument("--concurrent", type=int, default=128)
    training.add_argument("--iterations", type=int, default=4)
    training.add_argument("--minibatch", type=int, default=2048)
    training.add_argument("--microbatch", type=int, default=512)
    training.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    training.add_argument("--stop-after", type=int)
    training.add_argument(
        "--time-budget-seconds", type=int,
        help="Stop before starting another iteration once this segment's elapsed time reaches the budget",
    )
    training.add_argument("--resume", action="store_true")
    training.add_argument("--actors", type=int, choices=(1, 2), default=1)
    training.add_argument("--resident-rollout", action="store_true")
    training.add_argument("--policy-batch-size", type=int, default=0)
    training.add_argument("--defer-metrics", action="store_true")
    training.add_argument("--expected-gpu")
    training.add_argument("--original-only", action="store_true")
    training.add_argument("--lr-scale", type=float, default=1.0)
    training.add_argument("--bc-kl-scale", type=float, default=1.0)
    training.add_argument("--kl-check-every", type=int, default=1)
    training.add_argument("--sample-only", action="store_true")
    training.add_argument("--snapshot-every", type=int, default=0)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--test", type=Path, nargs="+", required=True)
    evaluation.add_argument("--arena-batch", type=int, default=192)
    evaluation.add_argument("--reference-pairs", type=Path)
    evaluation.add_argument("--panel", choices=("preflight", "development", "holdout"), default="preflight")
    comparison.add_argument("--compare-residency", action="store_true")
    comparison.add_argument("checkpoints", type=Path, nargs=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    {"train": train, "evaluate": evaluate, "compare": compare}[args.mode](args)
