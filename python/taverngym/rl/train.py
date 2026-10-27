"""BC -> recurrent PPO -> bounded historical opponent pool. Checkpoints at full-wave boundaries."""
import argparse
import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import signal
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config, identity
from .collect import collect, gae, InterruptedCollection
from .model import Policy, pack


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s["python"])
    n = s["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def load_policy(path, cfg, expected):
    state = torch.load(path, map_location=cfg.device, weights_only=True)
    if state["identity"]["digest"] != expected["digest"]:
        raise ValueError(f"checkpoint schema/engine/rules mismatch: {path}")
    # Loading opponents must not perturb the learner's sampling RNG.
    with torch.random.fork_rng(devices=[]):
        model = Policy(cfg).to(cfg.device)
    model.load_state_dict(state["model"])
    return model.eval()


def initialize_weights(model, path, cfg, expected):
    """Explicit new experiment: compatible weights only, never old optimizer/RNG/progress."""
    path = Path(path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    source = state["identity"]
    for key in ("schema", "engine", "layout", "vocabulary", "architecture"):
        if source.get(key) != expected.get(key):
            raise ValueError(f"init-from incompatible {key}")
    current = model.state_dict()
    canonicalized = []
    for key in ("numeric_map", "src", "dst", "positions", "utility"):
        wanted, supplied = current[key].cpu(), state["model"][key].cpu()
        # CUDA/CPU linspace can differ by one float32 ULP. This fixed utility vector
        # is a semantic constant, not learned weights; integer maps stay bit-exact.
        if key == "utility" and wanted.dtype == supplied.dtype == torch.float32 and wanted.shape == supplied.shape:
            if not torch.allclose(wanted, supplied, rtol=0, atol=2 * torch.finfo(torch.float32).eps):
                raise ValueError(f"init-from incompatible buffer {key}")
            if not torch.equal(wanted, supplied):
                canonicalized.append(key)
            state["model"][key] = current[key]
        elif not torch.equal(wanted, supplied):
            raise ValueError(f"init-from incompatible buffer {key}")
    model.load_state_dict(state["model"], strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_identity": source["digest"], "source_progress": state.get("progress"),
            "canonicalized_constant_buffers": canonicalized,
            "reset": ["optimizer", "rng", "progress", "opponent_pool", "validation_scores"]}


@torch.no_grad()
def prefix_hidden(model, chunks, device, horizon=128):
    """Exact current-weight prefix within explicit periodic memory resets used in collection."""
    if not getattr(model, "is_recurrent", True):
        return torch.zeros((1, len(chunks), model.hidden_size), device=device)
    hidden = []
    for tr, start in chunks:
        h = torch.zeros((1, 1, model.hidden_size), device=device)
        boundary = start - start % horizon
        for offset in range(boundary, start, 128):
            obs = [s["obs"] for s in tr["samples"][offset:min(start, offset + 128)]]
            summary, _ = model.encode(pack(obs, device))
            _, h = model.memory(summary[None], h)
        hidden.append(h[:, 0])
    return torch.stack(hidden, dim=1)


def update(model, optimizer, trajectories, cfg, *, bc=False, anchor=None):
    started = time.monotonic()
    prefix_seconds = 0.0
    executed = 0
    kl_values = []
    def report(stopped=False):
        return {**{k: float(np.mean([r[k] for r in logs])) for k in logs[0]},
                "updates": executed, "evaluated_batches": len(logs), "kl_early_stop": stopped,
                "kl_max": max(kl_values, default=0.0),
                "kl_trigger": kl_values[-1] if stopped else None,
                "seconds": time.monotonic()-started, "prefix_seconds": prefix_seconds}
    all_adv = []
    for tr in trajectories:
        adv, ret = gae(tr["samples"], tr["rank"], cfg.gamma, cfg.gae_lambda)
        tr["advantages"], tr["returns"] = adv, ret
        all_adv.extend(adv)
    if not all_adv:
        raise RuntimeError("no training decisions")
    mean, std = float(np.mean(all_adv)), max(float(np.std(all_adv)), 1e-6)
    chunks = [(tr, start) for tr in trajectories for start in range(0, len(tr["samples"]), cfg.seq_len)]
    logs = []
    model.train()
    for epoch in range(cfg.bc_epochs if bc else cfg.epochs):
        order = np.random.permutation(len(chunks))
        for offset in range(0, len(order), cfg.chunks_per_batch):
            group = [chunks[int(j)] for j in order[offset:offset + cfg.chunks_per_batch]]
            accumulation = 1 if bc else cfg.gradient_accumulation
            micro = offset // cfg.chunks_per_batch
            window_start = (micro // accumulation) * accumulation * cfg.chunks_per_batch
            window_stop = min(len(order), window_start + accumulation * cfg.chunks_per_batch)
            window_valid = sum(min(cfg.seq_len, len(chunks[int(j)][0]["samples"]) - chunks[int(j)][1])
                               for j in order[window_start:window_stop])
            if offset == window_start:
                optimizer.zero_grad(set_to_none=True)
            prefix_start = time.monotonic()
            h = prefix_hidden(model, group, cfg.device, cfg.memory_horizon)
            prefix_seconds += time.monotonic()-prefix_start
            rows, valid, actions, old, advantages, returns, ranks = [], [], [], [], [], [], []
            for tr, start in group:
                for t in range(cfg.seq_len):
                    pos = min(start + t, len(tr["samples"]) - 1)
                    s = tr["samples"][pos]
                    rows.append(s["obs"])
                    valid.append(start + t < len(tr["samples"]))
                    actions.append(s["action"])
                    old.append(s["logp"])
                    advantages.append((tr["advantages"][pos] - mean) / std)
                    returns.append(tr["returns"][pos])
                    ranks.append(tr["rank"] - 1)
            tensor = lambda x, dtype: torch.tensor(x, dtype=dtype, device=cfg.device)
            obs = pack(rows, cfg.device)
            summary, entities = model.encode(obs)
            states, _ = model.memory(summary.reshape(len(group), cfg.seq_len, -1), h)
            logits, values, rank = model.heads(states.reshape(-1, cfg.hidden), entities, obs["mask"])
            keep = tensor(valid, torch.bool)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(tensor(actions, torch.long))
            rank_loss = F.cross_entropy(rank[keep], tensor(ranks, torch.long)[keep])
            entropy = dist.entropy()[keep].mean()
            if bc:
                policy_keep = keep & (obs["mask"].sum(-1) > 1)
                if not policy_keep.any(): continue
                pg = -logp[policy_keep].mean()
                value_loss = torch.zeros((), device=cfg.device)
                kl = torch.zeros((), device=cfg.device)
                loss = pg + cfg.bc_rank_coef * rank_loss
            else:
                ratio_log = logp - tensor(old, torch.float32)
                ratio = ratio_log.exp()
                advantage = tensor(advantages, torch.float32)
                pg = -torch.minimum(ratio * advantage, ratio.clamp(1-cfg.clip, 1+cfg.clip) * advantage)[keep].mean()
                value_loss = F.mse_loss(values[keep], tensor(returns, torch.float32)[keep])
                kl = ((ratio - 1) - ratio_log)[keep].mean()
                loss = pg + cfg.value_coef * value_loss + cfg.rank_coef * rank_loss - cfg.entropy * entropy
            anchor_kl = torch.zeros((), device=cfg.device)
            if not bc and anchor is not None and cfg.anchor_coef:
                with torch.no_grad():
                    ah = prefix_hidden(anchor, group, cfg.device, cfg.memory_horizon)
                    az, ae = anchor.encode(obs)
                    astates, _ = anchor.memory(az.reshape(len(group), cfg.seq_len, -1), ah)
                    alogits, _, _ = anchor.heads(astates.reshape(-1, cfg.hidden), ae, obs["mask"])
                    ap = alogits.softmax(-1)
                    alp = alogits.log_softmax(-1)
                current_lp = logits.log_softmax(-1)
                anchor_kl = (ap * (alp - current_lp)).sum(-1)[keep].mean()
                loss = loss + cfg.anchor_coef * anchor_kl
            if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
            measured_kl = float(kl.detach())
            kl_values.append(measured_kl)
            row = {"loss": float(loss.detach()), "policy": float(pg.detach()),
                   "value": float(value_loss.detach()), "rank_ce": float(rank_loss.detach()),
                   "anchor_kl": float(anchor_kl.detach()),
                   "entropy": float(entropy.detach()), "kl": measured_kl, "grad": 0.0}
            if not bc and measured_kl > cfg.target_kl:
                optimizer.zero_grad(set_to_none=True)
                logs.append(row)
                return report(True)
            (loss * (int(keep.sum()) / window_valid)).backward()
            if offset + cfg.chunks_per_batch >= window_stop:
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                executed += 1
                row["grad"] = float(grad)
            logs.append(row)
    if not logs: raise RuntimeError("no trainable samples")
    return report()


def evaluate(cfg, model, seed=None, games=None, anchor=None, stop=None, artifact_dir=None):
    c = replace(cfg, n_games=games or cfg.eval_games, learner_seats=1)
    result, saved_rng = {}, rng_state()
    seeds = cfg.eval_seeds if seed is None else [seed]
    try:
        distributions = [("bots", replace(c, opponent_mode="bots"), None)]
        if cfg.opponent_mode == "library":
            if not cfg.library_validation_path or not cfg.library_validation_sha256:
                raise ValueError("library evaluation requires a separate validation bank")
            distributions.append(("library", replace(c, library_bank_path=cfg.library_validation_path,
                library_bank_sha256=cfg.library_validation_sha256), None))
        elif anchor is not None:
            distributions.append(("anchor_mix", c, {"bc": anchor}))
        for label, panel_cfg, opponents in distributions:
            for mode in ("greedy", "sample"):
                panels = []
                for panel_seed in seeds:
                    torch.manual_seed(panel_seed)
                    _, stats = collect(panel_cfg, model, panel_seed, opponents, greedy=mode == "greedy",
                                       opponent_greedy=True, record=False, fixed_opponents=True, stop=stop,
                                       allow_truncated=True,
                                       artifact_dir=Path(artifact_dir)/f"{label}-{mode}-{panel_seed}" if artifact_dir else None)
                    panels.append(stats)
                ranks = np.array([r for s in panels for r in s["ranks"]])
                rng = np.random.default_rng(seeds[0])
                means = ranks[rng.integers(0, len(ranks), size=(2000, len(ranks)))].mean(1) if len(ranks) else None
                truncated = sum(s.get("truncated_games", 0) for s in panels)
                result[f"{label}_{mode}"] = {
                    "games": len(ranks), "learner_episodes": len(ranks), "ranks": ranks.tolist(),
                    "requested_games": c.n_games * len(seeds), "truncated_games": truncated,
                    "eligible_for_selection": len(ranks) > 0 and truncated == 0,
                    "mean_rank": float(ranks.mean()) if len(ranks) else None,
                    "top4": float((ranks<=4).mean()) if len(ranks) else None,
                    "win": float((ranks==1).mean()) if len(ranks) else None,
                    "mean_rank_ci95": np.quantile(means,[.025,.975]).tolist() if means is not None else None,
                    "seconds": sum(s["seconds"] for s in panels), "seeds": list(seeds),
                    "learner_mode": mode, "opponent_mode": "greedy", "panels": panels}
        return result
    finally:
        restore_rng(saved_rng)


def selection_score(result):
    """只用完整评测选best；有截断时条件均值仅供诊断，避免幸存者偏差。"""
    if not result or any(not panel.get("eligible_for_selection", True) for panel in result.values()):
        return None
    sampled = [panel["mean_rank"] for name, panel in result.items() if name.endswith("_sample")]
    return float(np.mean(sampled)) if sampled and all(v is not None for v in sampled) else None


@contextmanager
def run_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run-lock").open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            if not f.read(1): f.write(b"0"); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def train(cfg, out, resume=False, init_from=None):
    with run_lock(out):
        return _train(cfg, out, resume, init_from)


def _train(cfg, out, resume=False, init_from=None):
    cfg.validate()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    torch.set_num_threads(cfg.torch_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    if latest.exists() and not resume:
        raise FileExistsError("output already has a checkpoint; use --resume")
    if not resume and any(p.name != ".run-lock" for p in out.iterdir()):
        raise FileExistsError("new experiment requires an empty output directory")
    signature = identity(cfg)
    model = Policy(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    progress = {"bc_done": 0, "iteration": 0, "best_rank": 9.0, "history": [], "last_eval": -1,
                "ppo_decisions": 0, "bc_decisions": 0}
    if resume and init_from:
        raise ValueError("resume and init-from are mutually exclusive")
    initialization = None
    if init_from:
        initialization = initialize_weights(model, init_from, cfg, signature)
    if resume:
        state = torch.load(latest, map_location=cfg.device, weights_only=True)
        if state["identity"]["digest"] != signature["digest"]:
            raise ValueError("engine/schema/rules changed; create a new run")
        allowed = {"device", "threads", "torch_threads", "iterations", "decision_budget"}
        if any(state["config"][k] != v for k, v in cfg.to_dict().items() if k not in allowed):
            raise ValueError("resume changed training semantics; use same config or a new run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        progress = state["progress"]
        restore_rng(state["rng"])
    else:
        (out / "manifest.json").write_text(json.dumps({"identity": signature, "config": cfg.to_dict(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "device": cfg.device, "initialization": initialization,
            "profile": "36.6.3 corrected first-attack baseline; no Naga; not certified full live parity"},
            ensure_ascii=False, indent=2), encoding="utf-8")
    interrupted = [False]
    def on_signal(signum, frame): interrupted[0] = True
    for sig in (signal.SIGTERM, signal.SIGINT, *([signal.SIGUSR1] if hasattr(signal, "SIGUSR1") else [])):
        signal.signal(sig, on_signal)
    def checkpoint(path=latest):
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": cfg.to_dict(), "identity": signature,
                     "progress": dict(progress), "rng": rng_state()}, path)
    def snapshot(path):
        atomic_save({"model": model.state_dict(), "config": cfg.to_dict(), "identity": signature}, path)
    def log(row):
        row["time_unix"] = time.time()
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    checkpoint()
    try:
        for wave in range(progress["bc_done"], cfg.bc_waves):
            tr, stats = collect(cfg, None, cfg.seed + wave * 700001, teacher=True,
                                stop=lambda: interrupted[0], artifact_dir=out)
            metrics = update(model, optimizer, tr, cfg, bc=True)
            progress["bc_done"] = wave + 1
            progress["bc_decisions"] = progress.get("bc_decisions", 0) + stats["samples"]
            checkpoint()
            log({"phase": "bc", "wave": wave + 1, "collection": stats, "training": metrics})
            if interrupted[0]: return
        anchor_path = out / "bc-anchor.pt"
        if not anchor_path.exists(): snapshot(anchor_path)
        anchor = load_policy(anchor_path, cfg, signature)
        anchor.requires_grad_(False)
        def validation(iteration):
            result = evaluate(cfg, model, anchor=anchor, stop=lambda: interrupted[0],
                              artifact_dir=out / "validation" / f"iteration-{iteration:06d}")
            selection_panels = ({k: v for k, v in result.items() if k.startswith("library_" if cfg.opponent_mode == "library" else "bots_")}
                                if cfg.opponent_mode in ("bots", "library") else result)
            score = selection_score(selection_panels)
            if score is not None and score < progress["best_rank"]:
                progress["best_rank"] = score
                progress["last_eval"] = iteration
                checkpoint(out / "best.pt")
            progress["last_eval"] = iteration
            checkpoint()
            log({"phase": "validation", "iteration": iteration, "selection_score": score,
                 "eligible_for_selection": score is not None,
                 "selection_protocol": ("library_sample against held-out source groups" if cfg.opponent_mode == "library" else "bots_sample against fixed scripted opponents" if cfg.opponent_mode == "bots"
                                        else "mean of sampled-learner panels; fixed greedy opponents"), "results": result})
        if progress["last_eval"] < 0:
            validation(0)
        elif progress["iteration"] % cfg.eval_every == 0 and progress["last_eval"] < progress["iteration"]:
            validation(progress["iteration"])
        for iteration in range(progress["iteration"] + 1, cfg.iterations + 1):
            if interrupted[0]: break
            if cfg.decision_budget and progress.get("ppo_decisions", 0) >= cfg.decision_budget:
                break
            opponents = {"bc": anchor}
            for name in progress["history"]:
                opponents[name] = load_policy(out / name, cfg, signature)
            tr, stats = collect(cfg, model, cfg.seed + iteration * 100003, opponents,
                                stop=lambda: interrupted[0], artifact_dir=out)
            metrics = update(model, optimizer, tr, cfg,
                             anchor=anchor if iteration <= cfg.anchor_iterations else None)
            progress["iteration"] = iteration
            progress["ppo_decisions"] = progress.get("ppo_decisions", 0) + stats["samples"]
            obsolete = []
            if iteration % cfg.history_every == 0:
                name = f"league/iteration-{iteration:06d}.pt"
                snapshot(out / name)
                history = progress["history"] + [name]
                obsolete, progress["history"] = history[:-cfg.history_size], history[-cfg.history_size:]
            # Save the completed iteration before evaluation, so a signal cannot lose training.
            checkpoint()
            for name in obsolete: (out / name).unlink(missing_ok=True)
            log({"phase": "ppo", "iteration": iteration, "collection": stats, "training": metrics})
            budget_done = cfg.decision_budget and progress["ppo_decisions"] >= cfg.decision_budget
            if (iteration % cfg.eval_every == 0 or budget_done or iteration == cfg.iterations) and not interrupted[0]:
                validation(iteration)
        if not interrupted[0]:
            log({"phase": "completed", "iteration": progress["iteration"],
                 "ppo_decisions": progress.get("ppo_decisions", 0),
                 "reason": "decision_budget" if cfg.decision_budget and progress.get("ppo_decisions", 0) >= cfg.decision_budget else "iteration_limit"})
    except InterruptedCollection as e:
        log({"phase": "interrupted", "reason": str(e)})
    finally:
        # Deliberately do not overwrite latest with partial collection/optimization.
        print(f"checkpoint: {latest}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", help="compatible weights into a NEW run; reset optimizer and progress")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--games", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--anchor", help="BC checkpoint; default sibling bc-anchor.pt")
    p.add_argument("--bots-only", action="store_true", help="explicit reduced protocol, not comparable to best selection")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate" and ((args.games is not None and args.games < 1)
                                       or (args.seed is not None and args.seed < 0)):
        parser.error("evaluation games must be positive and seed nonnegative")
    if args.command == "train":
        from .doctor import check
        cfg = Config.load(args.config)
        check(cfg)
        train(cfg, args.out, args.resume, args.init_from)
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = Config(**state["config"])
        cfg.device = args.device
        if args.split == "test": cfg.eval_seeds = [1900000000, 1901000000]
        cfg.validate()
        torch.set_num_threads(cfg.torch_threads)
        model = load_policy(args.checkpoint, cfg, identity(cfg))
        anchor = None if args.bots_only else load_policy(
            args.anchor or Path(args.checkpoint).parent / "bc-anchor.pt", cfg, identity(cfg))
        result = evaluate(cfg, model, args.seed, args.games, anchor=anchor)
        result = {"split": args.split, "bots_only": args.bots_only, "results": result}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__": main()
