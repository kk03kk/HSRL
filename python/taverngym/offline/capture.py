"""独立评测采集器；不修改训练 collector，日志关闭时使用相同决策路径。"""
import json
import time

import numpy as np
import torch
import taverngym as tg

from taverngym.rl.config import identity
from taverngym.rl.model import pack
from . import trace
from .actions import decode


@torch.no_grad()
def capture(cfg, *, seed, actor=None, greedy=True, path=None, seats=None):
    cfg.validate()
    seats = list(seats) if seats is not None else [(seed + g) % 8 for g in range(cfg.n_games)]
    if len(seats) != cfg.n_games or any(not 0 <= p < 8 for p in seats):
        raise ValueError("one learner seat in [0,8) is required per lobby")
    env = tg.BatchEnv(cfg.n_games, seed=seed, controlled=range(8), config=cfg.rules, threads=cfg.threads)
    env.auto_reset = False
    selected = [g * 8 + p for g, p in enumerate(seats)]
    library_mode = cfg.opponent_mode == "library"
    if library_mode:
        from taverngym.rl.library_bank import bank_text
        env.bind_library(bank_text(cfg.library_bank_path,cfg.library_bank_sha256),seats)
    rows = [{"type": "header", "format": trace.FORMAT, "identity": identity(cfg),
             "native": tg.build_info(), "seed": seed, "config": cfg.to_dict(),
             "learner_seats": seats, "policy": "network" if actor is not None else "native_bot",
             "greedy": greedy, "execution_order": "native lobby order then ascending seat",
             "debug_notice": "initial_debug_states contain privileged state; never policy input",
             "initial_debug_states": [json.loads(env.save(g)) for g in range(cfg.n_games)]}] if path else []
    device = torch.device(cfg.device)
    hidden = torch.zeros((env.n_slots, cfg.hidden), device=device)
    counts = np.zeros(env.n_slots, dtype=np.int64)
    done_all = np.zeros(env.n_slots, dtype=bool)
    ranks = np.zeros(env.n_slots, dtype=int)
    applied_total = deferred = 0
    started = time.monotonic()
    status, error = "completed", None
    if actor is not None:
        actor.eval()
    # 独立评测不改变调用者 RNG，包括随机策略采样。
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for tick in range(cfg.max_ticks):
            obs = env.observe()
            ready = np.flatnonzero(obs["ready"])
            if not len(ready):
                status, error = "failed", "no ready slot before termination"
                break
            extras = env.training_extras()
            initial_obs = {i: {"features": obs["features"][i].copy(), "cards": obs["cards"][i].copy(),
                               "mask": obs["action_mask"][i].copy(), "extras": extras[i]} for i in selected if obs["ready"][i]}
            before = {i: trace.public_view(env.game(i // 8), i % 8) for i in initial_obs} if path else {}
            previous = hidden.clone()
            reset = counts % cfg.memory_horizon == 0
            hidden[reset] = 0
            actions = env.bot_actions()
            from taverngym.rl.opponents import profile_action
            from taverngym.rl.collect import observation
            for i in ready:
                if library_mode and i not in selected:
                    actions[i] = int(np.flatnonzero(obs["action_mask"][i])[0])
                elif i not in selected or actor is None:
                    actions[i] = profile_action(observation(obs,extras,int(i)),int(actions[i]),cfg.bot_profile,int(i))
            estimates = {}
            indices = list(initial_obs)
            if actor is not None and indices:
                logits, value, _, new_hidden = actor(pack([initial_obs[i] for i in indices], device), hidden[indices])
                dist = torch.distributions.Categorical(logits=logits)
                chosen = logits.argmax(-1) if greedy else dist.sample()
                actions[indices] = chosen.cpu().numpy()
                hidden[indices] = new_hidden
                for j, i in enumerate(indices):
                    prob = dist.probs[j]
                    top = torch.topk(prob, min(5, int(initial_obs[i]["mask"].sum())))
                    estimates[i] = {"value": float(value[j]), "logp": float(dist.log_prob(chosen)[j]),
                                    "entropy": float(dist.entropy()[j]),
                                    "topk": [{"action": int(a), "probability": float(p)} for p, a in zip(top.values, top.indices)]}
            _, done, info = env.step(actions)
            hidden[~info["applied"]] = previous[~info["applied"]]
            counts += info["applied"]
            applied_total += int(info["applied"].sum())
            deferred += int(np.count_nonzero(obs["ready"] & ~info["applied"]))
            if path:
                decisions = []
                for i, o in initial_obs.items():
                    decisions.append({"slot": i, "lobby": i // 8, "seat": i % 8, "turn": before[i]["turn"],
                                      "decision": int(counts[i]), "action": int(actions[i]), "semantic": decode(int(actions[i])),
                                      "applied": bool(info["applied"][i]), "memory_reset": bool(reset[i]),
                                      "observation": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in o.items()},
                                      "before": before[i], "after": trace.public_view(env.game(i // 8), i % 8),
                                      "value": None, "logp": None, "entropy": None, "topk": None,
                                      **estimates.get(i, {})})
                rows.append({"type": "tick", "tick": tick, "actions": actions.tolist(),
                             "ready": obs["ready"].tolist(), "applied": info["applied"].tolist(),
                             "rejected": info["rejected"].tolist(), "done": done.tolist(),
                             "placement": info["placement"].tolist(), "errors": info["errors"],
                             "decisions": decisions, "post_hashes": [trace.digest(json.loads(env.save(g))) for g in range(cfg.n_games)]})
            diagnostics = env.diagnostics()
            if info["rejected"].any() or any(a or b or c for a, b, c in diagnostics) or any(done[i] and info["placement"][i] == 0 for i in (selected if library_mode else range(env.n_slots))):
                only_turn_cap = not info["rejected"].any() and any(a for a, _, _ in diagnostics) and not any(b or c for _, b, c in diagnostics)
                status, error = "truncated" if only_turn_cap else "failed", str({"errors": info["errors"], "diagnostics": diagnostics})
                break
            if np.any(done & done_all):
                status, error = "failed", "duplicate terminal receipt"
                break
            done_all |= done
            ranks[done] = info["placement"][done]
            if (all(done_all[i] for i in selected) if library_mode else done_all.all()):
                break
        else:
            status, error = "truncated", "collector tick limit"
    result = {"type": "summary", "status": status, "error": error, "seed": seed,
              "ranks": [int(ranks[i]) if done_all[i] else None for i in selected],
              "placements": ranks.reshape(cfg.n_games, 8).tolist(), "learner_seats": seats,
              "ticks": tick + 1, "applied": applied_total, "deferred": deferred,
              "seconds": time.monotonic() - started}
    if path:
        rows.append(result)
        trace.validate(rows)
        trace.write(path, rows)
    return result
