"""Full lobbies -> per-seat *effective-decision* trajectories, never empty batch ticks."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import taverngym as tg
from .model import pack


class InterruptedCollection(Exception):
    pass


def observation(batch, extras, i):
    return {"features": batch["features"][i].copy(), "cards": batch["cards"][i].copy(),
            "mask": batch["action_mask"][i].copy(), "extras": extras[i]}


def teacher_variant(obs, action, style):
    """Two visible-information buy heuristics; otherwise retain native bot decisions."""
    l = dict(tg.layout())
    if style == 2 or not l["A_BUY"] <= action < l["A_SELL"]:
        return action
    offset = (l["GLOBAL_F"] + l["N_QUESTS"] * l["QUEST_F"] + l["PENDING_KINDS"]
              + l["N_BOARD"] * l["MINION_F"] + l["N_HAND"] * l["ITEM_F"])
    best, score = action, -float("inf")
    for a in range(l["A_BUY"], l["A_SELL"]):
        f = offset + (a - l["A_BUY"]) * l["ITEM_F"]
        if obs["mask"][a] and obs["features"][f + 1]:
            atk, hp = obs["features"][f + 10:f + 12]
            value = (2 * atk + hp) if style == 0 else (atk + 2 * hp)
            if value > score:
                best, score = a, value
    return best


@torch.no_grad()
def collect(cfg, actor, seed, opponents=None, *, teacher=False, greedy=False,
            stop=None, artifact_dir=None, record=True, fixed_opponents=False,
            opponent_greedy=False, allow_truncated=False):
    if allow_truncated and (record or teacher):
        raise ValueError("truncation tolerance is evaluation-only; training trajectories require natural endings")
    env = tg.BatchEnv(cfg.n_games, seed=seed, controlled=range(8),
                      config=cfg.rules, threads=cfg.threads)
    env.auto_reset = False
    n = env.n_slots
    learning = {g * 8 + (seed + g + j) % 8
                for g in range(cfg.n_games) for j in range(cfg.learner_seats)}
    library_mode = cfg.opponent_mode == "library" and not teacher and actor is not None
    if library_mode:
        from .library_bank import bank_text
        env.bind_library(bank_text(cfg.library_bank_path, cfg.library_bank_sha256, "train" if record else "evaluation"),
                         [(seed + g) % 8 for g in range(cfg.n_games)])
    roles = {}
    # Each frozen opponent remains the same for the whole lobby/seat episode.
    opponents = opponents or {}
    history = [k for k in opponents if k != "bc"]
    if library_mode:
        keys, weights = ["library"], [1.0]
    elif teacher or actor is None:
        keys, weights = ["bot"], [1.0]
    elif fixed_opponents or cfg.opponent_mode == "bots":
        if cfg.opponent_mode == "bots" and not fixed_opponents:
            opponents = {}
        keys = ["bot", *(["bc"] if "bc" in opponents else [])]
        weights = [1.0] if len(keys) == 1 else [2/3, 1/3]
    else:
        keys = ["bot", "latest", *(["bc"] if "bc" in opponents else []), *history]
        weights = [.4, .2 + (0 if history else .2), *([.2] if "bc" in opponents else []),
                   *[.2 / len(history) for _ in history]]
        weights = np.asarray(weights) / sum(weights)
    rng = np.random.default_rng(seed)
    for i in range(n):
        roles[i] = "learner" if i in learning else str(rng.choice(keys, p=weights))
    policies = {"learner": actor, "latest": actor, **(opponents or {})}
    device = torch.device(cfg.device)
    hidden = torch.zeros((n, cfg.hidden), device=device)
    decision_counts = np.zeros(n, dtype=np.int64)
    initial = [env.save(g) for g in range(cfg.n_games)] if artifact_dir else []
    action_trace = []
    deferred = 0
    if actor is not None:
        for model in policies.values():
            model.eval()
    done = np.zeros(n, dtype=bool)
    ranks = np.zeros(n, dtype=np.int64)
    invalid_games = set()
    trajectories = {i: {"samples": [], "rank": 0, "slot": i, "seed": seed} for i in learning}
    start = time.monotonic()
    decisions = 0
    for tick in range(cfg.max_ticks):
        if stop and stop():
            raise InterruptedCollection("partial collection discarded; previous checkpoint remains valid")
        batch = env.observe()
        ready = np.flatnonzero(batch["ready"])
        if not len(ready):
            if (all(done[i] for i in learning) if library_mode else done.all()): break
            raise RuntimeError("no ready slots before full lobby termination")
        extras = env.training_extras()
        actions = env.bot_actions()
        previous_hidden = hidden.clone()
        hidden[decision_counts % cfg.memory_horizon == 0] = 0
        values, logps = {}, {}
        observations = {int(i): observation(batch, extras, i) for i in ready}
        from .opponents import profile_action
        for i in ready:
            if roles[i] == "library":
                actions[i] = int(np.flatnonzero(batch["action_mask"][i])[0])
            elif roles[i] == "bot":
                actions[i] = profile_action(observations[i], int(actions[i]), cfg.bot_profile, int(i))
        if teacher or actor is None:
            for i in ready:
                if teacher and i in learning:
                    actions[i] = teacher_variant(observations[i], int(actions[i]), i % 3)
        else:
            for role in sorted(set(roles[i] for i in ready)):
                if role in ("bot", "library"): continue
                indices = [int(i) for i in ready if roles[i] == role]
                model = policies[role]
                logits, val, _, nh = model(pack([observations[i] for i in indices], device), hidden[indices])
                if not torch.isfinite(logits).all() or not torch.isfinite(val).all():
                    raise RuntimeError("nonfinite policy/value")
                dist = torch.distributions.Categorical(logits=logits)
                choose_greedy = greedy if role == "learner" else opponent_greedy
                a = logits.argmax(-1) if choose_greedy else dist.sample()
                actions[indices] = a.cpu().numpy()
                hidden[indices] = nh
                for j, i in enumerate(indices):
                    values[i], logps[i] = float(val[j]), float(dist.log_prob(a)[j])
        for i in ready:
            if not observations[i]["mask"][actions[i]]:
                raise RuntimeError(f"illegal selected action {actions[i]} slot {i}")
        if artifact_dir:
            action_trace.append(actions.reshape(cfg.n_games, 8).tolist())
        _, dn, info = env.step(actions)
        applied = info["applied"]
        hidden[~applied] = previous_hidden[~applied]
        decision_counts += applied
        deferred += int(np.count_nonzero(batch["ready"] & ~applied))
        for i in ready:
            if record and i in learning and applied[i]:
                trajectories[i]["samples"].append({"obs": observations[i], "action": int(actions[i]),
                    "logp": logps.get(i, 0.0), "value": values.get(i, 0.0)})
        decisions += int(applied.sum())
        diagnostics = env.diagnostics()
        truncated = {g for g, d in enumerate(diagnostics) if d[0]}
        new_truncated = truncated - invalid_games
        invalid_games.update(truncated)
        bad = next((g for g, d in enumerate(diagnostics) if d[1] or d[2] or (d[0] and not allow_truncated)), None)
        missing_placement = any(dn[i] and info["placement"][i] == 0 and i//8 not in invalid_games for i in (learning if library_mode else range(n)))
        fatal = info["rejected"].any() or bad is not None or missing_placement
        if fatal or new_truncated:
            if artifact_dir:
                p = Path(artifact_dir)
                p.mkdir(parents=True, exist_ok=True)
                failed = {int(i)//8 for i in np.flatnonzero(info["rejected"] | (dn & (info["placement"] == 0)))}
                failed.update(g for g, d in enumerate(diagnostics) if d[1] or d[2])
                failed.update(new_truncated)
                for g in sorted(failed):
                    payload = {"format": 2, "seed": seed, "game_index": g, "tick": tick,
                               "native": tg.build_info(), "engine": tg.game_version(),
                               "initial": json.loads(initial[g]), "post": json.loads(env.save(g)),
                               "actions": [row[g] for row in action_trace], "errors": info["errors"][g],
                               "rejected": info["rejected"][g*8:(g+1)*8].tolist(),
                               "done": dn[g*8:(g+1)*8].tolist(),
                               "placement": info["placement"][g*8:(g+1)*8].tolist(),
                               "diagnostic": diagnostics[g]}
                    (p / f"failure-seed-{seed}-game-{g}.json").write_text(json.dumps(payload), encoding="utf-8")
            if fatal:
                raise RuntimeError(f"collection failure: errors={info['errors']}; diagnostics={diagnostics}")
        if np.any(dn & done):
            raise RuntimeError("duplicate terminal reward")
        done |= dn
        ranks[dn] = info["placement"][dn]
        if (all(done[i] for i in learning) if library_mode else done.all()): break
    else:
        raise RuntimeError("collector tick guard: truncated lobbies are not ranked as losses")
    for i, tr in trajectories.items():
        if i//8 in invalid_games:
            continue
        tr["rank"] = int(ranks[i])
        if not 1 <= tr["rank"] <= 8:
            raise RuntimeError("missing natural placement")
    valid_trajectories = [tr for i, tr in trajectories.items() if i//8 not in invalid_games]
    placements = [int(ranks[i]) for i in sorted(learning) if i//8 not in invalid_games]
    return valid_trajectories, {
        "games": cfg.n_games - len(invalid_games), "requested_games": cfg.n_games,
        "truncated_games": len(invalid_games), "truncated_game_indices": sorted(invalid_games),
        "learner_episodes": len(placements), "decisions": int(decisions),
        "samples": sum(len(t["samples"]) for t in trajectories.values()),
        "mean_rank": float(np.mean(placements)) if placements else None,
        "top4": float(np.mean(np.array(placements) <= 4)) if placements else None,
        "win": float(np.mean(np.array(placements) == 1)) if placements else None, "ranks": placements,
        "seconds": time.monotonic() - start, "ticks": tick + 1, "deferred": deferred,
        "opponent_counts": {k: sum(v == k for v in roles.values()) for k in keys},
        "bot_profile": cfg.bot_profile, "opponent_process": "library-v1" if library_mode else cfg.opponent_mode,
    }


def gae(samples, rank, gamma=1.0, lam=0.99):
    """Natural finite episodes only. No bootstrap across death/reset/waiting ticks."""
    reward = (9 - 2 * rank) / 7
    adv = np.zeros(len(samples), dtype=np.float32)
    next_value, last = 0.0, 0.0
    for i in reversed(range(len(samples))):
        value = samples[i]["value"]
        delta = (reward if i == len(samples) - 1 else 0.0) + gamma * next_value - value
        last = delta + gamma * lam * last
        adv[i], next_value = last, value
    return adv, adv + np.array([s["value"] for s in samples], dtype=np.float32)
