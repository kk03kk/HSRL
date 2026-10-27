"""Bounded supervised recovery with held-out readiness gates before any PPO.

This is a new experiment and weights-only initialization, not resume of the failed run.
The gate is a basic behavior check against scripted teachers, not a human-skill claim.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import taverngym as tg

from taverngym.rl.config import Config, identity
from taverngym.rl.model import Policy, pack
from taverngym.rl.collect import collect
from taverngym.rl.train import atomic_save, initialize_weights, update, evaluate, rng_state, restore_rng


GIFT_PAIR_BANS_3701 = [
    ["BG36_849", "BG36_MidGameEffect_000t60"],
    ["BG23_318", "BG36_MidGameEffect_000t16"],
    ["BG36_360t5", "BG36_MidGameEffect_000t15"],
]


@torch.no_grad()
def agreement(model, trajectories, cfg):
    model.eval()
    n = hits = 0
    nll = 0.
    groups, guesses = Counter(), Counter()
    for tr in trajectories:
        h = torch.zeros((1, 1, cfg.hidden), device=cfg.device)
        for start in range(0, len(tr["samples"]), 32):
            if start % cfg.memory_horizon == 0:
                h.zero_()
            samples = tr["samples"][start:start+32]
            obs = pack([s["obs"] for s in samples], cfg.device)
            z, entities = model.encode(obs)
            states, h = model.memory(z[None], h)
            logits, _, _ = model.heads(states[0], entities, obs["mask"])
            labels = torch.tensor([s["action"] for s in samples], device=cfg.device)
            keep = obs["mask"].sum(-1) > 1
            predicted = logits.argmax(-1)
            n += int(keep.sum()); hits += int(((labels == predicted) & keep).sum())
            nll += float(-logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)[keep].sum())
            for index in keep.nonzero().flatten().tolist():
                def category(a):
                    import re
                    return re.match(r"[A-Za-z]+", tg.action_name(int(a)))[0]
                groups[category(labels[index])] += 1
                guesses[category(predicted[index])] += 1
    return {"decisions": n, "top1": hits/n if n else None, "nll": nll/n if n else None,
            "teacher_actions": dict(groups), "greedy_actions_on_teacher_states": dict(guesses)}


def readiness(accuracy, panels, *, minimum_accuracy=.5, maximum_rank=6.5, minimum_decisions=128):
    checks = {"enough_teacher_decisions": accuracy["decisions"] >= minimum_decisions,
              "teacher_agreement": accuracy["top1"] is not None and accuracy["top1"] >= minimum_accuracy}
    for label in ("bots_greedy", "bots_sample"):
        p = panels.get(label, {})
        checks[label] = bool(p.get("eligible_for_selection") and p.get("mean_rank") is not None
                             and p["mean_rank"] <= maximum_rank and not p.get("truncated_games"))
    return {"ready_for_ppo": all(checks.values()), "checks": checks,
            "thresholds": {"minimum_accuracy": minimum_accuracy, "maximum_rank": maximum_rank,
                           "minimum_decisions": minimum_decisions},
            "notice": "Prerequisite for PPO on this scripted distribution; not expert or live-game qualification."}


def bootstrap(config, out, *, init_from, waves=16, epochs=4, bc_lr=3e-4, batch_chunks=8, gate_games=8, stop=None):
    cfg = Config.load(config)
    if cfg.memory_horizon % 32:
        raise ValueError("bootstrap evaluator needs memory horizon divisible by 32")
    if sorted(cfg.rules.get("dark_gift_pair_bans", [])) != sorted(GIFT_PAIR_BANS_3701):
        raise ValueError("recovery profile requires the three explicit 37.0.1 gift offering fixes")
    if not tg.build_is_current(Path.cwd()):
        raise ValueError("native source changed; rebuild before bootstrap")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("bootstrap requires a new output directory")
    out.mkdir(parents=True)
    torch.set_num_threads(cfg.torch_threads)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model = Policy(cfg).to(cfg.device)
    signature = identity(cfg)
    initialization = (initialize_weights(model, init_from, cfg, signature) if init_from else {"kind": "fresh architecture", "parameters": sum(p.numel() for p in model.parameters())})
    bc_cfg = replace(cfg, bc_epochs=epochs, bc_rank_coef=0., chunks_per_batch=batch_chunks)
    optimizer = torch.optim.Adam(model.parameters(), lr=bc_lr)
    progress = {"bc_done": 0, "bc_decisions": 0, "iteration": 0}
    def save(name):
        atomic_save({"model": model.state_dict(), "config": cfg.to_dict(), "identity": signature,
                     "progress": dict(progress), "bootstrap": True}, out/name)
    def log(row):
        row["time_unix"] = time.time()
        with (out/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    manifest = {"format": "taverngym.bootstrap.v1", "identity": signature, "initialization": initialization,
                "waves": waves, "epochs": epochs, "bc_lr": bc_lr, "batch_chunks": batch_chunks,
                "heldout_teacher_seed": 1907100000, "gate_seeds": [1908100000, 1908200000],
                "gate_games_per_seed": gate_games,
                "rank_auxiliary_coefficient": 0., "ppo_performed": False,
                "profile": "36.6.3 implemented baseline + three 37.0.1 gift offering exclusions; no full live parity"}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save('candidate.pt')
    # Fixed held-out teachers are never added to the supervised training stream.
    gate_cfg = replace(cfg, n_games=gate_games, learner_seats=1,
                       eval_games=gate_games, eval_seeds=[1908100000, 1908200000])
    held_rng = rng_state()
    teachers, _ = collect(gate_cfg, None, 1907100000, teacher=True, artifact_dir=out/'teacher-failures', stop=stop)
    restore_rng(held_rng)
    log({"phase": "initial_agreement", "result": agreement(model, teachers, cfg)})
    for wave in range(waves):
        trajectories, stats = collect(cfg, None, cfg.seed + wave*700001, teacher=True,
                                      artifact_dir=out/'collection-failures', stop=stop)
        metrics = update(model, optimizer, trajectories, bc_cfg, bc=True)
        progress["bc_done"] = wave+1; progress["bc_decisions"] += stats["samples"]
        save('candidate.pt')
        log({"phase": "bc_recovery", "wave": wave+1, "collection": stats, "training": metrics})
    accuracy = agreement(model, teachers, cfg)
    panels = evaluate(gate_cfg, model, artifact_dir=out/'gate-failures', stop=stop)
    result = {**readiness(accuracy, panels), "agreement": accuracy, "panels": panels,
              "source_initialization": initialization, "progress": progress}
    (out/'gate.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['ready_for_ppo']:
        save('bc-ready.pt')
    log({"phase": "bootstrap_gate", "result": result})
    return result
