"""Read-only first-model diagnosis. No optimizer, training, environment install or game input.

Run from the exact deployed release with its PYTHONPATH. This file can live in HOME
without being added to the release manifest or changing checkpoint identity.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics_summary(path):
    rows, malformed = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            malformed += 1
    ppo = [x for x in rows if x.get("phase") == "ppo"]
    bc = [x for x in rows if x.get("phase") == "bc"]
    return {"malformed_lines": malformed, "bc_waves": len(bc),
            "bc_samples": sum(x["collection"]["samples"] for x in bc),
            "ppo_waves": len(ppo), "ppo_samples": sum(x["collection"]["samples"] for x in ppo),
            "last_iteration": max((x["iteration"] for x in ppo), default=0),
            "completed_budget_log": any(x.get("phase") == "completed" for x in rows),
            "bc_losses": [{"wave": x["wave"], **{k: x["training"].get(k) for k in ("policy", "rank_ce", "updates")}} for x in bc],
            "validation": [x for x in rows if x.get("phase") == "validation"],
            "last_ppo": ppo[-3:]}


def replay_failure(path, out):
    import taverngym as tg
    f = json.loads(Path(path).read_text(encoding="utf-8"))
    if f.get("format") != 2:
        raise ValueError("exact replay requires a format-2 action trace")
    if f["native"]["source_hash"] != tg.build_info()["source_hash"]:
        raise ValueError("native source mismatch; do not bypass the failed experiment's identity")
    env = tg.BatchEnv(1, controlled=range(8), threads=1)
    env.auto_reset = False
    env.load(0, json.dumps(f["initial"]))
    result = {"source": Path(path).name, "expected_diagnostic": f.get("diagnostic"),
              "action_rows": len(f["actions"]), "guard_reproduced": False}
    for tick, actions in enumerate(f["actions"]):
        before = json.loads(env.save(0))
        _, dn, info = env.step(actions)
        diagnostic = env.diagnostics()[0]
        bad = bool(info["rejected"].any()) or any(diagnostic) or any(
            d and p == 0 for d, p in zip(dn, info["placement"]))
        if bad:
            post = json.loads(env.save(0))
            result.update(guard_reproduced=True, tick=tick, diagnostic=diagnostic,
                          rejected=bool(info["rejected"].any()), errors=info["errors"][0],
                          post_matches_saved=canonical(post) == canonical(f["post"]),
                          stopped_at_expected_row=tick == len(f["actions"]) - 1,
                          turn=before["game"]["turn"], settings=before["game"].get("cfg"))
            write(out / "before-guard.json", before)
            write(out / "replayed-post.json", post)
            break
    write(out / "replay.json", result)
    return result


def group(action):
    import taverngym as tg
    return re.match(r"[A-Za-z]+", tg.action_name(int(action)))[0]


def behavior(trajectories):
    import taverngym as tg
    layout = dict(tg.layout())
    offset = layout["GLOBAL_F"] + layout["N_QUESTS"] * layout["QUEST_F"] + layout["PENDING_KINDS"]
    counts, turns, ends, examples = Counter(), defaultdict(Counter), [], []
    for tr in trajectories:
        for sample in tr["samples"]:
            a, obs = sample["action"], sample["obs"]
            f = obs["features"]; turn = round(float(f[0]) * 20)
            category = group(a); counts[category] += 1; turns[turn][category] += 1
            state = {"turn": turn, "action": tg.action_name(int(a)), "gold": round(float(f[7]) * 10),
                     "tier": round(float(f[10]) * 6), "rank": tr["rank"],
                     "board_size": sum(bool(f[offset + i * layout["MINION_F"]]) for i in range(layout["N_BOARD"]))}
            if len(examples) < 16:
                examples.append(state)
            if category == "EndTurn":
                state["buy_available"] = bool(obs["mask"][layout["A_BUY"]:layout["A_SELL"]].any())
                ends.append(state)
    return {"effective_decisions": sum(counts.values()), "actions": dict(counts),
            "by_turn": {str(k): dict(v) for k, v in sorted(turns.items())},
            "end_turn_states": ends, "first_decisions": examples,
            "notice": "EndTurn with gold/buy available is diagnostic context, not automatically a mistake."}


def teacher_agreement(model, trajectories, cfg, limit=1024):
    import torch
    from taverngym.rl.model import pack
    n = correct = 0; nll = 0.; labels = Counter(); predicted = Counter()
    with torch.no_grad():
        for tr in trajectories:
            h = torch.zeros((1, cfg.hidden))
            for index, sample in enumerate(tr["samples"]):
                if index % cfg.memory_horizon == 0:
                    h.zero_()
                logits, _, _, h = model(pack([sample["obs"]], "cpu"), h)
                if sample["obs"]["mask"].sum() <= 1:
                    continue
                action = int(sample["action"]); choice = int(logits.argmax(-1))
                n += 1; correct += action == choice
                nll += float(-logits.log_softmax(-1)[0, action])
                labels[group(action)] += 1; predicted[group(choice)] += 1
                if n >= limit:
                    break
            if n >= limit:
                break
    return {"nonforced_decisions": n, "top1_agreement": correct / n if n else None,
            "negative_log_likelihood": nll / n if n else None,
            "teacher_groups": dict(labels), "predicted_groups": dict(predicted),
            "notice": "Teacher-forced scripted states, not an expert-skill metric."}


def audit_models(run, out, games=2):
    import torch
    import taverngym as tg
    from taverngym.rl.config import Config, identity
    from taverngym.rl.train import load_policy
    from taverngym.rl.collect import collect
    torch.set_num_threads(2)
    result = {}
    for name in ("bc-anchor.pt", "latest.pt", "best.pt"):
        path = run / name
        if not path.exists():
            result[name] = {"missing": True}; continue
        entry = {"sha256": sha(path)}
        result[name] = entry
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            original = Config(**state["config"])
            cfg = replace(original, device="cpu", threads=2, torch_threads=2,
                          n_games=games, learner_seats=1)
            cfg.validate()
            expected = identity(cfg)
            entry.update(progress=state.get("progress"), identity=state["identity"]["digest"],
                         architecture={k: state["config"][k] for k in ("policy_kind", "dim", "layers", "heads", "hidden")})
            model = load_policy(path, cfg, expected)
            entry["parameters"] = sum(p.numel() for p in model.parameters())
            entry["strict_identity_passed"] = True
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1903100000)
                teachers, teacher_stats = collect(cfg, None, 1903100000, teacher=True,
                                                 artifact_dir=out / name / "teacher-failures")
                entry["teacher_distribution"] = behavior(teachers)
                entry["teacher_stats"] = teacher_stats
                entry["teacher_agreement"] = teacher_agreement(model, teachers, cfg)
            entry["panels"] = []
            for mode in ("greedy", "sample"):
                for seed in (1903200000, 1903300000):
                    panel = {"mode": mode, "seed": seed}; entry["panels"].append(panel)
                    try:
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(seed)
                            trajectories, stats = collect(cfg, model, seed, greedy=mode == "greedy",
                                artifact_dir=out / name / f"{mode}-{seed}")
                        panel.update(status="completed", stats=stats, behavior=behavior(trajectories))
                    except Exception as e:
                        panel.update(status="failed", error=type(e).__name__ + ": " + str(e))
                    print(json.dumps({"checkpoint": name, "mode": mode, "seed": seed, "status": panel["status"]}), flush=True)
        except Exception as e:
            entry["error"] = type(e).__name__ + ": " + str(e)
        write(out / "models.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--games", type=int, default=2)
    p.add_argument("--replay-only", action="store_true")
    args = p.parse_args(argv)
    if not 1 <= args.games <= 8:
        p.error("diagnostic games must be in 1..8")
    args.run = args.run.resolve(); args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("diagnosis requires a new output directory")
    if args.run == args.out or args.run.is_relative_to(args.out) or args.out.is_relative_to(args.run):
        raise ValueError("diagnosis output must be separate from the original training run")
    args.out.mkdir(parents=True)
    import taverngym as tg
    report = {"format": "taverngym.pilot-diagnosis.v1", "training_performed": False,
              "native": tg.build_info(), "engine": tg.game_version(), "source_run": str(args.run),
              "failures": [], "started_unix": time.time()}
    inputs = {str(path): sha(path) for path in args.run.iterdir() if path.is_file()}
    try:
        if (args.run / "metrics.jsonl").exists():
            report["training_metrics"] = metrics_summary(args.run / "metrics.jsonl")
        for failure in sorted(args.run.glob("failure-*.json")):
            target = args.out / failure.stem; target.mkdir()
            try:
                report["failures"].append(replay_failure(failure, target))
            except Exception as e:
                report["failures"].append({"source": failure.name, "error": str(e)})
        if not args.replay_only:
            report["models"] = audit_models(args.run, args.out, args.games)
    except Exception:
        report["fatal_error"] = traceback.format_exc()
    report["source_files_unchanged"] = inputs == {str(path): sha(path) for path in args.run.iterdir() if path.is_file()}
    report["checks"] = {"source_files_unchanged": report["source_files_unchanged"],
        "failure_replay": all(x.get("guard_reproduced") and x.get("post_matches_saved") for x in report["failures"])}
    if not args.replay_only:
        models = report.get("models", {})
        report["checks"]["model_panels"] = all(
            models.get(n, {}).get("strict_identity_passed") and not models[n].get("error")
            and len(models[n].get("panels", [])) == 4
            and all(x["status"] == "completed" for x in models[n]["panels"])
            for n in ("bc-anchor.pt", "latest.pt"))
    report["finished_unix"] = time.time()
    write(args.out / "report.json", report)
    print(json.dumps({"report": str(args.out / "report.json"), "source_files_unchanged": report["source_files_unchanged"]}), flush=True)
    return 1 if report.get("fatal_error") or not all(report["checks"].values()) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd() / "python"))
    if sys.platform == "win32":
        sys.path.insert(0, str(Path.cwd() / "tools"))
        from local_offline import prepare
        handles = prepare()
    raise SystemExit(main())
