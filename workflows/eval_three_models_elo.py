#!/usr/bin/env python3
"""Four-model complete round-robin using 2 focus models + 6 engine bots.

Initial Elo is 1000 for every model.  A fixed standard Elo update is applied
after every game:
    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A += K * (S_A - E_A)
with K=16.  Only the final ranks of the two focus models count; the six bots
are not scored and never enter Elo.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.eval_round_robin_elo import (  # noqa: E402
    engine_fingerprint, pair_seats, run_pair, sha256_text,
)
from workflows.eval_two_models_six_bots import (  # noqa: E402
    audit_model, sha256_file, write_json,
)

SCHEMA = "hsai.eval_three_model_elo/v1"
N_PLAYERS = 8


def load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = ("engine", "models", "games_per_pair", "seed_base", "max_actions_per_turn",
                "move_cap", "move_repeat_limit", "bootstrap", "bootstrap_seed",
                "initial_elo", "k_factor")
    for key in required:
        if key not in cfg:
            raise ValueError(f"missing config key {key}")
    if len(cfg["models"]) != 4:
        raise ValueError("this runner expects exactly four models")
    ids = [str(x["id"]) for x in cfg["models"]]
    if len(set(ids)) != 4:
        raise ValueError("model ids must be unique")
    if int(cfg["max_actions_per_turn"]) != 80:
        raise ValueError("max-actions-per-turn must be 80")
    if int(cfg["move_cap"]) != 5 or int(cfg["move_repeat_limit"]) != 2:
        raise ValueError("move-cap/repeat rules must be 5/2")
    if int(cfg["games_per_pair"]) < 8 or int(cfg["games_per_pair"]) % 8:
        raise ValueError("games_per_pair must be divisible by 8")
    if int(cfg["bootstrap"]) < 100:
        raise ValueError("bootstrap too small")
    if float(cfg["initial_elo"]) != 1000.0:
        raise ValueError("initial Elo must be 1000")
    return cfg


def elo_update(ratings: Dict[str, float], a: str, b: str, score_a: float, k: float) -> None:
    ra, rb = ratings[a], ratings[b]
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    ratings[a] = ra + k * (score_a - ea)
    ratings[b] = rb + k * ((1.0 - score_a) - (1.0 - ea))


def run_elo(results: Sequence[Dict[str, Any]], ids: Sequence[str], initial: float,
            k: float, *, outcomes_override: Sequence[Sequence[float]] | None = None) -> Dict[str, float]:
    ratings = {x: float(initial) for x in ids}
    for pi, result in enumerate(results):
        a, b = str(result["i"]), str(result["j"])
        outcomes = (list(outcomes_override[pi]) if outcomes_override is not None
                    else list(result["outcomes"]))
        for score_a in outcomes:
            elo_update(ratings, a, b, float(score_a), k)
    return ratings


def bootstrap(results: Sequence[Dict[str, Any]], ids: Sequence[str], initial: float,
              k: float, n_boot: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    boots = {x: [] for x in ids}
    pair_scores = {(str(r["i"]), str(r["j"])): [] for r in results}
    for _ in range(int(n_boot)):
        override = []
        for result in results:
            out = np.asarray(result["outcomes"], dtype=np.float64)
            draw = out[rng.integers(0, out.size, size=out.size)]
            override.append(draw.tolist())
            pair_scores[(str(result["i"]), str(result["j"]))].append(
                float(np.sum(draw == 1.0) + 0.5 * np.sum(draw == 0.5)) / draw.size)
        ratings = run_elo(results, ids, initial, k, outcomes_override=override)
        for x in ids:
            boots[x].append(ratings[x])
    estimates = {}
    for x in ids:
        arr = np.asarray(boots[x], dtype=np.float64)
        estimates[x] = {
            "elo_ci95": [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))],
            "elo_bootstrap_mean": float(arr.mean()),
            "elo_bootstrap_std": float(arr.std(ddof=1)),
        }
    pair_ci = {}
    for (a, b), vals in pair_scores.items():
        arr = np.asarray(vals, dtype=np.float64)
        pair_ci[f"{a}::{b}"] = {
            "score_ci95": [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))],
            "score_mean": float(arr.mean()),
        }
    return {"estimates": estimates, "pair_score_ci": pair_ci,
            "method": "resample game outcomes within each pair; standard K=16 Elo in fixed pair order"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--games-per-pair", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.games_per_pair is not None:
        if args.games_per_pair < 8 or args.games_per_pair % 8:
            raise ValueError("smoke games_per_pair must be >=8 and divisible by 8")
        cfg["games_per_pair"] = int(args.games_per_pair)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    out.mkdir(parents=True)
    import torch
    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    fp = engine_fingerprint(cfg)
    ids = [str(x["id"]) for x in cfg["models"]]
    audits = {}
    for model_id, m in zip(ids, cfg["models"]):
        path = ROOT / str(m["path"])
        evaluator, audit = audit_model(path, model_id, expected_sha=str(m["file_sha256"]),
                                       device=str(device), model_id=model_id)
        audit["label"] = m.get("label", model_id)
        audit["staged_path"] = str(m["path"])
        audits[model_id] = {"evaluator": evaluator, "audit": audit}
    protocol = {
        "schema": SCHEMA, "config_path": str(args.config), "config_sha256": sha256_text(args.config),
        "runner_sha256": sha256_text(Path(__file__)), "engine": fp, "models": [
            {"id": x, "file_sha256": audits[x]["audit"]["file_sha256"],
             "params": audits[x]["audit"].get("params"), "arch": audits[x]["audit"].get("arch")}
            for x in ids],
        "model_order": ids, "pairs": [],
        "games_per_pair": int(cfg["games_per_pair"]),
        "seed_base": int(cfg["seed_base"]),
        "opponents": {"kind": "engine_bot", "count": 6, "action": "game.bot_play"},
        "decode": "greedy", "max_actions_per_turn": int(cfg["max_actions_per_turn"]),
        "move_guard": 1, "move_cap": int(cfg["move_cap"]),
        "move_repeat_limit": int(cfg["move_repeat_limit"]), "forbid_cancel": 1,
        "max_steps": int(cfg.get("max_steps", 50000)),
        "initial_elo": float(cfg["initial_elo"]), "k_factor": float(cfg["k_factor"]),
    }
    pairs = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    for p, (i, j) in enumerate(pairs):
        protocol["pairs"].append({"i": ids[i], "j": ids[j],
            "seed_start": int(cfg["seed_base"]) + p * int(cfg["games_per_pair"]),
            "seed_end": int(cfg["seed_base"]) + (p + 1) * int(cfg["games_per_pair"]) - 1})
    write_json(out / "protocol.json", protocol)
    write_json(out / "model-audit.json", {x: audits[x]["audit"] for x in ids})
    if args.audit_only:
        print(json.dumps({"audits": {x: audits[x]["audit"] for x in ids}, "protocol": protocol},
                         ensure_ascii=False, indent=2))
        return 0
    started = time.time()
    results = []
    with gzip.open(out / "games.jsonl.gz", "wt", encoding="utf-8") as f:
        for p, (i, j) in enumerate(pairs):
            print(f"[elo4] pair {p+1}/{len(pairs)} {ids[i]} vs {ids[j]}", flush=True)
            result = run_pair(cfg, fp, audits, p, i, j, int(cfg["games_per_pair"]), str(device))
            results.append(result)
            for row in result["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[elo4] pair done wins_i={result['wins_i']} wins_j={result['wins_j']} "
                  f"ties={result['ties']} valid={result['valid']}", flush=True)
    ratings = run_elo(results, ids, float(cfg["initial_elo"]), float(cfg["k_factor"]))
    boot = bootstrap(results, ids, float(cfg["initial_elo"]), float(cfg["k_factor"]),
                     int(cfg["bootstrap"]), int(cfg["bootstrap_seed"]))
    model_stats = {}
    for x in ids:
        wins = draws = losses = 0
        ranks = []
        for result in results:
            a, b = str(result["i"]), str(result["j"])
            for row in result["rows"]:
                if x == a:
                    s = float(row["outcome"]); ranks.append(int(row["ranks"][a]))
                elif x == b:
                    s = 1.0 - float(row["outcome"]); ranks.append(int(row["ranks"][b]))
                else:
                    continue
                if s == 1.0: wins += 1
                elif s == 0.0: losses += 1
                else: draws += 1
        games = wins + draws + losses
        model_stats[x] = {
            "elo": float(ratings[x]), **boot["estimates"][x],
            "games": games, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": (wins + 0.5 * draws) / games if games else None,
            "raw_win_rate": wins / games if games else None,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
        }
    pair_stats = []
    for result in results:
        a, b = str(result["i"]), str(result["j"])
        n = int(result["games"])
        score = (int(result["wins_i"]) + 0.5 * int(result["ties"])) / n
        pair_stats.append({
            "a": a, "b": b, "games": n, "wins_a": int(result["wins_i"]),
            "wins_b": int(result["wins_j"]), "draws": int(result["ties"]),
            "score_a": score, "score_ci95": boot["pair_score_ci"][f"{a}::{b}"]["score_ci95"],
            "valid": bool(result["valid"]),
            "seed_start": int(result["seed_start"]), "seed_end": int(result["seed_end"]),
        })
    order = sorted(ids, key=lambda x: ratings[x], reverse=True)
    leaderboard = [{"rank": i + 1, "id": x, **model_stats[x]} for i, x in enumerate(order)]
    # Pairwise improvement directions, especially V1.1 vs milestones.
    improvements = []
    for pair in pair_stats:
        a, b = pair["a"], pair["b"]
        improvements.append({**pair, "elo_delta_a_minus_b": ratings[a] - ratings[b]})
    report = {
        "schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "valid": all(p["valid"] for p in pair_stats),
        "engine": cfg["engine"], "protocol": protocol,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model_order": ids, "initial_elo": float(cfg["initial_elo"]),
        "k_factor": float(cfg["k_factor"]), "bootstrap": int(cfg["bootstrap"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
        "leaderboard": leaderboard, "models": model_stats, "pairs": pair_stats,
        "improvements": improvements, "bootstrap": boot,
        "audits": {x: audits[x]["audit"] for x in ids},
        "runtime": {"device": str(device), "torch": torch.__version__,
                    "python": platform.python_version(), "seconds": time.time() - started,
                    "games": len(results) * int(cfg["games_per_pair"]),
                    "pairs": len(results), "games_per_pair": int(cfg["games_per_pair"])},
    }
    write_json(out / "report.json", report)
    write_json(out / "pairs.json", pair_stats)
    write_json(out / "leaderboard.json", leaderboard)
    lines = ["# HSAI 四模型 pairwise Elo 评测", "",
             f"- 状态：**{'VALID' if report['valid'] else 'INVALID'}**",
             f"- 引擎：`{cfg['engine']['game_version']}` / `{cfg['engine']['source_hash']}`",
             f"- 初始 Elo：1000；K={cfg['k_factor']:g}；候选模型 4 个；每对 {cfg['games_per_pair']} 局",
             "- 每局：2 个焦点模型 + 6 个引擎 bot；只比较两个焦点模型的最终名次；bot 不计分。", "",
             "## 最新 Elo 排行榜", "",
             "| 排名 | 模型 | Elo | 95% CI | 总场次 | 胜/平/负 | 计分胜率 | 平均名次 |",
             "|---:|---|---:|---|---:|---:|---:|---:|"]
    for row in leaderboard:
        ci = row["elo_ci95"]
        lines.append(f"| {row['rank']} | {row['id']} | {row['elo']:.2f} | "
                     f"[{ci[0]:.2f}, {ci[1]:.2f}] | {row['games']} | "
                     f"{row['wins']}/{row['draws']}/{row['losses']} | "
                     f"{100*row['win_rate']:.2f}% | {row['mean_rank']:.3f} |")
    lines += ["", "## 两两对战", "",
              "| A | B | 局数 | A胜 | B胜 | 平 | A计分胜率 | 95% CI | A-B Elo |",
              "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for p in pair_stats:
        ci = p["score_ci95"]
        lines.append(f"| {p['a']} | {p['b']} | {p['games']} | {p['wins_a']} | {p['wins_b']} | "
                     f"{p['draws']} | {100*p['score_a']:.2f}% | "
                     f"[{100*ci[0]:.2f}%, {100*ci[1]:.2f}%] | "
                     f"{next(x['elo_delta_a_minus_b'] for x in improvements if x['a']==p['a'] and x['b']==p['b']):+.2f} |")
    lines += ["", "## 训练效果判断", ""]
    for p in improvements:
        a, b = p["a"], p["b"]
        delta = ratings[a] - ratings[b]
        sig = not (p["score_ci95"][0] <= 0.5 <= p["score_ci95"][1])
        lines.append(f"- **{a} vs {b}**：A 计分胜率 {100*p['score_a']:.2f}%，"
                     f"95% CI [{100*p['score_ci95'][0]:.2f}%, {100*p['score_ci95'][1]:.2f}%]，"
                     f"Elo 差 {delta:+.2f}；"
                     f"{'统计上可区分' if sig else '统计上不能排除平局/无差异'}。")
    lines += ["", "## 记录", "",
              f"- 机器可读报告：`report.json`", "- 完整逐局：`games.jsonl.gz`",
              "- 两两结果：`pairs.json`", "- 模型审计：`model-audit.json`",
              f"- 协议 SHA：`{report['protocol_sha256']}`", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"leaderboard": leaderboard, "pairs": pair_stats,
                      "protocol_sha256": report["protocol_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
