#!/usr/bin/env python3
"""最新 checkpoint vs N 个既有模型：2 焦点模型 + 6 引擎 bot 的配对评测。

每个 pair 使用 ``run_pair``（与 8 模型 Elo 相同的 8 座位轮转），只比较两个
焦点模型的最终名次；每个 pair 默认 128 局。用于回答“最新模型是否值得上实机”，
不把结果与之前不同 ``protocol_sha256`` 的 Elo 榜混在一起。

配置示例见 ``config/eval-latest-gauntlet.json``。输出：

- ``protocol.json`` / ``model-audit.json``
- ``games.jsonl.gz``（逐局）
- ``pairs.json``（每对胜负、计分胜率、bootstrap CI）
- ``report.json`` / ``report.md``（含 anchored Elo 参考与推荐）
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
    engine_fingerprint, run_pair, sha256_text,
)
from workflows.eval_two_models_six_bots import (  # noqa: E402
    audit_model, sha256_file, write_json,
)

SCHEMA = "hsai.eval_latest_gauntlet/v1"
N_PLAYERS = 8
ELO_SCALE = 400.0 / math.log(10.0)


def load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("engine", "latest", "models", "pairs", "games_per_pair", "seed_base",
                "max_actions_per_turn", "move_cap", "move_repeat_limit",
                "bootstrap", "bootstrap_seed")
    for key in required:
        if key not in cfg:
            raise ValueError(f"missing config key {key}")
    models = cfg["models"]
    ids = [str(m.get("id")) for m in models]
    if len(models) < 2 or len(set(ids)) != len(ids):
        raise ValueError("models must be unique and >=2")
    latest = str(cfg["latest"])
    if latest not in ids:
        raise ValueError(f"latest {latest!r} not in models")
    for m in models:
        for key in ("id", "path", "file_sha256", "bytes"):
            if key not in m:
                raise ValueError(f"model missing {key}: {m}")
    pairs = cfg["pairs"]
    if len(pairs) < 1:
        raise ValueError("pairs must not be empty")
    seen = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"pair must have 2 ids: {pair}")
        if pair[0] != latest:
            raise ValueError(f"latest must be the first id in each pair: {pair}")
        other = pair[1]
        if other == latest or other not in ids:
            raise ValueError(f"bad opponent in pair: {pair}")
        if other in seen:
            raise ValueError(f"duplicate opponent: {other}")
        seen.add(other)
    games = int(cfg["games_per_pair"])
    if games < 8 or games % 8:
        raise ValueError("games_per_pair must be >=8 and divisible by 8")
    if int(cfg["max_actions_per_turn"]) != 80:
        raise ValueError("max_actions_per_turn must be 80")
    if int(cfg["move_cap"]) != 5 or int(cfg["move_repeat_limit"]) != 2:
        raise ValueError("move_cap/repeat must be 5/2")
    if int(cfg["bootstrap"]) < 100:
        raise ValueError("bootstrap too small")
    return cfg


def bootstrap_score(outcomes: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    arr = np.asarray(outcomes, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        draw = arr[rng.integers(0, arr.size, size=arr.size)]
        vals[i] = (float(np.sum(draw == 1.0)) + 0.5 * float(np.sum(draw == 0.5))) / draw.size
    return {
        "score": (float(np.sum(arr == 1.0)) + 0.5 * float(np.sum(arr == 0.5))) / arr.size,
        "score_ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "score_bootstrap_mean": float(vals.mean()),
        "score_bootstrap_std": float(vals.std(ddof=1)),
    }


def implied_elo(prior: float | None, score: float | None) -> float | None:
    if prior is None or score is None or not (0.0 < score < 1.0):
        return None
    return float(prior) + ELO_SCALE * math.log10(score / (1.0 - score))


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
        games = int(args.games_per_pair)
        if games < 8 or games % 8:
            raise ValueError("smoke games_per_pair must be >=8 and divisible by 8")
        cfg["games_per_pair"] = games
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    out.mkdir(parents=True)

    import torch
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    fp = engine_fingerprint(cfg)
    ids = [str(m["id"]) for m in cfg["models"]]
    index = {model_id: i for i, model_id in enumerate(ids)}
    audits: Dict[str, Dict[str, Any]] = {}
    for model in cfg["models"]:
        model_id = str(model["id"])
        evaluator, audit = audit_model(
            ROOT / str(model["path"]), model_id,
            expected_sha=str(model["file_sha256"]), device=device, model_id=model_id)
        audit["label"] = model.get("label", model_id)
        audit["staged_path"] = str(model["path"])
        audit["prior_elo"] = model.get("prior_elo")
        audits[model_id] = {"evaluator": evaluator, "audit": audit}

    latest = str(cfg["latest"])
    latest_elo = next((m.get("prior_elo") for m in cfg["models"] if str(m["id"]) == latest), None)
    protocol = {
        "schema": SCHEMA,
        "config_path": str(args.config),
        "config_sha256": sha256_text(args.config),
        "runner_sha256": sha256_text(Path(__file__)),
        "engine": fp,
        "models": [
            {"id": model_id, "file_sha256": audits[model_id]["audit"]["file_sha256"],
             "params": audits[model_id]["audit"].get("params"),
             "arch": audits[model_id]["audit"].get("arch"),
             "prior_elo": audits[model_id]["audit"].get("prior_elo")}
            for model_id in ids
        ],
        "latest": latest,
        "pairs": [],
        "games_per_pair": int(cfg["games_per_pair"]),
        "seed_base": int(cfg["seed_base"]),
        "opponents": {"kind": "engine_bot", "count": 6, "action": "game.bot_play"},
        "seat_assignment": "pair_seats(gi, games_per_pair): all 8 seats cycled",
        "decode": str(cfg.get("decode", "greedy")),
        "max_actions_per_turn": int(cfg["max_actions_per_turn"]),
        "move_guard": 1,
        "move_cap": int(cfg["move_cap"]),
        "move_repeat_limit": int(cfg["move_repeat_limit"]),
        "forbid_cancel": 1,
        "max_steps": int(cfg.get("max_steps", 50000)),
        "bootstrap": int(cfg["bootstrap"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
        "device": device,
    }
    for p, pair in enumerate(cfg["pairs"]):
        i, j = str(pair[0]), str(pair[1])
        protocol["pairs"].append({
            "i": i, "j": j,
            "seed_start": int(cfg["seed_base"]) + p * int(cfg["games_per_pair"]),
            "seed_end": int(cfg["seed_base"]) + (p + 1) * int(cfg["games_per_pair"]) - 1,
        })
    write_json(out / "protocol.json", protocol)
    write_json(out / "model-audit.json", {x: audits[x]["audit"] for x in ids})
    if args.audit_only:
        print(json.dumps({"audits": {x: audits[x]["audit"] for x in ids},
                          "protocol": protocol}, ensure_ascii=False, indent=2))
        return 0

    started = time.time()
    results: List[Dict[str, Any]] = []
    with gzip.open(out / "games.jsonl.gz", "wt", encoding="utf-8") as f:
        for p, pair in enumerate(cfg["pairs"]):
            i, j = str(pair[0]), str(pair[1])
            print(f"[gauntlet] pair {p+1}/{len(cfg['pairs'])} {i} vs {j}", flush=True)
            result = run_pair(cfg, fp, audits, p, index[i], index[j],
                              int(cfg["games_per_pair"]), device)
            results.append(result)
            for row in result["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[gauntlet] pair done wins_i={result['wins_i']} "
                  f"wins_j={result['wins_j']} ties={result['ties']} valid={result['valid']}",
                  flush=True)

    pair_stats = []
    latest_outcomes: List[float] = []
    latest_ranks: List[int] = []
    for p, result in enumerate(results):
        a, b = str(result["i"]), str(result["j"])
        n = int(result["games"])
        score_a = (int(result["wins_i"]) + 0.5 * int(result["ties"])) / n
        boot = bootstrap_score(result["outcomes"], int(cfg["bootstrap"]),
                               int(cfg["bootstrap_seed"]) + p)
        if a == latest:
            latest_outcomes.extend(float(x) for x in result["outcomes"])
            other = b
            other_rows = [int(row["ranks"][b]) for row in result["rows"]]
        else:
            latest_outcomes.extend(1.0 - float(x) for x in result["outcomes"])
            other = a
            other_rows = [int(row["ranks"][a]) for row in result["rows"]]
        latest_rows = [int(row["ranks"][latest]) for row in result["rows"]]
        latest_ranks.extend(latest_rows)
        prior = next((m.get("prior_elo") for m in cfg["models"] if str(m["id"]) == other), None)
        pair_stats.append({
            "latest": latest,
            "opponent": other,
            "games": n,
            "latest_wins": int(result["wins_i"] if a == latest else result["wins_j"]),
            "opponent_wins": int(result["wins_j"] if a == latest else result["wins_i"]),
            "ties": int(result["ties"]),
            "latest_score": float(score_a if a == latest else 1.0 - score_a),
            "latest_score_ci95": boot["score_ci95"],
            "latest_mean_rank": float(np.mean(latest_rows)),
            "opponent_mean_rank": float(np.mean(other_rows)),
            "opponent_prior_elo": prior,
            "latest_implied_elo": implied_elo(
                prior, float(score_a if a == latest else 1.0 - score_a)),
            "valid": bool(result["valid"]),
            "seed_start": int(result["seed_start"]),
            "seed_end": int(result["seed_end"]),
        })

    aggregate = bootstrap_score(latest_outcomes, int(cfg["bootstrap"]),
                                int(cfg["bootstrap_seed"]) + 999)
    top = max(pair_stats, key=lambda x: (x["opponent_prior_elo"] is not None,
                                         x["opponent_prior_elo"] or -1e9))
    top_score = float(top["latest_score"])
    top_lo = float(top["latest_score_ci95"][0])
    beats_all = all(float(p["latest_score"]) > 0.5 for p in pair_stats)
    latest_significantly_better = beats_all and top_score > 0.5 and top_lo > 0.5
    best_model = latest if latest_significantly_better else top["opponent"]

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "valid": all(p["valid"] for p in pair_stats),
        "engine": cfg["engine"],
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "latest": latest,
        "latest_prior_elo": latest_elo,
        "pairs": pair_stats,
        "aggregate": {
            "games": len(latest_outcomes),
            "wins": int(sum(1 for x in latest_outcomes if x == 1.0)),
            "draws": int(sum(1 for x in latest_outcomes if x == 0.5)),
            "losses": int(sum(1 for x in latest_outcomes if x == 0.0)),
            "score": aggregate["score"],
            "score_ci95": aggregate["score_ci95"],
            "mean_rank": float(np.mean(latest_ranks)),
        },
        "recommendation": {
            "best_model": best_model,
            "latest_beats_all_incumbents": beats_all,
            "latest_beats_top_incumbent": latest_significantly_better,
            "top_incumbent": top["opponent"],
            "top_incumbent_score": top_score,
            "top_incumbent_score_ci95": top["latest_score_ci95"],
            "decision_rule": "latest is chosen only if it beats every incumbent (score>0.5) "
                             "and beats the strongest prior incumbent with 95% CI lower bound>0.5; "
                             "otherwise the strongest prior incumbent is kept",
            "note": "prior Elo values are anchors from the earlier 4-model 512-game run; "
                    "this gauntlet is not a new formal Elo leaderboard",
        },
        "runtime": {
            "device": device, "torch": torch.__version__,
            "python": platform.python_version(),
            "seconds": time.time() - started,
            "games": sum(int(r["games"]) for r in results),
            "pairs": len(results), "games_per_pair": int(cfg["games_per_pair"]),
        },
        "audits": {x: audits[x]["audit"] for x in ids},
    }
    write_json(out / "pairs.json", pair_stats)
    write_json(out / "report.json", report)

    lines = [
        "# 最新模型 vs 既有 Elo 模型：配对评测", "",
        f"- 状态：**{'VALID' if report['valid'] else 'INVALID'}**",
        f"- 最新模型：`{latest}`（prior Elo `{latest_elo}`）",
        f"- 引擎：`{cfg['engine']['game_version']}` / `{cfg['engine']['source_hash']}`",
        f"- 协议：每对 {cfg['games_per_pair']} 局，2 焦点模型 + 6 引擎 bot，8 座位轮转",
        f"- 协议 SHA：`{report['protocol_sha256']}`", "",
        "## 结论", "",
        f"- 推荐实机模型：**`{best_model}`**",
        f"- 最新模型对最强既有模型 `{top['opponent']}`：计分胜率 "
        f"{100*top_score:.2f}% (95% CI "
        f"[{100*top['latest_score_ci95'][0]:.2f}%, {100*top['latest_score_ci95'][1]:.2f}%])；"
        f"{'显著更好' if latest_significantly_better else '未达到显著更好'}。",
        f"- 最新模型 4 对合计：{report['aggregate']['wins']} 胜 / "
        f"{report['aggregate']['draws']} 平 / {report['aggregate']['losses']} 负，"
        f"计分胜率 {100*report['aggregate']['score']:.2f}% "
        f"(95% CI [{100*report['aggregate']['score_ci95'][0]:.2f}%, "
        f"{100*report['aggregate']['score_ci95'][1]:.2f}%])。", "",
        "## 逐对结果", "",
        "| 对手 | 局数 | 最新胜 | 对手胜 | 平 | 最新计分胜率 | 95% CI | 对手 prior Elo | 最新 anchored Elo |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for p in pair_stats:
        prior_text = (f"{p['opponent_prior_elo']:.2f}"
                      if p["opponent_prior_elo"] is not None else "—")
        implied_text = (f"{p['latest_implied_elo']:.2f}"
                        if p["latest_implied_elo"] is not None else "—")
        lines.append(
            f"| {p['opponent']} | {p['games']} | {p['latest_wins']} | {p['opponent_wins']} | "
            f"{p['ties']} | {100*p['latest_score']:.2f}% | "
            f"[{100*p['latest_score_ci95'][0]:.2f}%, {100*p['latest_score_ci95'][1]:.2f}%] | "
            f"{prior_text} | {implied_text} |")
    lines += ["", "## 记录", "",
              "- `report.json` / `pairs.json` / `games.jsonl.gz` / `protocol.json`",
              "- anchored Elo 只作参考：prior Elo 来自上一轮 4 模型 512 局/对，不是同一协议的新榜。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(out / "report.json"),
                      "recommendation": report["recommendation"],
                      "aggregate": report["aggregate"],
                      "protocol_sha256": report["protocol_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
