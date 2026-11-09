#!/usr/bin/env python3
"""8-model pairwise Elo evaluation with two focus models plus six engine bots.

Each of the 28 unordered model pairs is played for ``games_per_pair`` games.  The
two focus models occupy two seats; the remaining six seats use the native engine
bot.  Elo is fitted from the relative placement of the two focus models only.

Usage::

    python workflows/eval_round_robin_elo.py --config config/eval-elo-8.json --out artifacts/runs/elo-8-<job>
    python workflows/eval_round_robin_elo.py --config config/eval-elo-8.json --audit-only --out artifacts/models/elo-8
    python workflows/eval_round_robin_elo.py --config config/eval-elo-8.json --out /tmp/elo-smoke --pair-limit 1 --games-per-pair 2 --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.eval.duel_metrics import (  # noqa: E402
    final_state_summary,
    rank_summary,
    reach_tier_stats,
    turn_table,
)
from workflows.eval_two_models_six_bots import (  # noqa: E402
    CardNames,
    DuelSeatPolicy,
    audit_model,
    run_games,
    sha256_file,
    write_json,
)

SCHEMA = "hsai.eval_round_robin_elo/v1"
CONFIG_DEFAULT = ROOT / "config/eval-elo-8.json"
END_TURN = 0
N_PLAYERS = 8
ANCHOR_ID = "m1080772-v7357"
ELO_SCALE = 400.0 / math.log(10.0)


def sha256_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def protocol_sha256(protocol: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("engine", "models", "games_per_pair", "seed_base", "max_actions_per_turn",
                "move_cap", "move_repeat_limit", "bootstrap", "elo_anchor")
    for key in required:
        if key not in cfg:
            raise ValueError(f"配置缺少必需字段 {key!r}: {path}")
    models = cfg["models"]
    if len(models) != 8:
        raise ValueError(f"必须正好 8 个模型，实际 {len(models)}")
    ids = [str(m["id"]) for m in models]
    if len(set(ids)) != len(ids):
        raise ValueError("模型 id 重复")
    if str(cfg["elo_anchor"]) not in ids:
        raise ValueError(f"elo_anchor {cfg['elo_anchor']!r} 不在模型列表")
    for m in models:
        for key in ("id", "path", "file_sha256", "bytes"):
            if key not in m:
                raise ValueError(f"模型缺少 {key}: {m}")
    if int(cfg["games_per_pair"]) <= 0:
        raise ValueError("games_per_pair 必须 > 0")
    if int(cfg["max_actions_per_turn"]) != 80:
        raise ValueError("max_actions_per_turn 必须固定为 80")
    if int(cfg["move_cap"]) != 5 or int(cfg["move_repeat_limit"]) != 2:
        raise ValueError("move_cap 必须为 5，move_repeat_limit 必须为 2")
    if int(cfg["bootstrap"]) <= 0:
        raise ValueError("bootstrap 必须 > 0")
    return cfg


def model_index(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(m["id"]): m for m in cfg["models"]}


def all_pairs(cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    n = len(cfg["models"])
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pair_seats(gi: int, games_per_pair: int) -> Dict[str, int]:
    """Seat schedule for one pair: each model plays every seat once per 8 games."""
    block = max(1, int(games_per_pair) // 2)
    k = int(gi) % block
    a = k % N_PLAYERS
    b = (k + 1) % N_PLAYERS
    return {"A": b, "B": a} if int(gi) < block else {"A": a, "B": b}


def seat_counts(games_per_pair: int) -> Dict[str, Dict[int, int]]:
    counts = {"A": {s: 0 for s in range(N_PLAYERS)},
              "B": {s: 0 for s in range(N_PLAYERS)}}
    for gi in range(int(games_per_pair)):
        seats = pair_seats(gi, games_per_pair)
        counts["A"][seats["A"]] += 1
        counts["B"][seats["B"]] += 1
    return counts


def engine_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import hsbg_engine as E
    from hsbg.az.train.personal import fingerprint

    if not E.build_is_current(ROOT / "engine"):
        raise ValueError("原生扩展不是从当前 Rust 源码构建，先运行 `python -m pip install -e ./engine`")
    fp = fingerprint()
    expected = cfg["engine"]
    for key in ("game_version", "source_hash", "engine_cards_sha256", "vocab_sha256"):
        actual = fp["engine"].get(key) if key in fp["engine"] else fp.get(key)
        if str(actual) != str(expected.get(key)):
            raise ValueError(f"引擎指纹不符 {key}: expected={expected.get(key)} actual={actual}")
    if int(fp["actions"]) != int(expected["n_actions"]):
        raise ValueError("动作数不符")
    if int(fp["obs_floats"]) != int(expected["obs_floats"]):
        raise ValueError("观测浮点数不符")
    if int(fp["obs_cards"]) != int(expected["obs_cards"]):
        raise ValueError("观测卡数不符")
    return fp


def load_audits(cfg: Dict[str, Any], ids: Sequence[str], device: str) -> Dict[str, Dict[str, Any]]:
    audits: Dict[str, Dict[str, Any]] = {}
    for model_id in ids:
        model_cfg = model_index(cfg)[model_id]
        path = ROOT / str(model_cfg["path"])
        evaluator, audit = audit_model(
            path,
            model_id,
            expected_sha=str(model_cfg["file_sha256"]),
            device=device,
            model_id=model_id,
        )
        audit["label"] = model_cfg.get("label", "")
        audit["staged_path"] = str(model_cfg["path"])
        audits[model_id] = {"evaluator": evaluator, "audit": audit}
    return audits


def build_protocol(
    cfg: Dict[str, Any],
    cfg_path: Path,
    fp: Dict[str, Any],
    audits: Dict[str, Dict[str, Any]],
    games_per_pair: int,
    pair_limit: Optional[int],
    runner_sha: str,
    device: str,
) -> Dict[str, Any]:
    pairs = all_pairs(cfg)[:pair_limit] if pair_limit else all_pairs(cfg)
    return {
        "schema": SCHEMA,
        "config_path": str(cfg_path.relative_to(ROOT)) if str(cfg_path).startswith(str(ROOT))
        else str(cfg_path),
        "config_sha256": sha256_text(cfg_path),
        "runner_sha256": runner_sha,
        "engine": fp,
        "engine_unmodified": True,
        "models": [
            {
                "id": model_id,
                "file_sha256": entry["audit"]["file_sha256"],
                "weights_sha256": entry["audit"].get("weights_sha256"),
                "params": entry["audit"].get("params"),
                "arch": entry["audit"].get("arch"),
            }
            for model_id, entry in audits.items()
        ],
        "model_order": [str(m["id"]) for m in cfg["models"]],
        "pairs": [
            {
                "i": str(cfg["models"][i]["id"]),
                "j": str(cfg["models"][j]["id"]),
                "seed_start": int(cfg["seed_base"]) + p * int(games_per_pair),
                "seed_end": int(cfg["seed_base"]) + (p + 1) * int(games_per_pair) - 1,
            }
            for p, (i, j) in enumerate(pairs)
        ],
        "games_per_pair": int(games_per_pair),
        "pair_count": len(pairs),
        "opponents": {"kind": "engine_bot", "count": 6, "action": "game.bot_play"},
        "seat_assignment": "pair_seats(gi, games_per_pair): first half A=k%8/B=(k+1)%8, second half swapped",
        "decode": str(cfg.get("decode", "greedy")),
        "max_actions_per_turn": int(cfg["max_actions_per_turn"]),
        "move_guard": 1,
        "forbid_cancel": 1,
        "move_cap": int(cfg["move_cap"]),
        "move_repeat_limit": int(cfg["move_repeat_limit"]),
        "max_steps": int(cfg.get("max_steps", 50000)),
        "forced_tier_plan": "",
        "mirror_pool": None,
        "human_board_pool": None,
        "bootstrap": int(cfg["bootstrap"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
        "elo_anchor": str(cfg["elo_anchor"]),
        "elo_anchor_score": float(cfg["elo_anchor_score"]),
        "elo_scale": ELO_SCALE,
        "elo_l2": float(cfg.get("elo_l2", 0.01)),
        "primary_statistic": "P50(sum_attack+sum_health) per end-of-turn board",
        "device": str(device),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _fit_bt(wins: np.ndarray, games: np.ndarray, anchor: int,
            l2: float = 1e-8) -> np.ndarray:
    from scipy.optimize import root

    n = int(wins.shape[0])
    free = np.array([i for i in range(n) if i != int(anchor)], dtype=np.int64)

    def unpack(z: np.ndarray) -> np.ndarray:
        s = np.zeros(n, dtype=np.float64)
        s[free] = z
        return s

    def gradient_hessian(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = unpack(z)
        grad = np.zeros(n, dtype=np.float64)
        hess = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                nij = float(games[i, j])
                if nij <= 0.0:
                    continue
                d = float(s[i] - s[j])
                p = 1.0 / (1.0 + math.exp(-d)) if d >= 0 else math.exp(d) / (1.0 + math.exp(d))
                p = min(max(p, 1e-12), 1.0 - 1e-12)
                weight = nij * p * (1.0 - p)
                g = nij * p - float(wins[i, j])
                grad[i] += g
                grad[j] -= g
                hess[i, i] += weight
                hess[j, j] += weight
                hess[i, j] -= weight
                hess[j, i] -= weight
        grad_free = grad[free] + 2.0 * float(l2) * z
        hess_free = hess[np.ix_(free, free)] + 2.0 * float(l2) * np.eye(len(free))
        return grad_free, hess_free

    start = np.zeros(len(free), dtype=np.float64)
    res = root(gradient_hessian, start, jac=True, method="hybr",
               options={"xtol": 1e-12, "maxfev": 2000})
    return unpack(res.x if res.success else start)


def fit_elo(
    cfg: Dict[str, Any],
    pairs: Sequence[Dict[str, Any]],
    ids: Sequence[str],
) -> Dict[str, Any]:
    n = len(ids)
    index = {model_id: i for i, model_id in enumerate(ids)}
    anchor = index.get(str(cfg["elo_anchor"]), 0)
    wins = np.zeros((n, n), dtype=np.float64)
    games = np.zeros((n, n), dtype=np.float64)
    for pair in pairs:
        i, j = index[pair["i"]], index[pair["j"]]
        games[i, j] = games[j, i] = int(pair["games"])
        wins[i, j] = float(pair["wins_i"]) + 0.5 * float(pair["ties"])
        wins[j, i] = float(pair["wins_j"]) + 0.5 * float(pair["ties"])
    l2 = float(cfg.get("elo_l2", 0.01))
    strength = _fit_bt(wins, games, anchor, l2=l2)
    score = float(cfg["elo_anchor_score"]) + ELO_SCALE * strength

    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    samples = np.zeros((n, int(cfg["bootstrap"])), dtype=np.float64)
    for b in range(int(cfg["bootstrap"])):
        boot_wins = np.zeros((n, n), dtype=np.float64)
        boot_games = np.zeros((n, n), dtype=np.float64)
        for pair in pairs:
            i, j = index[pair["i"]], index[pair["j"]]
            outcomes = np.asarray(pair["outcomes"], dtype=np.float64)
            draw = outcomes[rng.integers(0, outcomes.size, size=outcomes.size)]
            games_ij = int(draw.size)
            w = float(np.sum(draw == 1.0) + 0.5 * np.sum(draw == 0.5))
            boot_games[i, j] = boot_games[j, i] = games_ij
            boot_wins[i, j] = w
            boot_wins[j, i] = games_ij - w
        boot_strength = _fit_bt(boot_wins, boot_games, anchor, l2=l2)
        samples[:, b] = float(cfg["elo_anchor_score"]) + ELO_SCALE * boot_strength

    estimates = []
    for i, model_id in enumerate(ids):
        lo, hi = np.quantile(samples[i], [0.025, 0.975])
        estimates.append({
            "id": model_id,
            "elo": float(score[i]),
            "elo_ci95": [float(lo), float(hi)],
            "elo_bootstrap_mean": float(samples[i].mean()),
            "elo_bootstrap_std": float(samples[i].std(ddof=1)) if samples.shape[1] > 1 else None,
            "ci_overlaps_anchor": bool(lo <= float(cfg["elo_anchor_score"]) <= hi),
        })
    estimates.sort(key=lambda x: x["elo"], reverse=True)
    empirical = {}
    elo_prob = {}
    for i, a in enumerate(ids):
        empirical[a] = {}
        elo_prob[a] = {}
        for j, b in enumerate(ids):
            if i == j:
                empirical[a][b] = None
                elo_prob[a][b] = None
            else:
                g = games[i, j]
                empirical[a][b] = (float(wins[i, j]) / g) if g > 0 else None
                elo_prob[a][b] = float(_sigmoid(np.array([score[i] - score[j]]))[0])
    return {
        "method": "Bradley-Terry maximum likelihood on pairwise placements; 2000 game-level bootstrap",
        "l2": float(cfg.get("elo_l2", 0.01)),
        "anchor": str(cfg["elo_anchor"]),
        "anchor_score": float(cfg["elo_anchor_score"]),
        "scale": ELO_SCALE,
        "estimates": estimates,
        "empirical_win_prob": empirical,
        "elo_win_prob": elo_prob,
        "pair_games": {
            a: {b: (int(games[index[a], index[b]]) if a != b else None) for b in ids}
            for a in ids
        },
    }


def run_pair(
    cfg: Dict[str, Any],
    fp: Dict[str, Any],
    audits: Dict[str, Dict[str, Any]],
    pair_index: int,
    i: int,
    j: int,
    games_per_pair: int,
    device: str,
) -> Dict[str, Any]:
    import hsbg_engine as E

    id_i, id_j = str(cfg["models"][i]["id"]), str(cfg["models"][j]["id"])
    seed_start = int(cfg["seed_base"]) + int(pair_index) * int(games_per_pair)
    policies = {
        "A": DuelSeatPolicy(id_i, audits[id_i]["evaluator"],
                            max_actions_per_turn=int(cfg["max_actions_per_turn"]),
                            move_cap=int(cfg["move_cap"]),
                            move_repeat_limit=int(cfg["move_repeat_limit"])),
        "B": DuelSeatPolicy(id_j, audits[id_j]["evaluator"],
                            max_actions_per_turn=int(cfg["max_actions_per_turn"]),
                            move_cap=int(cfg["move_cap"]),
                            move_repeat_limit=int(cfg["move_repeat_limit"])),
    }
    run = run_games(
        engine=E,
        cards=CardNames(E),
        policies=policies,
        games=int(games_per_pair),
        seed0=seed_start,
        max_steps=int(cfg.get("max_steps", 50000)),
        collect_opponents=bool(cfg.get("collect_opponents", False)),
        seats_for=lambda gi: pair_seats(gi, int(games_per_pair)),
        progress_every=8 if games_per_pair >= 8 else 1,
    )
    rows = []
    outcomes = []
    for local, row in enumerate(run["games"]):
        row = dict(row)
        gid = int(pair_index) * int(games_per_pair) + local
        row["game"] = gid
        row["pair_index"] = int(pair_index)
        row["pair"] = [id_i, id_j]
        row["ranks"] = {id_i: int(row["rank"]["A"]), id_j: int(row["rank"]["B"])}
        row["seats_by_model"] = {id_i: int(row["seats"]["A"]), id_j: int(row["seats"]["B"])}
        row["decisions_by_model"] = {id_i: int(row["decisions"]["A"]),
                                     id_j: int(row["decisions"]["B"])}
        row["guard_by_model"] = {id_i: dict(row["guard"]["A"]),
                                 id_j: dict(row["guard"]["B"])}
        row["final_by_model"] = {id_i: row["final"]["A"], id_j: row["final"]["B"]}
        row["final_board_by_model"] = {id_i: row["final_board"]["A"],
                                       id_j: row["final_board"]["B"]}
        if row["ranks"][id_i] < row["ranks"][id_j]:
            outcome = 1.0
        elif row["ranks"][id_i] > row["ranks"][id_j]:
            outcome = 0.0
        else:
            outcome = 0.5
        row["outcome"] = outcome
        outcomes.append(outcome)
        rows.append(row)
    wins_i = int(sum(1 for x in outcomes if x == 1.0))
    wins_j = int(sum(1 for x in outcomes if x == 0.0))
    ties = int(sum(1 for x in outcomes if x == 0.5))
    counts = seat_counts(int(games_per_pair))
    pair_valid = (
        int(run["illegal"]) == 0
        and int(run["truncated_games"]) == 0
        and int(run["engine_bad"]) == 0
        and len(rows) == int(games_per_pair)
        and all(bool(row["over"]) for row in rows)
        and all(1 <= int(row["ranks"][x]) <= N_PLAYERS for row in rows for x in (id_i, id_j))
        and all(int(row["decisions_by_model"][x]) > 0 for row in rows for x in (id_i, id_j))
    )
    if int(games_per_pair) % N_PLAYERS == 0:
        expected_seat = int(games_per_pair) // N_PLAYERS
        pair_valid = pair_valid and all(
            sum(1 for row in rows if row["seats_by_model"][x] == seat) == expected_seat
            for x in (id_i, id_j) for seat in range(N_PLAYERS)
        )
    return {
        "pair_index": int(pair_index),
        "i": id_i,
        "j": id_j,
        "seed_start": seed_start,
        "seed_end": seed_start + int(games_per_pair) - 1,
        "games": int(games_per_pair),
        "wins_i": wins_i,
        "wins_j": wins_j,
        "ties": ties,
        "outcomes": outcomes,
        "rank_delta_mean": float(np.mean([row["ranks"][id_i] - row["ranks"][id_j]
                                          for row in rows])),
        "valid": bool(pair_valid),
        "rows": rows,
        "totals": {
            "illegal": int(run["illegal"]),
            "truncated": int(run["truncated_games"]),
            "engine_bad": int(run["engine_bad"]),
            "decisions": int(sum(sum(row["decisions"].values()) for row in rows)),
        },
        "frames": {
            id_i: [dict(f, game=int(pair_index) * int(games_per_pair) + int(f["game"]))
                   for f in run["frames"]["A"]],
            id_j: [dict(f, game=int(pair_index) * int(games_per_pair) + int(f["game"]))
                   for f in run["frames"]["B"]],
        },
        "seat_counts": counts,
        "seconds": float(run["seconds"]),
        "run": run,
    }


def _model_game_view(row: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    final = row["final_by_model"].get(model_id) or {}
    return {
        "game": int(row["game"]),
        "seed": int(row["seed"]),
        "pair": list(row["pair"]),
        "seat": int(row["seats_by_model"][model_id]),
        "rank": int(row["ranks"][model_id]),
        "final": dict(final),
        "board": list(row.get("final_board_by_model", {}).get(model_id) or []),
    }


def _select_games(games: Sequence[Dict[str, Any]], *, limit: int,
                  order: str = "rank", only_win: bool = False) -> List[Dict[str, Any]]:
    rows = [g for g in games if not only_win or int(g["rank"]) == 1]
    if order == "rank":
        rows.sort(key=lambda g: (int(g["rank"]), -int((g["final"] or {}).get("body") or 0),
                                 int(g["seed"])))
    elif order == "body":
        rows.sort(key=lambda g: (-int((g["final"] or {}).get("body") or 0),
                                 int(g["rank"]), int(g["seed"])))
    else:
        raise ValueError(f"unknown order {order}")
    return rows[:int(limit)]


def aggregate(cfg: Dict[str, Any], results: Sequence[Dict[str, Any]],
              audits: Dict[str, Dict[str, Any]], per_pair: int) -> Dict[str, Any]:
    ids = [str(m["id"]) for m in cfg["models"]]
    ranks: Dict[str, List[int]] = {x: [] for x in ids}
    frames: Dict[str, List[Dict[str, Any]]] = {x: [] for x in ids}
    decisions: Dict[str, int] = {x: 0 for x in ids}
    guard: Dict[str, Dict[str, int]] = {x: {} for x in ids}
    final_rows: Dict[str, List[Dict[str, Any]]] = {x: [] for x in ids}
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.extend(result["rows"])
        for model_id in (result["i"], result["j"]):
            frames[model_id].extend(result["frames"][model_id])
            for row in result["rows"]:
                ranks[model_id].append(int(row["ranks"][model_id]))
                decisions[model_id] += int(row["decisions_by_model"][model_id])
                g = row["guard_by_model"][model_id]
                for key, value in g.items():
                    guard[model_id][key] = guard[model_id].get(key, 0) + int(value)
                final = row["final_by_model"][model_id]
                if final:
                    final_rows[model_id].append({
                        **final,
                        "rank": int(row["ranks"][model_id]),
                        "seed": int(row["seed"]),
                    })
    results_out = {}
    for model_id in ids:
        turns = sorted({int(f["turn"]) for f in frames[model_id]})
        model_games = [_model_game_view(row, model_id) for row in rows
                       if model_id in row["ranks"]]
        results_out[model_id] = {
            "model": model_id,
            "file_sha256": audits[model_id]["audit"]["file_sha256"],
            "rank": rank_summary(ranks[model_id], bootstrap=int(cfg["bootstrap"]),
                                 seed=int(cfg["bootstrap_seed"])),
            "turns": turn_table(frames[model_id], turns=turns),
            "reach_tier": reach_tier_stats(frames[model_id]),
            "final_state": final_state_summary(final_rows[model_id]),
            "decisions": int(decisions[model_id]),
            "guard": guard[model_id],
            "champion_games": _select_games(model_games, limit=3, order="body", only_win=True),
            "best_games": _select_games(model_games, limit=5, order="rank"),
            "top_power_games": _select_games(model_games, limit=3, order="body"),
            "per_seat": {
                str(seat): rank_summary(
                    [int(row["ranks"][model_id]) for row in rows
                     if row["seats_by_model"].get(model_id) == seat],
                    bootstrap=int(cfg["bootstrap"]), seed=int(cfg["bootstrap_seed"]) + seat)
                for seat in range(N_PLAYERS)
            },
        }
    pair_records = [{k: v for k, v in result.items()
                     if k not in ("rows", "frames", "run")} for result in results]
    elo = fit_elo(cfg, pair_records, ids)
    expected_games = len(results) * int(per_pair)
    expected_model = (len(ids) - 1) * int(per_pair)
    formal = (len(results) == len(all_pairs(cfg)) and int(per_pair) == int(cfg["games_per_pair"]))
    reasons: List[str] = []
    if len(rows) != expected_games:
        reasons.append(f"完成局数 {len(rows)} != {expected_games}")
    if any(int(result["totals"]["illegal"]) for result in results):
        reasons.append("存在非法动作")
    if any(int(result["totals"]["truncated"]) for result in results):
        reasons.append("存在截断局")
    if any(int(result["totals"]["engine_bad"]) for result in results):
        reasons.append("存在引擎异常")
    if any(not result["valid"] for result in results):
        reasons.append("存在无效模型对")
    for model_id in ids:
        if len(ranks[model_id]) != expected_model:
            reasons.append(f"{model_id} 名次局数 {len(ranks[model_id])} != {expected_model}")
        if int(decisions[model_id]) <= 0:
            reasons.append(f"{model_id} 没有任何决策")
        expected_seat_games = (len(ids) - 1) * (int(per_pair) // N_PLAYERS)
        if formal and any(int(results_out[model_id]["per_seat"][str(seat)]["games"]) !=
                          expected_seat_games for seat in range(N_PLAYERS)):
            reasons.append(f"{model_id} 座位轮转不完整")
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "valid": not reasons,
        "smoke": not formal,
        "leaderboard_eligible": bool(formal and not reasons),
        "reasons": reasons,
        "engine": cfg["engine"],
        "model_order": ids,
        "models": {x: audits[x]["audit"] for x in ids},
        "results": results_out,
        "elo": elo,
        "pairs": pair_records,
        "games": rows,
        "totals": {
            "games": len(rows),
            "pairs": len(results),
            "games_per_pair": int(per_pair),
            "ranked": int(sum(1 for row in rows if all(1 <= int(r) <= N_PLAYERS
                                                        for r in row["ranks"].values()))),
            "decisions": int(sum(int(x) for x in decisions.values())),
            "illegal": int(sum(int(x["totals"]["illegal"]) for x in results)),
            "truncated": int(sum(int(x["totals"]["truncated"]) for x in results)),
            "engine_bad": int(sum(int(x["totals"]["engine_bad"]) for x in results)),
            "seconds": float(sum(float(x["seconds"]) for x in results)),
        },
        "gate": {
            "formal": formal,
            "expected_games": expected_games,
            "expected_games_per_model": expected_model,
            "seat_rotation_ok": not any("座位轮转" in reason for reason in reasons),
        },
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 8 模型 pairwise Elo 评测")
    lines.append("")
    lines.append(f"- 状态：**{'VALID' if report['valid'] else 'INVALID'}**")
    lines.append(f"- 可入榜：**{'yes' if report['leaderboard_eligible'] else 'no'}**")
    lines.append(f"- 引擎：`{report['engine']['game_version']}` / `{report['engine']['source_hash']}`")
    lines.append(f"- 局数：{report['totals']['games']}；模型对：{report['totals']['pairs']}；"
                 f"每对：{report['totals']['games_per_pair']}")
    lines.append("")
    if report["reasons"]:
        lines.append("## 门禁失败")
        lines.append("")
        for reason in report["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    lines.append("## Elo")
    lines.append("")
    lines.append("| 排名 | 模型 | Elo | 95% CI | 平均名次 | 前四率 | 冠军率 | 与锚点 CI 重叠 |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---|")
    results = report["results"]
    for rank, row in enumerate(report["elo"]["estimates"], 1):
        model = results[row["id"]]
        ci = row["elo_ci95"]
        lines.append(
            f"| {rank} | {row['id']} | {row['elo']:.1f} | "
            f"[{ci[0]:.1f}, {ci[1]:.1f}] | "
            f"{_fmt(model['rank']['mean_rank'], 2)} | "
            f"{_fmt(None if model['rank']['top4_rate'] is None else 100 * model['rank']['top4_rate'], 1)}% | "
            f"{_fmt(None if model['rank']['win_rate'] is None else 100 * model['rank']['win_rate'], 1)}% | "
            f"{'yes' if row['ci_overlaps_anchor'] else 'no'} |"
        )
    lines.append("")
    lines.append("## 强模型吃鸡最终阵容")
    lines.append("")
    for estimate in report["elo"]["estimates"][:3]:
        model_id = estimate["id"]
        champion = report["results"][model_id].get("champion_games") or []
        lines.append(f"### {model_id}（Elo {estimate['elo']:.1f}）")
        lines.append("")
        if not champion:
            lines.append("无第一名对局。")
            lines.append("")
            continue
        for game in champion:
            final = game.get("final") or {}
            lines.append(
                f"- seed `{game['seed']}`，座位 {game['seat']}，名次 **1**，"
                f"最终等级 T{final.get('tier', '—')}，血量 {final.get('health', '—')}，"
                f"战团 P50 总值 {final.get('body', '—')}"
            )
            lines.append("")
            lines.append("| 位置 | 随从 | 卡 ID | 星级 | 攻/血 | 金色 | 关键词 |")
            lines.append("|---:|---|---|---:|---:|---|---|")
            for piece in game.get("board") or []:
                lines.append(
                    f"| {piece['position'] + 1} | {piece['name']} | {piece['card_id']} | "
                    f"{piece['card_tier']} | {piece['atk']}/{piece['hp']} | "
                    f"{'yes' if piece['golden'] else 'no'} | "
                    f"{', '.join(piece['keywords']) or '—'} |"
                )
            lines.append("")
    lines.append("## 两两经验胜率 P(i 名次高于 j)")
    lines.append("")
    ids = report["model_order"]
    lines.append("| i \\ j | " + " | ".join(ids) + " |")
    lines.append("|---|" + "---:|" * len(ids))
    for a in ids:
        cells = []
        for b in ids:
            p = report["elo"]["empirical_win_prob"][a][b]
            cells.append("—" if p is None else f"{100 * p:.1f}%")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 逐回合战团 P50(Σatk+Σhp)")
    lines.append("")
    turns = sorted({int(t) for model in results.values() for t in model["turns"]})
    lines.append("| 模型 | " + " | ".join(f"T{t}" for t in turns) + " |")
    lines.append("|---|" + "---:|" * len(turns))
    for model_id in ids:
        row = results[model_id]["turns"]
        cells = []
        for t in turns:
            value = (row.get(str(t)) or {}).get("body_p50")
            cells.append("—" if value is None else f"{value:.0f}")
        lines.append(f"| {model_id} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python workflows/eval_round_robin_elo.py --config config/eval-elo-8.json --out artifacts/runs/elo-8-<job>")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--games-per-pair", type=int, default=None)
    parser.add_argument("--pair-limit", type=int, default=None)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 但当前不可用")
    fp = engine_fingerprint(cfg)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"输出目录已存在，保留现场并使用新目录: {out}")
    out.mkdir(parents=True)
    started = time.time()
    ids = [str(m["id"]) for m in cfg["models"]]
    if args.audit_only:
        audits = load_audits(cfg, ids, device)
        payload = {
            "schema": "hsai.elo-model-audit/v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "engine": fp,
            "models": {model_id: audits[model_id]["audit"] for model_id in ids},
            "all_forward_ok": all(audits[x]["audit"].get("forward_ok") for x in ids),
            "all_card_tables_ok": all(audits[x]["audit"].get("card_table_ok") for x in ids),
        }
        write_json(out / "AUDIT.json", payload)
        protocol = build_protocol(cfg, cfg_path, fp, audits, int(cfg["games_per_pair"]),
                                  None, sha256_text(Path(__file__)), device)
        plan = {
            "schema": "hsai.elo-plan/v1",
            "protocol": protocol,
            "protocol_sha256": protocol_sha256(protocol),
            "expected_games": len(all_pairs(cfg)) * int(cfg["games_per_pair"]),
            "expected_games_per_model": (len(cfg["models"]) - 1) * int(cfg["games_per_pair"]),
            "pair_count": len(all_pairs(cfg)),
            "games_per_pair": int(cfg["games_per_pair"]),
        }
        write_json(out / "PLAN.json", plan)
        print(json.dumps({"audit": payload, "plan": plan}, ensure_ascii=False, indent=2))
        return 0

    games_per_pair = int(args.games_per_pair or cfg["games_per_pair"])
    pair_limit = int(args.pair_limit) if args.pair_limit is not None else len(all_pairs(cfg))
    if pair_limit <= 0 or pair_limit > len(all_pairs(cfg)):
        raise ValueError("pair_limit 越界")
    selected_pairs = all_pairs(cfg)[:pair_limit]
    audits = load_audits(cfg, ids, device)
    runner_sha = sha256_text(Path(__file__))
    protocol = build_protocol(cfg, cfg_path, fp, audits, games_per_pair, pair_limit,
                              runner_sha, device)
    results: List[Dict[str, Any]] = []
    for pair_index, (i, j) in enumerate(selected_pairs):
        print(f"[elo] pair {pair_index + 1}/{len(selected_pairs)} "
              f"{cfg['models'][i]['id']} vs {cfg['models'][j]['id']}", flush=True)
        result = run_pair(cfg, fp, audits, pair_index, i, j, games_per_pair, str(device))
        results.append(result)
        print(f"[elo] pair done: wins_i={result['wins_i']} wins_j={result['wins_j']} "
              f"ties={result['ties']} valid={result['valid']} seconds={result['seconds']:.1f}",
              flush=True)
    report = aggregate(cfg, results, audits, games_per_pair)
    report["protocol"] = protocol
    report["protocol_sha256"] = protocol_sha256(protocol)
    report["config"] = cfg
    report["runtime"] = {
        "os": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "seconds": time.time() - started,
    }
    write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    write_json(out / "model-audit.json", {x: audits[x]["audit"] for x in ids})
    print(f"[elo] report -> {out / 'report.json'} valid={report['valid']} "
          f"eligible={report['leaderboard_eligible']} smoke={report.get('smoke', False)}",
          flush=True)
    engineering_ok = (
        report["totals"]["illegal"] == 0
        and report["totals"]["truncated"] == 0
        and report["totals"]["engine_bad"] == 0
        and all(pair["valid"] for pair in report["pairs"])
    )
    if report.get("smoke"):
        return 0 if engineering_ok else 2
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
