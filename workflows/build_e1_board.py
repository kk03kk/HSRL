#!/usr/bin/env python3
"""Build the E1 four-model leaderboard from a round-robin report.

The evaluator collects games and fits a BT Elo; this script turns the validated
report into the permanent board artifact, including rank intervals and the
champion decision. It never edits the raw report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def wilson_ci(wins: float, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def load_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("valid") or not report.get("leaderboard_eligible"):
        raise SystemExit(f"report is not leaderboard-eligible: {path}")
    if len(report.get("elo", {}).get("estimates", [])) < 2:
        raise SystemExit("report needs at least two models")
    return report


def totals_by_model(report: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    totals: Dict[str, Dict[str, int]] = {}
    for model_id in report["model_order"]:
        totals[model_id] = {"games": 0, "wins": 0, "ties": 0, "losses": 0}
    for pair in report["pairs"]:
        i, j = str(pair["i"]), str(pair["j"])
        n = int(pair["games"])
        wi, wj, ties = int(pair["wins_i"]), int(pair["wins_j"]), int(pair["ties"])
        totals[i]["games"] += n
        totals[i]["wins"] += wi
        totals[i]["ties"] += ties
        totals[i]["losses"] += wj
        totals[j]["games"] += n
        totals[j]["wins"] += wj
        totals[j]["ties"] += ties
        totals[j]["losses"] += wi
    return totals


def _model_sha_map(report: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for model_id in report.get("model_order", []):
        audit = (report.get("models") or {}).get(model_id) or {}
        sha = audit.get("file_sha256")
        if sha:
            out[str(model_id)] = str(sha)
    return out


def validate_challenge(
    report: Dict[str, Any],
    challenge: Dict[str, Any],
    top_id: str,
    second_id: str,
) -> None:
    """A challenge may only decide a champion under the same rules/models."""
    if challenge.get("engine") != report.get("engine"):
        raise SystemExit("challenge engine fingerprint differs from the main report")
    main_protocol = report.get("protocol") or {}
    chal_protocol = challenge.get("protocol") or {}
    for key in ("games_per_pair", "seat_mode", "seed_mode", "decode",
                "max_actions_per_turn", "move_cap", "move_repeat_limit",
                "forbid_cancel", "max_steps"):
        if key in main_protocol and key in chal_protocol:
            if main_protocol[key] != chal_protocol[key]:
                raise SystemExit(
                    f"challenge protocol differs on {key}: "
                    f"{chal_protocol[key]!r} != {main_protocol[key]!r}")
    main_sha = _model_sha_map(report)
    chal_sha = _model_sha_map(challenge)
    for model_id in (top_id, second_id):
        if model_id not in chal_sha:
            raise SystemExit(f"challenge is missing model {model_id!r}")
        if model_id in main_sha and chal_sha[model_id] != main_sha[model_id]:
            raise SystemExit(
                f"challenge model {model_id!r} SHA differs: "
                f"{chal_sha[model_id]} != {main_sha[model_id]}")
    pair = (challenge.get("pairs") or [{}])[0]
    if {str(pair.get("i")), str(pair.get("j"))} != {top_id, second_id}:
        raise SystemExit(
            f"challenge must be exactly {top_id} vs {second_id}, got "
            f"{pair.get('i')!r} vs {pair.get('j')!r}")


def champion_from_challenge(
    top_id: str,
    second_id: str,
    challenge: Dict[str, Any],
) -> Dict[str, Any]:
    pair = challenge["pairs"][0]
    i, j = str(pair["i"]), str(pair["j"])
    n = int(pair["games"])
    wi, wj, ties = int(pair["wins_i"]), int(pair["wins_j"]), int(pair["ties"])
    if i != top_id or j != second_id:
        raise SystemExit(f"challenge pair order mismatch: {i} vs {j}")
    score = (wi + 0.5 * ties) / n
    lo, hi = wilson_ci(wi + 0.5 * ties, n)
    if lo > 0.5:
        champion, method = top_id, "challenge_ci_lower_gt_half"
    elif hi < 0.5:
        champion, method = second_id, "challenge_ci_upper_lt_half"
    else:
        champion, method = top_id, "challenge_unresolved_point_estimate"
    return {
        "champion": champion,
        "champion_method": method,
        "challenge": {
            "top": top_id,
            "second": second_id,
            "games": n,
            "wins_top": wi,
            "wins_second": wj,
            "ties": ties,
            "score_top": score,
            "score_ci95": [lo, hi],
            "protocol_sha256": challenge.get("protocol_sha256"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--challenge-report", type=Path, default=None)
    args = ap.parse_args()

    report = load_report(args.report)
    estimates = report["elo"]["estimates"]
    top, second = estimates[0], estimates[1]
    top_id, second_id = str(top["id"]), str(second["id"])
    # Separation is a joint-ranking statement: the top model must be rank 1 in
    # at least 97.5% of bootstrap resamples (upper rank CI < 2).  Comparing the
    # top Elo CI lower bound with other point estimates is not a valid test and
    # ignores multiplicity/selection.
    rank_ci = top.get("rank_ci95")
    separated = bool(rank_ci is not None and float(rank_ci[1]) < 1.5)
    if separated:
        decision = {
            "champion": top_id,
            "champion_method": "auto_rank_ci_separated",
            "challenge": None,
        }
    elif args.challenge_report is not None:
        challenge = load_report(args.challenge_report)
        validate_challenge(report, challenge, top_id, second_id)
        decision = champion_from_challenge(top_id, second_id, challenge)
    else:
        decision = {
            "champion": top_id,
            "champion_method": "point_estimate_pending_challenge",
            "challenge": None,
        }

    totals = totals_by_model(report)
    results = report.get("results", {})
    models: List[Dict[str, Any]] = []
    for rank, row in enumerate(estimates, 1):
        model_id = str(row["id"])
        t = totals[model_id]
        n = int(t["games"])
        score = (t["wins"] + 0.5 * t["ties"]) / n if n else None
        model_audit = report.get("models", {}).get(model_id, {})
        model_result = results.get(model_id, {})
        rank_summary = model_result.get("rank", {}) if isinstance(model_result, dict) else {}
        models.append({
            "rank": rank,
            "id": model_id,
            "elo": float(row["elo"]),
            "elo_ci95": [float(x) for x in row["elo_ci95"]],
            "rank_ci95": ([float(x) for x in row["rank_ci95"]]
                          if row.get("rank_ci95") is not None else None),
            "games": n,
            "wins": int(t["wins"]),
            "draws": int(t["ties"]),
            "losses": int(t["losses"]),
            "score": score,
            "mean_rank": rank_summary.get("mean_rank") if isinstance(rank_summary, dict) else None,
            "top4_rate": rank_summary.get("top4_rate") if isinstance(rank_summary, dict) else None,
            "win_rate": rank_summary.get("win_rate") if isinstance(rank_summary, dict) else None,
            "model_sha256": model_audit.get("file_sha256"),
        })

    board = {
        "schema": "hsai.leaderboard/e1-four/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": report.get("engine"),
        "protocol_sha256": report.get("protocol_sha256"),
        "protocol": {
            "games_per_pair": report.get("protocol", {}).get("games_per_pair"),
            "seat_mode": report.get("protocol", {}).get("seat_mode"),
            "seed_mode": report.get("protocol", {}).get("seed_mode"),
            "bootstrap": report.get("protocol", {}).get("bootstrap"),
            "elo_anchor": report.get("elo", {}).get("anchor"),
        },
        "source_report": str(args.report),
        "challenge_report": str(args.challenge_report) if args.challenge_report else None,
        "model_order": report["model_order"],
        "models": models,
        "pairwise_empirical": report.get("elo", {}).get("empirical_win_prob"),
        "pairwise_elo": report.get("elo", {}).get("elo_win_prob"),
        "champion": decision["champion"],
        "champion_method": decision["champion_method"],
        "challenge": decision["challenge"],
        "note": "E1 four-model board: V1.1, 12311, 13806, final only; other models are not ranked",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    board_path = args.out / "leaderboard.json"
    write_json(board_path, board)

    lines = [
        "# E1 four-model leaderboard",
        "",
        f"- generated: {board['generated_at']}",
        f"- protocol: `{board['protocol_sha256']}`",
        f"- champion: **{board['champion']}** ({board['champion_method']})",
        "",
        "| rank | model | Elo | 95% CI | rank 95% | games | score | mean rank |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for m in models:
        rank_ci = (f"[{m['rank_ci95'][0]:.1f}, {m['rank_ci95'][1]:.1f}]"
                   if m.get("rank_ci95") is not None else "—")
        lines.append(
            f"| {m['rank']} | {m['id']} | {m['elo']:.1f} | "
            f"[{m['elo_ci95'][0]:.1f}, {m['elo_ci95'][1]:.1f}] | "
            f"{rank_ci} | "
            f"{m['games']} | "
            f"{(100*m['score']):.2f}% | "
            f"{m['mean_rank'] if m['mean_rank'] is not None else '—'} |"
        )
    lines += ["", "Only V1.1, 12311, 13806, and final are ranked."]
    (args.out / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "hsai.leaderboard_manifest/e1-four/v1",
        "generated_at": board["generated_at"],
        "board": str(board_path),
        "report": str(args.report),
        "challenge_report": str(args.challenge_report) if args.challenge_report else None,
        "sha256": {
            "leaderboard.json": sha256_file(board_path),
            "leaderboard.md": sha256_file(args.out / "leaderboard.md"),
            "report.json": sha256_file(args.report),
            "challenge-report.json": sha256_file(args.challenge_report) if args.challenge_report else None,
        },
        "protocol_sha256": board["protocol_sha256"],
        "champion": board["champion"],
        "champion_method": board["champion_method"],
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
