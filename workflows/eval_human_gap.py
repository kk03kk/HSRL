#!/usr/bin/env python3
"""Human-gap panel: measure our models against the Firestone human baseline.

Why this exists
---------------
Every claim of the form "the model is weak late game" is meaningless until it is
expressed in the same units as a human. This workflow plays N games per
checkpoint against the seven engine bots, records the per-turn economy rows
(``EvalConfig.collect_economy``) and reduces them to the exact same statistics
that :mod:`tools.analyze_human_baseline` extracts from the replays:

* tavern tier by turn (level-up tempo);
* board strength by turn, where strength is ``Σ(atk + hp)`` of the
  recruitment-end board — byte-for-byte the same definition the human data uses
  (``stage == combat_start``);
* gold, board size, actions per turn;
* how deep games actually go (turn 12 / 14 survival).

Caveats this panel cannot remove (stated so nobody over-reads it)
----------------------------------------------------------------
* Humans played 8 real opponents; our models play 7 engine bots. Tier tempo and
  board-growth are mostly *intrinsic* to the player, but rank outcomes are not
  comparable at all — compare curves, never mean placement.
* The human feed is winner-biased (816/995 are 1st place), so the human curve is
  "what winners do", i.e. a demanding yardstick, not an average.

Usage
-----
::

    python workflows/eval_human_gap.py \
        --checkpoint v1.1=artifacts/models/v1.1/model.pt \
        --checkpoint final14858=artifacts/models/league-final-20261111-step014858/model.pt \
        --games 128 --device cuda:0 \
        --human-baseline artifacts/runs/human-baseline-20261112/baseline.json \
        --out artifacts/runs/human-gap-<date>/panel.json

    # tempo probe: force the human tier curve and see how much of the gap closes
    python workflows/eval_human_gap.py ... --forced-tier-plan "2:2,5:3,6:4,8:5"
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.common import engine_info  # noqa: E402
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy  # noqa: E402
from hsbg.az.train.model import load_checkpoint  # noqa: E402
from hsbg.az.train.ppo import PPOEvaluator, PPOGreedyPolicy  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def economy_curves(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn -> statistics. Uses the recruitment-end board, same as the human data."""
    by_turn: Dict[int, Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    tier_hist: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    for r in rows:
        t = int(r["turn"])
        b = by_turn[t]
        if isinstance(r.get("tier"), int):
            b["tier"].append(float(r["tier"]))
            tier_hist[t][int(r["tier"])] += 1
        if isinstance(r.get("gold"), int):
            b["gold"].append(float(r["gold"]))
        if isinstance(r.get("actions"), int):
            b["actions"].append(float(r["actions"]))
        # recruitment-end board == what the human snapshots call combat_start
        atk, hp = r.get("eot_attack"), r.get("eot_board_hp")
        if isinstance(atk, int) and isinstance(hp, int):
            b["strength"].append(float(atk + hp))
        board = r.get("eot_board") or r.get("board")
        if isinstance(board, list):
            b["board_n"].append(float(len(board)))

    out: Dict[str, Any] = {}
    for t in sorted(by_turn):
        b = by_turn[t]
        if len(b["tier"]) < 10:
            continue
        hist = tier_hist[t]
        n = len(b["tier"])
        out[str(t)] = {
            "n": n,
            "tier_mean": _r(sum(b["tier"]) / n, 2),
            "tier_mode": hist.most_common(1)[0][0],
            "tier6_share": _r(sum(1 for x in b["tier"] if x >= 6) / n, 3),
            "tier5plus_share": _r(sum(1 for x in b["tier"] if x >= 5) / n, 3),
            "tier_hist": dict(sorted(hist.items())),
            "gold_p50": _r(_pct(b["gold"], 0.5)),
            "strength_p50": _r(_pct(b["strength"], 0.5)),
            "strength_p90": _r(_pct(b["strength"], 0.90)),
            "strength_mean": _r(sum(b["strength"]) / len(b["strength"])) if b["strength"] else None,
            "board_n_p50": _r(_pct(b["board_n"], 0.5)),
            "actions_mean": _r(sum(b["actions"]) / len(b["actions"])) if b["actions"] else None,
        }
    return out


def survival(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reached = collections.defaultdict(int)
    games: Dict[Any, int] = {}
    for r in rows:
        g = r.get("seed")
        games[g] = max(games.get(g, 0), int(r["turn"]))
    for g, t in games.items():
        for milestone in (8, 10, 12, 14):
            if t >= milestone:
                reached[milestone] += 1
    n = max(len(games), 1)
    return {
        "games": len(games),
        "max_turn_p50": _pct(list(games.values()), 0.5),
        "reached": {str(k): _r(v / n, 3) for k, v in sorted(reached.items())},
    }


def run_checkpoint(name: str, path: Path, args) -> Dict[str, Any]:
    model, meta = load_checkpoint(str(path), device=args.device)
    evaluator = PPOEvaluator(model, device=args.device,
                             norm_mean=meta.get("norm_mean"),
                             norm_std=meta.get("norm_std"))
    policy = PPOGreedyPolicy(evaluator,
                             max_actions_per_turn=int(args.max_actions_per_turn),
                             forbid_unfreeze=bool(int(args.forbid_unfreeze)))
    cfg = EvalConfig(
        games=int(args.games),
        seed=int(args.seed),
        policy_seat=0,
        max_steps=int(args.max_steps),
        collect_economy=True,
        collect_action_hist=True,
        forced_tier_plan=str(args.forced_tier_plan or ""),
    )
    started = time.time()
    result = evaluate_policy(policy, cfg)
    rows = result.pop("economy", []) or []
    hist = result.pop("action_hist", None)
    return {
        "name": name,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "arch": meta.get("arch"),
        "games": int(args.games),
        "seconds": round(time.time() - started, 1),
        "outcome": {k: result.get(k) for k in
                    ("mean_rank", "top4_rate", "win_rate", "illegal_actions",
                     "truncated_games", "decisions", "decisions_per_game")},
        "action_hist": hist,
        "survival": survival(rows),
        "curves": economy_curves(rows),
    }


def compare(model: Dict[str, Any], human: Dict[str, Any]) -> Dict[str, Any]:
    """Per-turn model-vs-human deltas. Human side is the winner-biased baseline."""
    mc = model["curves"]
    hc = human.get("strength_by_turn") or {}
    ht = human.get("tier_by_turn") or {}
    rows = []
    for t in sorted(set(mc) & set(hc), key=int):
        m, h = mc[t], hc[t]
        tier_h = ht.get(t) or {}
        rows.append({
            "turn": int(t),
            "model_tier_mean": m["tier_mean"],
            "human_tier_mean": tier_h.get("mean"),
            "tier_gap": _r((m["tier_mean"] or 0) - (tier_h.get("mean") or 0), 2),
            "model_strength_p50": m["strength_p50"],
            "human_strength_p50": h.get("p50"),
            "strength_ratio": _r(m["strength_p50"] / h["p50"], 3)
            if m.get("strength_p50") and h.get("p50") else None,
            "model_strength_p90": m["strength_p90"],
            "human_strength_p90": h.get("p90"),
            "model_board_n_p50": m["board_n_p50"],
        })
    return {"per_turn": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="human-gap panel")
    ap.add_argument("--checkpoint", action="append", default=[], required=True,
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--max-actions-per-turn", type=int, default=80)
    ap.add_argument("--forbid-unfreeze", type=int, default=1, choices=[0, 1])
    ap.add_argument("--forced-tier-plan", default="",
                    help='e.g. "2:2,5:3,6:4,8:5" — diagnostic only, never for ranking')
    ap.add_argument("--human-baseline", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    specs = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"checkpoint not found: {p}")
        specs.append((name, p))

    human = {}
    if args.human_baseline:
        hp = Path(args.human_baseline)
        if hp.is_file():
            human = json.loads(hp.read_text(encoding="utf-8"))
        else:
            print(f"WARN: human baseline missing at {hp}; comparison disabled", file=sys.stderr)

    payload: Dict[str, Any] = {
        "schema": "hsai.human_gap_panel/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": engine_info(),
        "panel": {
            "opponent": "engine_bot x7",
            "games": int(args.games),
            "seed": int(args.seed),
            "max_actions_per_turn": int(args.max_actions_per_turn),
            "forbid_unfreeze": bool(int(args.forbid_unfreeze)),
            "forced_tier_plan": str(args.forced_tier_plan or ""),
        },
        "caveats": [
            "humans faced 8 real opponents; models face 7 engine bots - compare curves, not ranks",
            "human feed is winner-biased (816/995 1st place): a demanding yardstick, not an average",
        ],
        "models": [],
    }

    for name, path in specs:
        print(f"[{name}] evaluating {args.games} games ...", flush=True)
        entry = run_checkpoint(name, path, args)
        if human:
            entry["vs_human"] = compare(entry, human)
        payload["models"].append(entry)
        o = entry["outcome"]
        print(f"    mean_rank={o.get('mean_rank')} top4={o.get('top4_rate')} "
              f"win={o.get('win_rate')} illegal={o.get('illegal_actions')} "
              f"{entry['seconds']}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- compact side-by-side print ----
    if human:
        hc = human.get("strength_by_turn") or {}
        ht = human.get("tier_by_turn") or {}
        turns = [t for t in ("6", "8", "10", "12", "14") if t in hc]
        print()
        print("HUMAN-GAP TABLE  (board strength = sum(atk+hp) at recruitment end)")
        head = f"{'turn':>4} {'human p50':>10} {'human tier':>10}"
        for m in payload["models"]:
            head += f" {m['name'][:14]:>14}"
        print(head)
        for t in turns:
            h = hc[t]
            row = f"{t:>4} {h['p50']:>10.0f} {ht.get(t, {}).get('mean', float('nan')):>10.2f}"
            for m in payload["models"]:
                c = m["curves"].get(t)
                row += f" {c['strength_p50'] if c else float('nan'):>14.0f}"
            print(row)
        print()
        print(f"{'turn':>4} {'strength ratio (model/human)':>32}")
        for t in turns:
            line = f"{t:>4}"
            for m in payload["models"]:
                c = m["curves"].get(t)
                r = (c["strength_p50"] / hc[t]["p50"]) if (c and c.get("strength_p50")) else None
                line += f"  {m['name'][:10]}={r:.2f}" if r else f"  {m['name'][:10]}=NA"
            print(line)
        print()
        print(f"{'model':<16} {'survive>t12':>12} {'survive>t14':>12} {'T6@t12':>8}")
        for m in payload["models"]:
            s = m["survival"]["reached"]
            c = m["curves"].get("12") or {}
            print(f"{m['name']:<16} {str(s.get('12')):>12} {str(s.get('14')):>12} "
                  f"{str(c.get('tier6_share')):>8}")

    print(f"\nwritten: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
