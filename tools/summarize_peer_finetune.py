"""Paired natural-game evidence for a peer fine-tune and its unchanged parent."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from hsbg.runtime.artifacts import atomic_json
from tools.summarize_peer_distillation import body, estimate, p50, prepare

UTILITY = [1, .35, -1, -1.15, -1.3, -1.45, -1.6, -1.75]


def utility_limits(game):
    upper = game["tail_rank_upper"]
    rank = game["metrics"]["rank"]
    if upper:
        if not (1 <= rank <= upper <= 8 and game["tail_seats"] > 0):
            raise ValueError("invalid extrapolation bounds")
        return UTILITY[upper - 1], UTILITY[0]
    return UTILITY[rank - 1], UTILITY[rank - 1]


def compare(root, start, n):
    rows = [row for i in range(8) for row in json.loads((root / f"panel-{i}.json").read_bytes())]
    if sorted(row["parent"]["seed"] for row in rows) != list(range(start, start + n)):
        raise ValueError("seed coverage mismatch")
    for row in rows:
        if set(row) != {"parent", "candidate"} or len({(g["seed"], g["lobby"]) for g in row.values()}) != 1:
            raise ValueError("unpaired trajectory")
        for game in row.values():
            game["metrics"].update(prepare(game))
            game["metrics"]["reach12"] = int(body(game, 12) is not None)
    models = {}
    for name in rows[0]:
        games = [r[name] for r in rows]
        turns = {}
        for turn in range(6, 19):
            values = [v for g in games if (v := body(g, turn)) is not None]
            turns[turn] = dict(n=len(values), p50=p50(values))
        models[name] = dict(turns=turns,
            means={k: np.mean([g["metrics"][k] for g in games]) for k in games[0]["metrics"]},
            extrapolated=sum(bool(g["tail_rank_upper"]) for g in games))
    delta = {k: estimate([r["candidate"]["metrics"][k] - r["parent"]["metrics"][k] for r in rows])
             for k in rows[0]["parent"]["metrics"]}
    for turn in (12, 14, 15):
        pairs = [(r["parent"], r["candidate"]) for r in rows if all(body(g, turn) is not None for g in r.values())]
        def log_growth(g):
            return np.log1p(body(g, turn)) - np.log1p(body(g, 9))
        delta[f"log{turn}"] = estimate([np.log1p(body(b, turn)) - np.log1p(body(a, turn)) for a,b in pairs])
        delta[f"growth9to{turn}"] = estimate([log_growth(b) - log_growth(a) for a,b in pairs])
    limits = [(utility_limits(r["parent"]), utility_limits(r["candidate"])) for r in rows]
    strict_bounds = [np.mean([b[0]-a[1] for a,b in limits]), np.mean([b[1]-a[0] for a,b in limits])]
    return dict(seed=start, n=n, models=models, delta=delta, exact_field_utility_bounds=strict_bounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--games", required=True, type=int)
    args = parser.parse_args()
    atomic_json(args.root / "paired-growth.json", compare(args.root, args.seed, args.games))
