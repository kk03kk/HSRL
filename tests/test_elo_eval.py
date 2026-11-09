import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.eval_round_robin_elo import (  # noqa: E402
    all_pairs,
    fit_elo,
    pair_seats,
    seat_counts,
)


def test_pair_seats_cover_all_seats_and_swap_roles():
    games = 64
    counts = seat_counts(games)
    for label in ("A", "B"):
        assert counts[label] == {seat: 8 for seat in range(8)}
    for gi in range(32):
        first = pair_seats(gi, games)
        second = pair_seats(gi + 32, games)
        assert {first["A"], first["B"]} == {second["A"], second["B"]}
        assert first["A"] == second["B"] and first["B"] == second["A"]
        assert first["A"] != first["B"]


def test_pair_seats_are_distinct_for_odd_game_counts():
    for games in (1, 2, 3, 7, 8, 64):
        for gi in range(games):
            seats = pair_seats(gi, games)
            assert seats["A"] != seats["B"]
            assert 0 <= seats["A"] < 8 and 0 <= seats["B"] < 8


def test_pair_seed_ranges_are_disjoint():
    models = [{"id": f"m{i}"} for i in range(8)]
    cfg = {"models": models}
    pairs = all_pairs(cfg)
    assert len(pairs) == 28
    per_pair = 64
    base = 88000000
    ranges = [(base + p * per_pair, base + (p + 1) * per_pair - 1)
              for p in range(len(pairs))]
    assert ranges[0][0] == base
    assert ranges[-1][1] == base + 28 * per_pair - 1
    for left, right in zip(ranges, ranges[1:]):
        assert left[1] < right[0]


def _synthetic_pairs(ids, true_strength, games=512):
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p = 1.0 / (1.0 + math.exp(-(true_strength[i] - true_strength[j])))
            wins = int(round(games * p))
            outcomes = [1.0] * wins + [0.0] * (games - wins)
            pairs.append({
                "i": ids[i],
                "j": ids[j],
                "games": games,
                "wins_i": wins,
                "wins_j": games - wins,
                "ties": 0,
                "outcomes": outcomes,
            })
    return pairs


def test_fit_elo_recovers_order_and_anchor():
    ids = ["anchor", "strong", "middle", "weak"]
    true = np.array([0.0, 0.8, 0.2, -0.8])
    cfg = {
        "elo_anchor": "anchor",
        "elo_anchor_score": 1500.0,
        "bootstrap": 80,
        "bootstrap_seed": 7,
    }
    elo = fit_elo(cfg, _synthetic_pairs(ids, true), ids)
    scores = {row["id"]: row["elo"] for row in elo["estimates"]}
    assert scores["anchor"] == pytest.approx(1500.0)
    assert scores["strong"] > scores["middle"] > scores["weak"]
    assert all(row["elo_ci95"][0] <= row["elo"] <= row["elo_ci95"][1]
               for row in elo["estimates"])
    assert elo["empirical_win_prob"]["strong"]["weak"] > 0.8
    assert 0.0 <= elo["elo_win_prob"]["strong"]["weak"] <= 1.0
