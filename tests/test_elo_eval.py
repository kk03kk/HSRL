import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.eval_round_robin_elo import (  # noqa: E402
    ELO_SCALE,
    all_pairs,
    fit_elo,
    load_config,
    pair_seats,
    seat_counts,
)
from workflows.eval_two_models_six_bots import semantic_code_sha256  # noqa: E402


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
    # The Elo win probability must be sigmoid((Elo_i - Elo_j)/ELO_SCALE),
    # not sigmoid(Elo_i - Elo_j).  True log-odds gap is 1.6 -> ~0.832.
    pair_p = elo["elo_win_prob"]["strong"]["weak"]
    expected = 1.0 / (1.0 + math.exp(-1.6))
    assert pair_p == pytest.approx(expected, abs=0.03)
    assert 0.0 < elo["elo_win_prob"]["weak"]["strong"] < 1.0
    assert elo["elo_win_prob"]["weak"]["strong"] == pytest.approx(1.0 - pair_p, abs=1e-12)


def test_shared_seed_bootstrap_and_missing_protocol_code(tmp_path):
    ids = ["anchor", "a", "b"]
    true = np.array([0.0, 0.5, -0.5])
    cfg = {
        "elo_anchor": "anchor",
        "elo_anchor_score": 1000.0,
        "bootstrap": 40,
        "bootstrap_seed": 11,
        "seat_mode": "ordered56",
        "seed_mode": "shared",
        "games_per_pair": 64,
    }
    elo = fit_elo(cfg, _synthetic_pairs(ids, true, games=64), ids)
    assert [row["id"] for row in elo["estimates"]][0] == "a"
    assert all(row["rank_ci95"] is not None for row in elo["estimates"])

    cfg["seed_mode"] = "disjoint"
    elo_disjoint = fit_elo(cfg, _synthetic_pairs(ids, true, games=64), ids)
    assert [row["id"] for row in elo_disjoint["estimates"]][0] == "a"

    cfg["seed_mode"] = "shared"
    pair = _synthetic_pairs(ids, true, games=32)
    pair[-1]["outcomes"] = pair[-1]["outcomes"][:-1]
    with pytest.raises(ValueError):
        fit_elo(cfg, pair, ids)

    code = semantic_code_sha256()
    assert "workflows/eval_two_models_six_bots.py" in code
    assert "hsbg/az/train/ppo.py" in code


def test_load_config_rejects_unknown_modes(tmp_path):
    models = [{"id": x, "path": "x", "file_sha256": "0" * 64, "bytes": 1}
              for x in ("anchor", "other")]
    base = {
        "engine": {},
        "models": models,
        "games_per_pair": 8,
        "seed_base": 1,
        "max_actions_per_turn": 80,
        "move_cap": 5,
        "move_repeat_limit": 2,
        "bootstrap": 10,
        "elo_anchor": "anchor",
        "elo_anchor_score": 1000,
        "bootstrap_seed": 1,
    }
    good = tmp_path / "good.json"
    good.write_text(__import__("json").dumps(base), encoding="utf-8")
    load_config(good)
    for key, value in (("seat_mode", "ordered_56"), ("seed_mode", "share")):
        bad = dict(base, **{key: value})
        bad_path = tmp_path / f"bad-{key}.json"
        bad_path.write_text(__import__("json").dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(bad_path)


def test_fit_elo_two_model_challenge_is_finite():
    """A 2-model challenge: bootstrap all-win samples must not crash or fake a tie."""
    ids = ["league-final-14858", "league-latest-13806"]
    true = np.array([0.25, -0.25])
    pairs = _synthetic_pairs(ids, true, games=448)
    cfg = {
        "elo_anchor": ids[0],
        "elo_anchor_score": 1000.0,
        "bootstrap": 200,
        "bootstrap_seed": 5,
        "seat_mode": "ordered56",
        "seed_mode": "shared",
        "games_per_pair": 448,
    }
    elo = fit_elo(cfg, pairs, ids)
    assert len(elo["estimates"]) == 2
    assert all(np.isfinite(row["elo"]) for row in elo["estimates"])
    assert all(np.isfinite(row["elo_ci95"]).all() for row in elo["estimates"])
    assert elo["estimates"][0]["id"] == ids[0]
