"""Regression tests for mirror opponent sampling diversity (audit 2026-11-11)."""
from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.mirror_snapshots import (MirrorSnapshotPool, SnapshotFrame,  # noqa: E402
                                            parse_turn_bands, turn_band_percentile)


def _frame(seed: int, strength: int) -> SnapshotFrame:
    player = {"board": [{"atk": int(strength), "hp": 0}]}
    return SnapshotFrame(turn=8, seat=0, player=player,
                         source_game_seed=int(seed), strength=int(strength))


def _pool(mode: str) -> MirrorSnapshotPool:
    pool = MirrorSnapshotPool(sample=mode)
    pool.index[(8, 0)] = [_frame(i, i) for i in range(1, 101)]
    pool.frames = 100
    return pool


def test_p75_is_deterministic_when_strength_is_unique():
    pool = _pool("p75")
    rng = random.Random(3)
    picks = [pool.get(8, 0, rng=rng) for _ in range(50)]
    assert {f.strength for f in picks} == {75}


def test_p75band_restores_opponent_diversity():
    pool = _pool("p75band")
    rng = random.Random(3)
    picks = [pool.get(8, 0, rng=rng) for _ in range(400)]
    assert len({f.source_game_seed for f in picks}) > 10
    mean_strength = sum(f.strength for f in picks) / len(picks)
    assert 70.0 <= mean_strength <= 80.0
    assert min(f.strength for f in picks) >= 70
    assert max(f.strength for f in picks) <= 81


def test_unknown_mode_falls_back_to_random():
    pool = _pool("not-a-mode")
    rng = random.Random(3)
    picks = [pool.get(8, 0, rng=rng) for _ in range(200)]
    assert len({f.source_game_seed for f in picks}) > 10


def test_turn_bands_parse_and_percentile():
    bands = parse_turn_bands(
        "turnband:1-4=55,5-8=60,9-12=70,13-16=75,17-20=80,21-99=75")
    assert [round(turn_band_percentile(bands, t), 2) for t in (1, 8, 12, 16, 20, 25)] == [
        0.55, 0.60, 0.70, 0.75, 0.80, 0.75]


def test_turn_bands_reject_gaps():
    try:
        parse_turn_bands("turnband:1-4=55,6-8=60,9-24=75")
    except ValueError as exc:
        assert "连续" in str(exc)
    else:
        raise AssertionError("gap should fail")


def test_turnband_uses_configured_band():
    pool = MirrorSnapshotPool(
        sample="turnband:1-4=55,5-8=60,9-12=70,13-16=75,17-20=80,21-99=75")
    for turn, lo, hi in ((8, 55, 65), (12, 65, 75), (16, 70, 80), (20, 75, 85)):
        pool.index[(turn, 0)] = [
            SnapshotFrame(turn=turn, seat=0,
                          player={"board": [{"atk": i, "hp": 0}]},
                          source_game_seed=i, strength=i)
            for i in range(1, 101)]
    rng = random.Random(7)
    for turn, lo, hi in ((8, 55, 65), (12, 65, 75), (16, 70, 80), (20, 75, 85)):
        picks = [pool.get(turn, 0, rng=rng) for _ in range(200)]
        assert min(f.strength for f in picks) >= lo
        assert max(f.strength for f in picks) <= hi + 1
        assert len({f.source_game_seed for f in picks}) > 5
