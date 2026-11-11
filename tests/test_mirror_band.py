"""Regression tests for mirror opponent sampling diversity (audit 2026-11-11)."""
from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool, SnapshotFrame  # noqa: E402


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
