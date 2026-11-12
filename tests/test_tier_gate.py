"""``tier_target`` 的**板面门控**单测（2026-11-13 新增）。

背景：无门控的等级奖励会被「把金币全砸升级、不买随从」打穿（tier 超额达成、
板面 −33%、覆盖 −70%，见 `docs/analysis/REWARD_GAMING_20261113.md`）。
门控要求 ``Σ(atk+hp) >= gate_k * gate_ref[turn]``，不通过时回**中性**（不奖不罚）。

覆盖点：默认曲线解析、门控不通过 → 0 且不计 hits/misses、门控通过 → 正常记账、
``board_strength`` 缺失按不通过处理、``coef=0`` 时零影响。
"""

from __future__ import annotations

import json

import pytest

from hsbg.az.train.tier_target import (
    DEFAULT_GATE_REF,
    TierTargetTracker,
    board_strength_of,
    parse_gate_ref,
)


class FakeGame:
    def __init__(self, atk_hp=(), tier=6, turn=10):
        self.board = [{"atk": a, "hp": h} for a, h in atk_hp]
        self.tier = tier
        self.turn = turn

    def player_json(self, seat):
        return json.dumps({"board": self.board, "tier": self.tier})

    def observe(self, seat):
        raise RuntimeError("this fake has no obs path")   # 强制走 player_json


def test_parse_gate_ref():
    assert parse_gate_ref("") == DEFAULT_GATE_REF
    assert parse_gate_ref(None) == DEFAULT_GATE_REF
    assert parse_gate_ref("10:500,12:1000") == {10: 500, 12: 1000}
    with pytest.raises(ValueError):
        parse_gate_ref("garbage")


def test_board_strength_of():
    g = FakeGame([(100, 200), (50, 50)])
    assert board_strength_of(g, 0) == 400.0


def test_gate_blocks_weak_board():
    # 参考：T10 = 591。门控 0.9 → 需要 ≥532。这里板面只有 400。
    t = TierTargetTracker(coef=1.0, table="10:6", gate_k=0.9)
    g = FakeGame([(100, 200), (50, 50)], tier=6, turn=10)
    assert t.observe(g, 0) == 0.0
    st = t.stats()
    assert st["tier_gate_blocked"] == 1
    assert st["tier_hits"] == 0 and st["tier_misses"] == 0   # 中性，不记账


def test_gate_passes_strong_board():
    t = TierTargetTracker(coef=1.0, table="10:6", gate_k=0.9)
    g = FakeGame([(300, 300)], tier=6, turn=10)             # 600 >= 532
    assert t.observe(g, 0) == pytest.approx(t.coef * t.bonus)
    st = t.stats()
    assert st["tier_hits"] == 1 and st["tier_gate_blocked"] == 0


def test_gate_without_strength_is_neutral():
    t = TierTargetTracker(coef=1.0, table="10:6", gate_k=0.9)

    class NoBoard:
        turn = 10

        def player_json(self, seat):
            raise RuntimeError("boom")

    assert t.observe(NoBoard(), 0) == 0.0
    assert t.stats()["tier_gate_blocked"] == 1


def test_gate_off_keeps_old_behaviour():
    t = TierTargetTracker(coef=1.0, table="10:6")
    g = FakeGame([(1, 1)], tier=4, turn=10)
    assert t.observe(g, 0) == pytest.approx(-t.coef * t.penalty)   # 未达标 → 罚
    assert t.stats()["tier_gate_blocked"] == 0


def test_coef_zero_is_noop():
    t = TierTargetTracker(coef=0.0)
    assert t.enabled is False
    assert t.stats() == {}
    assert t.observe(FakeGame([(1, 1)]), 0) == 0.0
