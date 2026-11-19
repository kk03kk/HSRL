"""`forced_spell` 的单测：判定 / 配额 / 合法性 / 概率门控 / 记账 / 关闭逐位一致。

为什么这些测试必须在接线前就全绿：这是**改 PPO 热路径**的前置条件，
而 2026-11-14 已经踩过一次"奖励项对零频动作无效"（`RUNS_20261113.md` §5.79）。
"""
from __future__ import annotations

import random

import pytest

from hsbg.az.train.forced_spell import (
    DEFAULT_FORCED_SPELL_PER_GAME,
    DEFAULT_FORCED_SPELL_TURN_MIN,
    ForcedSpellTracker,
    aggregate_forced_spell,
    forced_spell_rates,
    summarize_forced_spell,
)

AVAILABLE_SPELL = 11


@pytest.mark.parametrize("action", [8, 11, 15])
def test_tracker_uses_the_supplied_buy_action(action):
    decision = ForcedSpellTracker(1, turn_min=0).decide(turn=8, spell_action=action)
    assert decision.force and decision.action == action


@pytest.mark.parametrize("per_game", [0, "0"])
def test_disabled_tracker_is_inert(per_game):
    """关闭时：不判定、不记账、不新增 meta 字段（逐位一致是硬约定）。"""
    t = ForcedSpellTracker(per_game)
    assert t.enabled is False
    for _ in range(10):
        d = t.decide(turn=8, spell_action=AVAILABLE_SPELL)
        assert d.force is False
    assert t.stats() == {}
    assert t.game_stats() == {}
    t.note_policy_action(is_spell_purchase=True)
    assert t.stats() == {}


def test_quota_per_game_is_respected():
    t = ForcedSpellTracker(3, turn_min=0)
    forced = [t.decide(turn=8, spell_action=AVAILABLE_SPELL).force for _ in range(6)]
    assert forced == [True, True, True, False, False, False]
    assert t.stats()["forced_actions"] == 3


def test_reset_restores_quota():
    t = ForcedSpellTracker(2, turn_min=0)
    assert t.decide(turn=8, spell_action=AVAILABLE_SPELL).force is True
    assert t.decide(turn=8, spell_action=AVAILABLE_SPELL).force is True
    assert t.decide(turn=8, spell_action=AVAILABLE_SPELL).force is False
    t.reset()
    assert t.decide(turn=8, spell_action=AVAILABLE_SPELL).force is True


def test_turn_min_blocks_early_shop():
    t = ForcedSpellTracker(3, turn_min=6)
    for turn in range(1, 6):
        assert t.decide(turn=turn, spell_action=AVAILABLE_SPELL).force is False
    assert t.decide(turn=6, spell_action=AVAILABLE_SPELL).force is True
    assert t.stats()["forced_actions"] == 1


def test_never_forces_when_mask_says_illegal():
    """法术槽不合法就不强制，并记 blocked（不读金币、不补金币）。"""
    t = ForcedSpellTracker(3, turn_min=0)
    d = t.decide(turn=8, spell_action=None)
    assert d.force is False and d.blocked is True
    assert t.stats()["blocked"] == 1
    assert t.stats()["forced_actions"] == 0


def test_prob_zero_never_forces_and_counts_skips():
    t = ForcedSpellTracker(3, turn_min=0)
    rng = random.Random(0)
    for _ in range(5):
        assert t.decide(turn=8, spell_action=AVAILABLE_SPELL, force_prob=0.0, rng=rng).force is False
    assert t.stats()["prob_skips"] == 5
    assert t.stats()["forced_actions"] == 0


def test_prob_gate_still_forces_sometimes_and_writes_teacher_on_skip():
    t = ForcedSpellTracker(3, turn_min=0)
    rng = random.Random(1234)
    decs = [t.decide(turn=8, spell_action=AVAILABLE_SPELL, force_prob=0.5, rng=rng)
            for _ in range(20)]
    assert any(d.force for d in decs)
    assert any(d.teacher for d in decs)
    assert t.stats()["teacher_labels"] >= 1


def test_prob_schedule_decays_by_decisions():
    """累计决策数过阈值后，概率从 0 退化为 0（把控制权交还策略）。"""
    t = ForcedSpellTracker(100, turn_min=0, prob_spec="0:1.0,1000:0.0")
    assert t.decide(turn=8, spell_action=AVAILABLE_SPELL, decisions=0.0).force is True
    assert t.decide(turn=8, spell_action=AVAILABLE_SPELL, decisions=1e6).force is False


def test_policy_spell_actions_only_counts_unforced_choices():
    t = ForcedSpellTracker(3, turn_min=0)
    t.note_policy_action(is_spell_purchase=True, forced=False)
    t.note_policy_action(is_spell_purchase=True, forced=True)
    t.note_policy_action(is_spell_purchase=False, forced=False)
    assert t.stats()["policy_spell_actions"] == 1


def test_legal_decisions_proves_reachability():
    """`legal_decisions > 0` 是"这个动作真的可达"的证据（§5.79 的第一条判据）。"""
    t = ForcedSpellTracker(3, turn_min=0)
    for _ in range(4):
        t.decide(turn=9, spell_action=AVAILABLE_SPELL)
    assert t.stats()["legal_decisions"] == 4


def test_game_isolation_by_game_id_and_seat():
    t = ForcedSpellTracker(1, turn_min=0)
    assert t.decide(game_id=1, seat=0, turn=8, spell_action=AVAILABLE_SPELL).force is True
    assert t.decide(game_id=1, seat=0, turn=8, spell_action=AVAILABLE_SPELL).force is False
    # 另一局/另一座位配额独立
    assert t.decide(game_id=2, seat=0, turn=8, spell_action=AVAILABLE_SPELL).force is True
    assert t.decide(game_id=1, seat=1, turn=8, spell_action=AVAILABLE_SPELL).force is True


def test_aggregate_sums_per_game_rows():
    metas = [
        {"forced_spell_game": {"forced": 3, "blocked": 0, "prob_skips": 1,
                              "policy_spell_actions": 0, "legal_decisions": 40}},
        {"forced_spell_game": {"forced": 2, "blocked": 1, "prob_skips": 0,
                              "policy_spell_actions": 2, "legal_decisions": 38}},
    ]
    agg = aggregate_forced_spell(metas)
    assert agg["forced"] == 5
    assert agg["blocked"] == 1
    assert agg["policy_spell_actions"] == 2
    assert agg["legal_decisions"] == 78
    assert agg["games"] == 2
    rates = forced_spell_rates(agg)
    assert rates["forced"] == pytest.approx(2.5)


def test_aggregate_is_empty_when_disabled():
    assert aggregate_forced_spell([{"other": 1}, {}]) == {}
    assert summarize_forced_spell([]) == {}
    assert forced_spell_rates({}) == {}


def test_defaults_are_the_documented_ones():
    assert DEFAULT_FORCED_SPELL_PER_GAME == 3
    assert DEFAULT_FORCED_SPELL_TURN_MIN == 6


@pytest.mark.parametrize("bad", [{"per_game": -2}, {"turn_min": -1}, {"turn_min": 10_000}])
def test_invalid_arguments_fail_fast(bad):
    kwargs = {"per_game": 3}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        ForcedSpellTracker(**kwargs)
