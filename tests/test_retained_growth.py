"""`retained_growth` 单测：口径、三条不可作弊性、有界性、关闭逐位一致。

为什么这些必须钉死：这是 D-052 之后唯一"即使名次不敏感也能学"的奖励项。
历史上的奖励实验各有死法——`Φ=log Σ` 变方差机器（§5.74）、终局板面被"拿板面换命"
（§5.78）、动作条件补贴对零频动作为零（§5.79）。所以这里把"不可作弊"和"有界"
写成断言，而不是写在注释里。
"""
from __future__ import annotations

import json

import pytest

from hsbg.az.train.retained_growth import (
    DEFAULT_CAP,
    DEFAULT_MIN_TURN,
    DEFAULT_NORMALIZER,
    RetainedGrowthTracker,
    board_strength_by_card,
    kept_gain,
)


class _Game:
    """最小 game 替身：`observe` 只读 `.turn` 和 `.player_json(seat)`。"""

    def __init__(self, turn: int, board):
        self.turn = int(turn)
        self._board = board

    def player_json(self, seat: int) -> str:
        return json.dumps({"board": self._board})


def _m(card, atk, hp):
    return {"card": card, "atk": atk, "hp": hp}


# ── 口径：与 report_stat_growth / retained_gain_by_turn 一致 ──────────────

def test_board_strength_sums_atk_plus_hp_and_strips_golden():
    b = board_strength_by_card([_m("BGS_034", 3, 4), _m("BG26_817_G", 10, 10)])
    assert b == {"BGS_034": 7.0, "BG26_817": 20.0}


def test_duplicates_are_aggregated_as_one_stack():
    b = board_strength_by_card([_m("A", 1, 1), _m("A", 2, 2)])
    assert b == {"A": 6.0}


def test_kept_gain_only_counts_cards_on_both_turns():
    prev = {"A": 10.0, "B": 10.0}
    cur = {"A": 30.0, "C": 99.0}       # B 被卖掉，C 是新买的
    gain, n = kept_gain(prev, cur)
    assert n == 1
    assert gain == 20.0                # 只有 A 算，(30-10)/1


def test_new_units_do_not_count_as_growth():
    """买一堆新随从 ≠ 长大 —— 这是本奖励与"终局板面"最关键的区别。"""
    prev = {"A": 10.0}
    cur = {"A": 10.0, "N1": 500.0, "N2": 500.0}
    gain, n = kept_gain(prev, cur)
    assert n == 1 and gain == 0.0


def test_shrinking_is_not_punished_but_not_rewarded():
    gain, n = kept_gain({"A": 100.0}, {"A": 50.0})
    assert n == 1 and gain == 0.0


# ── 不可作弊性 ────────────────────────────────────────────────────────────

def test_dying_early_earns_less_than_growing_long():
    """早死 = 没回合 = 没奖励；这是"不能靠早死刷分"的断言。"""
    short = RetainedGrowthTracker(coef=0.5, min_turn=9)
    short.observe(_Game(9, [_m("A", 10, 10)]))
    short.observe(_Game(10, [_m("A", 10, 10)]))   # 没增长
    total_short = short.reward_sum

    long = RetainedGrowthTracker(coef=0.5, min_turn=9)
    long.observe(_Game(9, [_m("A", 10, 10)]))
    for t in range(10, 14):                       # 连续 4 回合把它养大
        long.observe(_Game(t, [_m("A", 10 + 500 * (t - 9), 10)]))
    assert long.reward_sum > total_short
    assert long.turns > short.turns


def test_selling_the_unit_resets_growth_credit():
    """卖掉再买回来不算"留住并养大"。"""
    t = RetainedGrowthTracker(coef=0.5, min_turn=9)
    t.observe(_Game(9, [_m("A", 10, 10)]))
    t.observe(_Game(10, []))                      # 卖掉 -> 没有留存
    d = t.observe(_Game(11, [_m("A", 900, 900)]))
    assert d == 0.0


# ── 有界性与量级 ──────────────────────────────────────────────────────────

def test_episode_total_is_bounded_by_cap():
    """每局上界 = cap（默认 0.5 = 名次全距 2.0 的 25%）。"""
    t = RetainedGrowthTracker(coef=0.5, min_turn=9, cap=0.5)
    t.observe(_Game(9, [_m("A", 1, 1)]))
    for k in range(1, 30):
        t.observe(_Game(9 + k, [_m("A", 1 + 5000 * k, 1)]))
    assert t.reward_sum <= 0.5 + 1e-9


def test_per_turn_score_is_normalized_by_the_human_690():
    """每只留存随从 +690 ⇒ 该回合拿满 coef 的 1/1（score=1）。"""
    t = RetainedGrowthTracker(coef=0.5, min_turn=9, cap=10.0)
    t.observe(_Game(9, [_m("A", 0, 0)]))
    d = t.observe(_Game(10, [_m("A", 690, 0)]))
    assert d == pytest.approx(0.5)


def test_half_human_growth_gets_half_credit():
    t = RetainedGrowthTracker(coef=0.5, min_turn=9, cap=10.0)
    t.observe(_Game(9, [_m("A", 0, 0)]))
    d = t.observe(_Game(10, [_m("A", 345, 0)]))
    assert d == pytest.approx(0.25)


# ── 入场局：注入的战团不算自己养的 ────────────────────────────────────────

def test_entry_floor_ignores_the_injected_board():
    t = RetainedGrowthTracker(coef=0.5, min_turn=9)
    t.set_floor(11)
    t.observe(_Game(11, [_m("A", 900, 900)]))     # 注入帧：只设基线
    assert t.reward_sum == 0.0
    d = t.observe(_Game(12, [_m("A", 900 + 690, 900)]))
    assert d > 0.0


# ── 关闭时逐位一致 ────────────────────────────────────────────────────────

def test_disabled_is_inert():
    t = RetainedGrowthTracker(coef=0.0)
    assert t.enabled is False
    assert t.observe(_Game(10, [_m("A", 999, 999)])) == 0.0
    assert t.queue(0, 0, 1.0) is False
    recs = [{"game_id": 0, "reward": 1.0}]
    assert t.flush(recs) == 0
    assert recs[0]["reward"] == 1.0
    assert t.stats() == {}


def test_same_turn_is_only_counted_once():
    t = RetainedGrowthTracker(coef=0.5, min_turn=9, cap=10.0)
    t.observe(_Game(9, [_m("A", 0, 0)]))
    first = t.observe(_Game(10, [_m("A", 690, 0)]))
    again = t.observe(_Game(10, [_m("A", 690, 0)]))
    assert first > 0.0 and again == 0.0
    assert t.turns == 1


def test_errors_never_interrupt_collection():
    class _Bad:
        turn = 10

        def player_json(self, seat):
            raise RuntimeError("boom")

    t = RetainedGrowthTracker(coef=0.5, min_turn=9)
    assert t.observe(_Bad(), 0) == 0.0
    assert t.stats()["retained_growth_errors"] == 1


def test_queue_and_flush_apply_to_the_right_record():
    t = RetainedGrowthTracker(coef=0.5, min_turn=9)
    recs = [{"game_id": 3, "reward": 0.0}, {"game_id": 4, "reward": 0.0}]
    assert t.queue(1, 4, 0.25) is True
    assert t.flush(recs) == 1
    assert recs[0]["reward"] == 0.0 and recs[1]["reward"] == pytest.approx(0.25)


def test_defaults_are_the_documented_ones():
    assert DEFAULT_NORMALIZER == 690.0
    assert DEFAULT_MIN_TURN == 9
    assert DEFAULT_CAP == 0.5
