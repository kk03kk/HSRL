"""训练健康门（``hsbg.az.train.health_gate``）的单测。

钉住"硬门 fail-closed / 软指标只警告"这条边界：审计 P2-11 发现
``TRAINING_RECIPE §5`` 的红线**一条都没实现**，而把"策略选择"（熵/KL/吞吐）
误做成自动停机又会违反"不因超参看起来不对就改配方"的授权。
"""

from __future__ import annotations

import pytest

from hsbg.az.train.health_gate import (HealthLimits, format_failures,
                                       hard_collect_failures, hard_update_failures,
                                       soft_warnings)


# ────────────────────────── 采集侧硬门 ──────────────────────────
def test_clean_collect_passes():
    meta = {"illegal_actions": 0, "invalid_games": 0, "truncated_games": 0}
    assert hard_collect_failures(meta) == []


@pytest.mark.parametrize("key", ["illegal_actions", "invalid_games",
                                 "truncated_games", "truncated_combats",
                                 "engine_stopped"])
def test_any_engineering_error_blocks(key):
    out = hard_collect_failures({key: 1})
    assert out and key in out[0]


def test_missing_keys_are_tolerated():
    """老作业/新加字段不能因为缺键就判失败。"""
    assert hard_collect_failures({}) == []
    assert hard_collect_failures({"decisions": 10}) == []


def test_mirror_apply_rate_collapse_blocks():
    meta = {"mirror_attempt": 1000, "mirror_applied": 500}
    out = hard_collect_failures(meta)
    assert out and "mirror_applied/attempt" in out[0]


def test_mirror_counts_inside_collect_are_found():
    meta = {"collect": {"mirror_attempt": 100, "mirror_applied": 100}}
    assert hard_collect_failures(meta) == []
    bad = {"collect": {"mirror_attempt": 100, "mirror_applied": 10}}
    assert hard_collect_failures(bad)


def test_mirror_check_can_be_disabled_for_pure_bot_runs():
    limits = HealthLimits(mirror_apply_min=None)
    assert hard_collect_failures({"mirror_attempt": 10, "mirror_applied": 0},
                                 limits) == []


# ────────────────────────── 更新侧硬门 ──────────────────────────
def test_non_finite_loss_blocks():
    stats = {"epochs": [{"total": float("nan"), "policy_loss": 0.1,
                         "value_loss": 1.0}]}
    out = hard_update_failures(stats)
    assert out and "非有限" in out[0]


def test_illegal_prob_leak_blocks():
    out = hard_update_failures({"epochs": [{"illegal_prob_max": 1e-3}]})
    assert out and "illegal_prob_max" in out[0]


def test_flat_stats_without_epochs_are_supported():
    assert hard_update_failures({"total": 0.7, "illegal_prob_max": 0.0}) == []


def test_healthy_update_passes():
    stats = {"epochs": [{"total": 0.72, "policy_loss": 0.004,
                         "value_loss": 1.42, "illegal_prob_max": 0.0,
                         "entropy": 0.89, "approx_kl": 0.008}]}
    assert hard_update_failures(stats) == []
    assert soft_warnings(stats) == []


# ────────────────────────── 软警告：只记录，不停机 ──────────────────────────
def test_low_entropy_is_only_a_warning():
    """熵崩塌（< 0.05）才告警。默认阈值 0.8 会误报：本游戏早期只有 2 个合法
    动作，最大熵就是 ln2 = 0.693（2026-11-17 smoke 实测每轮都报）。"""
    stats = {"epochs": [{"entropy": 0.01}]}
    assert hard_update_failures(stats) == []
    assert any("entropy" in w for w in soft_warnings(stats))
    # ln2 级别的"正常"熵不该告警
    assert soft_warnings({"epochs": [{"entropy": 0.69}]}) == []


def test_high_kl_is_only_a_warning():
    stats = {"epochs": [{"approx_kl": 0.09}]}
    assert hard_update_failures(stats) == []
    assert any("approx_kl" in w for w in soft_warnings(stats))


def test_low_throughput_warning_is_off_by_default():
    """吞吐告警默认关闭（0）：小规模 smoke 必然低于任何固定阈值。"""
    meta = {"decisions_per_s": 900}
    assert hard_collect_failures(meta) == []
    assert soft_warnings({}, meta) == []
    # 显式打开后才告警
    lim = HealthLimits(throughput_warn_below=1500.0)
    assert any("decisions_per_s" in w for w in soft_warnings({}, meta, limits=lim))


# ────────────────────────── 参数与格式化 ──────────────────────────
def test_limits_from_args_defaults_and_override():
    class A:
        health_mirror_apply_min = 0.0        # 0 = 关掉该项
    lim = HealthLimits.from_args(A())
    assert lim.mirror_apply_min is None
    assert lim.illegal_actions == 0 and lim.entropy_warn_below == 0.05
    assert lim.throughput_warn_below == 0.0


def test_format_failures_is_empty_when_clean():
    assert format_failures([]) == ""
    msg = format_failures(["illegal_actions=2 > 允许 0"])
    assert "health-gate" in msg and "illegal_actions" in msg
