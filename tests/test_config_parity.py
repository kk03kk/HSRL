"""配置漂移门禁 + 预检吞吐门禁（2026-11-07 GPU 验收 1091647 性能事故的防复发）。

背景：作业 1091647（强制升本 GPU 验收）两轮 PPO 用了 33m48s，而配方几乎相同的基线
1085078 三轮只用 4m54s。审计要求把两件事变成**机检项**而不是人工记忆：

1. 本轮配方相对快速基线（``config/p50-7357-train.json``）只允许改动显式批准的键；
2. 预检必须给出"单轮墙钟 / 采样速率"，超过阈值当场失败，不能静默吃掉 12 小时长训。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsbg.az.train.personal import FORCED_ALLOWED_CONFIG_DIFF as TRAIN_ALLOWED
from hsbg.az.train.personal import config_drift as train_config_drift
from tools.nus_submit import FORCED_ALLOWED_CONFIG_DIFF as SUBMIT_ALLOWED
from tools.nus_submit import config_drift as submit_config_drift
from workflows.run_7357_preflight import (MAX_ROUND_SECONDS, MIN_DECISIONS_PER_S,
                                          _check, check_throughput, history_tail)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/p50-7357-train.json"
FORCED = ROOT / "config/p50-forced-tier.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── 配置漂移 ─────────────────────────────────────────────────────────────
def test_allow_lists_are_identical():
    """训练侧与登录节点侧的两份允许清单必须逐项一致（否则一处放行一处拒绝）。"""
    assert set(TRAIN_ALLOWED) == set(SUBMIT_ALLOWED)


def test_shipped_forced_config_has_no_unapproved_drift():
    cfg, base = read(FORCED), read(BASELINE)
    assert train_config_drift(cfg, base) == []
    assert submit_config_drift(cfg, base) == []


def test_changing_a_performance_knob_is_flagged():
    cfg, base = read(FORCED), read(BASELINE)
    for key, value in (("games", 256), ("processes", 4), ("pipeline", 0),
                       ("max-actions-per-turn", 40), ("mirror-strict", 0),
                       ("minibatch", 2048), ("leaf-fused", 0)):
        broken = dict(cfg, **{key: value})
        assert train_config_drift(broken, base) == [key], key


def test_adding_an_unknown_key_is_flagged():
    cfg, base = read(FORCED), read(BASELINE)
    broken = dict(cfg, **{"rust-collector": 1})
    assert train_config_drift(broken, base) == ["rust-collector"]


def test_approved_keys_are_not_flagged():
    cfg, base = read(FORCED), read(BASELINE)
    broken = dict(cfg, **{"entropy-coef": 0.02, "anchor-beta": 0.02,
                          "forced-tier-plan": "3:2,6:3"})
    assert train_config_drift(broken, base) == []


def test_removing_a_baseline_key_is_flagged():
    cfg, base = read(FORCED), read(BASELINE)
    broken = dict(cfg)
    broken.pop("move-cap")
    assert train_config_drift(broken, base) == ["move-cap"]


# ── 吞吐门禁 ─────────────────────────────────────────────────────────────
def _round(**overrides) -> dict:
    row = dict(round=0, key="warm0", wall_s=98.0, collect_wall_s=11.5, decisions=98029,
               decisions_per_s=8594.3, ppo_s=9.0, samples=112751, updates=111)
    row.update(overrides)
    return row


def test_throughput_passes_on_baseline_like_rounds():
    result = check_throughput({"inputs_s": 12.0,
                               "rounds": [_round(round=0), _round(round=1),
                                          _round(round=2, key="train")]})
    assert result["ok"] is True
    assert result["rounds"][0]["decisions_per_s"] == pytest.approx(8594.3)


def test_throughput_fails_when_a_round_is_slow():
    slow = _round(wall_s=MAX_ROUND_SECONDS + 1.0, decisions_per_s=120.0)
    result = check_throughput({"rounds": [_round(), slow]})
    assert result["ok"] is False
    assert "单轮墙钟" in result["reason"]
    assert result["slow_rounds"] == [0]


def test_throughput_fails_when_sampling_rate_collapses():
    thin = _round(wall_s=100.0, decisions_per_s=MIN_DECISIONS_PER_S - 1)
    result = check_throughput({"rounds": [thin]})
    assert result["ok"] is False
    assert "采样速率" in result["reason"]
    assert result["thin_rounds"] == [0]


def test_throughput_fails_without_rounds():
    result = check_throughput({"rounds": []})
    assert result["ok"] is False and "计时" in result["reason"]


def test_thresholds_are_not_looser_than_three_times_baseline():
    """防退化：阈值不能为了防止失败而被悄悄放宽（基线 98 s / 8,600 dec·s⁻¹）。"""
    assert MAX_ROUND_SECONDS <= 300.0
    assert MIN_DECISIONS_PER_S >= 3000.0


# ── history 尾部读取 ─────────────────────────────────────────────────────
def test_history_tail_reads_last_iteration(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    rows = [{"type": "header", "args": {}},
            {"iter": 7357, "collect": {"wall_s": 25.4}, "ppo": {"seconds": 11.0}},
            {"iter": 7358, "collect": {"wall_s": 12.1}, "ppo": {"seconds": 9.2}}]
    with (train / "history.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tail = history_tail(train)
    assert tail["iter"] == 7358
    assert tail["collect"]["wall_s"] == 12.1


def test_history_tail_missing_returns_none(tmp_path: Path):
    assert history_tail(tmp_path / "nope") is None


def test_check_helper_shape():
    assert _check("x", True, "").keys() >= {"name", "ok", "reason"}
