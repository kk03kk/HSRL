"""`retained_gain_by_turn` 单测：把"留存随从每回合涨多少"算对。

背景（`docs/analysis/RUNS_20261113.md` §5.48）：真人 T10→11 每只留存随从平均 **+690 Σ**，
整局增长几乎全部来自这一项；模型侧此前没有逐卡属性，算不出来。
`evaluate.py` 现在写 `board_stats`（card_id → [atk, hp]），本函数把它按 seed 串成回合序列配对。

注：`workflows.eval_human_gap` 会 import torch（`hsbg.az.train.model`），
所以本地（torch-cpu 且无引擎）跑这套测试会被 skip，服务器上正常执行。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from workflows.eval_human_gap import retained_gain_by_turn  # noqa: E402


def _row(seed, turn, stats):
    return {"seed": seed, "turn": turn,
            "board_stats": {k: list(v) for k, v in stats.items()}}


def test_kept_minion_gain():
    rows = [
        _row(1, 10, {"A": [10, 10], "B": [5, 5]}),      # A: 20 → 30（+10）；B 被换掉
        _row(1, 11, {"A": [20, 10], "C": [50, 50]}),    # 只有 A 留存
    ]
    out = retained_gain_by_turn(rows, min_turn=6)
    assert out["10"]["games"] == 1
    assert out["10"]["kept_gain"] == 10.0      # (30 - 20) / 1 只
    assert out["10"]["kept_before"] == 20.0
    assert out["10"]["kept_after"] == 30.0


def test_averages_over_kept_only():
    rows = [
        _row(2, 10, {"A": [10, 10], "B": [10, 10], "X": [1, 1]}),
        _row(2, 11, {"A": [20, 20], "B": [10, 20]}),   # A +20, B +10 → 平均 +15
    ]
    out = retained_gain_by_turn(rows, min_turn=6)
    assert out["10"]["kept_gain"] == 15.0


def test_no_overlap_is_skipped():
    rows = [_row(3, 10, {"A": [10, 10]}), _row(3, 11, {"Z": [99, 99]})]
    assert retained_gain_by_turn(rows, min_turn=6) == {}


def test_multi_seed_accumulates():
    rows = [
        _row(1, 10, {"A": [10, 10]}), _row(1, 11, {"A": [12, 10]}),   # +2
        _row(2, 10, {"B": [10, 10]}), _row(2, 11, {"B": [18, 10]}),   # +8
    ]
    out = retained_gain_by_turn(rows, min_turn=6)
    assert out["10"]["games"] == 2
    assert out["10"]["kept_gain"] == 5.0        # (2 + 8) / 2


def test_min_turn_filters():
    rows = [_row(4, 4, {"A": [1, 1]}), _row(4, 5, {"A": [2, 1]})]
    assert retained_gain_by_turn(rows, min_turn=6) == {}
