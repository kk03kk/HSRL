"""训练循环里**健康门接线**的静态回归（审计 P2-11）。

``tests/test_health_gate.py`` 已经覆盖门本身的判定逻辑；这里只防"门被悄悄摘掉"：
采集后与更新后各必须有一次调用，且 ``--health-gate`` 默认开。

这类"读源码断言"的测试在本仓库已有先例（``tests/test_teacher_bc_static.py``）；
它不是行为测试，但能挡住"重构时把调用删了、文档还写着有门"的静默退化。
"""

from __future__ import annotations

from pathlib import Path

SRC = (Path(__file__).resolve().parents[1]
       / "workflows" / "train_entity_sync.py").read_text(encoding="utf-8")


def test_collect_gate_is_called_before_update():
    assert "hard_collect_failures(collect_meta, health_limits)" in SRC


def test_update_gate_is_called_after_update():
    assert "hard_update_failures(stats, health_limits)" in SRC


def test_soft_warnings_are_recorded_not_fatal():
    assert "soft_warnings(stats, collect_meta, health_limits)" in SRC
    # 软警告必须进 history（否则没人看得见）
    assert '"health": {"soft_warnings"' in SRC


def test_health_gate_defaults_to_on():
    from hsbg.az.train.cli import build_parser
    assert build_parser().parse_args([]).health_gate == 1


def test_teacher_loss_normalize_flag_exists_and_defaults_to_legacy():
    from hsbg.az.train.cli import build_parser
    assert build_parser().parse_args([]).teacher_loss_normalize == 'labeled'


def test_limits_are_built_once_from_args():
    assert "health_limits = HealthLimits.from_args(args)" in SRC
