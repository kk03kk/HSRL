"""采集质量门的回归测试：**只有本该有镜像时才要求 mirror_applied > 0**。

2026-11-14 事故：`l9anneal2` 的退火课程把 `mirror_p` 按 ramp 降到 0.0，
旧写法把"`mirror_applied=0`"一律判成采集质量失败并 abort —— 恰好在 iter 300（ramp 终点）
把一条跑了 300 轮的臂误杀。修复后：`config["mirror_p"]` 是 ramp 之后的**有效值**，
它等于 0 时不该要求镜像；>0 时仍然保留原有保护。
"""
from __future__ import annotations

import pytest

from hsbg.az.train.collect_diagnostics import require_valid_mirror_collect


def _meta(mirror_applied: int) -> dict:
    return {
        "mirror_applied": mirror_applied,
        "mirror_rejected_by_reason": {},
        "mirror_strict_counts": {},
        "games": 256,
    }


def test_annealed_to_zero_does_not_trip_the_gate(tmp_path):
    """ramp 把有效 mirror_p 降到 0 ⇒ 不该报 mirror_applied=0。"""
    cfg = {"mirror_p": 0.0, "mirror_strict": 0}
    require_valid_mirror_collect(_meta(0), str(tmp_path), iteration=300, config=cfg)


def test_expected_mirrors_still_enforced(tmp_path):
    """有效 mirror_p > 0 且一个镜像都没戴上 ⇒ 必须仍然报错（保护不能丢）。"""
    cfg = {"mirror_p": 0.5, "mirror_strict": 0}
    with pytest.raises(RuntimeError) as exc:
        require_valid_mirror_collect(_meta(0), str(tmp_path), iteration=7, config=cfg)
    assert "mirror_applied" in str(exc.value)


def test_partial_anneal_with_mirrors_applied_is_fine(tmp_path):
    cfg = {"mirror_p": 0.3, "mirror_strict": 0}
    require_valid_mirror_collect(_meta(120), str(tmp_path), iteration=200, config=cfg)
