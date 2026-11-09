"""采集质量门：失败时先保存计数和现场，再拒绝 PPO 更新。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .common import to_jsonable


def probe_config(config: Dict[str, Any], *, iteration: int):
    """与单进程训练入口保持相同的迭代种子及采集参数；不创建优化器。"""
    from .ppo import CollectConfig
    if iteration < 0:
        raise ValueError("iteration 必须 >= 0")
    return CollectConfig(games=int(config.get("games", 8)),
        seed=int(config.get("seed", 100000000)) + int(iteration) * 1000,
        opponent="bot", behavior="net", lockstep=True, fast_act=True,
        max_steps=20000, max_actions_per_turn=int(config.get("max-actions-per-turn", 80)),
        mirror_seats=7, mirror_p=1.0,
        mirror_sample=str(config.get("mirror-sample", "random")))


def require_valid_mirror_collect(meta: Dict[str, Any], out: str, *,
                                 iteration: int, config: Dict[str, Any]) -> None:
    keys = ("illegal_actions", "truncated_games", "mirror_errors",
            "mirror_pool_unbalanced", "mirror_pool_species_unbalanced",
            "truncated_combats", "truncated_recruit_contexts", "turn_limit_end", "engine_stopped")
    issues = {k: int(meta.get(k, 0)) for k in keys if meta.get(k, 0)}
    # 非预期引用/载荷拒绝不能被mirror_errors=0掩盖；预期pending/死亡仍允许。
    from .mirror_snapshots import EXPECTED_REJECT_REASONS
    unexpected = {k: int(v) for k, v in meta.get("mirror_rejected_by_reason", {}).items()
                  if k not in EXPECTED_REJECT_REASONS and v}
    if unexpected:
        issues["unexpected_mirror_rejections"] = sum(unexpected.values())
    if not meta.get("mirror_applied", 0):
        issues["mirror_applied"] = 0
    if config.get("mirror_strict"):
        counts = meta.get("mirror_strict_counts") or {}
        if (meta.get("mirror_fallback") or meta.get("mirror_rejected")
                or meta.get("mirror_applied") != counts.get("eligible_live_seats")
                or not (counts.get("live_combats", 0) + counts.get("ghost_combats", 0))):
            issues["strict_mirror_coverage"] = 1
    if not issues:
        return
    path = Path(out) / "diagnostics" / f"collect_{int(iteration):06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(to_jsonable({"schema": "hsrl.collect_failure/v1",
        "iteration": int(iteration), "issues": issues, "config": config,
        "collect": meta}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    summary = ", ".join(f"{k}={v}" for k, v in issues.items())
    raise RuntimeError(f"采集质量门失败：{summary}；拒绝 PPO 更新；诊断：{path}")
