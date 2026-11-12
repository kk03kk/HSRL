#!/usr/bin/env python3
"""Fail-fast health check for one league training segment.

Run on a compute node after every segment.  It reads only the tail of
``history.jsonl`` plus ``latest.json``; it never imports torch and is safe to
run between segments.  A non-zero exit means the caller must stop the long run
and preserve the workspace for diagnosis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

ZERO_KEYS = (
    "illegal_actions",
    "truncated_games",
    "invalid_games",
    "dropped_decisions",
    "search_failures",
)
MIRROR_ZERO_KEYS = (
    "mirror_missing",
    "mirror_errors",
    "mirror_pool_unbalanced",
    "mirror_pool_species_unbalanced",
)

#: 这些计数**允许**非零，但必须能被 ``mirror_rejected_by_reason`` 完全解释。
#: ``not_alive``（座位已死/已结束）与 ``pending_choice``（饰品/任务回合不许替换状态）
#: 都是预期内的情况，不是缺陷。
MIRROR_EXPLAINABLE_KEYS = ("mirror_rejected", "mirror_fallback")
PPO_FINITE_KEYS = (
    "total",
    "policy_loss",
    "value_loss",
    "anchor_kl",
    "entropy",
    "approx_kl",
    "clip_frac",
)


def read_last(path: Path) -> Dict[str, Any]:
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    if last is None:
        raise ValueError(f"empty history: {path}")
    return json.loads(last)


def finite(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check(history: Path, latest: Path, *, min_version: int = 0,
          expect_mirror: bool = True) -> Dict[str, Any]:
    issues: List[str] = []
    record = read_last(history)
    collect = record.get("collect") or {}
    ppo = record.get("ppo") or {}
    for key in ZERO_KEYS:
        if int(collect.get(key, 0) or 0) != 0:
            issues.append(f"collect.{key}={collect.get(key)}")
    decisions = int(collect.get("decisions", 0) or 0)
    if decisions <= 0:
        issues.append("collect.decisions=0")
    workers = collect.get("workers") or []
    worker_errors = sum(int(w.get("errors", 0) or 0) for w in workers)
    if worker_errors:
        issues.append(f"worker_errors={worker_errors}")
    if expect_mirror:
        attempt = int(collect.get("mirror_attempt", 0) or 0)
        applied = int(collect.get("mirror_applied", 0) or 0)
        if attempt <= 0:
            issues.append("mirror_attempt=0")
        # 被拒的两个原因是**预期内**的：座位已经死亡/已结束，或该座位正卡在
        # pending choice（饰品/任务回合不允许替换状态）。它们不是缺陷，但必须
        # **可解释**——真正要拦的是"没解释的差额"，那才代表注入在静默失效。
        # 2026-11-12：只有 not_alive + pending_choice 两类时，旧门禁会误判为失败。
        reasons = collect.get("mirror_rejected_by_reason") or {}
        try:
            explained = sum(int(v or 0) for v in
                            (reasons.get("not_alive", 0), reasons.get("pending_choice", 0)))
        except (AttributeError, TypeError):
            explained = 0
        unexplained = (attempt - applied) - explained
        if applied <= 0:
            issues.append(f"mirror_applied={applied} (no frame was ever injected)")
        elif unexplained > max(1, int(0.01 * attempt)):
            issues.append(
                f"mirror_applied={applied} != attempt={attempt} and "
                f"{unexplained} rejection(s) are unexplained "
                f"(reasons={reasons})")
        for key in MIRROR_ZERO_KEYS:
            value = int(collect.get(key, 0) or 0)
            # ``mirror_missing`` 允许极小比例：库只覆盖到某个回合（本库 turn_max=24），
            # 个别对局会走到更晚，那里没有帧可取。0.1% 以内且 <=2 次视为噪声，
            # 超过就说明库的回合覆盖明显不够，仍要拦。
            if key == "mirror_missing":
                if value > max(2, int(0.001 * attempt)):
                    issues.append(f"collect.{key}={value} (library turn coverage gap?)")
                continue
            if value != 0:
                issues.append(f"collect.{key}={value}")
        for key in MIRROR_EXPLAINABLE_KEYS:
            value = int(collect.get(key, 0) or 0)
            if value and not explained:
                issues.append(
                    f"collect.{key}={value} but no rejection reason was recorded")
    epochs = ppo.get("epochs") or []
    if not epochs:
        issues.append("ppo.epochs is empty")
    for idx, epoch in enumerate(epochs):
        for key in PPO_FINITE_KEYS:
            if not finite(epoch.get(key)):
                issues.append(f"ppo.epochs[{idx}].{key}={epoch.get(key)!r}")
    if not finite(ppo.get("lr")):
        issues.append(f"ppo.lr={ppo.get('lr')!r}")
    latest_data: Dict[str, Any] = {}
    if not latest.is_file():
        issues.append(f"missing latest.json: {latest}")
    else:
        latest_data = json.loads(latest.read_text(encoding="utf-8"))
        version = int(latest_data.get("version", -1) or -1)
        if version <= int(min_version):
            issues.append(f"latest.version={version} <= previous={min_version}")
        for key in ("model", "trainer"):
            path = latest_data.get(key)
            if not path or not Path(path).is_file():
                issues.append(f"latest.{key} missing: {path}")
    return {
        "ok": not issues,
        "issues": issues,
        "history": str(history),
        "latest": str(latest),
        "iter": record.get("iter"),
        "decisions": decisions,
        "worker_errors": worker_errors,
        "mirror_attempt": int(collect.get("mirror_attempt", 0) or 0),
        "mirror_applied": int(collect.get("mirror_applied", 0) or 0),
        "trainer_updates": ppo.get("trainer_updates"),
        "lr": ppo.get("lr"),
        "latest_version": latest_data.get("version"),
        "latest_cum_decisions": latest_data.get("cum_decisions"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--latest", type=Path, required=True)
    ap.add_argument("--min-version", type=int, default=0)
    ap.add_argument("--expect-mirror", type=int, choices=[0, 1], default=1)
    args = ap.parse_args()
    result = check(args.history, args.latest, min_version=args.min_version,
                   expect_mirror=bool(args.expect_mirror))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
