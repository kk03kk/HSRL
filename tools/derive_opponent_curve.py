#!/usr/bin/env python3
"""从一次双模型配对评测报告推导「稍弱于模型」的对手战团库曲线（P50 口径）。

用途：上一轮训练库（``config/opponent-p50-next.planned.json``）把 T12–T19 的目标定到
起点的 **3.1–21.6 倍**、真实 bot 池的 **8–50 倍**，模型在必败区里拿不到梯度。本工具按
**被选模型自己的逐回合 P50** 生成一条始终不超过它的阶梯曲线。

规则（写进产物，供 ``build_p50_bank.py`` 直接使用 ``stage_targets``）：

1. **T1–T8 不动**：沿用旧曲线的 train 目标（本来就≈真实 bot 水平，且都不高于模型，
   只有 T1 的 5 vs 4 有 1 点差）；
2. **T9–T19 = round(ratio × 模型自身 P50)**，四档阶梯对应训练场换库点
   ``0 / 40M / 80M / 120M``，``dev/test`` 用同级（1.00×）作评测参照；
3. 硬性不变量：逐回合阶梯单调不减、train 档 ≤1.00×模型、T1–T8 与旧曲线逐值相同。

用法::

    python tools/derive_opponent_curve.py \
        --report artifacts/imports/nus-duel-eval-1092254/report.json \
        --model A --out config/opponent-p50-slightly-weaker.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hsai.opponent_curve/v2"
PROFILE = "p50-v2-weaker"
#: 阶梯：与 ``workflows/run_forced_tier_long.SCHEDULE`` 同界（0/40M/80M/120M）
STAGES = ("train-start", "train-ramp1", "train-ramp2", "train", "dev", "test")
DEFAULT_RATIOS = {"train-start": 0.70, "train-ramp1": 0.80, "train-ramp2": 0.88,
                  "train": 0.95, "dev": 1.00, "test": 1.00}
#: 阶梯必须单调不减（旧库到新库都按这个顺序加难度）
LADDER = ("train-start", "train-ramp1", "train-ramp2", "train", "dev", "test")
EARLY_TURNS = range(1, 9)      # T1–T8 沿用旧值
LATE_TURNS = range(9, 20)      # T9–T19 按比例


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def model_p50(report: Dict[str, Any], label: str) -> Dict[int, float]:
    """评测报告里某模型逐回合的 ``turns[turn].body_p50``（条件分布，随淘汰下降）。"""
    turns = report["results"][label]["turns"]
    out = {int(t): float(row["body_p50"]) for t, row in turns.items()
           if row.get("body_p50") is not None}
    if not out:
        raise ValueError(f"报告里没有模型 {label} 的逐回合 P50")
    return out


def build_curve(*, report: Dict[str, Any], old_curve: Dict[str, Any], label: str,
                ratios: Dict[str, float], report_path: Path) -> Dict[str, Any]:
    model = model_p50(report, label)
    old_train = {int(k): int(v) for k, v in old_curve["train_targets"].items()}
    missing = [t for t in LATE_TURNS if t not in model]
    if missing:
        raise ValueError(f"模型 {label} 缺少 T9–T19 的 P50：{missing}")
    stage_targets: Dict[str, Dict[str, int]] = {}
    for stage in STAGES:
        ratio = float(ratios[stage])
        row = {}
        for turn in range(1, 20):
            row[str(turn)] = (int(old_train[turn]) if turn in EARLY_TURNS
                              else int(round(ratio * model[turn])))
        stage_targets[stage] = row
    # ── 不变量 ────────────────────────────────────────────────────────────
    for turn in range(1, 20):
        ladder = [stage_targets[s][str(turn)] for s in LADDER]
        if any(b < a for a, b in zip(ladder, ladder[1:])):
            raise ValueError(f"T{turn} 阶梯不单调：{dict(zip(LADDER, ladder))}")
    for turn in LATE_TURNS:
        if stage_targets["train"][str(turn)] > model[turn] + 1e-9:
            raise ValueError(f"T{turn} train 目标 {stage_targets['train'][str(turn)]} "
                             f"超过模型自身 P50 {model[turn]}")
    for turn in EARLY_TURNS:
        if stage_targets["train"][str(turn)] != old_train[turn]:
            raise ValueError(f"T{turn} 早期目标被改动")
    # ── 记录每档对模型的比例，便于审计 ────────────────────────────────────
    ratios_observed = {
        stage: {str(t): round(stage_targets[stage][str(t)] / model[t], 4)
                for t in LATE_TURNS} for stage in STAGES}
    return {
        "schema": SCHEMA,
        "status": "implemented_pending_formal_nus_build_and_gpu_acceptance",
        "calibration_profile": PROFILE,
        "date": time.strftime("%Y-%m-%d"),
        "primary_statistic": "P50(sum_attack+sum_health) per board",
        "reference": {
            "source": "artifacts/imports/nus-duel-eval-1092254/report.json (NUS Slurm 1092254)",
            "report": str(report_path.as_posix()),
            "report_sha256": sha256_file(report_path),
            "protocol_sha256": report.get("protocol_sha256"),
            "model": report["results"][label]["model"],
            "model_label": label,
            "model_file_sha256": report["results"][label]["file_sha256"],
            "games": int(report.get("totals", {}).get("games") or 0),
            "model_p50": {str(t): model[t] for t in sorted(model)},
            "note": "模型自身逐回合 P50 取自该评测；它是对局条件的（越晚的回合只剩活下来的局），"
                    "但正是对手需要达到的'存活棋盘'水平。",
        },
        "stage_ratios": {stage: float(ratios[stage]) for stage in STAGES},
        "stage_targets": stage_targets,
        # 兼容旧字段：build_p50_bank 的旧路径按 split 取整条曲线
        "train_targets": stage_targets["train"],
        "dev_test_targets": stage_targets["dev"],
        "early_turns_policy": "T1–T8 沿用旧曲线 train_targets（≈真实 bot 水平）",
        "ratios_observed": ratios_observed,
        "changes_running_training": True,
        "performance_stop_policy": "正常名次波动不停止；只保留工程与数据完整性门禁",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="A", choices=["A", "B"])
    parser.add_argument("--old-curve", type=Path,
                        default=ROOT / "config/opponent-p50-next.planned.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "config/opponent-p50-slightly-weaker.json")
    parser.add_argument("--ratio-start", type=float, default=DEFAULT_RATIOS["train-start"])
    parser.add_argument("--ratio-ramp1", type=float, default=DEFAULT_RATIOS["train-ramp1"])
    parser.add_argument("--ratio-ramp2", type=float, default=DEFAULT_RATIOS["train-ramp2"])
    parser.add_argument("--ratio-train", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--ratio-devtest", type=float, default=DEFAULT_RATIOS["dev"])
    args = parser.parse_args(argv)
    ratios = {"train-start": args.ratio_start, "train-ramp1": args.ratio_ramp1,
              "train-ramp2": args.ratio_ramp2, "train": args.ratio_train,
              "dev": args.ratio_devtest, "test": args.ratio_devtest}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    old_curve = json.loads(args.old_curve.read_text(encoding="utf-8"))
    curve = build_curve(report=report, old_curve=old_curve, label=args.model,
                        ratios=ratios, report_path=args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curve, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    model = curve["reference"]["model_p50"]
    print(f"[curve] profile={curve['calibration_profile']} model={curve['reference']['model']}")
    print("[curve] turn |   model | " + " | ".join(f"{s:>11}" for s in STAGES))
    for turn in range(1, 20):
        cells = " | ".join(f"{curve['stage_targets'][s][str(turn)]:>11}" for s in STAGES)
        print(f"[curve] T{turn:<4} | {float(model[str(turn)]):>7.0f} | {cells}")
    print(f"[curve] -> {args.out}  sha256={sha256_text(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
