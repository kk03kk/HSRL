#!/usr/bin/env python
"""两/多条训练臂的**同口径对照**读数（只用标准库）。

为什么需要它
------------
单条 history 的曲线只能说明"它在动"；要判断"新课程是不是真的更好"，必须把两条臂
放在**同一张表**上，并且看出三件不同的事：

1. **行为**（`tel_*`）：自建板面、升本节奏、活到第几回合、掉了多少血——与对手无关；
2. **难度**（`arena_*`）：控制器把对手顶到了什么强度/分位——它上升说明"能打赢更强的
   真人战团"，是**唯一不会被"对手变弱"污染**的进步刻度（训练胜率会被控制器钉在目标附近）；
3. **健康**（`ppo.*`）：KL / 熵 / 损失，确认不是靠崩溃换来的数字。

用法::

    python tools/compare_runs.py artifacts/runs/A/train artifacts/runs/B/train [--tail 20]
    python tools/compare_runs.py A=trainA B=trainB --json

输出是等宽表 + 一段 `[对照]` 判读。缺数据一律 `-`，绝不因为某条臂少键就崩。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: 对照用的指标： (键, 显示名, 小数位, 分组)。分组决定打几张表。
METRICS: Tuple[Tuple[str, str, int, str], ...] = (
    ("tel_own_str_8", "自建板面 T8", 0, "行为"),
    ("tel_own_str_10", "自建板面 T10", 0, "行为"),
    ("tel_own_str_12", "自建板面 T12", 0, "行为"),
    ("tel_tier_8", "酒馆等级 T8", 2, "行为"),
    ("tel_tier_10", "酒馆等级 T10", 2, "行为"),
    ("tel_tier_12", "酒馆等级 T12", 2, "行为"),
    ("tel_reach9", "到达 T9 占比", 3, "行为"),
    ("tel_reach12", "到达 T12 占比", 3, "行为"),
    ("tel_cum_dmg_12", "累计掉血 T12", 1, "行为"),
    ("learner_final_str_mean", "终局板面（粗）", 0, "行为"),
    ("arena_early_str", "对手强度 early", 0, "难度"),
    ("arena_mid_str", "对手强度 mid", 0, "难度"),
    ("arena_late_str", "对手强度 late", 0, "难度"),
    ("arena_early_win", "胜率 early", 3, "难度"),
    ("arena_mid_win", "胜率 mid", 3, "难度"),
    ("arena_late_win", "胜率 late", 3, "难度"),
    ("arena_reach_target", "前期压力目标", 2, "难度"),
    ("arena_ratchet", "棘轮档位", 0, "难度"),
    ("human_board_applied", "注入成功（座位次数）", 0, "难度"),
    ("human_board_arena_dead", "注入被淘汰席位数", 0, "难度"),
    ("approx_kl", "approx_kl", 4, "健康"),
    ("entropy", "熵", 3, "健康"),
    ("anchor_kl", "anchor_kl", 4, "健康"),
    ("loss", "loss", 4, "健康"),
)

#: 这些键也可能落在 ppo/profile 段里。
_PPO_KEYS = frozenset({"approx_kl", "entropy", "anchor_kl", "loss", "adv_std"})


def _get(row: Mapping[str, Any], key: str) -> Any:
    """依次在本层 / collect / stat / stats / meta / ppo / profile 里找键。"""
    for where in (row, row.get("collect"), row.get("stat"), row.get("stats"),
                  row.get("meta"), row.get("ppo"), row.get("profile")):
        if isinstance(where, Mapping) and key in where:
            return where[key]
    return None


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def load_history(path: str) -> List[Dict[str, Any]]:
    """读 history.jsonl（也接受 run 目录）；坏行跳过，返回可解析的行。"""
    p = Path(path)
    if p.is_dir():
        for cand in (p / "history.jsonl", p / "train" / "history.jsonl"):
            if cand.is_file():
                p = cand
                break
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], tail: int = 20) -> Dict[str, Any]:
    """把一条臂压成 ``{key: 末尾均值}`` + 首末趋势（首末都取窗口内的真实行）。"""
    window = list(rows[-tail:]) if tail > 0 else list(rows)
    out: Dict[str, Any] = {"rows": len(rows), "window": len(window), "mean": {}, "first": {},
                           "last": {}, "trend": {}}
    for key, _label, _digits, _group in METRICS:
        vals = [_num(_get(r, key)) for r in window]
        vals = [v for v in vals if v is not None]
        out["mean"][key] = (sum(vals) / len(vals)) if vals else None
        if vals:
            out["first"][key], out["last"][key] = vals[0], vals[-1]
            out["trend"][key] = (vals[-1] - vals[0])
    return out


def _fmt(value: Any, digits: int) -> str:
    v = _num(value)
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _delta(a: Any, b: Any, digits: int) -> str:
    va, vb = _num(a), _num(b)
    if va is None or vb is None:
        return "-"
    return f"{vb - va:+.{digits}f}"


def render(names: Sequence[str], sums: Sequence[Dict[str, Any]]) -> List[str]:
    """等宽表：每个指标一行，每条臂两列（窗口均值 / 首→末），外加 A→B 的差。"""
    lines: List[str] = []
    label_w = max(len(m[1]) for m in METRICS) + 2
    arm_w = 18
    head = f"{'指标':<{label_w}}" + "".join(f"{n:>{arm_w}}" for n in names)
    if len(sums) >= 2:
        head += f"{'末臂-首臂':>14}"
    for group in ("行为", "难度", "健康"):
        lines.append(f"\n[{group}]  臂列 = 末尾窗口均值（括号内 首→末）")
        lines.append(head)
        lines.append("-" * _w(head))
        for key, label, digits, g in METRICS:
            if g != group:
                continue
            row = f"{label:<{label_w}}"
            for s in sums:
                mean = _fmt(s["mean"].get(key), digits)
                first = _fmt(s["first"].get(key), digits)
                last = _fmt(s["last"].get(key), digits)
                row += f"{mean + ' (' + first + '→' + last + ')':>{arm_w}}"
            if len(sums) >= 2:
                row += f"{_delta(sums[0]['mean'].get(key), sums[-1]['mean'].get(key), digits):>14}"
            lines.append(row)
    return lines


def _w(text: str) -> int:
    """等宽宽度：CJK 记 2 列。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def verdict(names: Sequence[str], sums: Sequence[Dict[str, Any]]) -> List[str]:
    """``[对照]``：只看**方向**，不吹单个数字（转型期名次会掉，见 D-054）。"""
    if len(sums) < 2:
        return ["[对照] 少于两条臂，只能看单臂读数"]
    a, b = sums[0], sums[-1]
    out = [f"[对照] {names[-1]} vs {names[0]}（末尾窗口均值差）"]
    checks = [
        ("自建板面 T12", "tel_own_str_12", 0, "↑"),
        ("到达 T12 占比", "tel_reach12", 3, "↑"),
        ("酒馆等级 T12", "tel_tier_12", 2, "↑"),
        ("累计掉血 T12", "tel_cum_dmg_12", 1, "↓"),
    ]
    for label, key, digits, want in checks:
        d = _delta(a["mean"].get(key), b["mean"].get(key), digits)
        good = "-"
        va, vb = _num(a["mean"].get(key)), _num(b["mean"].get(key))
        if va is not None and vb is not None:
            better = (vb > va) if want == "↑" else (vb < va)
            good = "更好" if better else "更差"
        out.append(f"  {label}: {_fmt(a['mean'].get(key), digits)} → "
                   f"{_fmt(b['mean'].get(key), digits)}（{d}）{good}")
    late_a, late_b = _num(a["mean"].get("arena_late_str")), _num(b["mean"].get("arena_late_str"))
    if late_a is not None or late_b is not None:
        out.append(f"  对手强度 late: {_fmt(late_a, 0)} → {_fmt(late_b, 0)}"
                   "（上升=能打赢更强的真人战团，这是最硬的进步刻度）")
    out.append("  注意：训练胜率会被难度控制器钉在目标附近，**不能**当进步证据；"
               "名次要看 768 局冻结面板。")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="compare training arms on one scale")
    ap.add_argument("runs", nargs="+", help="path 或 name=path（run 目录 / history.jsonl）")
    ap.add_argument("--tail", type=int, default=20, help="末尾窗口行数（默认 20，0=全部）")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON")
    a = ap.parse_args(argv)
    names: List[str] = []
    sums: List[Dict[str, Any]] = []
    for spec in a.runs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).parent.name or spec, spec
        rows = load_history(path)
        names.append(name)
        sums.append(summarize(rows, tail=a.tail))
    if a.json:
        json.dump({"schema": "hsrl.compare_runs/v1", "arms": names, "summaries": sums},
                  sys.stdout, ensure_ascii=False, indent=1, default=str)
        sys.stdout.write("\n")
        return 0
    print("对照臂: " + ", ".join(f"{n}(rows={s['rows']})" for n, s in zip(names, sums)))
    for line in render(names, sums):
        print(line)
    print()
    for line in verdict(names, sums):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
