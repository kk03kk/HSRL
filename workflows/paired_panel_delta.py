#!/usr/bin/env python3
"""同 seed/同 seat 的两个人类战团面板 → 配对 rank Δ 与 bootstrap 95% CI。

也支持个人 ``hsrl.eval/v1`` 输出：严格核对协议、seed和座位，按seed簇bootstrap，
同时给出rank/top4/win差分区间；16局smoke不能通过512局工程门槛。

``workflows/human_panel_eval.py --smoke --games N`` 的产物里有
``panels[0].details.per_seat.seat_k.model.rank_sequence``（逐局名次，顺序 = 该 seat 的
seed 递增）。本脚本按 ``(seat, 局序)`` 严格配对，报：

- ``mean_rank_before / mean_rank_after``、配对 ``delta = after − before``（名次越小越好，
  所以负值 = 变强）；
- 配对 bootstrap（默认 5000 次）95% CI、SE；
- 变好/变差局数、top4 率变化、非法/截断计数校验。

用法::

    python workflows/paired_panel_delta.py \
        --before runs/az_entity_s19/eval/human_baseline_s17_128.json \
        --after  runs/az_entity_s19/eval/human_005000000.json \
        --out    runs/az_entity_s19/eval/paired_5M.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def compare_personal(before, after, *, n_boot=5000, seed=20261102):
    """适配个人固定库协议；按seed成簇重采样，保留八座位的相关性。"""
    if not before.get("valid") or not after.get("valid"):
        raise ValueError("含无效评测，拒绝比较")
    if before.get("protocol_sha256") != after.get("protocol_sha256") or \
            before.get("protocol") != after.get("protocol"):
        raise ValueError("评测协议不同，拒绝配对")
    protocol = before["protocol"]
    seats = protocol["seats"]
    count = int(protocol["games_per_seat"])
    start = int(protocol["seed"])
    if n_boot < 2 or count < 1 or seats != list(range(8)):
        raise ValueError("需要8座位和有效bootstrap/seed数")
    def matrix(result):
        rows = {int(r["policy_seat"]): r for r in result["per_seat"]}
        if len(rows) != 8 or len(result["per_seat"]) != 8:
            raise ValueError("座位缺失或重复")
        out = []
        for s in seats:
            row = rows[s]
            if row["seed_start"] != start or row["seed_end"] != start + count - 1:
                raise ValueError("seed范围不匹配")
            ranks = row["rank_sequence"]
            if len(ranks) != count or any(int(r) not in range(1, 9) for r in ranks):
                raise ValueError("名次缺失或非法")
            if any(row.get(k, 0) for k in ("illegal_actions", "truncated_games", "truncated_combats")):
                raise ValueError("质量门未通过")
            out.append(ranks)
        return np.asarray(out, dtype=np.float64)
    a, b = matrix(before), matrix(after)
    rng = np.random.default_rng(seed)
    metrics = {"rank": b - a, "top4": (b <= 4).astype(float) - (a <= 4),
               "win": (b == 1).astype(float) - (a == 1)}
    intervals = {}
    for key, values in metrics.items():
        cluster_means = values.mean(axis=0)
        boot = np.array([cluster_means[rng.integers(0, count, count)].mean()
                         for _ in range(n_boot)])
        intervals[key] = np.quantile(boot, [.025, .975]).tolist()
    delta = float((b - a).mean())
    return {"schema": "hsrl.paired_delta/v1", "protocol_sha256": before["protocol_sha256"],
        "n": int(a.size), "independent_seeds": count, "bootstrap": n_boot,
        "mean_rank_before": float(a.mean()), "mean_rank_after": float(b.mean()),
        "delta": delta, "ci_low": intervals["rank"][0], "ci_high": intervals["rank"][1],
        "top4_delta": float(metrics["top4"].mean()), "top4_delta_ci95": intervals["top4"],
        "win_delta": float(metrics["win"].mean()), "win_delta_ci95": intervals["win"],
        "before_sha256": (before.get("checkpoint") or {}).get("sha256"),
        "after_sha256": (after.get("checkpoint") or {}).get("sha256"),
        "passed_hard_criterion": bool(a.size >= 512 and delta <= -.10 and intervals["rank"][1] < 0),
        "note": "负rank差表示候选改善；晋级参考不决定训练是否继续，也不替代多训练seed和封存测试"}


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not d.get("panels"):
        raise SystemExit(f"{path}: 没有 panels 段（面板失败？）")
    p = d["panels"][0]
    det = p.get("details") or {}
    per_seat = det.get("per_seat") or {}
    ranks = []
    seats = []
    for name in sorted(per_seat, key=lambda s: int(str(s).split("_")[-1])):
        row = per_seat[name]
        m = (row.get("model") or {})
        seq = m.get("rank_sequence")
        if seq is None:
            raise SystemExit(f"{path}: {name} 缺 rank_sequence")
        ranks.extend(int(r) for r in seq)
        seats.extend([int(str(name).split("_")[-1])] * len(seq))
    if not ranks:
        raise SystemExit(f"{path}: 逐局面板为空")
    return {"path": os.path.abspath(path), "ranks": np.asarray(ranks, np.int64),
            "seats": np.asarray(seats, np.int64),
            "mean_rank": float(np.mean(ranks)),
            "top4": float(np.mean(np.asarray(ranks) <= 4)),
            "n": int(len(ranks)),
            "checkpoint": d.get("checkpoint"),
            "sha256": d.get("checkpoint_sha256"),
            "config": d.get("config") or {},
            "illegal": int(p.get("illegal") or 0),
            "truncated": int(p.get("truncated") or 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.before, encoding="utf-8") as f:
        personal_before = json.load(f)
    with open(args.after, encoding="utf-8") as f:
        personal_after = json.load(f)
    if "protocol_sha256" in personal_before or "protocol_sha256" in personal_after:
        out = compare_personal(personal_before, personal_after,
                               n_boot=args.bootstrap, seed=args.seed)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        return 0

    a = _load(args.before)
    b = _load(args.after)
    if a["n"] != b["n"]:
        raise SystemExit(f"样本数不一致: before={a['n']} after={b['n']}")
    if not np.array_equal(a["seats"], b["seats"]):
        raise SystemExit("seat 顺序不一致：不是同一配对面板")
    diff = (b["ranks"] - a["ranks"]).astype(np.float64)
    rng = np.random.default_rng(int(args.seed))
    n = diff.size
    idx = rng.integers(0, n, size=(int(args.bootstrap), n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {
        "before": a["path"], "after": b["path"],
        "before_checkpoint": a["checkpoint"], "after_checkpoint": b["checkpoint"],
        "before_sha256": a["sha256"], "after_sha256": b["sha256"],
        "n": int(n), "bootstrap": int(args.bootstrap),
        "mean_rank_before": round(float(a["mean_rank"]), 6),
        "mean_rank_after": round(float(b["mean_rank"]), 6),
        "delta": round(float(diff.mean()), 6),
        "se_bootstrap": round(float(boot.std(ddof=1)), 6),
        "ci_low": round(float(lo), 6), "ci_high": round(float(hi), 6),
        "better_games": int((diff < 0).sum()),
        "worse_games": int((diff > 0).sum()),
        "tie_games": int((diff == 0).sum()),
        "top4_before": round(float(a["top4"]), 6),
        "top4_after": round(float(b["top4"]), 6),
        "top4_delta": round(float(b["top4"] - a["top4"]), 6),
        "illegal_before": a["illegal"], "illegal_after": b["illegal"],
        "truncated_before": a["truncated"], "truncated_after": b["truncated"],
        "passed_hard_criterion": bool(diff.mean() <= -0.10 and hi < 0.0),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
