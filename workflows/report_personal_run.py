#!/usr/bin/env python3
"""仅用标准库读取训练日志或 cProfile；可在登录节点运行，不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pstats


def summarize(rows):
    """只使用每轮的单调时钟差，恢复前后的 seconds 不相减。"""
    totals = Counter()
    rejected = Counter()
    strict = Counter()
    iterations = []
    missing_timeline = 0
    for row in rows:
        if row.get("iter") is None:
            continue  # history 里的启动元数据行
        collect, ppo, profile = (row.get(k) or {} for k in ("collect", "ppo", "profile"))
        decisions = int(collect["decisions"])
        collect_s, ppo_s = (float(profile[k]) for k in ("collect_s", "ppo_s"))
        if decisions < 0 or any(not math.isfinite(t) or t < 0 for t in (collect_s, ppo_s)):
            raise ValueError("决策数或耗时非法")
        totals.update(decisions=decisions, collect_s=collect_s, ppo_s=ppo_s,
                      processed_samples=int(ppo["processed_samples"]),
                      optimizer_steps=int(ppo.get("updates", 0)),
                      kl_early_stop_iterations=int(bool(ppo.get("stopped_early"))))
        for key in ("forced_end_turns", "illegal_actions", "truncated_games", "mirror_errors",
                    "engine_quality_checked_games", "truncated_combats", "truncated_recruit_contexts",
                    "turn_limit_end", "engine_stopped",
                    "mirror_missing", "mirror_attempt", "mirror_applied", "mirror_fallback",
                    "mirror_pool_unbalanced", "mirror_pool_species_unbalanced",
                    "mirror_pool_preexisting", "mirror_pool_species_preexisting"):
            totals[key] += int(collect.get(key, 0))
        rejected.update(collect.get("mirror_rejected_by_reason") or {})
        strict.update(collect.get("mirror_strict_counts") or {})
        timeline = profile.get("timeline") or {}
        names = ("t_iter_start", "t_collect_done", "t_ppo_start", "t_ppo_done", "t_history_done")
        if all(k in timeline for k in names):
            times = [float(timeline[k]) for k in names]
            if any(not math.isfinite(t) for t in times) or times != sorted(times):
                raise ValueError("单轮 timeline 非单调或非有限值")
            totals["timed_decisions"] += decisions
            totals["iteration_s"] += times[-1] - times[0]
            totals["post_ppo_s"] += times[-1] - times[-2]
        else:
            missing_timeline += 1
        iterations.append(int(row["iter"]))
    if not iterations:
        raise ValueError("没有完整训练记录")
    def rate(n, d):
        return totals[n] / totals[d] if totals[d] else None
    unexpected = {k: v for k, v in rejected.items() if k not in {"pending_choice", "not_alive", "dying"} and v}
    return {"schema": "hsrl.run_report/v1", "iterations": iterations, "totals": dict(totals),
            "rates": {"collect_decisions_per_s": rate("decisions", "collect_s"),
                      "ppo_processed_samples_per_s": rate("processed_samples", "ppo_s"),
                      "collect_plus_ppo_decisions_per_s": totals["decisions"] / (totals["collect_s"] + totals["ppo_s"])
                          if totals["collect_s"] + totals["ppo_s"] else None,
                      "iteration_decisions_per_s": rate("timed_decisions", "iteration_s"),
                      "forced_end_turns_per_decision": rate("forced_end_turns", "decisions")},
            **({"mirror_strict_counts": dict(strict)} if strict else {}),
            "mirror_rejected_by_reason": dict(rejected), "unexpected_mirror_rejections": unexpected,
            "missing_timeline_iterations": missing_timeline,
            "notes": ["速度为样本总数/耗时总和；不是每轮速度的算术平均。",
                      "iteration 只覆盖已记录单轮，排除进程启动、最终导出和作业间等待；不是完整Slurm吞吐。",
                      "post_ppo_s 含版本快照、导出、轮转及记账，不能全部归因于磁盘。",
                      "预期拒绝不计入异常；unexpected 拒绝与 mirror_errors 分开报告，0 errors不等于0回退。"]}


def profile_report(path, limit=30):
    stats = pstats.Stats(str(path))
    def rows(index):
        items = sorted(stats.stats.items(), key=lambda kv: kv[1][index], reverse=True)[:limit]
        return [{"function": f"{file}:{line}:{name}", "primitive_calls": data[0], "calls": data[1],
                 "self_s": data[2], "cumulative_s": data[3]}
                for (file, line, name), data in items]
    return {"schema": "hsrl.cpu_profile/v1", "total_self_s": stats.total_tt,
            "top_self": rows(2), "top_cumulative": rows(3),
            "note": "全CLI剖析含import/模型加载；累计耗时有嵌套，禁止相加。CUDA异步，CPU耗时不能替代GPU kernel计时。"}


def summarize_probes(root):
    groups, contract = {}, None
    for path in sorted(Path(root).rglob("probe.json")):
        if "profile-run" in path.relative_to(root).parts:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["ppo_updates"] != 0:
            raise ValueError("基准不允许PPO更新")
        hardware = row["hardware"]
        config = {k: v for k, v in row["config"].items() if k not in {"games", "local_batch_inference"}}
        current = (row["checkpoint"]["sha256"], row["fingerprint"], row["library_digest"],
                   config, {k: v for k, v in hardware.items() if k not in {"torch_threads", "cuda_peak_allocated_bytes"}})
        if contract is not None and current != contract:
            raise ValueError("不同模型/引擎/库/采样配置/硬件的基准不能混合")
        contract = current
        key = (int(hardware["torch_threads"]), int(row["config"]["games"]), int(row["config"].get("local_batch_inference", 0)))
        group = groups.setdefault(key, {"torch_threads": key[0], "games": key[1], "local_batch_inference": key[2], "runs": [],
                                       "decisions": 0, "collect_s": 0.0, "unexpected_rejections": Counter()})
        seconds, decisions = float(row["collect_seconds"]), int(row["collect"]["decisions"])
        if not math.isfinite(seconds) or seconds <= 0 or decisions <= 0:
            raise ValueError("基准耗时或决策数非法")
        quality_keys = ("illegal_actions", "truncated_games", "mirror_errors", "mirror_pool_unbalanced", "mirror_pool_species_unbalanced")
        if any(row["collect"].get(k, 0) for k in quality_keys) or not row["collect"].get("mirror_applied"):
            raise ValueError(f"采集质量门未通过：{path}")
        group["decisions"] += decisions
        group["collect_s"] += seconds
        group["runs"].append({"path": str(path), "decisions_per_s": decisions / seconds,
                              "inference": row.get("inference")})
        group["unexpected_rejections"].update({k: v for k, v in row["collect"].get("mirror_rejected_by_reason", {}).items()
                                               if k not in {"not_alive", "pending_choice", "dying"}})
    if not groups:
        raise ValueError("没有未剖析的采集基准")
    for group in groups.values():
        rates = [r["decisions_per_s"] for r in group["runs"]]
        group.update(decisions_per_s=group["decisions"] / group["collect_s"],
                     min_decisions_per_s=min(rates), max_decisions_per_s=max(rates))
    return {"schema": "hsrl.probe_report/v1", "groups": list(groups.values()),
            "note": "同seed重复只衡量性能波动；排除profile-run；同一GPU名称不保证跨作业同等负载。"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history", type=Path)
    source.add_argument("--pstats", type=Path)
    source.add_argument("--probes", type=Path)
    parser.add_argument("--last", type=int, default=0, help="只统计最后N轮；0为全部")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.last < 0:
        parser.error("last 必须 >= 0")
    if args.history:
        records = []
        with args.history.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{args.history}:{number}: 日志不完整，先确认作业已停止写入") from exc
                if row.get("iter") is not None:
                    records.append(row)
        result = summarize(records[-args.last:] if args.last else records)
        result["source"] = str(args.history)
    elif args.pstats:
        result = profile_report(args.pstats)
    else:
        result = summarize_probes(args.probes)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"报告已存在：{args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
