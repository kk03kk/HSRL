#!/usr/bin/env python3
"""Read-only reconstruction of one `artifacts/runs/live-*` directory.

Never launches the game, never reads a live log, and never edits the run.
It answers the questions needed after a real test:

* which model decisions and gestures happened;
* how long inference and input took;
* which stages were blocked/cancelled and why;
* whether an action repeated without a state change;
* whether receipts and archives were complete.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def pct(values: list[float]) -> dict:
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {"count": 0, "p50": None, "p90": None, "mean": None, "max": None}

    def q(p: float) -> float:
        index = (len(values) - 1) * p
        lower = int(index)
        upper = min(len(values) - 1, lower + 1)
        fraction = index - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction

    return {
        "count": len(values),
        "p50": q(0.5),
        "p90": q(0.9),
        "mean": statistics.fmean(values),
        "max": values[-1],
    }


def audit(run: Path) -> dict:
    run = Path(run)
    events = read_jsonl(run / "agent-events.jsonl")
    decisions = []
    dispatches = []
    applied = []
    statuses = []
    input_stages = []
    for event in events:
        kind = event.get("event")
        if kind in ("model_decision", "decision_reused"):
            decisions.append(event)
        elif kind == "dispatch_intent":
            dispatches.append(event)
        elif kind == "applied":
            applied.append(event)
        elif kind == "status":
            statuses.append(event)
        elif kind == "input_stage":
            input_stages.append(event)

    # Consecutive decision -> dispatch -> applied latency.  This is the
    # actual user-visible cost, not only the model forward time.
    pairs = []
    dispatch_at = None
    for event in events:
        if event.get("event") == "dispatch_intent":
            dispatch_at = event.get("at")
        elif event.get("event") == "input_stage":
            if dispatch_at is not None:
                pairs.append({"seconds_to_input_stage": event.get("at") - dispatch_at})
                dispatch_at = None
    decision_to_dispatch = []
    last_decision = None
    for event in events:
        if event.get("event") in ("model_decision", "decision_reused"):
            last_decision = event.get("at")
        elif event.get("event") == "dispatch_intent" and last_decision is not None:
            decision_to_dispatch.append({
                "decision_seconds": event.get("at") - last_decision,
            })
            last_decision = None

    reasons = Counter(status.get("reason", "") for status in statuses)
    repeated = []
    history = defaultdict(list)
    # Repetition is measured from actual dispatch intents (with their
    # decision snapshot), never from speculative model records that stopped
    # at the two-frame gate.
    for event in events:
        if event.get("event") != "dispatch_intent":
            continue
        plan = event.get("plan") or []
        if not plan:
            continue
        key = event.get("game_id") or "game"
        history[key].append((json.dumps(plan, sort_keys=True), plan[0][0], None))
    for key, rows in history.items():
        run_length = 1
        for i in range(1, len(rows)):
            if rows[i][0] == rows[i - 1][0]:
                run_length += 1
                if run_length >= 3:
                    repeated.append({"game_id": key, "dispatch_plan": rows[i][0],
                                     "primitive": rows[i][1], "run_length": run_length})
            else:
                run_length = 1

    model = [d for d in decisions if d.get("event") == "model_decision"]
    reused = [d for d in decisions if d.get("event") == "decision_reused"]
    result = read_json(run / "result.json") or {}
    status = read_json(run / "status.json") or {}
    session = read_json(run / "session.json") or {}
    exit_info = read_json(run / "exit.json") or {}
    return {
        "schema": "hsai.live-run-audit/v1",
        "run": str(run),
        "session": session,
        "exit": exit_info,
        "result": result,
        "last_status": status,
        "counts": {
            "events": len(events),
            "model_decision": len(model),
            "decision_reused": len(reused),
            "dispatch_intent": len(dispatches),
            "input_stage": len(input_stages),
            "applied": len(applied),
            "status": len(statuses),
        },
        "inference_ms": pct([d.get("policy_ms") for d in model]),
        "decision_to_dispatch_seconds": pct([p["decision_seconds"] for p in decision_to_dispatch]),
        "dispatch_to_input_stage_seconds": pct([p["seconds_to_input_stage"] for p in pairs]),
        "status_reasons": dict(reasons.most_common()),
        "repeated_action_runs": repeated[:50],
        "archive_status": read_json(run / "archive-status.json"),
        "event_archive_status": read_json(run / "event-archive-status.json"),
        "latency": read_json(run / "latency.json"),
    }


def markdown(report: dict) -> str:
    counts = report["counts"]
    result = report.get("result") or {}
    lines = [
        "# 实机运行审计",
        "",
        f"- 目录：`{report['run']}`",
        f"- 结果：`{result.get('result')}`，名次 `{result.get('placement')}`，回合 `{result.get('turn')}`",
        f"- 退出：`{(report.get('exit') or {}).get('completed_normally')}` / "
        f"归档完成 `{(report.get('exit') or {}).get('archive_complete')}`",
        "",
        "## 计数与耗时",
        "",
        f"- 模型决策 {counts['model_decision']} 次，复用 {counts['decision_reused']} 次",
        f"- 推理耗时 p50/p90/max：{report['inference_ms']['p50']} / "
        f"{report['inference_ms']['p90']} / {report['inference_ms']['max']} ms",
        f"- 输入阶段 {counts['input_stage']} 次，服务端因果回执 {counts['applied']} 次",
        f"- 决策到派发 p50/p90：{report['decision_to_dispatch_seconds']['p50']} / "
        f"{report['decision_to_dispatch_seconds']['p90']} s",
        f"- 派发到 input_stage p50/p90：{report['dispatch_to_input_stage_seconds']['p50']} / "
        f"{report['dispatch_to_input_stage_seconds']['p90']} s",
        "",
        "## 状态原因",
        "",
        "| 原因 | 次数 |",
        "|---|---:|",
    ]
    for reason, count in report["status_reasons"].items():
        lines.append(f"| `{reason}` | {count} |")
    repeated = report.get("repeated_action_runs") or []
    lines += ["", "## 重复动作", ""]
    if repeated:
        lines += ["| 局 | 原始输入计划 | 原语 | 连续次数 |", "|---|---|---|---:|"]
        for row in repeated[:20]:
            lines.append(f"| `{row['game_id']}` | `{row['dispatch_plan']}` | "
                         f"{row['primitive']} | {row['run_length']} |")
    else:
        lines.append("未发现连续三次以上的完全相同物理输入计划。")
    lines += [
        "",
        "## 判读",
        "",
        "模型决策与脚本执行是两层：`model_decision` 表示网络输出；只有 `applied` 才表示服务端",
        "因果回执。状态原因中的等待是证据不足时的暂停，不是重新采样或重新点击。",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("session", "latency")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
