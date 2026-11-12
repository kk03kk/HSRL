#!/usr/bin/env python3
"""Summarise an overnight live root into a markdown report.

Counts only games that actually reached the client's ``game_over`` phase (proved
by ``result.json``, which the runtime writes only from that branch) so a
mid-game disconnect can never be reported as a finished game.

Usage:
    python client/tools/live_night_report.py --root artifacts/runs/live-night-20261112 \
      --out docs/analysis/LIVE_NIGHT_REPORT_20261112.md
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    except OSError:
        return rows
    return rows


def games_in(batch: Path) -> list[dict]:
    out = []
    for result in sorted(batch.glob("match-*/result.json")):
        payload = read_json(result) or {}
        match = result.parent
        perf = read_json(match / "performance.json") or {}
        waits = (read_json(match / "wait-summary.json") or {}).get("seconds_by_category") or {}
        out.append(
            dict(
                match=match.name,
                game_id=payload.get("game_id"),
                result=payload.get("result"),
                placement=payload.get("placement"),
                placement_source=payload.get("placement_source"),
                terminal=bool(payload.get("terminal") or payload.get("game_id")),
                terminal_proof=payload.get("terminal_proof"),
                turn=payload.get("turn"),
                decisions=perf.get("decisions"),
                executions=perf.get("executions"),
                applied=perf.get("applied"),
                success_rate=perf.get("action_success_rate"),
                waits=waits,
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    batches = sorted(p for p in root.glob("batch-*") if p.is_dir())
    singles = [root / d for d in ("e0-002",) if (root / d).is_dir()]
    groups = batches + singles

    rows: list[dict] = []
    signatures: collections.Counter = collections.Counter()
    for group in groups:
        for game in games_in(group):
            game["batch"] = group.name
            rows.append(game)
        for incident in read_jsonl(group / "incidents.jsonl"):
            if incident.get("severity") not in (None, "ok"):
                signatures[incident.get("signature") or incident.get("severity")] += 1

    overnight = read_jsonl(root / "overnight.jsonl")
    stalls = [e for e in overnight if e.get("event") == "stall_detected"]
    starts = [e for e in overnight if e.get("event") == "batch_start"]

    placements = [int(g["placement"]) for g in rows if str(g.get("placement", "")).isdigit()]
    lines: list[str] = []
    lines.append("# 夜间实机报告（自动生成）")
    lines.append("")
    lines.append(f"- root: `{root}`")
    lines.append(f"- 批次目录: {len(groups)}（supervisor 启动 {len(starts)} 批；卡死重启 {len(stalls)} 次）")
    lines.append(f"- **真实终局对局: {len(rows)}**")
    if placements:
        avg = sum(placements) / len(placements)
        top4 = sum(1 for p in placements if p <= 4)
        wins = sum(1 for p in placements if p == 1)
        lines.append(
            f"- 平均名次 **{avg:.2f}**；前四 {top4}/{len(placements)} "
            f"({100.0 * top4 / len(placements):.0f}%)；吃鸡 {wins}"
        )
    lines.append("")
    lines.append("## 逐局")
    lines.append("")
    lines.append("| 批次 | 局 | 结果 | 名次 | 终局回合 | 决策 | 执行 | 动作成功率 | 终局证据 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for g in rows:
        rate = g.get("success_rate")
        lines.append(
            "| {b} | {m} | {r} | {p} | {t} | {d} | {e} | {s} | {proof} |".format(
                b=g["batch"], m=g["match"], r=g.get("result"), p=g.get("placement"),
                t=g.get("turn"), d=g.get("decisions"), e=g.get("executions"),
                s=("-" if rate is None else f"{100.0 * float(rate):.1f}%"),
                proof=g.get("terminal_proof") or "-",
            )
        )
    lines.append("")
    lines.append("## 失败签名")
    lines.append("")
    if signatures:
        lines.append("| 签名 | 次数 |")
        lines.append("|---|---:|")
        for name, count in signatures.most_common(15):
            lines.append(f"| `{name}` | {count} |")
    else:
        lines.append("（无失败）")
    lines.append("")
    lines.append("## 等待类别合计（秒）")
    lines.append("")
    totals: collections.Counter = collections.Counter()
    for g in rows:
        for key, value in (g.get("waits") or {}).items():
            try:
                totals[key] += float(value)
            except (TypeError, ValueError):
                continue
    if totals:
        lines.append("| 类别 | 合计 | 每局均值 |")
        lines.append("|---|---:|---:|")
        for key, value in totals.most_common():
            lines.append(f"| {key} | {value:.1f} | {value / len(rows):.1f} |")
    else:
        lines.append("（无数据）")
    lines.append("")
    if overnight:
        lines.append("## supervisor 事件")
        lines.append("")
        lines.append("| 时间 | 事件 | 批次 | 说明 |")
        lines.append("|---|---|---|---|")
        for event in overnight:
            detail = event.get("outcome") or event.get("reason") or event.get("stop_why") or ""
            lines.append(
                f"| {event.get('at', '')} | {event.get('event', '')} | {event.get('batch', '')} | {detail} |"
            )
        lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(rows)} terminal games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
