"""离线审计 / 跨端对照工具（只读，不启动游戏、不跑模拟器）。

    # 1) 校验一条对局记录是否完整可还原（模拟器、实机通用）
    python client/tools/trace_audit.py check artifacts/local/sim-trace-20261111-001/game-001
    python client/tools/trace_audit.py check artifacts/runs/live-xxx/match-0001

    # 2) 用新的"复用即等待"规则重放旧实机日志，看清理前后的决策数
    python client/tools/trace_audit.py replay <run-or-match-dir>

    # 3) 逐回合对照模拟器与实机（动作类型、金币/等级节奏）
    python client/tools/trace_audit.py compare <sim-trace-dir> <live-match-dir>

所有命令只读 JSONL，不改任何记录。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsrl_agent import trace_schema as S


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def find_root(path: Path) -> Path:
    """给目录或文件，返回含 decisions.jsonl 的对局目录。"""
    path = Path(path)
    if path.is_file():
        return path.parent
    if (path / "decisions.jsonl").exists():
        return path
    matches = sorted(p.parent for p in path.glob("*/decisions.jsonl"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"找不到唯一的 decisions.jsonl：{path}")


def load_run(root: Path) -> dict:
    return {
        "root": root,
        "decisions": read_jsonl(root / "decisions.jsonl"),
        "executions": [r for r in read_jsonl(root / "actions.jsonl") if r.get("event") == "execution"],
        "waits": read_jsonl(root / "waits.jsonl"),
        "events": read_jsonl(root / "events.jsonl"),
    }


def cmd_check(args) -> int:
    root = find_root(Path(args.run))
    run = load_run(root)
    decisions = run["decisions"]
    result = S.audit(decisions, run["executions"], run["waits"],
                     side=(decisions[0].get("side") if decisions else "unknown"))
    result["directory"] = str(root)
    result["turns"] = S.turns_summary(decisions)
    if args.summarize:
        from hsrl_agent.live.trace import summarize
        summarize(root)
    if args.table:
        print(f"{result['directory']}  side={result['side']}  complete={result['complete']}  "
              f"decisions={result['decisions']} waits={result['waits']} "
              f"executions={result['executions']}")
        print(f"{'turn':>4} {'phase':<10} {'gold':>4} {'tier':>4} {'dec':>4} {'distinct':>8}  kinds")
        for turn, row in result["turns"].items():
            kinds = ", ".join(f"{k}x{n}" for k, n in sorted((row["kinds"] or {}).items()))
            print(f"{turn:>4} {str(row.get('phase')):<10} {row.get('gold')!s:>4} "
                  f"{row.get('tier')!s:>4} {row['decisions']:>4} {row.get('distinct')!s:>8}  {kinds}")
        for problem in result["problems"]:
            print("problem:", problem)
        return 0 if result["complete"] else 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] else 2


def cmd_replay(args) -> int:
    """旧实机日志（可能含轮询污染）按新规则重放，输出清理前后对比。"""
    root = find_root(Path(args.run))
    rows = read_jsonl(root / "decisions.jsonl")
    kept, waits = [], 0
    last = None
    synthetic = 0
    for row in rows:
        signature = (row.get("stamp"), row.get("action"))
        if row.get("reused") and signature == last:
            waits += 1
            continue
        if not row.get("side"):
            row = dict(row, side="live", action_class=row.get("action_class")
                       or S.action_class(row.get("kind")))
            synthetic += 1
        kept.append(row)
        last = signature
    before = S.turns_summary(rows)
    after = S.turns_summary(kept)
    report = {
        "directory": str(root),
        "legacy_rows": len(rows),
        "rows_after_dedupe": len(kept),
        "waits_moved_to_waits_jsonl": waits,
        "rows_normalized": synthetic,
        "per_turn": {
            turn: {
                "before": before.get(turn, {}).get("decisions"),
                "after": after.get(turn, {}).get("decisions"),
                "distinct_after": after.get(turn, {}).get("distinct"),
                "kinds_after": after.get(turn, {}).get("kinds"),
            }
            for turn in sorted(set(before) | set(after), key=lambda t: int(t) if t.isdigit() else 10 ** 6)
        },
        "old_summary_decisions": _old_summary(root),
        "consistency": {
            "same_row_count": len(kept) + waits == len(rows),
            "turns_monotonic_after": _monotonic(after),
        },
    }
    if args.table:
        print(f"legacy rows: {report['legacy_rows']}  ->  decisions: {report['rows_after_dedupe']}"
              f"  waits: {report['waits_moved_to_waits_jsonl']}")
        print(f"{'turn':>4} {'before':>7} {'after':>6} {'distinct':>9}  kinds")
        total = [0, 0, 0]
        for turn, row in report["per_turn"].items():
            kinds = ", ".join(f"{k}x{n}" for k, n in sorted((row["kinds_after"] or {}).items()))
            print(f"{turn:>4} {row['before']!s:>7} {row['after']!s:>6} "
                  f"{row['distinct_after']!s:>9}  {kinds}")
        for row in report["per_turn"].values():
            total[0] += row["before"] or 0
            total[1] += row["after"] or 0
            total[2] += row["distinct_after"] or 0
        print(f"{'ALL':>4} {total[0]:>7} {total[1]:>6} {total[2]:>9}")
        return 0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _old_summary(root: Path) -> dict:
    path = root / "summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts = data.get("counts") or {}
    return {"decisions": counts.get("decisions"), "action_kinds": data.get("action_kinds")}


def _monotonic(turns: dict) -> bool:
    keys = [int(k) for k in turns if str(k).isdigit()]
    return keys == sorted(keys)


def cmd_compare(args) -> int:
    sim = load_run(find_root(Path(args.sim)))
    live = load_run(find_root(Path(args.live)))
    report = {
        "sim": str(sim["root"]),
        "live": str(live["root"]),
        "counts": {
            "sim": {"decisions": len(sim["decisions"]), "executions": len(sim["executions"]),
                    "waits": len(sim["waits"])},
            "live": {"decisions": len(live["decisions"]), "executions": len(live["executions"]),
                     "waits": len(live["waits"])},
        },
        "per_turn": {},
    }
    sim_turns = S.turns_summary(sim["decisions"])
    live_turns = S.turns_summary(live["decisions"])
    for turn in sorted(set(sim_turns) | set(live_turns),
                       key=lambda t: int(t) if str(t).isdigit() else 10 ** 6):
        a, b = sim_turns.get(turn, {}), live_turns.get(turn, {})
        report["per_turn"][turn] = {
            "sim": {"decisions": a.get("decisions"), "classes": a.get("classes"),
                    "gold": a.get("gold"), "tier": a.get("tier"), "phase": a.get("phase")},
            "live": {"decisions": b.get("decisions"), "classes": b.get("classes"),
                     "gold": b.get("gold"), "tier": b.get("tier"), "phase": b.get("phase")},
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("run")
    p.add_argument("--table", action="store_true", help="打印逐回合表格")
    p.add_argument("--summarize", action="store_true",
                   help="顺带重算 summary/audit/report（只读原始流）")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("replay")
    p.add_argument("run")
    p.add_argument("--table", action="store_true", help="打印紧凑表格而不是 JSON")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("compare")
    p.add_argument("sim")
    p.add_argument("live")
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
