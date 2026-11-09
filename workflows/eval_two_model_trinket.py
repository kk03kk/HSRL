#!/usr/bin/env python3
"""Two-model + six-bot audit with trinket telemetry and every final lineup.

Config example: ``config/eval-9691-vs-7936.json``.  The two focus models play 20
games; the other six seats use the native engine bot.  Every end-of-turn frame
records the owning player's trinkets and their per-card counters.  The report
keeps all 20 final boards for each focus model.

Usage::

    python workflows/eval_two_model_trinket.py --config config/eval-9691-vs-7936.json \
        --out artifacts/runs/9691-vs-7936-trinket-<job>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.eval.duel_metrics import final_state_summary, rank_summary  # noqa: E402
from workflows.eval_two_models_six_bots import (  # noqa: E402
    CardNames,
    DuelSeatPolicy,
    audit_model,
    run_games,
    sha256_file,
    write_json,
)

SCHEMA = "hsai.eval_two_model_trinket/v1"
CONFIG_DEFAULT = ROOT / "config/eval-9691-vs-7936.json"
N_PLAYERS = 8
END_TURN = 0
TARGET_DEFAULT = "BG36_MagicItem_414"


def sha256_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def protocol_sha256(protocol: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("models", "games", "seed", "max_actions_per_turn", "move_cap",
                "move_repeat_limit"):
        if key not in cfg:
            raise ValueError(f"配置缺少 {key}: {path}")
    if int(cfg["max_actions_per_turn"]) != 80:
        raise ValueError("max_actions_per_turn 必须固定为 80")
    if int(cfg["move_cap"]) != 5 or int(cfg["move_repeat_limit"]) != 2:
        raise ValueError("move_cap 必须为 5，move_repeat_limit 必须为 2")
    for label in ("A", "B"):
        model = cfg["models"].get(label) or {}
        for key in ("id", "path", "file_sha256", "bytes"):
            if key not in model:
                raise ValueError(f"models.{label} 缺少 {key}")
    if int(cfg["games"]) <= 0:
        raise ValueError("games 必须 > 0")
    return cfg


def seat_schedule(gi: int, games: int) -> Dict[str, int]:
    block = max(1, int(games) // 2)
    k = int(gi) % block
    a = k % N_PLAYERS
    b = (k + 1) % N_PLAYERS
    return {"A": b, "B": a} if int(gi) < block else {"A": a, "B": b}


def engine_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import hsbg_engine as E
    from hsbg.az.train.personal import fingerprint

    if not E.build_is_current(ROOT / "engine"):
        raise ValueError("原生扩展不是从当前 Rust 源码构建")
    fp = fingerprint()
    expected = cfg.get("engine") or {}
    for key in ("game_version", "source_hash"):
        actual = fp["engine"].get(key)
        if expected.get(key) and str(actual) != str(expected[key]):
            raise ValueError(f"engine {key} 不符: {actual} != {expected[key]}")
    return fp


def trinkets_of(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(frame.get("trinkets") or [])


def has_target(frame: Dict[str, Any], target: str) -> bool:
    return any(str(t.get("id")) == target for t in trinkets_of(frame))


def build_protocol(cfg: Dict[str, Any], cfg_path: Path, fp: Dict[str, Any],
                   audits: Dict[str, Any], runner_sha: str, device: str) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "config_path": str(cfg_path.relative_to(ROOT)) if str(cfg_path).startswith(str(ROOT))
        else str(cfg_path),
        "config_sha256": sha256_text(cfg_path),
        "runner_sha256": runner_sha,
        "engine": fp,
        "engine_unmodified": True,
        "models": {
            label: {"id": audits[label]["id"], "file_sha256": audits[label]["file_sha256"],
                    "params": audits[label].get("params"), "arch": audits[label].get("arch")}
            for label in ("A", "B")
        },
        "games": int(cfg["games"]),
        "seed_start": int(cfg["seed"]),
        "seed_end": int(cfg["seed"]) + int(cfg["games"]) - 1,
        "seat_schedule": "first half A=k%8/B=(k+1)%8; second half swapped",
        "opponents": {"kind": "engine_bot", "count": 6, "action": "game.bot_play"},
        "decode": str(cfg.get("decode", "greedy")),
        "max_actions_per_turn": int(cfg["max_actions_per_turn"]),
        "move_guard": 1,
        "forbid_cancel": 1,
        "move_cap": int(cfg["move_cap"]),
        "move_repeat_limit": int(cfg["move_repeat_limit"]),
        "max_steps": int(cfg.get("max_steps", 50000)),
        "forced_tier_plan": "",
        "mirror_pool": None,
        "human_board_pool": None,
        "target_trinket": str(cfg.get("target_trinket", TARGET_DEFAULT)),
        "device": str(device),
    }


def lineup(frame: Dict[str, Any], board: Sequence[Dict[str, Any]], *, seed: int,
           seat: int, rank: int, target: str) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "seat": int(seat),
        "rank": int(rank),
        "turn": int(frame.get("turn") or 0),
        "tier": int(frame.get("tier") or 0),
        "health": int(frame.get("health") or 0),
        "body": int(frame.get("body") or 0),
        "trinkets": trinkets_of(frame),
        "target_present": has_target(frame, target),
        "board": list(board or []),
    }


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def p50(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), 50))


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 9691 vs 7936：2 model + 6 bot 饰品审计")
    lines.append("")
    lines.append(f"- 状态：**{'VALID' if report['valid'] else 'INVALID'}**")
    lines.append(f"- 引擎：`{report['engine']['engine']['game_version']}` / "
                 f"`{report['engine']['engine']['source_hash']}`")
    lines.append(f"- 局数：{report['totals']['games']}；种子："
                 f"`{report['protocol']['seed_start']}`–`{report['protocol']['seed_end']}`")
    lines.append(f"- 协议 SHA：`{report['protocol_sha256']}`")
    lines.append("")
    if report["reasons"]:
        lines.append("## 门禁失败")
        lines.append("")
        for reason in report["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    lines.append("## 名次")
    lines.append("")
    lines.append("| 模型 | 平均名次 | 前四率 | 冠军率 | 名次分布 1..8 |")
    lines.append("|---|---:|---:|---:|---|")
    for label in ("A", "B"):
        row = report["models"][label]
        counts = "/".join(str(row["rank"]["rank_counts"][str(i)]) for i in range(1, 9))
        lines.append(f"| {row['id']} | {fmt(row['rank']['mean_rank'], 2)} | "
                     f"{percent(row['rank']['top4_rate'])} | {percent(row['rank']['win_rate'])} | "
                     f"{counts} |")
    lines.append("")
    lines.append("## 加重护手（`BG36_MagicItem_414`）")
    lines.append("")
    lines.append("| 模型 | 出现过 | 终局持有 | 首次出现回合 | 终局倍率 | 有护手终局 P50 | 无护手终局 P50 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in ("A", "B"):
        row = report["models"][label]["weighted_gauntlet"]
        lines.append(f"| {report['models'][label]['id']} | {row['games_seen']} | "
                     f"{row['final_games']} | {row['first_turn'] if row['first_turn'] is not None else '—'} | "
                     f"{row['final_counter'] if row['final_counter'] is not None else '—'} | "
                     f"{fmt(row['body_p50_with'], 0)} | {fmt(row['body_p50_without'], 0)} |")
    lines.append("")
    for label in ("A", "B"):
        row = report["models"][label]
        lines.append(f"## {row['id']}：20 局最终阵容")
        lines.append("")
        for item in row["lineups"]:
            gw = [t for t in item["trinkets"] if str(t.get("id")) == report["target"]]
            extra = ""
            if gw:
                extra = "；加重护手 counter=" + ", ".join(str(t.get("counter")) for t in gw)
            comp = ", ".join(
                f"{p['name']} {p['atk']}/{p['hp']}" for p in item["board"]
            )
            lines.append(f"- seed `{item['seed']}`，座位 {item['seat']}，名次 {item['rank']}，"
                         f"T{item['tier']}，血量 {item['health']}，终局总值 {item['body']}"
                         f"{extra}：{comp}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch
    import hsbg_engine as E

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 但当前不可用")
    fp = engine_fingerprint(cfg)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"输出目录已存在: {out}")
    out.mkdir(parents=True)

    audits: Dict[str, Any] = {}
    evaluators: Dict[str, Any] = {}
    for label in ("A", "B"):
        model = cfg["models"][label]
        evaluator, audit = audit_model(
            ROOT / str(model["path"]), label,
            expected_sha=str(model["file_sha256"]), device=str(device),
            model_id=str(model["id"]),
        )
        evaluators[label] = evaluator
        audits[label] = audit
    policies = {
        label: DuelSeatPolicy(
            str(cfg["models"][label]["id"]), evaluators[label],
            max_actions_per_turn=int(cfg["max_actions_per_turn"]),
            move_cap=int(cfg["move_cap"]),
            move_repeat_limit=int(cfg["move_repeat_limit"]),
        ) for label in ("A", "B")
    }
    started = time.time()
    run = run_games(
        engine=E, cards=CardNames(E), policies=policies,
        games=int(cfg["games"]), seed0=int(cfg["seed"]),
        max_steps=int(cfg.get("max_steps", 50000)),
        collect_opponents=False,
        seats_for=lambda gi: seat_schedule(gi, int(cfg["games"])),
        progress_every=4,
    )
    target = str(cfg.get("target_trinket", TARGET_DEFAULT))
    rows = run["games"]
    ranked = [r for r in rows if r["over"]]
    reasons: List[str] = []
    if len(ranked) != int(cfg["games"]):
        reasons.append(f"完成局数 {len(ranked)} != {cfg['games']}")
    if int(run["illegal"]):
        reasons.append(f"非法动作 illegal={run['illegal']}")
    if int(run["truncated_games"]):
        reasons.append(f"截断局={run['truncated_games']}")
    if int(run["engine_bad"]):
        reasons.append(f"引擎异常={run['engine_bad']}")
    models_out: Dict[str, Any] = {}
    for label in ("A", "B"):
        model_id = str(cfg["models"][label]["id"])
        lines = [
            lineup(r["final"][label], r["final_board"][label], seed=r["seed"],
                   seat=r["seats"][label], rank=r["rank"][label], target=target)
            for r in ranked if r["final"].get(label) is not None
        ]
        ranks = [int(r["rank"][label]) for r in ranked]
        frames = run["frames"][label]
        seen_games = sorted({int(f["game"]) for f in frames if has_target(f, target)})
        final_games = [x for x in lines if x["target_present"]]
        first_turn = min((int(f["turn"]) for f in frames if has_target(f, target)),
                         default=None)
        final_counter = None
        for item in reversed(lines):
            gw = [t for t in item["trinkets"] if str(t.get("id")) == target]
            if gw:
                final_counter = gw[-1].get("counter")
                break
        with_body = [x["body"] for x in lines if x["target_present"]]
        without_body = [x["body"] for x in lines if not x["target_present"]]
        if len(ranks) != int(cfg["games"]):
            reasons.append(f"{model_id} 有效名次 {len(ranks)} != {cfg['games']}")
        models_out[label] = {
            "id": model_id,
            "file_sha256": audits[label]["file_sha256"],
            "rank": rank_summary(ranks),
            "final_state": final_state_summary([
                dict(r["final"][label], rank=int(r["rank"][label]), seed=int(r["seed"]))
                for r in ranked if r["final"].get(label)
            ]),
            "weighted_gauntlet": {
                "games_seen": len(seen_games),
                "final_games": len(final_games),
                "first_turn": first_turn,
                "final_counter": final_counter,
                "body_p50_with": p50(with_body),
                "body_p50_without": p50(without_body),
            },
            "lineups": lines,
        }
    protocol = build_protocol(cfg, cfg_path, fp, audits,
                              sha256_text(Path(__file__)),
                              str(device))
    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "valid": not reasons,
        "reasons": reasons,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256(protocol),
        "engine": fp,
        "models": models_out,
        "target": target,
        "totals": {
            "games": len(rows), "ranked": len(ranked),
            "illegal": int(run["illegal"]),
            "truncated": int(run["truncated_games"]),
            "engine_bad": int(run["engine_bad"]),
            "decisions": int(sum(sum(r["decisions"].values()) for r in rows)),
            "seconds": float(run["seconds"]),
        },
        "runtime": {"os": platform.system(), "python": platform.python_version(),
                    "torch": torch.__version__, "device": str(device),
                    "seconds": time.time() - started},
        "games": rows,
    }
    write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"[two-model] report -> {out / 'report.json'} valid={report['valid']}", flush=True)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
