#!/usr/bin/env python3
"""行为层诊断：模型 vs 真人赢家，**每局的采购/搜牌/升本/出牌量**对比。

为什么需要它：`HUMAN_GAP` 证明战力差，`analyze_composition_coverage` 证明不组阵容，
但都没回答"卡在哪一层"。是买得少？搜得少？升得不敢？还是买了很多但拼不起来？

这个问题**不需要 GPU**：两边都已经有现成数据。

* 模型侧：`eval_human_gap` 面板的 `models[].action_hist`（245 个动作下标的计数），
  除以局数即得每局各类动作次数。动作分段来自 `hsbg/env/action_space_std.py`。
* 真人侧：Firestone PMS 的 `minionsBoughtOverTurn` / `rerollsOverTurn` /
  `minionsSoldOverTurn` / `coinsWastedOverTurn` 逐回合序列求和。

口径差异（必须写在结论里）：真人面对 8 个真人、模型面对 7 个引擎 bot；
真人对局平均 13–14 回合，模型在 bot 面板上活得更久。所以**绝对次数不可直接比**，
要看"每回合"的比值，并且两边一起看。

用法::

    PYTHONPATH=engine/python:. python tools/analyze_action_economy.py \
        --panel artifacts/runs/human-gap-humancurve-20261112/panel.json \
        --pms data/firestone_replays/refresh-20261112T031450Z/pms \
        --human data/real_library/firestone-20261112-1000/library.jsonl \
        --out artifacts/runs/human-baseline-20261112/action-economy.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: 245 个原生动作的分段（与 hsbg/env/action_space_std.py 一致）
SEGMENTS = {
    "end_turn": (0, 1),
    "refresh": (1, 2),
    "freeze": (2, 3),
    "upgrade": (3, 4),
    "hero_power": (4, 6),
    "dark_gift": (6, 7),
    "cancel": (7, 8),
    "buy": (8, 16),
    "sell": (16, 23),
    "activate": (23, 30),
    "choose": (30, 46),
    "play": (46, 126),
    "magnetize": (126, 196),
    "move": (196, 245),
}


def model_actions(hist: Dict[str, Any], games: int) -> Dict[str, float]:
    per_game = {}
    for name, (lo, hi) in SEGMENTS.items():
        total = sum(int(hist.get(str(i), 0) or 0) for i in range(lo, hi))
        per_game[name] = round(total / max(games, 1), 2)
    return per_game


def human_games(pms_dir: Path) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if not pms_dir.is_dir():
        return out
    for path in sorted(pms_dir.glob("*.json")):
        try:
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            d = json.loads(raw)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        row: Dict[str, float] = {}
        for key, field in (("buy", "minionsBoughtOverTurn"),
                           ("sell", "minionsSoldOverTurn"),
                           ("refresh", "rerollsOverTurn"),
                           ("gold_wasted", "coinsWastedOverTurn")):
            seq = d.get(field)
            row[key] = float(sum(v for v in seq if isinstance(v, (int, float)))) if isinstance(seq, list) else 0.0
        out.append(row)
    return out


def human_turns(library: Path) -> Dict[str, int]:
    """每局的最大回合（用于把'绝对次数'换成'每回合次数'）。"""
    turns: Dict[str, int] = {}
    with library.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("perspective") != "self":
                continue
            g = r["game_id"]
            turns[g] = max(turns.get(g, 0), int(r["turn"]))
    return turns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="模型 vs 真人的行为量对比")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--pms", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    models = []
    for m in panel.get("models") or []:
        hist = m.get("action_hist")
        if not hist:
            continue
        games = int(m.get("games") or panel.get("panel", {}).get("games") or 1)
        models.append((m["name"], model_actions(hist, games)))

    humans = human_games(Path(args.pms))
    turns = human_turns(Path(args.human))
    mean_turns = statistics.fmean(turns.values()) if turns else float("nan")
    h_mean = {k: round(statistics.fmean([r[k] for r in humans]), 2) for k in ("buy", "sell", "refresh", "gold_wasted")} if humans else {}

    print(f"human games with PMS : {len(humans)}   mean turns/game = {mean_turns:.2f}")
    print(f"model games          : {panel.get('panel', {}).get('games')}")
    print()
    print(f"{'metric':<14} {'human/game':>11} {'human/turn':>11} "
          + " ".join(f"{n[:12]:>13}" for n, _ in models))
    for key in ("buy", "sell", "refresh", "gold_wasted"):
        if key not in h_mean:
            continue
        hpt = h_mean[key] / mean_turns if mean_turns else float("nan")
        row = f"{key:<14} {h_mean[key]:>11.1f} {hpt:>11.2f} "
        for _, acts in models:
            row += f"{acts.get(key, 0.0):>13.1f}"
        print(row)

    # 模型侧没有 gold_wasted；其余按"每回合"换算需要模型自己的回合数，
    # 面板 survival 里有 max_turn_p50，用它做粗略换算并在结论里标注。
    print()
    for name, acts in models:
        print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in acts.items() if v > 0))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "hsai.action_economy/v1",
            "human_games": len(humans), "human_mean_turns": mean_turns,
            "human_per_game": h_mean,
            "human_per_turn": {k: (h_mean[k] / mean_turns if mean_turns else None) for k in h_mean},
            "models": [{"name": n, "per_game": a} for n, a in models],
            "caveats": ["humans faced 8 humans, models 7 engine bots",
                        "models live longer on the bot panel; compare per-turn, not raw totals"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
