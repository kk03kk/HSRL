"""资源吞吐对比：真人库的逐回合动作量与模型的 `action_mix_by_turn`。

动机（§5.37）：引擎/留存/集中度都被否掉后，剩下的候选是**资源吞吐**
（引擎的作用是让金币变多 → 每回合能买/施放/贴的东西更多 → buff 总量更大）。
本工具把两边的**逐回合动作次数**放在一张表上比。

两套动作命名不同，这里做显式映射：

| 真人库 `action_counts` | 模型 `action_mix_by_turn.mean` |
|---|---|
| `buy_minion` | `buy` |
| `play_minion` + `play_spell` | `play`（模型侧 play 含法术） |
| `reroll` / `refresh` | `refresh` |
| `upgrade` | `upgrade` |
| `hero_power` | `hero_power` |
| `sell_minion` | `sell` |

例：
    python3 tools/report_action_volume.py \
      --bank data/real_library/firestone-bank-6k/train-m6k.jsonl \
      --panel base=artifacts/runs/gap-mech-base-20261113/panel.json --turns 8-12
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Dict, List

BANK_TO_MODEL = {
    "buy_minion": "buy",
    "play_minion": "play",
    "play_spell": "play",
    "reroll": "refresh",
    "refresh": "refresh",
    "upgrade": "upgrade",
    "hero_power": "hero_power",
    "sell_minion": "sell",
}


def bank_volume(bank: str, lo: int, hi: int) -> Dict[int, Dict[str, float]]:
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    games = collections.defaultdict(set)
    for line in open(bank, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except BaseException:  # noqa: BLE001
            continue
        if str(r.get("perspective") or "") != "self":
            continue                    # 只统计被跟踪的那一位玩家
        try:
            t = int(r.get("turn") or 0)
        except BaseException:  # noqa: BLE001
            continue
        if not (lo <= t <= hi):
            continue
        counts = r.get("action_counts")
        if not isinstance(counts, dict):
            continue
        gid = str(r.get("game_id") or f"anon{t}")
        games[t].add(gid)
        for k, v in counts.items():
            target = BANK_TO_MODEL.get(str(k))
            if target and isinstance(v, (int, float)):
                per[t][target] += float(v)
    out: Dict[int, Dict[str, float]] = {}
    for t, d in per.items():
        n = max(len(games[t]), 1)
        out[t] = {k: v / n for k, v in d.items()}
    return out


def panel_volume(panel: str, lo: int, hi: int) -> Dict[str, Dict[int, Dict[str, float]]]:
    with open(panel, "r", encoding="utf-8-sig") as fh:
        doc = json.load(fh)
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for m in doc.get("models") or []:
        am = m.get("action_mix_by_turn") or {}
        per: Dict[int, Dict[str, float]] = {}
        for t, d in am.items():
            try:
                ti = int(t)
            except (TypeError, ValueError):
                continue
            if lo <= ti <= hi:
                per[ti] = {k: float(v) for k, v in (d.get("mean") or {}).items()}
        out[str(m.get("name"))] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="")
    ap.add_argument("--panel", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--turns", default="8-12")
    args = ap.parse_args()
    lo, _, hi = args.turns.partition("-")
    lo, hi = int(lo), int(hi or lo)

    groups = ("buy", "play", "refresh", "sell", "upgrade", "hero_power")
    rows: List[tuple] = []
    if args.bank:
        rows.append(("human", bank_volume(args.bank, lo, hi)))
    for spec in args.panel:
        tag, _, path = spec.partition("=")
        for name, per in panel_volume(path, lo, hi).items():
            rows.append((f"{tag}:{name}", per))

    for label, per in rows:
        print()
        print(f"[{label}]  每回合平均动作次数")
        print(f"{'turn':>5} " + " ".join(f"{g:>8}" for g in groups))
        for t in range(lo, hi + 1):
            d = per.get(t)
            if not d:
                continue
            print(f"{t:>5} " + " ".join(f"{d.get(g, 0.0):>8.2f}" for g in groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
