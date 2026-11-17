#!/usr/bin/env python3
"""从「含多模型的子面板」重建一份合并面板（不重跑评测）。

背景（2026-11-17 的 bug）：``eval_human_gap.py --dump-rows`` 把 rows 落成
``rows.jsonl.gz``，一个进程跑 4 个模型时**后者覆盖前者**，于是 ``merge_panels.py``
只能重建出最后一个模型。但**每个子面板的 ``models[]`` 本身是完整的**（每个模型各有
自己的 ``trace``），所以直接从子面板的 models 聚合即可，无需重跑。

策略：按模型名分组 → 拼接逐局部分（trace / rank_sequence / final_boards /
boards_by_turn / strength_by_seed）→ 从合并后的 trace 重算 curves / survival /
trinkets / gold / heroes → opponents_by_turn 按 n 加权合并。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.eval_human_gap import (  # noqa: E402
    _counts_all, _counts_by_turn, _gold_by_turn, _strength_by_seed, _trace,
    economy_curves,
)


def merge(srcs, out_path: Path):
    by_model: dict[str, list[dict]] = collections.OrderedDict()
    for p in srcs:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        for m in doc.get("models") or []:
            by_model.setdefault(str(m.get("name")), []).append(m)

    merged_models = []
    for name, parts in by_model.items():
        rows = []
        ranks = []
        fb = []
        for m in parts:
            for r in m.get("trace") or []:
                rows.append(r)
            rs = m.get("rank_sequence")
            if isinstance(rs, list):
                ranks.extend(rs)
            fb.extend(m.get("final_boards") or [])

        # 从 trace 还原成聚合所需的 rows（补上 eot_attack / eot_board_hp）
        ec = []
        for r in rows:
            s = r.get("strength")
            ro = r.get("roster") or []
            ec.append({
                "seed": r.get("seed"), "turn": r.get("turn"),
                "tier": r.get("tier"), "gold": r.get("gold"),
                "health": r.get("health"), "hand_n": r.get("hand_n"),
                "actions": r.get("actions"),
                "eot_attack": int(s) // 2 if isinstance(s, (int, float)) else None,
                "eot_board_hp": int(s) - (int(s) // 2) if isinstance(s, (int, float)) else None,
                "eot_board": ro, "board": ro,
                "trinkets": r.get("trinkets"), "hero": r.get("hero"),
                "consecutive_repeats": r.get("repeats"),
                "gold_events": [],
            })

        # survival：每局最大回合
        games = collections.defaultdict(int)
        for r in rows:
            try:
                g, t = str(r.get("seed")), int(r.get("turn"))
            except (TypeError, ValueError):
                continue
            games[g] = max(games[g], t)
        surv = collections.Counter()
        for t in games.values():
            for k in range(1, t + 1):
                surv[k] += 1

        # opponents_by_turn：按 n 加权合并
        opp_acc = collections.defaultdict(lambda: {"n": 0, "sum": 0.0})
        for m in parts:
            for t, d in (m.get("opponents_by_turn") or {}).items():
                n = int((d or {}).get("n") or 0)
                if n:
                    opp_acc[t]["n"] += n
                    opp_acc[t]["sum"] += n * float((d or {}).get("mean") or 0)
        opp = {t: {"n": v["n"], "mean": round(v["sum"] / v["n"], 1) if v["n"] else None}
               for t, v in sorted(opp_acc.items(), key=lambda kv: int(kv[0]))}

        entry = {
            "name": name,
            "games": len(games),
            "shards": len(parts),
            "outcome": parts[0].get("outcome") or {},
            "rank_sequence": ranks or None,
            "curves": economy_curves(ec),
            "survival": dict(sorted(surv.items())),
            "final_boards": fb,
            "trace": rows,
            "strength_by_seed": _strength_by_seed(ec),
            "trinkets_by_turn": _counts_by_turn(ec, "trinkets"),
            "gold_by_turn": _gold_by_turn(ec),
            "heroes": _counts_all(ec, "hero"),
            "opponents_by_turn": opp,
        }
        # 名次类指标按拼接后的逐局名次重算（多个分片时 parts[0] 的只是局部）
        ok = [int(x) for x in ranks if 1 <= int(x) <= 8]
        if ok:
            entry["outcome"] = dict(entry["outcome"])
            entry["outcome"].update({
                "mean_rank": round(sum(ok) / len(ok), 4),
                "top4_rate": round(sum(1 for x in ok if x <= 4) / len(ok), 4),
                "win_rate": round(sum(1 for x in ok if x == 1) / len(ok), 4),
                "games": len(ok),
            })
        merged_models.append(entry)

    doc = {"models": merged_models, "merged_from": [str(p) for p in srcs],
           "merge": {"schema": "hsai.panel.multimodel/v1", "shards": len(srcs)}}
    out_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    for m in merged_models:
        cv = m["curves"]
        print("merged %-10s games=%-5s rows=%-6d T12=%-7s T16=%-7s rank=%.3f"
              % (m["name"], m["games"], len(m["trace"]),
                 (cv.get("12") or {}).get("strength_p50"),
                 (cv.get("16") or {}).get("strength_p50"),
                 (m["outcome"] or {}).get("mean_rank") or 0))
    print("wrote", out_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    merge(a.srcs, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
