#!/usr/bin/env python3
"""给真人对局库补上「同一玩家逐回合强度序列」，用于专家相对的过程奖励。

动机（`docs/analysis/ENTRY_S1_20261112.md`）：入场课程给了模型一张真的 T10–T11 阵容，
但它几回合内就把 Σ(atk+hp) 从 ≈2,000 做到 ≈1,500，而真人同回合是 5,600 → 27,000。
只靠终局名次去学「这一回合该长多少」太稀疏，所以把**同一个真人这一局每一回合的板面
Σ** 附到帧上，让训练侧可以问「第 t 回合，这个真人在同一局里做到了多少」。

只在 ``perspective == "self"`` 的行上建序列：``opponent`` 行的战团属于谁没有稳定标识
（同一英雄可能同场出现两次），无法安全地串成序列，因此该视角的帧不进本库。

输出：与输入同 schema，额外两个字段

- ``chain_turn``：该帧所属的「每回合序列」覆盖到的回合列表，例如 ``[9,10,11,12]``；
- ``chain_strength``：与 ``chain_turn`` 对齐的 Σ(atk+hp) 列表。

同一回合多行时只保留一行（优先 ``combat_start``），保证「一回合一个强度」。

用法::

    python tools/build_human_chain.py \
        --library data/real_library/firestone-bank-1000/train-m8k.jsonl \
        --out     data/real_library/firestone-bank-1000/train-m8k-selfchain.jsonl \
        --summary data/real_library/firestone-bank-1000/summary-m8k-selfchain.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os

STAGE_RANK = {"combat_start": 0, "recruitment_end_at_phase_switch": 1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--min-series", type=int, default=2,
                    help="序列至少覆盖几个回合才保留（默认 2）")
    ap.add_argument("--min-mmr", type=int, default=0,
                    help="只要 mmr >= 该值的 self 行（默认 0 = 不过滤）。"
                         "**入场课程必须设**：不设就会把全部 995 局（含 3k MMR）"
                         "当成 p75 参考，等于把弱阵容当专家教学。")
    args = ap.parse_args(argv)

    rows = []
    with open(args.library, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # game -> turn -> 选中的行（同 turn 多行按 stage 优先级取最好的一条）
    best = collections.defaultdict(dict)
    dropped_mmr = 0
    for r in rows:
        if r.get("perspective") != "self":
            continue
        if int(args.min_mmr) > 0:
            try:
                if float(r.get("mmr")) < float(args.min_mmr):
                    dropped_mmr += 1
                    continue
            except (TypeError, ValueError):
                continue
        gid = str(r.get("game_id") or "")
        if not gid:
            continue
        t = int(r.get("turn") or 0)
        if t <= 0:
            continue
        rank = STAGE_RANK.get(str(r.get("stage")), 9)
        cur = best[gid].get(t)
        if cur is None or rank < STAGE_RANK.get(str(cur.get("stage")), 9):
            best[gid][t] = r

    series = {}
    for gid, per_turn in best.items():
        turns = sorted(per_turn)
        vals = [int(per_turn[t].get("strength") or 0) for t in turns]
        cards = []
        ok = True
        for t in turns:
            detail = per_turn[t].get("board_detail") or []
            try:
                cards.append([int(m["card"]) for m in detail])
            except (KeyError, TypeError, ValueError):
                ok = False
                break
        if not ok:
            dropped["bad_cards"] += 1
            continue
        if len(turns) >= int(args.min_series) and all(v > 0 for v in vals):
            series[gid] = (turns, vals, cards)

    kept = 0
    dropped = collections.Counter()
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("perspective") != "self":
                dropped["not_self"] += 1
                continue
            gid = str(r.get("game_id") or "")
            s = series.get(gid)
            if s is None:
                dropped["short_series"] += 1
                continue
            out = dict(r)
            out["chain_turn"] = list(s[0])
            out["chain_strength"] = list(s[1])
            out["chain_cards"] = [list(c) for c in s[2]]
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1
    os.replace(tmp, args.out)

    lens = [len(v[0]) for v in series.values()]
    summ = {
        "schema": "hsai.human_chain/v1",
        "source": os.path.abspath(args.library),
        "rows_in": len(rows),
        "rows_out": int(kept),
        "games_with_series": len(series),
        "series_len_min": min(lens) if lens else 0,
        "series_len_max": max(lens) if lens else 0,
        "dropped": dict(sorted(dropped.items())),
        "min_mmr": int(args.min_mmr),
        "dropped_mmr_below_floor": int(dropped_mmr),
        "note": "只串 perspective=self；同回合多行按 combat_start 优先取一条。",
    }
    print(json.dumps(summ, ensure_ascii=False, indent=2), flush=True)
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)) or ".", exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
