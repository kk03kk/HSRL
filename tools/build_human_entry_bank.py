#!/usr/bin/env python3
"""把 Firestone 真人解析库过滤成**可直接注入引擎**的人类战团/入场库。

背景（``docs/analysis/HUMAN_GAP_20261112.md``）：模型 T12 板面战力只有真人的
0.20–0.23 倍，且**从未见过真人战团**——训练对手全是自对弈镜像；而
``humancurve-*`` 库只是把**模型自己的牌**按真人攻血分位放大数字，牌面构成仍是我们
自己的弱阵容。因此需要一份**真人卡组的完整帧**：

- 对手侧：``--human-boards`` 用它做压力（真人阵容 + 真人数值）；
- 学习侧：``--entry-boards`` 用它做「中后期入场」课程（docs/LATE_GAME_ENTRY.md）。

输入是 ``workflows/parse_firestone_replay_xml.py`` 产出的 ``library.jsonl``
（schema ``hsbg.human_state_quasi_full``，一行 = 某座位某回合末的真实战团）。
本工具**只做过滤与归类，不改任何数值**：不重分配攻血、不补造字段。

过滤口径（全部显式写进 summary，便于复现）：

1. ``split`` 按游戏哈希已经分好（train/dev/test）；训练用 ``train``，评测用 ``dev``/``test``，
   两者**绝不混用**；
2. ``stage`` 只留 ``combat_start`` / ``recruitment_end_at_phase_switch``（回合末战团）；
3. ``is_ghost`` 一类的空/幽灵战团、``board_detail`` 缺失或长度不符、非 minion 卡、
   ``atk<0`` / ``hp<=0`` / 单随从 ``atk+hp>100000`` / 全队 ``Σ>250000`` 一律丢弃
   （与 ``hsbg.az.train.human_board_pool`` 的过滤阈值一致）；
4. ``mmr >= --min-mmr``（默认 8000）。``perspective=="opponent"`` 的行没有自己的 MMR，
   按**同一局上传者的 MMR** 继承（同房匹配，误差 << 档位宽度），并把
   ``mmr_source="game_uploader"`` 写清楚——不假装那是观测到的本人 MMR。

用法::

    python tools/build_human_entry_bank.py \
        --library data/real_library/firestone-20261112-1000/library.jsonl \
        --out-train data/real_library/firestone-bank-1000/train.jsonl \
        --out-eval  data/real_library/firestone-bank-1000/eval.jsonl \
        --summary   data/real_library/firestone-bank-1000/summary.json \
        --min-mmr 8000
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: 允许的回合末阶段（与 workflows/build_human_curve_bank.py 的同名过滤一致）。
STAGES = ("combat_start", "recruitment_end_at_phase_switch")
#: 与 hsbg.az.train.human_board_pool 完全一致的离群阈值。
MAX_MINION_STAT = 100_000
MAX_BOARD_STRENGTH = 250_000
MAX_BOARD = 7


def _pct(vals, q):
    if not vals:
        return None
    xs = sorted(vals)
    i = min(len(xs) - 1, max(0, int(round(float(q) * (len(xs) - 1)))))
    return int(xs[i])


def clean_row(row, min_mmr):
    """返回 (row|None, reason|None)；不改数值，只判断可用性。"""
    if row.get("stage") not in STAGES:
        return None, "stage"
    detail = row.get("board_detail")
    if not isinstance(detail, list) or not detail:
        return None, "no_board"
    if len(detail) > MAX_BOARD:
        return None, "board_over_7"
    if len(detail) != len(row.get("board") or []):
        return None, "board_detail_mismatch"
    total = 0
    for m in detail:
        try:
            card = int(m["card"])
            atk = int(m["atk"])
            hp = int(m["hp"])
        except (KeyError, TypeError, ValueError):
            return None, "bad_entry"
        if card <= 0 or atk < 0 or hp <= 0:
            return None, "bad_stats"
        if atk + hp > MAX_MINION_STAT:
            return None, "stat_outlier"
        total += atk + hp
    if total > MAX_BOARD_STRENGTH:
        return None, "strength_outlier"
    persp = row.get("perspective")
    mmr = row.get("mmr")
    src = "self"
    if mmr is None:
        if persp != "opponent":
            return None, "no_mmr"
        mmr = row.get("mmr_imputed")
    try:
        mmr = int(mmr)
    except (TypeError, ValueError):
        return None, "no_mmr"
    if mmr < int(min_mmr):
        return None, "mmr_below_floor"
    out = dict(row)
    out["mmr"] = int(mmr)
    out["mmr_source"] = src if src == "self" and row.get("mmr") is not None else "game_uploader"
    out["strength"] = int(total)
    return out, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", default="")
    ap.add_argument("--train-splits", default="train",
                    help="训练用 split 名（逗号分隔；解析库实际只有 train/evaluation）")
    ap.add_argument("--eval-splits", default="evaluation,dev,test",
                    help="评测用 split 名（逗号分隔；必须与训练不相交）")
    ap.add_argument("--summary", default="")
    ap.add_argument("--min-mmr", type=int, default=8000)
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument("--max-turn", type=int, default=99)
    args = ap.parse_args(argv)

    kept = collections.Counter()
    dropped = collections.Counter()
    per_turn = collections.defaultdict(list)
    per_split = collections.Counter()
    arche = collections.Counter()
    # 第一遍：给 opponent 行补上「同一局上传者 MMR」，再整体过滤。
    game_mmr = {}
    rows = []
    with open(args.library, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except BaseException:  # noqa: BLE001
                dropped["bad_json"] += 1
                continue
            gid = str(row.get("game_id") or "")
            if row.get("perspective") == "self" and row.get("mmr") is not None:
                try:
                    game_mmr[gid] = max(int(game_mmr.get(gid, 0)), int(row["mmr"]))
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    for row in rows:
        row["mmr_imputed"] = game_mmr.get(str(row.get("game_id") or ""))
        clean, why = clean_row(row, args.min_mmr)
        if clean is None:
            dropped[why] += 1
            continue
        t = int(clean.get("turn") or 0)
        if not (args.min_turn <= t <= args.max_turn):
            dropped["turn_range"] += 1
            continue
        kept["rows"] += 1
        per_split[str(clean.get("split"))] += 1
        per_turn[t].append(int(clean["strength"]))
        if clean.get("perspective") == "self" and clean.get("archetype"):
            arche[str(clean["archetype"])] += 1
        row["_keep"] = clean

    def write(path, splits):
        n = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                clean = row.get("_keep")
                if clean is None or str(clean.get("split")) not in splits:
                    continue
                f.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
        os.replace(tmp, path)
        return n

    train_splits = tuple(s.strip() for s in str(args.train_splits).split(",") if s.strip())
    eval_splits = tuple(s.strip() for s in str(args.eval_splits).split(",") if s.strip())
    overlap = set(train_splits) & set(eval_splits)
    if overlap:
        raise SystemExit(f"train/eval split 重叠：{sorted(overlap)}")
    n_train = write(args.out_train, train_splits)
    n_eval = write(args.out_eval, eval_splits) if args.out_eval else 0

    summary = {
        "schema": "hsai.human_entry_bank/v1",
        "source": os.path.abspath(args.library),
        "min_mmr": int(args.min_mmr),
        "train_splits": list(train_splits),
        "eval_splits": list(eval_splits),
        "kept_rows": int(kept["rows"]),
        "train_rows": int(n_train),
        "eval_rows": int(n_eval),
        "by_split": dict(sorted(per_split.items())),
        "dropped": dict(sorted(dropped.items())),
        "per_turn": {str(t): {"n": len(v), "p50": _pct(v, 0.50), "p75": _pct(v, 0.75),
                              "p90": _pct(v, 0.90), "max": max(v)}
                     for t, v in sorted(per_turn.items())},
        "archetypes": dict(arche.most_common()),
        "note": "只做过滤与归类，不改数值；opponent 行的 mmr 继承同局上传者并标 "
                "mmr_source=game_uploader。",
    }
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_turn", "archetypes")},
                     ensure_ascii=False, indent=2), flush=True)
    print("per_turn:", json.dumps(summary["per_turn"], ensure_ascii=False), flush=True)
    print("archetypes:", json.dumps(dict(arche.most_common(12)), ensure_ascii=False),
          flush=True)
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)) or ".", exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
