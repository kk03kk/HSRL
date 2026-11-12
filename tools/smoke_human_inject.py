#!/usr/bin/env python3
"""验证「真人完整状态 → 引擎座位」这条注入链真的通（数据↔引擎兼容性体检）。

它回答三个必须先用实测确认的问题，再谈训练效果：

1. 真人库的每一帧，引擎**接受**吗？（``apply_turn_snapshots`` 返回 applied）
2. 接受之后，座位上的战团/手牌/酒馆等级**真的**是库里的那一帧吗？（逐字段复核）
3. 一次注入的 Python + 引擎成本是多少？（决定入场课程能否塞进采集热路径）

不需要 torch，只需要 ``hsbg_engine``：所以它能在计算节点上单独跑，也可以当作
``--entry-boards`` 换库后的前置门禁。

用法::

    PYTHONPATH=engine/python:. python tools/smoke_human_inject.py \
        --library data/real_library/firestone-bank-1000/train-m8k.jsonl \
        --turns 8,9,10,11,12 --per-turn 6 --out artifacts/runs/smoke-inject.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fast_forward(game, target_turn, n_players=8, max_steps=200000):
    """让引擎 bot 把对局推进到 ``target_turn`` 回合开始（recruit 阶段）。"""
    steps = 0
    while int(game.turn) < int(target_turn) and not game.is_over() and steps < max_steps:
        q = next((p for p in range(n_players) if game.needs_action(p)), None)
        if q is None:
            break
        game.bot_play(int(q))
        steps += 1
    return steps


def board_of(game, seat):
    pj = json.loads(game.player_json(int(seat)))
    return {
        "board": [[int(m.get("card")), int(m.get("atk")), int(m.get("hp"))]
                  for m in (pj.get("board") or [])],
        "hand_n": len(pj.get("hand") or []),
        "tier": int(pj.get("tier") or 0),
        "health": int(pj.get("health") or 0),
        "gold": int(pj.get("gold") or 0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    ap.add_argument("--turns", default="8,9,10,11,12,13")
    ap.add_argument("--per-turn", type=int, default=4)
    ap.add_argument("--sample", default="p50",
                    choices=["auto", "p50", "p75", "p90", "max"])
    ap.add_argument("--fidelity", default="full", choices=["board", "hand", "full"])
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from hsbg.az.train.common import load_engine
    from hsbg.az.train.human_board_pool import load_pool
    from hsbg.az.train.human_obs_full import inject_quasi_state

    E = load_engine()
    pool = load_pool(args.library, sample=args.sample)
    if pool is None:
        raise SystemExit(f"库为空/不存在：{args.library}")
    print(f"[bank] kept={pool.kept} payload={pool.payload_kept} "
          f"turns={pool.turns()[:6]}...{pool.turns()[-3:]}", flush=True)

    rng = random.Random(args.seed)
    report = {"schema": "hsai.smoke_inject/v1", "library": os.path.abspath(args.library),
              "fidelity": args.fidelity, "sample": args.sample,
              "seat": int(args.seat), "rows": [], "by_turn": {}}
    agg = collections.Counter()
    for t in [int(x) for x in str(args.turns).split(",") if x.strip()]:
        per_turn = {"attempt": 0, "applied": 0, "board_match": 0, "hand_match": 0,
                    "tier_match": 0, "rejected": 0, "reasons": {},
                    "ms_mean": None, "strength_p50": None}
        ms = []
        strengths = []
        for k in range(int(args.per_turn)):
            payload = pool.sample_payload(t, rng=rng)
            if payload is None:
                per_turn["rejected"] += 1
                per_turn["reasons"]["no_payload"] = \
                    per_turn["reasons"].get("no_payload", 0) + 1
                continue
            want = payload.get("board_flat") or []
            g = E.Game(seed=int(args.seed) + t * 1000 + k)
            fast_forward(g, t)
            if int(g.turn) < t:
                per_turn["reasons"]["never_reached_turn"] = \
                    per_turn["reasons"].get("never_reached_turn", 0) + 1
                continue
            before = board_of(g, args.seat)
            t0 = time.perf_counter()
            rep = inject_quasi_state(g, args.seat, payload, args.fidelity,
                                     reset_ended=True, tier_fn=None)
            ms.append((time.perf_counter() - t0) * 1000.0)
            per_turn["attempt"] += 1
            if not rep.get("applied"):
                per_turn["rejected"] += 1
                r = str(rep.get("reason") or "?")[:80]
                per_turn["reasons"][r] = per_turn["reasons"].get(r, 0) + 1
                continue
            per_turn["applied"] += 1
            after = board_of(g, args.seat)
            ok_board = [list(x[:3]) for x in after["board"]] == \
                       [list(x[:3]) for x in want]
            ok_hand = (after["hand_n"] == len(payload.get("hand") or []))
            ok_tier = (payload.get("tier") is None
                       or int(after["tier"]) == int(payload["tier"]))
            per_turn["board_match"] += int(ok_board)
            per_turn["hand_match"] += int(ok_hand)
            per_turn["tier_match"] += int(ok_tier)
            strengths.append(sum(int(x[1]) + int(x[2]) for x in want))
            report["rows"].append({
                "turn": int(t), "idx": int(k), "applied": True,
                "want_strength": int(sum(int(x[1]) + int(x[2]) for x in want)),
                "want_board": [list(x) for x in want],
                "got_board": after["board"], "board_match": bool(ok_board),
                "want_hand_n": len(payload.get("hand") or []),
                "got_hand_n": after["hand_n"], "hand_match": bool(ok_hand),
                "want_tier": payload.get("tier"), "got_tier": after["tier"],
                "tier_match": bool(ok_tier),
                "health_before": before["health"], "health_after": after["health"],
                "tier_before": before["tier"], "tier_after": after["tier"],
                "overlay": rep.get("overlay"),
                "pool_balance_ok": rep.get("pool_balance_ok"),
            })
        if ms:
            per_turn["ms_mean"] = round(sum(ms) / len(ms), 3)
        if strengths:
            strengths.sort()
            per_turn["strength_p50"] = int(strengths[len(strengths) // 2])
        report["by_turn"][str(t)] = per_turn
        for key in ("attempt", "applied", "board_match", "hand_match", "tier_match",
                    "rejected"):
            agg[key] += int(per_turn[key])
        print(f"[turn {t:2d}] attempt={per_turn['attempt']} applied={per_turn['applied']} "
              f"board_match={per_turn['board_match']} hand={per_turn['hand_match']} "
              f"tier={per_turn['tier_match']} rejected={per_turn['rejected']} "
              f"ms={per_turn['ms_mean']} str50={per_turn['strength_p50']}"
              + (f" reasons={per_turn['reasons']}" if per_turn["reasons"] else ""),
              flush=True)
    report["totals"] = dict(agg)
    print("[totals]", json.dumps(dict(agg), ensure_ascii=False), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[out] {args.out}", flush=True)
    # 门禁：接受率与逐字段一致性都必须接近满分，否则不要拿它去训练。
    bad = (agg["attempt"] > 0 and agg["applied"] < agg["attempt"]
           or agg["applied"] > 0 and agg["board_match"] < agg["applied"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
