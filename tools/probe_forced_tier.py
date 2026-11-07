"""强制升本的本地可行性/效果探针（真实 7357 网络 + 引擎 bot 对手，CPU 即可）。

回答三个问题：

1. 计划 ``2:2,5:3,6:4,8:5`` 在正常经济规则下能否稳定执行（on_time / deferred / missed）；
2. 强制执行后模型真的到了 5 本吗，5 本之后战团有没有成长（逐回合总攻血 P50）；
3. 对照（不强制）的 tier 曲线与成长差多少。

只跑评测、不更新参数；不作为棋力结论，也不代替 NUS 的库注入评测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from hsbg.az.train.forced_tier import (  # noqa: E402
    DEFAULT_FORCED_TIER_PLAN_SPEC,
    ForcedTierTracker,
    UPGRADE_ACTION,
    forced_tier_rates,
    plan_spec_of,
)
from hsbg.az.train.ppo import PPOEvaluator  # noqa: E402
from hsbg.az.train.model import load_checkpoint  # noqa: E402


def _p50(vals):
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), 50))


def run_arm(model, *, games: int, seed0: int, plan: str, carry: int,
            greedy: bool = True) -> dict:
    import hsbg_engine as E

    ev = PPOEvaluator(model, device="cpu")
    tracker = ForcedTierTracker(plan, carry=carry) if plan else None
    per_turn = defaultdict(lambda: dict(tier=[], gold=[], strength=[], n=0))
    ranks = []
    taken = Counter()
    for gi in range(games):
        game = E.Game(seed=seed0 + gi)
        seat = 0
        steps = 0
        turn_actions = 0
        last_turn = -1
        if tracker is not None:
            tracker.reset()
        while not game.is_over() and steps < 4000:
            acted = False
            for p in range(8):
                if not game.needs_action(p):
                    continue
                if p != seat:
                    game.bot_play(p)
                    acted = True
                    continue
                turn = int(game.turn)
                if turn != last_turn:
                    last_turn = turn
                    turn_actions = 0
                turn_actions += 1
                mask = np.frombuffer(game.legal_mask(seat), dtype=np.uint8).astype(bool)
                if turn_actions > 80 and bool(mask[0]):
                    # 与训练同口径的每回合动作上限，防 Play/Cancel 循环
                    game.step(seat, 0)
                    acted = True
                    steps += 1
                    continue
                obs = game.observe(seat)
                f = np.frombuffer(obs[0], dtype="<f4").copy()
                c = np.frombuffer(obs[1], dtype="<i4").copy()
                pl = json.loads(game.player_json(seat))
                tier = int(pl.get("tier") or 1)
                fd = (tracker.decide(game_id=gi, seat=seat, turn=turn, tier=tier,
                                     upgrade_legal=bool(mask[UPGRADE_ACTION]),
                                     force_prob=1.0)
                      if tracker is not None else None)
                if fd is not None and fd.force and bool(mask[UPGRADE_ACTION]):
                    action = UPGRADE_ACTION
                    taken["forced"] += 1
                else:
                    prior, _v = ev(f, c, mask, seat)
                    p_arr = np.asarray(prior, dtype=np.float64).reshape(-1)
                    p_arr = np.where(mask, np.maximum(p_arr, 0.0), 0.0)
                    if p_arr.sum() <= 0:
                        p_arr = mask.astype(np.float64)
                    if greedy:
                        action = int(np.argmax(p_arr))
                    else:
                        action = int(np.random.default_rng(steps).choice(
                            p_arr.size, p=p_arr / p_arr.sum()))
                    if turn in (2, 5, 6, 8) and int(pl.get("tier") or 1) >= (
                            {2: 2, 5: 3, 6: 4, 8: 5}.get(turn) or 99):
                        pass
                if action == UPGRADE_ACTION:
                    taken["model"] += 1
                if int(game.turn) != turn:
                    pass
                game.step(seat, action)
                board = pl.get("board") or []
                per_turn[turn]["n"] += 1
                per_turn[turn]["tier"].append(tier)
                per_turn[turn]["gold"].append(int(pl.get("gold") or 0))
                per_turn[turn]["strength"].append(
                    sum(int(m["atk"]) + int(m["hp"]) for m in board))
                acted = True
                steps += 1
                if game.is_over():
                    break
            if not acted:
                break
        ranks.append(int(game.placements()[seat]) if game.is_over() else 0)

    rows = {}
    for turn in sorted(per_turn):
        v = per_turn[turn]
        ti = np.asarray(v["tier"], dtype=np.float64)
        rows[turn] = dict(
            n=len(ti),
            tier_p50=float(np.percentile(ti, 50)),
            tier_ge4=float(np.mean(ti >= 4)),
            tier_ge5=float(np.mean(ti >= 5)),
            tier_ge6=float(np.mean(ti >= 6)),
            gold_p50=_p50(v["gold"]),
            strength_p50=_p50(v["strength"]),
        )
    known = [r for r in ranks if r >= 1]
    out = dict(
        games=games, plan=plan, carry=carry,
        mean_rank=float(np.mean(known)) if known else None,
        top4_rate=float(np.mean([r <= 4 for r in known])) if known else None,
        finish_turn_p50=float(np.percentile([max(per_turn)] * len(known), 50)) if known else None,
        turns=rows,
        upgrade_actions=dict(taken),
        forced_tier=(tracker.stats() if tracker is not None else {}),
    )
    if tracker is not None:
        out["forced_tier_rates"] = forced_tier_rates(out["forced_tier"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(
        ROOT / "artifacts/models/nus-1080772/step-007357-360m/model.pt"))
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11000000)
    ap.add_argument("--plan", default=DEFAULT_FORCED_TIER_PLAN_SPEC)
    ap.add_argument("--carry", type=int, default=1)
    ap.add_argument("--out", default=str(
        ROOT / "records/research/long1085131/forced-tier-probe.json"))
    args = ap.parse_args()

    import torch
    # 小批量前向在 CPU 上多线程会互相抢核（实测 16 线程比 4 线程慢一个数量级）
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))
    model, meta = load_checkpoint(args.checkpoint, device="cpu")
    print(f"[probe] checkpoint={args.checkpoint} games={args.games} "
          f"plan={args.plan!r} carry={args.carry}", flush=True)
    free = run_arm(model, games=args.games, seed0=args.seed, plan="", carry=args.carry)
    forced = run_arm(model, games=args.games, seed0=args.seed, plan=args.plan,
                     carry=args.carry)
    result = {
        "schema": "hsai.forced_tier_probe/v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": meta.get("sha256"),
        "games_per_arm": int(args.games),
        "seed": int(args.seed),
        "opponents": "engine_bot (8 seats, learner=seat0, greedy)",
        "note": "本地探针：无镜像库注入、无参数更新；不能当作 NUS 长训的棋力结论",
        "free": free,
        "forced": forced,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n[turn]  free(tier_p50 / >=5 / str_p50)   forced(tier_p50 / >=5 / str_p50)")
    for turn in sorted(set(free["turns"]) | set(forced["turns"])):
        a = free["turns"].get(turn) or {}
        b = forced["turns"].get(turn) or {}
        fmt = lambda r: ("%4.1f / %3.0f%% / %6.0f" % (
            r.get("tier_p50") or 0, 100 * (r.get("tier_ge5") or 0),
            r.get("strength_p50") or 0)) if r else "   -"
        print(f"  T{turn:<2}  {fmt(a):>26}   {fmt(b):>26}")
    print("\nfree  : mean_rank=%s top4=%s upgrades=%s" % (
        free["mean_rank"], free["top4_rate"], free["upgrade_actions"]))
    print("forced: mean_rank=%s top4=%s upgrades=%s" % (
        forced["mean_rank"], forced["top4_rate"], forced["upgrade_actions"]))
    print("forced plan rates:", json.dumps(forced.get("forced_tier_rates"), ensure_ascii=False))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
