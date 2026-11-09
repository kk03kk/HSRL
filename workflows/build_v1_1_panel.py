#!/usr/bin/env python3
"""V1.1 对 7 个引擎 bot 的完整对局采集、统计与 EOT 战团库构建。

固定口径：
- 每局恰好一个 V1.1 学习席，其余 7 席为引擎内置 bot；
- 学习席按 game_index % 8 轮转，200 局即每席 25 局；
- 学习席 greedy（temperature=0），避免采样温度改变强度口径；
- 每个招募阶段 EndTurn 前保存学习席完整 EOT 战团；
- 动作、金币、等级、商店、买卖结果写入 actions/trajectory；
- EOT 库不套 Firestone/P50 校准，直接保存原始战团。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.common import engine_info, mask_from_bytes, obs_arrays  # noqa: E402
from hsbg.az.train.engine_quality import require_clean_engine  # noqa: E402
from hsbg.az.train.mirror_snapshots import (  # noqa: E402
    MirrorSnapshotPool, capture_eot, engine_stamp, write_snapshot_file,
)
from hsbg.az.train.personal import fingerprint, library_digest, sha256, write_json  # noqa: E402
from hsbg.az.train.model import load_checkpoint  # noqa: E402
from hsbg.az.train.ppo import PPOEvaluator, PPOGreedyPolicy  # noqa: E402


def _pct(n: int, d: int) -> float:
    return 0.0 if not d else 100.0 * float(n) / float(d)


def _mean(values: List[float]) -> Optional[float]:
    return None if not values else float(mean(values))


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    a = sorted(float(v) for v in values)
    return float(a[min(len(a) - 1, max(0, int(math.floor(q * (len(a) - 1) + 0.5))))])


class Cards:
    def __init__(self, E: Any) -> None:
        self.E = E
        self.cache: Dict[int, tuple] = {}

    def get(self, index: int) -> tuple:
        i = int(index)
        if i not in self.cache:
            self.cache[i] = tuple(self.E.card_info((i & 0x0fff) + 1))
        return self.cache[i]

    def name(self, index: int) -> str:
        try:
            return str(self.get(index)[2])
        except BaseException:
            return f"card#{index}"

    def info(self, index: int) -> Dict[str, Any]:
        try:
            cid, en, zh, tier = self.get(index)
            return {"card_id": cid, "name": zh, "name_en": en, "card_tier": int(tier)}
        except BaseException:
            return {"card_id": f"card#{index}", "name": f"card#{index}",
                    "name_en": f"card#{index}", "card_tier": 0}


def minion_desc(cards: Cards, m: Dict[str, Any], position: int) -> Dict[str, Any]:
    info = cards.info(int(m["card"]))
    return {
        "position": int(position),
        **info,
        "atk": int(m["atk"]),
        "hp": int(m["hp"]),
        "max_hp": int(m.get("max_hp") or m["hp"]),
        "golden": bool(m.get("golden")),
        "races": int(m.get("races") or 0),
        "keywords": int(m.get("kw") or 0) | int(m.get("bonus_kw") or 0) | int(m.get("temp_kw") or 0),
    }


def board_desc(cards: Cards, board: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [minion_desc(cards, m, i) for i, m in enumerate(board or [])]


def shop_desc(cards: Cards, shop: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, slot in enumerate(shop or []):
        item = slot.get("item") if isinstance(slot, dict) else None
        if not isinstance(item, dict) or not item:
            continue
        kind = next(iter(item))
        obj = item[kind]
        if not isinstance(obj, dict):
            continue
        d = {"slot": i, "kind": kind, **cards.info(int(obj.get("card", 0)))}
        for k in ("atk", "hp", "tier", "golden", "cost"):
            if k in obj:
                d[k] = obj[k]
        out.append(d)
    return out


def write_jsonl_gz(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    import hsbg_engine as E
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    from hsbg.az.train.policies import EngineBotPolicy
    from workflows.mirror_hard_gates import _drive_to_turn

    out = Path(args.out).resolve()
    lib = Path(args.library).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录必须为空: {out}")
    if lib.exists() and any(lib.iterdir()):
        raise ValueError(f"库目录必须为空: {lib}")
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)
    (out / "report").mkdir(parents=True, exist_ok=True)
    (lib / "train").mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).resolve()
    model_sha = sha256(model_path)
    model, meta = load_checkpoint(str(model_path), device=args.device)
    evaluator = PPOEvaluator(model, device=args.device,
                             norm_mean=meta.get("norm_mean"), norm_std=meta.get("norm_std"))
    policy = PPOGreedyPolicy(evaluator, max_actions_per_turn=int(args.max_actions_per_turn),
                             move_guard=True, move_cap=int(args.move_cap),
                             move_repeat_limit=int(args.move_repeat_limit),
                             pending_guard=True, cancel_cap=int(args.cancel_cap),
                             commit_play=True, forbid_cancel=bool(int(args.forbid_cancel)))
    cards = Cards(E)
    rng = np.random.default_rng(int(args.act_seed))
    sample_rng = random.Random(int(args.sample_seed))
    t0 = time.time()
    games: List[Dict[str, Any]] = []
    snaps_total = snaps_skipped = 0
    action_counts_by_turn: Dict[int, List[int]] = defaultdict(list)
    non_end_by_turn: Dict[int, List[int]] = defaultdict(list)
    type_by_turn: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    type_games_by_turn: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reached_by_turn: Dict[int, int] = defaultdict(int)
    eot_by_turn: Dict[int, int] = defaultdict(int)
    tier_by_turn: Dict[int, Counter] = defaultdict(Counter)
    gold_by_turn: Dict[int, List[int]] = defaultdict(list)
    strength_by_turn: Dict[int, List[int]] = defaultdict(list)
    board_size_by_turn: Dict[int, List[int]] = defaultdict(list)
    purchases_by_tier: Dict[int, Counter] = defaultdict(Counter)
    purchase_meta: Dict[tuple, Dict[str, Any]] = {}
    purchases_all: Counter = Counter()
    final_rows: List[Dict[str, Any]] = []
    cancel_events = 0
    move_counts: Counter = Counter()
    max_move_per_turn = 0

    def reached_turn(turn: int) -> None:
        if not reached_by_turn.get(turn):
            reached_by_turn[turn] = 0
        reached_by_turn[turn] += 1

    for gi in range(int(args.games)):
        seed = int(args.seed) + gi
        model_seat = gi % 8
        game = E.Game(seed=seed)
        policy.reset()
        snaps: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        traj_frames: List[Dict[str, Any]] = []
        non_end_per_turn: Counter = Counter()
        game_types_per_turn: Dict[int, set] = defaultdict(set)
        pending_model_actions: List[List[int]] = []
        cur_turn = int(game.turn)
        cur_state = game.state_json()
        reached = set()
        if int(game.health()[model_seat]) > 0:
            reached.add(cur_turn)
            reached_turn(cur_turn)
        rounds = 0
        while not game.is_over() and rounds < int(args.max_steps):
            rounds += 1
            acted = False
            for p in range(8):
                if game.is_over() or not game.needs_action(p):
                    continue
                acted = True
                turn_before = int(game.turn)
                if int(game.health()[model_seat]) > 0 and turn_before not in reached:
                    reached.add(turn_before)
                    reached_turn(turn_before)
                if p == model_seat:
                    decision = policy.decide(game, int(p))
                    a = int(decision.action)
                    forced = bool((decision.info or {}).get("forced"))
                    info = {"forced_end": True} if forced else {}
                    before = json.loads(game.player_json(int(p)))
                    name = E.action_name(int(a))
                    atype = name.split("(", 1)[0].split(" {", 1)[0].strip()
                    bought = sold = None
                    if atype == "Buy":
                        slot = int(name.split("(", 1)[1].split(")", 1)[0])
                        shop = shop_desc(cards, before.get("shop") or [])
                        hit = next((x for x in shop if x["slot"] == slot), None)
                        if hit and hit.get("kind") == "Minion":
                            bought = hit
                    elif atype == "Sell":
                        slot = int(name.split("(", 1)[1].split(")", 1)[0])
                        b = before.get("board") or []
                        if 0 <= slot < len(b):
                            sold = minion_desc(cards, b[slot], slot)
                    cap_turn = len([x for x in pending_model_actions if x[0] == p]) + 1
                    event = {
                        "game": gi, "seed": seed, "model_seat": model_seat,
                        "turn": turn_before, "seat": p, "role": "model",
                        "action": int(a), "action_name": name, "action_type": atype,
                        "model_action_index": cap_turn,
                        "gold_before": int(before.get("gold", 0)),
                        "tier_before": int(before.get("tier", 0)),
                        "actions_this_turn_before": int(before.get("actions_this_turn", 0)),
                        "board_size_before": len(before.get("board") or []),
                        "hand_size_before": len(before.get("hand") or []),
                        "shop_before": shop_desc(cards, before.get("shop") or []),
                        "bought_minion": bought, "sold_minion": sold,
                        **info,
                    }
                    if atype == "Move":
                        k = int(a)
                        # 196..244, from/to 7x7
                        k2 = k - 196
                        event["move_from"] = k2 // 7
                        event["move_to"] = k2 % 7
                    # EOT：必须在 step 前抓取；clone 内保持战斗前状态。
                    if a == 0 and turn_before >= int(args.snap_from_turn):
                        fr = capture_eot(game, int(p), action=0)
                        if fr is not None and len(fr.get("player", {}).get("board") or []) >= int(args.snap_min_board):
                            fr["source_game_seed"] = seed
                            fr["source_seat"] = int(p)
                            fr["seat"] = 0
                            fr["model_turn"] = turn_before
                            snaps.append(fr)
                        else:
                            snaps_skipped += 1
                    game.step(int(p), int(a))
                    after = json.loads(game.player_json(int(p))) if int(game.health()[p]) > 0 else before
                    event.update({
                        "gold_after": int(after.get("gold", 0)),
                        "tier_after": int(after.get("tier", 0)),
                        "board_size_after": len(after.get("board") or []),
                        "hand_size_after": len(after.get("hand") or []),
                    })
                    event["action_type"] = atype
                    events.append(event)
                    pending_model_actions.append([int(p), int(a)])
                    turn_key = turn_before
                    type_by_turn[turn_key][atype] += 1
                    game_types_per_turn[turn_key].add(atype)
                    if atype == "Cancel":
                        cancel_events += 1
                    if atype == "Move":
                        move_counts[(seed, turn_before, int(p))] += 1
                        max_move_per_turn = max(max_move_per_turn, move_counts[(seed, turn_before, int(p))])
                    if atype == "Buy" and bought:
                        tier = int(bought.get("card_tier") or bought.get("tier") or 0)
                        nm = str(bought.get("name"))
                        purchases_by_tier[tier][nm] += 1
                        purchases_all[(nm, tier)] += 1
                        purchase_meta[(nm, tier)] = bought
                    if atype != "EndTurn":
                        non_end_per_turn[turn_key] += 1
                else:
                    game.bot_play(int(p))
                if game.turn != cur_turn:
                    traj_frames.append({"turn": cur_turn, "state": cur_state,
                                        "model_actions": list(pending_model_actions)})
                    cur_turn = int(game.turn)
                    cur_state = game.state_json()
                    pending_model_actions = []
                    if int(game.health()[model_seat]) > 0 and cur_turn not in reached:
                        reached.add(cur_turn)
                        reached_turn(cur_turn)
                if game.is_over():
                    break
            if not acted:
                break
        if pending_model_actions:
            traj_frames.append({"turn": cur_turn, "state": cur_state,
                                "model_actions": list(pending_model_actions)})
        require_clean_engine(game)
        if not game.is_over():
            raise RuntimeError(f"游戏未正常结束 seed={seed} rounds={rounds}")
        placements = [int(x) for x in game.placements()]
        rank = int(placements[model_seat])
        end_turn = int(game.turn)
        final_turn = int(snaps[-1]["turn"]) if snaps else None
        final_board = board_desc(cards, snaps[-1]["player"].get("board") or []) if snaps else []
        # turn-level model action totals (including EndTurn); one entry per reached game.
        per_turn_model = Counter()
        for x in events:
            if x["role"] == "model":
                per_turn_model[int(x["turn"])] += 1
        for t in reached:
            t = int(t)
            action_counts_by_turn[t].append(int(per_turn_model.get(t, 0)))
            non_end_by_turn[t].append(int(non_end_per_turn.get(t, 0)))
            for atype in game_types_per_turn.get(t, set()):
                type_games_by_turn[t][atype] += 1
        # EOT stats
        for fr in snaps:
            t = int(fr["turn"])
            pj = fr["player"]
            eot_by_turn[t] += 1
            tier_by_turn[t][int(pj.get("tier", 0))] += 1
            gold_by_turn[t].append(int(pj.get("gold", 0)))
            b = pj.get("board") or []
            board_size_by_turn[t].append(len(b))
            strength_by_turn[t].append(sum(int(m["atk"]) + int(m["hp"]) for m in b))
        game_row = {
            "game": gi, "seed": seed, "model_seat": model_seat, "rank": rank,
            "end_turn": end_turn, "final_eot_turn": final_turn,
            "final_board": final_board, "reached_turns": sorted(reached),
            "model_actions": len(events), "eot_frames": len(snaps),
            "placements": placements,
        }
        games.append(game_row)
        if snaps:
            write_snapshot_file(str(lib / "train" / f"snap_{seed:08d}.json.gz"), snaps,
                                seed=seed, learner_seat=model_seat,
                                engine=engine_stamp(),
                                meta={"driver": "model_eot_v1.1", "checkpoint": str(model_path),
                                      "checkpoint_sha256": model_sha, "model_seat": model_seat,
                                      "temperature": float(args.temperature),
                                      "max_actions_per_turn": int(args.max_actions_per_turn),
                                      "snap_from_turn": int(args.snap_from_turn),
                                      "min_board": int(args.snap_min_board)})
            snaps_total += len(snaps)
        with gzip.open(out / "trajectories" / f"traj_{seed:08d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"schema": "hsai.v1.1-panel-trajectory/v1", "seed": seed,
                       "model_seat": model_seat, "placements": placements,
                       "rank": rank, "end_turn": end_turn, "frames": traj_frames,
                       "final_eot_turn": final_turn, "final_board": final_board,
                       "meta": {"checkpoint": str(model_path), "checkpoint_sha256": model_sha,
                                "engine": engine_info(), "model_actions": len(events)}},
                      f, ensure_ascii=False)
        write_jsonl_gz(out / "actions" / f"actions_{seed:08d}.jsonl.gz", events)
        print(json.dumps({"game": gi, "seed": seed, "seat": model_seat, "rank": rank,
                          "end_turn": end_turn, "snaps": len(snaps)}, ensure_ascii=False), flush=True)

    # Overall stats.
    ranks = [int(g["rank"]) for g in games]
    rank_dist = Counter(ranks)
    overall = {
        "games": len(games), "mean_rank": float(mean(ranks)),
        "top4_rate": float(sum(r <= 4 for r in ranks) / len(ranks)),
        "win_rate": float(sum(r == 1 for r in ranks) / len(ranks)),
        "rank_distribution": {str(k): int(rank_dist.get(k, 0)) for k in range(1, 9)},
        "mean_end_turn": float(mean(int(g["end_turn"]) for g in games)),
        "engine": engine_info(),
        "model": {"path": str(model_path), "sha256": model_sha, "version": 9691,
                  "cum_decisions": 810501828, "temperature": float(args.temperature)},
    }

    # Per-turn table. action_counts_by_turn now has one entry per reached game.
    max_turn = max(reached_by_turn) if reached_by_turn else 0
    action_rows = []
    for t in range(1, max_turn + 1):
        reached = int(reached_by_turn.get(t, 0))
        ac = action_counts_by_turn.get(t, [])
        ne = non_end_by_turn.get(t, [])
        tc = type_by_turn.get(t, {})
        hit80 = sum(1 for n in ac if n >= int(args.max_actions_per_turn))
        action_rows.append({
            "turn": t, "reached": reached,
            "avg_actions": _mean(ac), "avg_non_end_actions": _mean(ne),
            "max_actions": max(ac) if ac else 0,
            "hit_80": int(hit80), "hit_80_rate": _pct(hit80, reached),
            "avg_upgrade": float(tc.get("Upgrade", 0) / reached) if reached else 0.0,
            "upgrade_rate": _pct(type_games_by_turn[t].get("Upgrade", 0), reached),
            "avg_refresh": float(tc.get("Refresh", 0) / reached) if reached else 0.0,
            "avg_buy_minion": float(type_games_by_turn[t].get("Buy_minion", 0) / reached) if reached else 0.0,
            "avg_sell": float(tc.get("Sell", 0) / reached) if reached else 0.0,
            "avg_move": float(tc.get("Move", 0) / reached) if reached else 0.0,
            "by_type": dict(tc),
        })
    rule_audit = {
        "cancel_actions": int(cancel_events),
        "max_moves_per_turn": int(max_move_per_turn),
        "move_violations": int(sum(1 for (seed, t, s), n in move_counts.items() if n > 5)),
        "max_actions_per_turn": int(max((r["max_actions"] for r in action_rows), default=0)),
        "turns_over_80": int(sum(1 for r in action_rows for n in action_counts_by_turn.get(r["turn"], []) if n > 80)),
        "forced_actions": int(getattr(policy, "forced_actions", 0)),
        "move_guard": (policy._guard.to_meta() if getattr(policy, "_guard", None) is not None else {}),
        "pending_guard": (policy.pending_guard_meta() if hasattr(policy, "pending_guard_meta") else {}),
        "forbid_cancel": bool(int(args.forbid_cancel)),
    }
    if cancel_events:
        raise RuntimeError(f"规则审计失败：检测到 {cancel_events} 个 Cancel 动作")
    if max_move_per_turn > 5:
        raise RuntimeError(f"规则审计失败：单回合 Move 超限 {max_move_per_turn}")
    # Recompute purchase/turn stats from compact action logs.
    buy_minion_by_turn = defaultdict(int)
    # events were not retained globally; re-read compact actions files.
    for seed in [g["seed"] for g in games]:
        with gzip.open(out / "actions" / f"actions_{seed:08d}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e.get("action_type") == "Buy" and e.get("bought_minion"):
                    buy_minion_by_turn[int(e["turn"])] += 1
    for row in action_rows:
        t = int(row["turn"])
        reached = int(row["reached"])
        row["avg_buy_minion"] = float(buy_minion_by_turn.get(t, 0) / reached) if reached else 0.0

    # Tier distribution and cumulative reach.
    tier_rows = []
    for t in range(1, max_turn + 1):
        eot = int(eot_by_turn.get(t, 0))
        dist = tier_by_turn.get(t, Counter())
        row = {"turn": t, "eot": eot,
               "tier_distribution": {str(k): _pct(dist.get(k, 0), eot) for k in range(1, 7)}}
        for k in range(1, 7):
            row[f"reach_tier{k}"] = _pct(sum(v for tier, v in dist.items() if tier >= k), eot)
        tier_rows.append(row)

    # Gold / strength / survival.
    turn_rows = []
    for t in range(1, max_turn + 1):
        reached = int(reached_by_turn.get(t, 0))
        eot = int(eot_by_turn.get(t, 0))
        gold = gold_by_turn.get(t, [])
        strength = strength_by_turn.get(t, [])
        sizes = board_size_by_turn.get(t, [])
        turn_rows.append({
            "turn": t, "reached": reached, "eot": eot,
            "avg_gold": _mean(gold), "median_gold": median(gold) if gold else None,
            "gold_positive_rate": _pct(sum(1 for x in gold if x > 0), len(gold)),
            "avg_board_size": _mean(sizes), "avg_attack_health_total": _mean(strength),
            "p50_attack_health_total": _quantile(strength, .5),
            "p90_attack_health_total": _quantile(strength, .9),
            "avg_gold_per_reached": (sum(gold) / reached) if reached else None,
        })

    # Purchase top lists.
    tier_top = {}
    for tier in range(1, 7):
        rows = []
        for (nm, tr), cnt in purchases_all.items():
            if int(tr) == tier:
                rows.append({"name": nm, "tier": int(tr), "count": int(cnt)})
        rows.sort(key=lambda x: (-x["count"], x["name"]))
        tier_top[str(tier)] = rows[:10]
    overall_top = [{"name": nm, "tier": int(tr), "count": int(cnt)}
                   for (nm, tr), cnt in purchases_all.most_common(10)]

    # 10 reproducible final lineups.
    n_sample = min(10, len(games))
    sample_idx = sorted(sample_rng.sample(range(len(games)), n_sample))
    final_sample = [games[i] for i in sample_idx]

    # Library validation and manifest.
    pool = MirrorSnapshotPool(str(lib / "train"), sample="random")
    if pool.bad_files:
        raise RuntimeError(f"库存在坏文件: {pool.bad_files}")
    if pool.frames != snaps_total:
        raise RuntimeError(f"库帧数不符 pool={pool.frames} expected={snaps_total}")
    # 全帧注入检查
    template = _drive_to_turn(E.Game(seed=730000), 1).state_json()
    checked = 0
    for frames in pool.index.values():
        for fr in frames:
            game = E.Game.from_json(template)
            payload = json.dumps(fr.player, ensure_ascii=False)
            result = game.apply_turn_snapshots([[0, payload]], False)
            if (result.get("applied") != 1 or result.get("rejected")
                    or not result.get("pool_balance_ok") or not result.get("pool_species_ok")
                    or game.invariant_violations()):
                raise RuntimeError(f"EOT 帧注入失败 turn={fr.turn} seed={fr.source_game_seed} {result}")
            checked += 1
    strict = evaluate_policy(EngineBotPolicy(), EvalConfig(
        games=8, seed=int(args.seed) + 777777, mirror_pool=pool, mirror_strict=True,
        collect_exposure=True))
    ms = strict["mirror"]
    if (strict["illegal_actions"] or strict["truncated_games"] or strict["truncated_combats"]
            or any(ms[k] for k in ("mirror_errors", "mirror_missing", "mirror_rejected",
                                   "mirror_fallback", "mirror_pool_unbalanced",
                                   "mirror_pool_species_unbalanced"))
            or not ms["mirror_attempt"] == ms["mirror_applied"]):
        raise RuntimeError(f"严格镜像验收失败: {ms} illegal={strict['illegal_actions']}")
    last_turn = max(int(x.turn) for frames in pool.index.values() for x in frames)
    manifest = {
        "schema": "hsrl.library/v1", "split": "train", "driver": "model_eot_v1.1",
        "digest": library_digest(lib / "train"),
        "fingerprint": fingerprint(),
        "source_checkpoint": str(model_path), "source_checkpoint_sha256": model_sha,
        "source_version": 9691, "games": len(games), "last_turn": last_turn,
        "smoke": bool(args.games < 20),
        "model_seats": sorted({int(g["model_seat"]) for g in games}),
        "temperature": float(args.temperature),
        "max_actions_per_turn": int(args.max_actions_per_turn),
        "rules_config": json.loads(E.Game(seed=0).state_json())["cfg"],
        "validation": {"checked": checked, "accepted": checked, "quarantined": 0},
        "strict_audit": {"mirror": ms, "illegal_actions": strict["illegal_actions"],
                         "truncated_games": strict["truncated_games"],
                         "truncated_combats": strict["truncated_combats"]},
    }
    write_json(lib / "train" / "manifest.json", manifest)
    write_json(lib / "manifest.json", {"schema": "hsrl.library/v1", "split": "train",
                                       "driver": "model_eot_v1.1", "digest": manifest["digest"],
                                       "source_checkpoint": str(model_path),
                                       "source_checkpoint_sha256": model_sha,
                                       "games": len(games), "last_turn": last_turn,
                                       "validation": manifest["validation"],
                                       "fingerprint": manifest["fingerprint"]})
    stats = {
        "schema": "hsai.v1.1-panel/v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "config": vars(args), "overall": overall,
        "actions_by_turn": action_rows, "tier_by_turn": tier_rows,
        "turn_stats": turn_rows, "purchase_top_by_tier": tier_top,
        "purchase_top_all": overall_top,
        "purchase_counts_total": int(sum(purchases_all.values())),
        "guard": (policy._guard.to_meta() if getattr(policy, "_guard", None) is not None else {}),
        "pending_guard": (policy.pending_guard_meta() if hasattr(policy, "pending_guard_meta") else {}),
        "forbid_cancel": bool(int(args.forbid_cancel)),
        "forced_actions": int(getattr(policy, "forced_actions", 0)),
        "rule_audit": rule_audit,
        "final_sample": final_sample,
        "library": {"path": str(lib), "train": str(lib / "train"),
                    "frames": snaps_total, "skipped": snaps_skipped,
                    "manifest": manifest},
    }
    write_json(out / "report" / "report.json", stats)
    write_json(out / "report" / "actions_by_turn.json", action_rows)
    write_json(out / "report" / "turn_stats.json", turn_rows)
    write_json(out / "report" / "tier_by_turn.json", tier_rows)
    write_json(out / "report" / "purchase_top.json", {"by_tier": tier_top, "all": overall_top})
    write_json(out / "report" / "final_sample.json", final_sample)
    write_json(out / "report" / "game_summaries.json", games)
    write_json(out / "report" / "library_manifest.json", manifest)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="artifacts/models/v1.1/model.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=240000000)
    p.add_argument("--out", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-actions-per-turn", type=int, default=80)
    p.add_argument("--move-cap", type=int, default=5)
    p.add_argument("--move-repeat-limit", type=int, default=2)
    p.add_argument("--cancel-cap", type=int, default=2)
    p.add_argument("--forbid-cancel", type=int, default=1, choices=[0, 1])
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--snap-from-turn", type=int, default=1)
    p.add_argument("--snap-min-board", type=int, default=1)
    p.add_argument("--act-seed", type=int, default=880000)
    p.add_argument("--sample-seed", type=int, default=20261109)
    a = p.parse_args()
    if (a.games < 1 or a.temperature < 0 or a.max_actions_per_turn != 80
            or a.move_cap != 5 or a.move_repeat_limit != 2
            or a.forbid_cancel != 1):
        p.error("强制动作规则：games>=1, temperature>=0, max-actions=80, "
                "move-cap=5, move-repeat=2, forbid-cancel=1")
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
