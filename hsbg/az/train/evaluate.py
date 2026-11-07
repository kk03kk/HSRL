"""固定 seed 的策略评测：seat0 = 被评策略，其余 7 座 = 引擎 bot。

评测协议（docs/AZ_MCTS_SELFPLAY_DESIGN.md §9）：
- 评测 seed 集与自对弈 seed 分离（默认从 ``--seed`` 递增，调用方保证不重叠）；
- 记录：平均名次（主）、前四率、冠军率、期望效用、合法动作率、truncated、耗时、决策数；
- 引擎 bot 基线 = 同一循环里 seat0 也走 ``bot_action``（特权，仅作基线）。

**人类战团对手面板**（``docs/EVAL_PANELS.md``）：``EvalConfig.human_board_pool`` 非空时，
每个**新回合开始、任何座位行动前**，把该 turn 分位的人类战团（board-only）注入全部
非 learner 座位。注入是 board-only 近似（无 hand/shop/池差量），只用于对手强度评测，
不改变被评策略的观测/动作口径。注入 RNG 按 ``(seed, game, turn, seat)`` 独立派生，
因此同一 seed 的 model / bot 基线面对的是同一批人类战团。

只用 ``observe/legal_mask`` 作为被评策略的决策输入；``bot_action`` 只出现在基线策略里。
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .common import (
    ACTION_SEGMENTS,
    N_PLAYERS,
    group_of,
    load_engine,
    mask_from_bytes,
    obs_arrays,
    utility_from_rank,
)
from .human_board_pool import HumanBoardStats, inject_turn
from .policies import EngineBotPolicy, Policy
from .forced_tier import ForcedTierTracker
from .mirror_snapshots import MirrorStats, inject_turn as inject_mirror_turn


@dataclass
class EvalConfig:
    """评测配置。"""

    games: int = 20
    seed: int = 90000
    policy_seat: int = 0
    max_steps: int = 50000
    opponent: str = "engine_bot"   # 其余座位策略（当前只支持引擎 bot）
    collect_examples: int = 0      # >0 时保留若干决策样本用于人工抽查
    #: 人类战团池（``HumanBoardPool``）；None = 纯引擎 bot 面板（零影响）。
    human_board_pool: Any = None
    #: 注入时是否同时写入按 turn 派生的 tier（人类数据没有 tier 字段）。
    human_board_set_tier: bool = True
    #: 固定回合末库；与 human_board_pool 互斥。逐座位派生 RNG 保证配对可比。
    mirror_pool: Any = None
    mirror_strict: bool = False
    collect_economy: bool = False
    collect_exposure: bool = False  # 有界研究面板才读取实际战斗，不影响默认热路径
    #: 评测期强制升本计划（``"2:2,5:3,6:4,8:5"``；空 = 关闭，模型完全自主）。
    #: 只用于「规则内」探针；正式 Dev 评测必须保持空，否则棋力不可比。
    forced_tier_plan: str = ""
    forced_tier_carry: int = 1


def _mix_seed(*parts: int) -> int:
    """把 ``(seed, game, turn, seat)`` 混合成稳定的 63-bit 注入 RNG 种子。

    **不能**依赖调用顺序：model / bot 两条基线局内死亡座位不同，若用单一 RNG 顺序
    消耗，同一 seed 的两条基线在后期会抽到不同的战团，配对差分就失去了可比性。
    """
    h = 0x9E3779B97F4A7C15
    for p in parts:
        h ^= (int(p) + 0x9E3779B97F4A7C15 + (h << 6) + (h >> 2)) & 0xFFFFFFFFFFFFFFFF
        h &= 0xFFFFFFFFFFFFFFFF
    return int(h % ((1 << 63) - 1))


def _inject_human_turn(game: Any, config: "EvalConfig", *, seed: int, gi: int,
                       turn: int, stats: HumanBoardStats) -> int:
    """新回合开始：给全部存活、非 learner 座位各注入该 turn 的一条人类战团。

    逐座位独立调用 :func:`inject_turn`（``seats=[seat]``），这样每个座位的采样 RNG
    只由 ``(seed, game, turn, seat)`` 决定，与其它座位是否存活/注入成功无关。
    返回尝试注入的座位数。
    """
    pool = getattr(config, "human_board_pool", None)
    if pool is None:
        return 0
    learner = int(config.policy_seat)
    try:
        health = [int(h) for h in game.health()]
    except Exception:  # noqa: BLE001 - 旧构建没有 health() -> 不按血量过滤
        health = [1] * N_PLAYERS
    seats = [s for s in range(N_PLAYERS)
             if s != learner and 0 <= s < len(health) and health[s] > 0]
    for seat in seats:
        rng = random.Random(_mix_seed(int(seed) + int(gi), int(turn), int(seat)))
        inject_turn(game, pool, int(turn), seats=[int(seat)], p=1.0, rng=rng,
                    learner_seat=learner, stats=stats,
                    set_tier=bool(getattr(config, "human_board_set_tier", True)))
    return len(seats)


def evaluate_policy(policy: Policy, config: EvalConfig) -> Dict[str, Any]:
    """评测一个策略（固定 seeds，seat0 vs 7 引擎 bot / 可选人类战团注入）。"""
    E = load_engine()
    strict_library = None
    if config.mirror_strict:
        if config.mirror_pool is None:
            raise ValueError("严格镜像评测需要校准库")
        from .strict_mirror import StrictLibrary, StrictMirror
        strict_library = StrictLibrary(config.mirror_pool)
        strict_counts = Counter()
    if config.mirror_pool is not None and config.human_board_pool is not None:
        raise ValueError("两个战团注入源不能同时启用")
    seat0 = int(config.policy_seat)
    ft_plan = str(getattr(config, "forced_tier_plan", "") or "").strip()
    ft_tracker = (ForcedTierTracker(ft_plan,
                                    carry=int(getattr(config, "forced_tier_carry", 1) or 0))
                  if ft_plan else None)
    forced_tier_actions = 0
    ranks: List[int] = []
    truncated = 0
    decisions = 0
    illegal = 0
    examples: List[Dict[str, Any]] = []
    truncated_combats = 0
    hb_stats = HumanBoardStats()
    mirror_stats = MirrorStats()
    hb_turns = 0
    mirror_skipped_ended = 0
    exposure = []
    economy = []
    card_names = {}
    def describe_board(board):
        result = []
        for position, minion in enumerate(board):
            index = int(minion['card'])
            if index not in card_names:
                card_names[index] = E.card_info(index)[:2]
            card_id, name = card_names[index]
            result.append({**minion, 'position':position, 'card_id':card_id, 'name':name})
        return result
    t0 = time.time()
    for gi in range(int(config.games)):
        game = E.Game(seed=int(config.seed) + gi)
        game_seed = int(config.seed) + gi
        policy.reset()
        steps = 0
        last_turn = -1
        state: Dict[str, int] = {"turns": 0}
        injected = {}
        combat_seen = set()
        turn_actions = 0
        economy_turn = -1
        action_counts = Counter()
        previous_action = None
        repeated_actions = 0
        upgrade_legal_steps = 0
        gold_events = []
        strict = None
        if strict_library is not None:
            strict = StrictMirror(
                strict_library, seat0, random.Random(_mix_seed(game_seed, seat0)), mirror_stats,
                frame_rng=lambda turn, target: random.Random(_mix_seed(game_seed, turn, target)),
            )

        def _record_combats():
            if not config.collect_exposure:
                return
            for combat in json.loads(game.state_json()).get('last_combats', []):
                a, b = combat['a'], combat['b']
                key = (combat['turn'], a, b)
                if key in combat_seen or seat0 not in (a, b):
                    continue
                combat_seen.add(key)
                opponent = b if a == seat0 else a
                info = injected.get((combat['turn'], opponent), {})
                winner = combat['winner']
                won = winner == (0 if a == seat0 else 1)
                exposure.append(dict(seed=game_seed, learner=seat0, turn=combat['turn'],
                    opponent=opponent, ghost=b is None, mirror_applied=bool(info),
                    snapshot=info.get('snapshot'), source_strength=info.get('strength'),
                    source_turn=info.get('source_turn'),
                    source=info.get('source'), mixture_source=info.get('mixture_source'),
                    win=won, tie=winner == 2, damage=combat['damage'],
                    damage_taken=0 if won or winner == 2 else combat['damage'],
                    opponent_board=info.get('board'), opponent_board_stage='injected_snapshot_eot'))

        def _maybe_inject() -> None:
            """在**任何座位行动前**检查是否进入新回合；新回合则注入一次。

            必须在 while 顶层和 ``for p`` 内都调用：引擎可能在 ``for p`` 中途推进
            turn（例如 learner 结束本回合后，后续座位的 ``bot_play`` 已经属于新回合），
            只在 while 顶层注入会错过窗口、被 ``already_ended`` 拒绝。
            """
            nonlocal last_turn, mirror_skipped_ended
            if config.human_board_pool is None and config.mirror_pool is None:
                return
            turn_now = int(game.turn)
            if turn_now == last_turn:
                return
            last_turn = turn_now
            if turn_now < 1:
                # 引擎 turn=0 是开局准备帧（人类数据从 turn=1 起，且该帧注入会被
                # 引擎以 not_recruit 拒绝）；直接从 turn=1 开始注入。
                return
            if strict is not None:
                strict.prepare(game)
                for target, frame in strict.frames.items():
                    injected[(turn_now, target)] = dict(snapshot=frame.path, strength=frame.strength,
                        source_turn=frame.source_turn or frame.turn,
                        source=frame.source_label, mixture_source=frame.mixture_source,
                        board=describe_board(json.loads(game.player_json(target))['board']) if config.collect_exposure else None)
            elif config.mirror_pool is not None:
                for target in range(N_PLAYERS):
                    if target != seat0:
                        # 若目标已经ended，原生注入必拒绝且不改状态。
                        # 跳过无效目标，不能把already_ended混进载荷/引用故障。
                        if bool(json.loads(game.player_json(target)).get("ended_turn")):
                            mirror_skipped_ended += 1
                            continue
                        old_applied = mirror_stats.applied
                        inject_mirror_turn(
                            game, config.mirror_pool, turn_now, seats=[target],
                            rng=random.Random(_mix_seed(game_seed, turn_now, target)),
                            learner_seat=seat0, stats=mirror_stats, advance=False)
                        if config.collect_exposure and mirror_stats.applied > old_applied:
                            # 独立重建同一采样，不消耗环境或策略RNG。
                            frame = config.mirror_pool.get(turn_now, target,
                                rng=random.Random(_mix_seed(game_seed, turn_now, target)))
                            injected[(turn_now, target)] = dict(snapshot=frame.path, strength=frame.strength,
                                source=frame.source_label, mixture_source=frame.mixture_source)
                # learner 已淘汰时，七个 ended 快照可能使全桌结束；显式触发战斗推进。
                game.apply_turn_snapshots([], True)
                _record_combats()
            else:
                _inject_human_turn(game, config, seed=int(config.seed), gi=gi,
                                   turn=turn_now, stats=hb_stats)
            state["turns"] += 1

        while not game.is_over() and steps < int(config.max_steps):
            _maybe_inject()
            acted = False
            for p in range(N_PLAYERS):
                _maybe_inject()
                if not game.needs_action(p):
                    continue
                seat = int(p)
                if seat == seat0:
                    mask = mask_from_bytes(game.legal_mask(seat))
                    dec = policy.decide(game, seat)
                    action = int(dec.action)
                    forced_tier_here = 0
                    if ft_tracker is not None:
                        _pj = json.loads(game.player_json(seat))
                        _fd = ft_tracker.decide(
                            game_id=gi, seat=seat, turn=int(game.turn),
                            tier=int(_pj.get("tier") or 1),
                            upgrade_legal=bool(mask[3]), force_prob=1.0)
                        if bool(getattr(_fd, "force", False)) and bool(mask[3]):
                            action, forced_tier_here = 3, 1
                            forced_tier_actions += 1
                    if not (0 <= action < mask.size and bool(mask[action])):
                        illegal += 1
                        legal = np.flatnonzero(mask)
                        action = int(legal[0]) if legal.size else 0
                    if len(examples) < int(config.collect_examples):
                        f, c = obs_arrays(game.observe(seat))
                        examples.append({"features": f, "cards": c, "mask": mask,
                                         "action": action, "seat": seat,
                                         "turn": int(game.turn), "seed": game_seed,
                                         "info": dec.info})
                    if config.collect_economy:
                        turn = int(game.turn)
                        if turn != economy_turn:
                            economy_turn, turn_actions = turn, 0
                            action_counts.clear()
                            previous_action = None
                            repeated_actions = 0
                            upgrade_legal_steps = 0
                            gold_events = []
                        gold_before = int(json.loads(game.player_json(seat))['gold'])
                        turn_actions += 1
                        action_counts[group_of(action)] += 1
                        upgrade_legal_steps += int(mask[ACTION_SEGMENTS["upgrade"][0]])
                        repeated_actions += action == previous_action
                        previous_action = action
                        if action == 0:
                            player = json.loads(game.player_json(seat))
                            board = player["board"]
                            from .mirror_snapshots import capture_eot
                            eot = capture_eot(game, seat)
                            eot_board = eot['player']['board'] if eot is not None else None
                            economy.append(dict(
                                seed=game_seed, turn=turn, tier=int(player["tier"]),
                                gold=int(player["gold"]), health=int(player["health"]),
                                attack=sum(int(m["atk"]) for m in board),
                                board_hp=sum(int(m["hp"]) for m in board),
                                actions=turn_actions, action_counts=dict(action_counts),
                                consecutive_repeats=repeated_actions,
                                upgrade_legal_steps=upgrade_legal_steps,
                                forced=bool(dec.info.get("forced", False)),
                                board=describe_board(board), board_stage='before_end_turn',
                                eot_available=eot_board is not None,
                                eot_board=describe_board(eot_board) if eot_board is not None else None,
                                eot_attack=sum(int(m['atk']) for m in eot_board) if eot_board is not None else None,
                                eot_board_hp=sum(int(m['hp']) for m in eot_board) if eot_board is not None else None,
                                eot_board_stage='recruitment_end_before_combat',
                                gold_events=list(gold_events),
                            ))
                            turn_actions = 0
                    game.step(seat, action)
                    if config.collect_economy and action != 0:
                        after = json.loads(game.player_json(seat))
                        gold_events.append(dict(action=action, kind=group_of(action),
                            before=gold_before, after=int(after['gold']),
                            net_change=int(after['gold'])-gold_before))
                    if strict is not None:
                        strict.after_learner_step(game)
                    decisions += 1
                else:
                    if strict is not None and int(game.turn) > 0:
                        raise RuntimeError(f"严格镜像评测：非库席行动 turn={game.turn} seat={seat}")
                    game.bot_play(seat)
                if config.collect_exposure and (int(game.turn) != last_turn or game.is_over()):
                    _record_combats()
                acted = True
                steps += 1
                if game.is_over():
                    break
            if not acted:
                if strict is not None and not game.is_over():
                    before_turn = int(game.turn)
                    game.apply_turn_snapshots([], True)
                    strict.after_learner_step(game)
                    _record_combats()
                    if game.is_over() or int(game.turn) != before_turn:
                        continue
                if config.mirror_pool is not None and int(game.turn) != last_turn:
                    continue
                break
        if strict is not None:
            strict_counts.update(strict.counts)
        hb_turns += int(state["turns"])
        if game.is_over():
            ranks.append(int(game.placements()[seat0]))
        else:
            truncated += 1
            ranks.append(0)
        from .engine_quality import require_clean_engine
        quality = require_clean_engine(game)
        truncated_combats += quality['truncated_combats']
    elapsed = time.time() - t0
    known = np.asarray([r for r in ranks if r >= 1], dtype=np.float32)
    n_known = int(known.size)
    util = utility_from_rank(known) if n_known else np.zeros(0, dtype=np.float32)
    rank_counts: Dict[int, int] = {}
    for r in ranks:
        rank_counts[int(r)] = rank_counts.get(int(r), 0) + 1
    hb_meta = hb_stats.to_meta()
    return {
        "policy": getattr(policy, "name", "unknown"),
        "gameplay": getattr(policy, "playstyle", None),
        "games": int(config.games),
        "games_finished": int(n_known),
        "truncated_games": int(truncated),
        "seed_start": int(config.seed),
        "seed_end": int(config.seed) + int(config.games) - 1,
        "policy_seat": seat0,
        "opponent": str(config.opponent),
        "mean_rank": float(known.mean()) if n_known else None,
        "top4_rate": float((known <= 4).mean()) if n_known else None,
        "win_rate": float((known == 1).mean()) if n_known else None,
        "mean_utility": float(util.mean()) if n_known else None,
        "rank_counts": rank_counts,
        "rank_sequence": [int(r) for r in ranks],
        "decisions": int(decisions),
        "decisions_per_game": float(decisions / max(config.games, 1)),
        "illegal_actions": int(illegal),
        "legal_action_rate": float(1.0 - illegal / max(decisions, 1)),
        "truncated_combats": int(truncated_combats),
        "seconds": float(elapsed),
        "ms_per_decision": float(1000.0 * elapsed / max(decisions, 1)),
        "games_per_s": float(config.games / max(elapsed, 1e-9)),
        # 人类战团注入计数（纯 bot 面板时全为 0；写进 panel details 便于核对是否真的生效）
        "human_board": hb_meta,
        "mirror": mirror_stats.to_meta(),
        "mirror_skipped_ended": mirror_skipped_ended,
        "combat_exposure": exposure,
        **({"mirror_strict_counts": dict(strict_counts)} if strict_library is not None else {}),
        **({"end_turn_economy": economy} if config.collect_economy else {}),
        "human_board_turns": int(hb_turns),
        "examples": examples,
        # 强制升本探针（默认关闭 -> 0，不影响默认热路径与报告口径）
        "forced_tier_actions": int(forced_tier_actions),
        **({"forced_tier": ft_tracker.stats()} if ft_tracker is not None else {}),
    }


def evaluate_baseline(config: EvalConfig) -> Dict[str, Any]:
    """引擎 bot 基线（同协议：seat0=bot，其余 bot）。"""
    cfg = EvalConfig(games=config.games, seed=config.seed,
                     policy_seat=config.policy_seat, max_steps=config.max_steps,
                     opponent=config.opponent)
    return evaluate_policy(EngineBotPolicy(), cfg)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def run_comparison(policy_factory: Dict[str, Any], config: EvalConfig) -> Dict[str, Any]:
    """跑一组策略的对比评测。

    ``policy_factory``：``{name: zero-arg callable -> Policy}``；
    返回 ``{"protocol":..., "policies": {name: metrics}, "comparison": {...}}``
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, factory in policy_factory.items():
        policy = factory()
        print(f"[eval] policy={name} games={config.games} seeds={config.seed}"
              f"..{config.seed + config.games - 1}", flush=True)
        res = evaluate_policy(policy, config)
        results[name] = res
        print(f"[eval] {name}: mean_rank={res['mean_rank']} top4={res['top4_rate']} "
              f"win={res['win_rate']} trunc={res['truncated_games']} "
              f"legal={res['legal_action_rate']:.4f} {res['seconds']:.1f}s", flush=True)
    base = results.get("bot") or results.get("engine_bot")
    comparison: Dict[str, Any] = {}
    for name, res in results.items():
        if name == "bot" or name == base:
            continue
        comparison[f"{name}_vs_bot"] = {
            "mean_rank_delta": _delta(res.get("mean_rank"),
                                      base.get("mean_rank") if base else None),
            "top4_rate_delta": _delta(res.get("top4_rate"),
                                      base.get("top4_rate") if base else None),
            "win_rate_delta": _delta(res.get("win_rate"),
                                     base.get("win_rate") if base else None),
            "mean_utility_delta": _delta(res.get("mean_utility"),
                                         base.get("mean_utility") if base else None),
        }
    return {
        "protocol": {
            "games_per_policy": int(config.games),
            "seed_start": int(config.seed),
            "seed_end": int(config.seed) + int(config.games) - 1,
            "policy_seat": int(config.policy_seat),
            "opponent": str(config.opponent),
            "max_steps": int(config.max_steps),
        },
        "policies": results,
        "comparison": comparison,
    }


__all__ = ["EvalConfig", "evaluate_policy", "evaluate_baseline", "run_comparison"]
