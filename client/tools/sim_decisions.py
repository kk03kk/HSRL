"""模拟器侧对局录制：与实机共用 ``hsai.game-trace/v2`` 记录规范。

    python client/tools/sim_decisions.py --games 1 --checkpoint <model.pt> --out <dir>
    python client/tools/sim_decisions.py --checkpoint <model.pt> --out <dir> --sample

一条对局 = 一个完整的 trace 目录（``<out>/game-001/``），里面是与实机完全同名同义
的 ``decisions.jsonl`` / ``actions.jsonl`` / ``waits.jsonl`` / ``events.jsonl`` /
``observations.npy`` / ``summary.json`` / ``audit.json``。

记录约定（与实机一致，代码见 ``hsrl_agent/trace_schema.py``）：

* 学习席的每一次前向 = 一条 ``decisions`` 行，带执行前状态、输入向量指纹、
  合法动作掩码、原始 logits 指纹、动作绑定、执行后状态与差值；
* 其余 7 席也由同一个网络驱动（与训练自对弈一致），但它们的动作写进
  ``events.jsonl`` 的 ``other_player_action``，**不计入**模型决策；
* 模拟器没有等待轮询，``waits.jsonl`` 为空；被 MoveGuard/80 上限规则挡掉的
  动作写 ``events.jsonl`` 的 ``action_masked``，不算决策也不改状态。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hsrl_agent import trace_schema as S
from hsrl_agent.live.trace import GameTrace
from hsbg.az.train.pending_guard import without_cancel

MAX_ACTIONS_PER_TURN = 80
#: 引擎 bot 的安全上限（长训不限制 bot；实测单席单回合最多 37 次，这里只防死循环）。
BOT_ACTIONS_PER_TURN_GUARD = 200


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vector_meta(array) -> dict:
    flat = np.ascontiguousarray(array).reshape(-1)
    return dict(sha256=sha(flat.tobytes()),
                shape=[int(v) for v in np.shape(array)],
                dtype=str(flat.dtype),
                nonzero=int(np.count_nonzero(flat)),
                abs_sum=round(float(np.abs(flat).sum()), 6))


def binding_of(action: int, view: dict) -> dict:
    """引擎动作下标 -> 与实机 ``Binding`` 同义的 {kind, source, target, position}。"""
    kind = S.action_kind(action)
    params = S.action_params(action)
    source = target = position = None

    def eid(zone: str, index):
        rows = view.get(zone) or []
        if index is None or index >= len(rows):
            return None
        return rows[index].get("entity_id")

    if kind == "buy":
        source = eid("shop", params.get("shop_slot"))
        position = params.get("shop_slot")
    elif kind in ("sell", "activate"):
        source = eid("board", params.get("board_slot"))
        position = params.get("board_slot")
    elif kind == "choose":
        idx = params.get("choice_index")
        source = eid("choices", idx)
        if source is None:
            # Target 类 pending 的候选是实体 id 列表（不是卡牌）。
            rows = view.get("targets") or []
            source = rows[idx] if idx is not None and idx < len(rows) else None
        position = idx
    elif kind == "play":
        source = eid("hand", params.get("hand_slot"))
        position = params.get("position")
    elif kind == "magnetize":
        source = eid("hand", params.get("hand_slot"))
        target = eid("board", params.get("board_slot"))
        position = params.get("board_slot")
    elif kind == "move":
        source = eid("board", params.get("from"))
        position = params.get("to")
    return {"kind": kind, "source": source, "target": target, "position": position}


def intents_of(view: dict, binding: dict) -> str:
    zones = {z: {c.get("entity_id"): c.get("card_id") for c in view.get(z) or []}
             for z in ("board", "hand", "shop", "choices")}
    src = binding.get("source")
    tgt = binding.get("target")
    src_card = next((zones[z].get(src) for z in zones if src in zones[z]), None) \
        or (f"entity:{src}" if src is not None else None)
    tgt_card = next((zones[z].get(tgt) for z in zones if tgt in zones[z]), None) \
        or (f"entity:{tgt}" if tgt is not None else None)
    text = f"{binding.get('kind')} {src_card}"
    if tgt is not None:
        text += f" -> {tgt_card}"
    if binding.get("position") is not None:
        text += f" @slot {binding['position']}"
    return text


def phase_pending(state: dict, view: dict) -> int:
    """引擎 pending 类型 -> 实机同一套 pending 编码（1 选英雄/2 发现/4 指向）。"""
    kind = view.get("pending_kind")
    return {"HeroSelect": 1, "HeroPowerChoice": 1, "Discover": 2, "Trinket": 2,
            "ChooseOne": 2, "Target": 4}.get(kind, 0)


class _Binding:
    """trace 需要的只读绑定视图（实机用 contracts.Binding）。"""

    __slots__ = ("action", "kind", "source", "target", "position")

    def __init__(self, action, kind, source=None, target=None, position=None):
        self.action = int(action)
        self.kind = kind
        self.source = source
        self.target = target
        self.position = position


def run_game(game_index: int, args, net, meta, guard, catalog, engine) -> dict:
    seed = args.seed + game_index
    game = engine.Game(seed)
    seat = args.learner_seat % game.num_players
    game_id = f"sim-{seed}"
    out = Path(args.out) / f"game-{game_index + 1:03d}"
    trace = GameTrace(out, side="sim", store_vectors=True)
    norm_mean = meta.get("norm_mean")
    norm_std = meta.get("norm_std")
    per_turn: dict = {}
    revision = 0
    decisions = 0
    other = 0
    masked = 0
    freeze_masked = 0
    open_parent = None
    started = time.time()
    stalled = None
    eliminated_turn = None
    was_alive = True
    seen_combats = set()
    last_turn = None

    def snapshot_of(client: dict, pending: int) -> dict:
        return {
            "game_id": game_id,
            "revision": revision,
            "client": client,
            "options": [],
            "view": {"pending": pending, "unknown": []},
            "approximation": {"mode": "sim-native",
                              "fields": dict(S.SIM_APPROXIMATION_FIELDS)},
            "visibility": S.VISIBILITY["sim"],
        }

    while not game.is_over():
        acted = False
        if decisions >= args.max_decisions:
            stalled = f"decision cap {args.max_decisions} reached"
            break
        for current in range(game.num_players):
            if not game.needs_action(current):
                continue
            acted = True
            revision += 1
            state = json.loads(game.state_json())
            client = S.sim_view(state, current, catalog)
            turn = int(state.get("turn") or 0)
            key = (current, turn)
            if last_turn is not None and turn != last_turn:
                # 回合切换（谁是最后一个结束招募的席位不重要，边界要记下来）。
                trace.turn_change(turn=last_turn, next_turn=turn,
                                  phase=state.get("phase"))
            last_turn = turn
            if current != seat:
                # 其他席位由引擎 bot 驱动（与 36h 长训同构：1 学习席 + 7 引擎 bot，
                # 不注入对手库）。bot 动作不是模型决策，写事件流、不进 decisions。
                action_bot = game.bot_action(current)
                if not isinstance(action_bot, int) or not 0 <= action_bot < engine.N_ACTIONS:
                    stalled = f"turn {turn}: engine bot returned invalid action {action_bot!r}"
                    break
                binding = binding_of(action_bot, client)
                other += 1
                per_turn[key] = per_turn.get(key, 0) + 1
                trace.note("other_player_action", turn=turn, phase=state.get("phase"),
                           seat=current, action=action_bot, kind=binding["kind"],
                           intent=intents_of(client, binding), engine="bot",
                           gold=client.get("gold"), tier=client.get("tavern_tier"),
                           board_size=len(client.get("board") or []),
                           turn_action_index=per_turn[key])
                if per_turn[key] > BOT_ACTIONS_PER_TURN_GUARD:
                    stalled = (f"turn {turn}: engine bot seat {current} exceeded "
                               f"{BOT_ACTIONS_PER_TURN_GUARD} actions")
                    break
                game.step(current, action_bot)
                continue
            observation = game.observe(current)
            features_raw = np.frombuffer(observation[0], dtype=np.float32).copy()
            cards = np.frombuffer(observation[1], dtype=np.int32).copy()
            raw_mask = np.frombuffer(game.legal_mask(current), dtype=np.uint8).astype(bool)
            legal_count = int(raw_mask.sum())
            # 与 36h 长训同一条规则：采样前清掉 Cancel(7)（forbid_cancel=1）。
            mask = without_cancel(raw_mask)
            cancel_removed = bool(raw_mask[7]) and not mask[7]
            if cancel_removed:
                # 与长训 forbid_cancel=1 同规则：Cancel(7) 在采样前被清掉。
                trace.note("cancel_masked", turn=turn, seat=current, action=7,
                           reason="forbid_cancel (matches 36h training)")
            # 与实机 bind_actions() 同规则：商店全冻时不提供 Freeze（它是开关）。
            # 规则文字与 client/hsrl_agent/live/actions.py:77 完全一致。
            shop_slots = client.get("shop") or []
            if mask[2] and shop_slots and all(
                    "FROZEN" in (slot.get("tags") or {}) for slot in shop_slots):
                mask[2] = False
                freeze_masked += 1
                trace.note("freeze_masked", turn=turn, seat=current, action=2,
                           reason="all shop slots frozen (matches live bind_actions)")
            if not mask.any():
                stalled = f"turn {turn}: forbid-cancel left no legal action"
                break

            budget = per_turn.get(key, 0)
            if budget >= MAX_ACTIONS_PER_TURN:
                if mask[0]:
                    # 回合动作预算用尽：只能停手；这一步不是模型决策。
                    trace.note("action_budget_exhausted", turn=turn, seat=current,
                               actions=budget, forced="EndTurn")
                    game.step(current, 0)
                    per_turn[key] = budget + 1
                    continue
                stalled = f"turn {turn}: {budget} actions, EndTurn illegal"
                break
            guarded, guard_forced = guard.guarded_mask(mask, game_id=seed, seat=current,
                                                       turn=turn)
            guarded = np.asarray(guarded, dtype=bool)
            if guarded.size == mask.size and guarded.any():
                blocked = np.flatnonzero(mask & ~guarded)
                if blocked.size:
                    masked += int(blocked.size)
                    trace.note("action_masked", turn=turn, seat=current,
                               actions=[int(a) for a in blocked], reason="move_guard",
                               forced_end=bool(guard_forced))
                mask = guarded
            if not mask.any():
                stalled = f"turn {turn}: every legal action masked by move guard"
                break

            features = features_raw
            if norm_mean is not None:
                features = (features - norm_mean) / np.maximum(norm_std, 1e-6)
            import torch
            from hsbg.az.train.ppo import forward_model, split_model_outputs
            clock = time.perf_counter()
            with torch.inference_mode():
                logits_t, _, value_t = split_model_outputs(forward_model(
                    net,
                    torch.from_numpy(np.asarray(features, np.float32))[None],
                    torch.from_numpy(cards.astype(np.int64))[None],
                    torch.from_numpy(mask)[None]))
            policy_ms = (time.perf_counter() - clock) * 1000.
            logits = logits_t[0].float().cpu().numpy()
            if not np.isfinite(logits).all():
                stalled = f"turn {turn}: nonfinite logits"
                break
            action = int(np.argmax(np.where(mask, logits, -np.inf)))
            if not mask[action]:
                stalled = f"turn {turn}: argmax action {action} is not legal"
                break
            binding = binding_of(action, client)
            value = (None if value_t is None
                     else round(float(np.asarray(value_t).reshape(-1)[0]), 6))

            snapshot = snapshot_of(client, phase_pending(state, client))
            inputs = dict(features=np.asarray(features, np.float32), cards=cards,
                          features_meta=vector_meta(features), cards_meta=vector_meta(cards),
                          mask=mask)
            decision = {
                "game_id": game_id,
                "revision": revision,
                "stamp": S.state_key(client),
                "action": action,
                "binding": binding,
                "checkpoint_sha256": args.checkpoint_sha256,
                "observation_sha256": sha(np.ascontiguousarray(
                    np.asarray(features, np.float32)).tobytes() + cards.tobytes()),
                "mask_sha256": sha(np.ascontiguousarray(mask).tobytes()),
                "logits_sha256": sha(logits.tobytes()),
                "value": value,
                "engine_action_name": engine.action_name(action),
                "raw_observation_sha256": sha(observation[0]),
                "legal_count": legal_count,
                "guarded_legal_count": int(mask.sum()),
                "cancel_removed": cancel_removed,
                "guard_forced_end": bool(guard_forced),
            }
            did = trace.decision(
                snapshot, decision, policy_ms=policy_ms, reused=False,
                bindings=[_Binding(a, S.action_kind(a)) for a in np.flatnonzero(mask)],
                role="model", seat=current, inputs=inputs,
                # 两步动作链：play 打开 Target 后，紧随的 choose 挂在这个 play 下面，
                # 这样"一次施法 = 两条决策"在记录里是一条可追溯的链，不是两次无关动作。
                parent=open_parent if client.get("pending_kind") else None,
                plan=[{"channel": "engine", "action": action}])
            decisions += 1
            guard.note_action(action, game_id=seed, seat=current, turn=turn)
            per_turn[key] = budget + 1
            phase_before = state.get("phase")
            game.step(current, action)
            after_state = json.loads(game.state_json())
            after_client = S.sim_view(after_state, current, catalog)
            open_parent = did if after_client.get("pending_kind") else None
            trace.execution(did, snapshot=None,
                            plan=[{"channel": "engine", "action": action}],
                            status="applied", outcome="applied", physical=True,
                            channel="engine", after=after_client)
            if int(after_state.get("turn") or 0) != turn or after_state.get("phase") != phase_before:
                trace.note("phase_transition", turn=turn, phase_before=phase_before,
                           phase_after=after_state.get("phase"),
                           next_turn=after_state.get("turn"),
                           gold=after_client.get("gold"),
                           tavern_tier=after_client.get("tavern_tier"),
                           health=after_client.get("health"),
                           alive=(after_state.get("players") or [{}])[current].get("alive"))
            # 战斗在 EndTurn 内部结算：把引擎给出的确定性战果写成事件。
            me = (after_state.get("players") or [{}])[current] or {}
            for entry in after_state.get("last_combats") or []:
                key = (entry.get("turn"), entry.get("a"), entry.get("b"), entry.get("winner"))
                if key in seen_combats:
                    continue
                seen_combats.add(key)
                if current not in (entry.get("a"), entry.get("b")):
                    continue
                trace.combat(
                    turn=entry.get("turn"), seat=current,
                    opponent=entry.get("b") if entry.get("a") == current else entry.get("a"),
                    winner=entry.get("winner"), damage=entry.get("damage"),
                    # 引擎的 damage 是总伤害；护甲先吸收，所以 health 差值通常更小。
                    health_after=me.get("health"), armor_after=me.get("armor"),
                    health_before_battle=client.get("health"),
                    damage_note="engine total damage; armor absorbs first, health delta may be smaller",
                    truncated=entry.get("truncated"),
                    frames=len(entry.get("frames") or []),
                    alive_after=me.get("alive"),
                    board_after=len(S.sim_view(after_state, current, catalog).get("board") or []))
        if stalled:
            break
        if not acted:
            stalled = "no seat needs an action but the game is not over"
            break
        # 记录被记录席位在哪一回合被淘汰（回合/名次是跨端比较的终点）。
        try:
            alive_now = bool(json.loads(game.state_json())["players"][seat].get("alive"))
        except Exception:
            alive_now = was_alive
        if was_alive and not alive_now:
            eliminated_turn = int(game.turn)
            trace.note("seat_eliminated", seat=seat, turn=eliminated_turn,
                       phase=json.loads(game.state_json()).get("phase"))
        was_alive = alive_now

    placement = int(list(game.placements())[seat]) if game.is_over() else None
    turns = int(game.turn)
    trace.game_end(seed=seed, seat=seat, placement=placement, turns=turns,
                   decisions=decisions, other_actions=other, masked=masked,
                   freeze_masked=freeze_masked,
                   stalled=stalled, truncated_combats=int(game.truncated_combats),
                   placements=[int(v) for v in game.placements()] if game.is_over() else None,
                   eliminated_turn=eliminated_turn,
                   wall_seconds=round(time.time() - started, 2))
    status = trace.close()
    record = {
        "game": game_index + 1, "seed": seed, "seat": seat, "game_id": game_id,
        "placement": placement, "turns": turns, "decisions": decisions,
        "other_player_actions": other, "masked_actions": masked,
        "over": bool(game.is_over()), "stalled": stalled,
        "truncated_combats": int(game.truncated_combats),
        "wall_seconds": round(time.time() - started, 2),
        "trace": {k: status.get(k) for k in ("events_dropped", "write_errors",
                                             "decisions", "executions", "vectors")},
        "audit_complete": (status.get("summary") or {}).get("complete"),
        "problems": (status.get("summary") or {}).get("problems"),
        "directory": str(out),
    }
    print(f"[game {game_index + 1}] place={placement} turns={turns} "
          f"decisions={decisions} other={other} "
          f"complete={record['audit_complete']} {record['wall_seconds']}s", flush=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--learner-seat", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=40000,
                        help="安全上限：学习席决策数（正常一局约 200-400）")
    parser.add_argument("--sample", action="store_true",
                        help="结束前打印第一条合并记录")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    import hsbg_engine as engine
    from hsbg.az.train.model import load_checkpoint
    from hsbg.az.train.move_guard import MoveGuard

    checkpoint = Path(args.checkpoint)
    args.checkpoint_sha256 = sha(checkpoint.read_bytes())
    net, meta = load_checkpoint(str(checkpoint), device=args.device)
    net.eval()
    guard = MoveGuard()
    catalog = S.build_catalog(json.loads(engine.catalog_json()))
    os.makedirs(args.out, exist_ok=True)
    build = engine.build_info()
    header = {
        "schema": S.SCHEMA,
        "side": "sim",
        "engine": {"version": engine.version(), "game_version": engine.game_version(),
                   "source_hash": build.get("source_hash"), "profile": build.get("profile")},
        "checkpoint": {"path": str(checkpoint), "sha256": args.checkpoint_sha256,
                       "policy_id": meta.get("policy_id") or meta.get("id")},
        "contract": {"n_actions": int(engine.N_ACTIONS), "obs_floats": int(engine.OBS_FLOATS),
                     "obs_cards": int(engine.OBS_CARDS), "n_cards": int(engine.N_CARDS)},
        "panel": {"learner": "policy", "others": "engine bot (game.bot_action, no opponent library)",
                  "note": "本局不是 36h 长训面板：长训的 7 席是引擎 bot + 每回合 p75 严格 EOT 快照注入"},
        "decoding": {"mode": "argmax", "temperature": 0, "sims": 0},
        "rules": {"max_actions_per_turn": MAX_ACTIONS_PER_TURN,
                  "forbid_cancel": True,
                  "move_guard": {"cap": 5, "repeat_limit": 2, "no_immediate_inverse": True}},
        "safety": {"bot_actions_per_turn_guard": BOT_ACTIONS_PER_TURN_GUARD},
        "started_at": time.time(),
    }
    (Path(args.out) / "trace-header.json").write_text(
        json.dumps(header, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    games = []
    for index in range(args.games):
        games.append(run_game(index, args, net, meta, guard, catalog, engine))
    summary = {"schema": "hsai.sim-trace-summary/v2", "header": header, "games": games}
    placements = [g["placement"] for g in games if g["placement"]]
    if placements:
        summary["placement"] = {"mean": round(sum(placements) / len(placements), 3),
                                "values": placements}
    (Path(args.out) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample and games:
        first = Path(games[0]["directory"]) / "decisions-resolved.jsonl"
        if first.exists():
            print(first.read_text(encoding="utf-8").splitlines()[0][:1500])
    print("done", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
