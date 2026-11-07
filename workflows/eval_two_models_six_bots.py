#!/usr/bin/env python3
"""2 个训练好的模型 + 6 个引擎启发式 bot 的 128 局配对评测（只读，不训练）。

评测对象（协议见 ``config/eval-duel-7357-vs-10004.json``，说明见 ``docs/EVAL_DUEL.md``）::

    A = entity16m-nus-7357        （NUS 1080772 末段 checkpoint，AGENTS 批准的续训起点）
    B = entity16m-p50-10004-820m  （P50 1085131 末段 checkpoint）
    bot ×6 = hsbg_engine 内置启发式 bot（``game.bot_play``，与历史评测/P50 库同口径）

一局 = 8 座位：A、B 各占 1 座（**逐局交换**，抵消座位偏差），其余 6 座为 bot。
两模型同 seed、同 bot 面板对打，因此名次差分是**配对样本**。

规则（用户指定，写进 protocol 参与 ``protocol_sha256``）::

    每回合 80 个动作上限（第 81 个决策在 EndTurn 合法时强制结束回合）
    每回合 5 次换位上限 + 同一 (from,to) 累计 2 次去抖（当前研究配方）
    greedy argmax 解码；不启用强制升本计划；不注入人类/镜像战团；引擎代码不改

记录（``report.json`` + ``report.md``）::

    - 排名：逐局名次、名次分布、均值/中位数、前四率、冠军率、效用、bootstrap CI；
    - 酒馆等级：逐回合 tier P50 / ≥4 / ≥5 / ≥6 占比、达标回合、最终等级分布；
    - 阵容战力：逐回合 **P50(Σatk+Σhp)**（主参考）+ P25/P75/P90/均值，
      同时给出 frames / games_at_turn / survivors 与 bot 面板参照；
    - 表现最佳的几局：完整最终战团（位置/卡名/等级/攻血/金色/关键词）。

用法::

    python workflows/eval_two_models_six_bots.py --out artifacts/runs/duel-eval-001
    python workflows/eval_two_models_six_bots.py --out /tmp/duel --games 2 --device cpu  # 冒烟
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "hsai.eval_duel/v1"
N_PLAYERS = 8
END_TURN_ACTION = 0
CONFIG_DEFAULT = ROOT / "config/eval-duel-7357-vs-10004.json"
LABELS = ("A", "B")

#: 引擎 ``types::Kw`` 位名（bit0..bit11），用于把战团关键词解成可读文本。
KW_NAMES = ("TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS", "POISONOUS",
            "STEALTH", "MAGNETIC", "MEGA_WINDFURY", "CLEAVE", "IMMUNE", "IMMUNE_ATTACKING")


# ══════════════════════════════════════════════════════════════════════════
# 纯工具
# ══════════════════════════════════════════════════════════════════════════
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    """按 LF 归一化计算源码指纹（Windows/Linux 同一内容同一指纹）。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def keyword_names(bits: int) -> List[str]:
    return [name for i, name in enumerate(KW_NAMES) if int(bits) & (1 << i)]


def fmt(value: Any, digits: int = 1, percent: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if percent:
        return f"{100.0 * number:.{digits}f}%"
    return f"{number:.{digits}f}"


def seats_for_game(gi: int) -> Dict[str, int]:
    """座位轮换：偶数局 A=0/B=1，奇数局互换（两模型各拿一半的早位）。"""
    return {"A": 0, "B": 1} if int(gi) % 2 == 0 else {"A": 1, "B": 0}


# ══════════════════════════════════════════════════════════════════════════
# 配置 / 协议
# ══════════════════════════════════════════════════════════════════════════
def load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("models", "games", "seed", "max_actions_per_turn", "move_cap",
                "move_repeat_limit"):
        if key not in cfg:
            raise ValueError(f"配置缺少必需字段 {key!r}: {path}")
    for label in LABELS:
        model = cfg["models"].get(label) or {}
        if not model.get("id") or not model.get("path"):
            raise ValueError(f"配置 models.{label} 需要 id/path: {path}")
    if int(cfg["games"]) <= 0:
        raise ValueError("games 必须 > 0")
    if int(cfg["max_actions_per_turn"]) <= 0:
        raise ValueError("max_actions_per_turn 必须 > 0")
    if int(cfg["move_cap"]) < 0 or int(cfg["move_repeat_limit"]) < 1:
        raise ValueError("move_cap>=0 且 move_repeat_limit>=1")
    sha_a = str(cfg["models"]["A"].get("file_sha256") or "")
    sha_b = str(cfg["models"]["B"].get("file_sha256") or "")
    if sha_a and sha_a == sha_b:
        raise ValueError("A/B 两个模型的 file_sha256 相同：这不是一次两模型评测")
    return cfg


def build_protocol(cfg: Dict[str, Any], *, fingerprint: Dict[str, Any],
                   source_sha: str, model_audits: Dict[str, Dict[str, Any]],
                   runner_sha: str) -> Dict[str, Any]:
    """协议字典：只放**决定可比性**的字段（不放路径/时间/设备）。"""
    return {
        "schema": SCHEMA,
        "fingerprint": fingerprint,
        "training_source_sha256": str(source_sha),
        "runner_sha256": str(runner_sha),
        "models": {label: {"id": audit["id"], "file_sha256": audit["file_sha256"],
                           "weights_sha256": audit.get("weights_sha256"),
                           "params": audit.get("params")}
                   for label, audit in model_audits.items()},
        "opponents": {"kind": "engine_bot", "action": "game.bot_play", "count": 6,
                      "implementation": "hsbg_engine::bot::choose_action / play_turn"},
        "games": int(cfg["games"]),
        "seed_start": int(cfg["seed"]),
        "seed_end": int(cfg["seed"]) + int(cfg["games"]) - 1,
        "seat_assignment": "A=seat0/B=seat1 on even games, swapped on odd games",
        "decode": str(cfg.get("decode", "greedy")),
        "max_actions_per_turn": int(cfg["max_actions_per_turn"]),
        "move_guard": 1,
        "move_cap": int(cfg["move_cap"]),
        "move_repeat_limit": int(cfg["move_repeat_limit"]),
        "max_steps": int(cfg.get("max_steps", 50000)),
        "forced_tier_plan": "",
        "mirror_pool": None,
        "human_board_pool": None,
        "engine_unmodified": True,
        "primary_statistic": "P50(sum_attack+sum_health) per end-of-turn board",
    }


def protocol_sha256(protocol: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# 模型审计（严格加载 + 卡表一致 + 真实观测前向）
# ══════════════════════════════════════════════════════════════════════════
def audit_model(path: Path, label: str, *, expected_sha: str, device: str,
                model_id: str) -> tuple[Any, Dict[str, Any]]:
    import torch

    import hsbg_engine as E
    from hsbg.az.train.common import mask_from_bytes, obs_arrays
    from hsbg.az.train.model import load_checkpoint
    from hsbg.az.train.ppo import PPOEvaluator

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"模型 {label} 不存在: {path}")
    file_sha = sha256_file(path)
    if expected_sha and file_sha != expected_sha:
        raise ValueError(f"模型 {label} 文件 SHA256 不符：期望 {expected_sha}，实际 {file_sha}；"
                         "上传的权重与配置不是同一份")
    model, meta = load_checkpoint(str(path), device=device)
    if meta.get("arch") in ("entity", "entity18m"):
        for key in ("d_model", "n_layers", "n_heads"):
            if key not in meta:
                raise ValueError(f"模型 {label} 的 entity checkpoint 缺少 {key}")
        from hsbg.az.train.entity_model import load_card_tables
        tables = load_card_tables()
        if not np.array_equal(model.card_static.detach().cpu().numpy(), tables.static) or \
                not np.array_equal(model.card_text.detach().cpu().numpy(), tables.text):
            raise ValueError(f"模型 {label} 内嵌卡牌表与当前 37.0.2 数据不一致（不能与其他模型同榜）")
    game = E.Game(seed=12345)
    features, cards = obs_arrays(game.observe(0))
    mask = mask_from_bytes(game.legal_mask(0))
    evaluator = PPOEvaluator(model, device=device, norm_mean=meta.get("norm_mean"),
                            norm_std=meta.get("norm_std"))
    prior, value = evaluator(features, cards, mask, 0)
    if prior.shape != (245,) or not np.isfinite(prior).all() or not np.isfinite(value):
        raise ValueError(f"模型 {label} 真实观测前向输出异常")
    if not np.isclose(float(prior.sum()), 1.0) or np.any(prior[~mask] != 0):
        raise ValueError(f"模型 {label} 合法动作概率异常（未在合法掩码上归一）")
    audit = {
        "id": str(model_id),
        "path": str(path),
        "file_sha256": file_sha,
        "weights_sha256": meta.get("weights_sha256"),
        "params": int(sum(p.numel() for p in model.parameters())),
        "arch": meta.get("arch"),
        "meta": {k: v for k, v in meta.items()
                 if not str(k).startswith("norm_") and isinstance(v, (int, float, str, bool))},
        "forward_ok": True,
        "card_table_ok": True,
        "torch": torch.__version__,
        "device": str(device),
    }
    return evaluator, audit


# ══════════════════════════════════════════════════════════════════════════
# 单座位策略：MoveGuard(5, 2) + 每回合 80 动作上限 + greedy
# ══════════════════════════════════════════════════════════════════════════
class DuelSeatPolicy:
    """一个模型的座位策略。

    与 ``PPOGreedyPolicy`` 用**同一个** :class:`MoveGuard`（规则完全一致），差异只有两点，
    都如实计数并写进报告：

    1. ``hard_cap=False``：换位上限触发后若引擎只给换位动作，退回原掩码继续（记
       ``move_cap_overflow``），而不是抛 ``RuntimeError`` 中断整局评测；
    2. 换位与回合计数在本类里可见（``stats``），便于把 5 换位 / 80 动作规则写进报告。
    """

    def __init__(self, label: str, evaluator: Any, *, max_actions_per_turn: int = 80,
                 move_cap: int = 5, move_repeat_limit: int = 2) -> None:
        from hsbg.az.train.move_guard import MoveGuard, is_move

        self.label = str(label)
        self.evaluator = evaluator
        self.max_actions_per_turn = int(max_actions_per_turn)
        self.guard = MoveGuard(cap=int(move_cap), repeat_limit=int(move_repeat_limit),
                               hard_cap=False)
        self._is_move = is_move
        self.turn_counts: Dict[tuple, int] = {}
        self.turn_moves: Dict[tuple, int] = {}
        self.stats: Counter = Counter()
        self._guard_snapshot: Dict[str, int] = {}

    def reset(self) -> None:
        self.guard.clear_state()
        self.turn_counts.clear()
        self.turn_moves.clear()

    def take_guard_delta(self) -> Dict[str, int]:
        """取自上局以来的守卫计数增量（守卫累计统计跨局保留）。"""
        current = {
            "decisions": int(self.guard.stats.decisions),
            "moves": int(self.guard.stats.moves),
            "blocked_inverse": int(self.guard.stats.blocked_inverse),
            "blocked_repeat": int(self.guard.stats.blocked_repeat),
            "blocked_cap": int(self.guard.stats.blocked_cap),
            "forced": int(self.guard.stats.forced),
        }
        delta = {key: current[key] - int(self._guard_snapshot.get(key, 0)) for key in current}
        self._guard_snapshot = current
        return delta

    def decide(self, game: Any, seat: int) -> tuple[int, Dict[str, Any]]:
        """守卫(5 换位/2 去抖) → 80 动作上限 → masked softmax + argmax。"""
        from hsbg.az.train.common import mask_from_bytes, masked_normalize, obs_arrays

        seat, turn = int(seat), int(game.turn)
        mask = mask_from_bytes(game.legal_mask(seat))
        guarded, guard_forced = self.guard.guarded_mask(
            mask, game_id=0, seat=seat, turn=turn)
        key = (seat, turn)
        n_turn = self.turn_counts.get(key, 0) + 1
        self.turn_counts[key] = n_turn
        cap_forced = bool(self.max_actions_per_turn > 0
                          and n_turn > self.max_actions_per_turn and bool(guarded[0]))
        info: Dict[str, Any] = {"turn_actions": n_turn, "forced": bool(cap_forced or guard_forced),
                                "forced_reason": None}
        if cap_forced:
            self.stats["forced_cap80"] += 1
            info["forced_reason"] = "action_cap"
        elif guard_forced:
            self.stats["forced_move_guard"] += 1
            info["forced_reason"] = "move_guard"
        if info["forced"]:
            action = END_TURN_ACTION
        else:
            features, cards = obs_arrays(game.observe(seat))
            prior, value = self.evaluator(features, cards, guarded, seat)
            pi = masked_normalize(prior, guarded)
            info["value"] = float(value)
            if float(pi.sum()) > 0.0:
                action = int(np.argmax(pi))
            else:  # 掩码被清空（不应发生）：退到守卫掩码/原掩码的第一个合法动作
                legal = np.flatnonzero(guarded)
                if legal.size == 0:
                    legal = np.flatnonzero(mask)
                action = int(legal[0]) if legal.size else END_TURN_ACTION
                self.stats["empty_mask_fallback"] += 1
        if self._is_move(action):
            if int(self.turn_moves.get(key, 0)) >= int(self.guard.cap):
                # 到上限后引擎只给换位动作：规则只能让步，如实记账
                self.stats["move_cap_overflow"] += 1
                info["move_cap_overflow"] = True
            self.turn_moves[key] = int(self.turn_moves.get(key, 0)) + 1
            self.stats["moves_taken"] += 1
        self.guard.note_action(action, game_id=0, seat=seat, turn=turn)
        return int(action), info


# ══════════════════════════════════════════════════════════════════════════
# 战团快照
# ══════════════════════════════════════════════════════════════════════════
class CardNames:
    """卡索引 -> (id, enUS, zhCN, tier) 缓存（``game.player_json`` 只给索引）。"""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.cache: Dict[int, tuple] = {}

    def get(self, index: int) -> tuple:
        index = int(index)
        if index not in self.cache:
            try:
                self.cache[index] = tuple(self.engine.card_info(index))
            except Exception:  # noqa: BLE001 - 卡表外索引不应出现，但不能中断整局评测
                self.cache[index] = (f"card#{index}", f"card#{index}", f"card#{index}", 0)
        return self.cache[index]


def describe_board(cards: CardNames, board: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """战团逐格详情（位置/卡名/等级/攻血/金色/关键词）。"""
    out = []
    for position, minion in enumerate(board or ()):
        card_id, name_en, name_zh, tier = cards.get(int(minion["card"]))
        bits = int(minion.get("kw") or 0) | int(minion.get("bonus_kw") or 0) \
            | int(minion.get("temp_kw") or 0)
        out.append({
            "position": int(position),
            "card": int(minion["card"]),
            "card_id": card_id,
            "name": name_zh,
            "name_en": name_en,
            "card_tier": int(tier),
            "atk": int(minion["atk"]),
            "hp": int(minion["hp"]),
            "max_hp": int(minion.get("max_hp") or minion["hp"]),
            "golden": bool(minion.get("golden")),
            "keywords": keyword_names(bits),
        })
    return out


def snapshot(game: Any, cards: CardNames, seat: int, *, gi: int, turn: int,
             detail: bool) -> Dict[str, Any]:
    """回合末帧：等级/金币/血量 + 战团总值；``detail=True`` 时附逐格详情。"""
    player = json.loads(game.player_json(int(seat)))
    board = player.get("board") or []
    atk = sum(int(m["atk"]) for m in board)
    hp = sum(int(m["hp"]) for m in board)
    frame: Dict[str, Any] = {
        "game": int(gi), "turn": int(turn),
        "tier": int(player.get("tier") or 1),
        "gold": int(player.get("gold") or 0),
        "health": int(player.get("health") or 0),
        "alive": bool(player.get("alive")),
        "size": len(board), "atk": int(atk), "hp": int(hp), "body": int(atk + hp),
        "hero": int(player.get("hero") or 0),
    }
    if detail:
        frame["board"] = describe_board(cards, board)
    return frame


# ══════════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════════
def run_games(*, engine: Any, cards: CardNames, policies: Dict[str, DuelSeatPolicy],
              games: int, seed0: int, max_steps: int, collect_opponents: bool,
              progress_every: int = 16) -> Dict[str, Any]:
    """跑 ``games`` 局：A/B 各 1 座（逐局交换）+ 6 个引擎 bot。

    帧时点约定：模型在**执行 EndTurn 之前**取帧（招募结束、战斗未结算），bot 在
    ``bot_play`` 返回后取帧；若该 bot 是最后一个结束回合的座位，引擎已经结算战斗，
    这一帧记 ``stage="post_combat"`` 且不进入 P50 参照（如实计数 ``bot_eot_skipped``）。
    """
    from hsbg.az.train.common import mask_from_bytes
    from hsbg.az.train.engine_quality import diagnostics

    frames: Dict[str, List[Dict[str, Any]]] = {label: [] for label in LABELS}
    bot_frames: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    illegal_total = 0
    truncated_games = 0
    engine_bad = 0
    bot_eot_skipped = 0
    t0 = time.time()
    for gi in range(int(games)):
        seed = int(seed0) + gi
        seats = seats_for_game(gi)
        game = engine.Game(seed=int(seed))
        for policy in policies.values():
            policy.reset()
        decisions = {label: 0 for label in LABELS}
        last_frame: Dict[str, Optional[Dict[str, Any]]] = {label: None for label in LABELS}
        last_board: Dict[str, Optional[List[Dict[str, Any]]]] = {label: None for label in LABELS}
        game_illegal = 0
        steps = 0
        game_t0 = time.time()
        while not game.is_over() and steps < int(max_steps):
            acted = False
            for p in range(N_PLAYERS):
                if not game.needs_action(p):
                    continue
                seat, turn = int(p), int(game.turn)
                label = next((k for k, s in seats.items() if s == seat), None)
                if label is not None:
                    mask = mask_from_bytes(game.legal_mask(seat))
                    action, _info = policies[label].decide(game, seat)
                    if not (0 <= action < int(mask.size) and bool(mask[action])):
                        game_illegal += 1
                        legal = np.flatnonzero(mask)
                        action = int(legal[0]) if legal.size else END_TURN_ACTION
                    if action == END_TURN_ACTION:
                        # EndTurn 之前取帧：招募结束、战斗与回合推进都还没发生
                        frame = snapshot(game, cards, seat, gi=gi, turn=turn, detail=False)
                        frames[label].append(frame)
                        last_frame[label] = frame
                        last_board[label] = describe_board(
                            cards, json.loads(game.player_json(seat)).get("board") or [])
                    game.step(seat, int(action))
                    decisions[label] += 1
                else:
                    game.bot_play(seat)
                    if collect_opponents and not game.is_over():
                        same_turn = int(game.turn) == turn
                        if same_turn:
                            bot_frames.append(snapshot(game, cards, seat, gi=gi, turn=turn,
                                                       detail=False))
                        else:
                            bot_eot_skipped += 1
                acted = True
                steps += 1
                if game.is_over():
                    break
            if not acted:
                break
        over = bool(game.is_over())
        placements = [int(x) for x in game.placements()] if over else [0] * N_PLAYERS
        diag = diagnostics(game)
        if any(int(v) for v in diag.values()):
            engine_bad += 1
        if not over:
            truncated_games += 1
        illegal_total += game_illegal
        rows.append({
            "game": int(gi), "seed": int(seed), "seats": dict(seats),
            "turns": int(game.turn), "steps": int(steps), "decisions": decisions,
            "illegal": int(game_illegal), "over": over,
            "rank": {label: int(placements[seat]) for label, seat in seats.items()},
            "placements": placements,
            "final": {label: (dict(last_frame[label]) if last_frame[label] else None)
                      for label in LABELS},
            "final_board": {label: last_board[label] for label in LABELS},
            "guard": {label: policies[label].take_guard_delta() for label in LABELS},
            "engine_diagnostics": diag,
            "seconds": round(time.time() - game_t0, 3),
        })
        if progress_every and (gi + 1) % int(progress_every) == 0:
            means = {label: np.mean([r["rank"][label] for r in rows if r["over"]])
                     for label in LABELS}
            print(f"[duel] {gi + 1}/{games} games  " +
                  "  ".join(f"{k}: mean_rank={v:.3f}" for k, v in means.items()) +
                  f"  {time.time() - t0:.0f}s", flush=True)
    return {
        "frames": frames, "bot_frames": bot_frames, "games": rows,
        "illegal": int(illegal_total), "truncated_games": int(truncated_games),
        "engine_bad": int(engine_bad), "bot_eot_skipped": int(bot_eot_skipped),
        "seconds": float(time.time() - t0),
        "rule_counters": {label: dict(policies[label].stats) for label in LABELS},
        "guard_totals": {
            label: {key: int(getattr(policies[label].guard.stats, key))
                    for key in ("decisions", "moves", "blocked_inverse", "blocked_repeat",
                                "blocked_cap", "forced")}
            for label in LABELS},
    }


# ══════════════════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════════════════
def build_report(*, cfg: Dict[str, Any], protocol: Dict[str, Any], audits: Dict[str, Any],
                 run: Dict[str, Any], best_k: int, power_k: int) -> Dict[str, Any]:
    from hsbg.eval.duel_metrics import (
        final_state_summary, paired_delta, rank_summary, reach_tier_stats,
        select_best_games, turn_table)

    rows = run["games"]
    ranked = [r for r in rows if r["over"]]
    ranks = {label: [int(r["rank"][label]) for r in ranked] for label in LABELS}
    per_model: Dict[str, Any] = {}
    for label in LABELS:
        model_frames = run["frames"][label]
        turns = sorted({int(f["turn"]) for f in model_frames})
        bot_reference: Dict[str, Any] = {}
        if run["bot_frames"]:
            bot_table = turn_table(run["bot_frames"], turns=turns)
            bot_reference = {turn: {"body_p50": row.get("body_p50"),
                                    "frames": row.get("frames")}
                             for turn, row in bot_table.items()
                             if int(turn) in turns}
        model_rows = [dict(row, label=label, rank=int(row["rank"][label]),
                           final_body=int((row["final"].get(label) or {}).get("body") or 0))
                      for row in ranked]
        per_model[label] = {
            "model": audits[label]["id"],
            "file_sha256": audits[label]["file_sha256"],
            "rank": rank_summary(ranks[label]),
            "turns": turn_table(model_frames, turns=turns),
            "bot_reference_body_p50": bot_reference,
            "reach_tier": reach_tier_stats(model_frames),
            "final_state": final_state_summary(
                [dict(row["final"][label], rank=row["rank"][label], seed=row["seed"])
                 for row in ranked if row["final"].get(label)]),
            "decisions": int(sum(row["decisions"][label] for row in rows)),
            "guard": {key: int(sum(int(row["guard"][label].get(key, 0)) for row in rows))
                      for key in ("decisions", "moves", "blocked_inverse", "blocked_repeat",
                                  "blocked_cap", "forced")},
            "best_games": select_best_games(model_rows, limit=int(best_k), order="rank"),
            "top_power_games": select_best_games(model_rows, limit=int(power_k),
                                                 order="body"),
            "per_seat": {str(seat): rank_summary([int(row["rank"][label]) for row in ranked
                                                  if row["seats"][label] == seat])
                         for seat in (0, 1)},
        }
    paired = paired_delta(ranks["A"], ranks["B"])
    learner_seats = {seat for row in ranked for seat in row["seats"].values()}
    bot_ranks = [int(placement) for row in ranked
                 for seat, placement in enumerate(row["placements"])
                 if seat not in learner_seats]
    illegal = int(run["illegal"])
    truncated = int(run["truncated_games"])
    engine_bad = int(run["engine_bad"])
    decided = [row["rank"][label] for row in ranked for label in LABELS]
    reasons: List[str] = []
    if len(ranked) != int(cfg["games"]):
        reasons.append(f"完成局数 {len(ranked)} != games={cfg['games']}")
    if illegal:
        reasons.append(f"非法动作 illegal={illegal} != 0")
    if truncated:
        reasons.append(f"截断局 truncated={truncated} != 0")
    if engine_bad:
        reasons.append(f"引擎非正常结束 engine_bad={engine_bad} != 0")
    if any(not (1 <= int(rank) <= N_PLAYERS) for rank in decided):
        reasons.append("存在名次越界")
    for label in LABELS:
        if int(per_model[label]["decisions"]) <= 0:
            reasons.append(f"{label} 没有任何决策")
        if int(per_model[label]["rank"]["games_ranked"]) != int(cfg["games"]):
            reasons.append(f"{label} 有效名次局数 {per_model[label]['rank']['games_ranked']} "
                           f"!= games={cfg['games']}")
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "valid": not reasons,
        "reasons": reasons,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256(protocol),
        "models": audits,
        "config": cfg,
        "runtime": {"os": platform.system(), "python": platform.python_version(),
                    "device": str(audits["A"].get("device")), "seconds": run["seconds"]},
        "results": per_model,
        "paired": paired,
        "bot_field": {"seats": 6, "rank": rank_summary(bot_ranks),
                      "eot_skipped_frames": int(run["bot_eot_skipped"]),
                      "note": "bot 面板参照（回合末帧；最后一个结束回合的座位已结算战斗，"
                              "该帧不进入参照）"},
        "rule_counters": run["rule_counters"],
        "guard_totals": run["guard_totals"],
        "totals": {"games": len(rows), "ranked": len(ranked),
                   "decisions": int(sum(sum(row["decisions"].values()) for row in rows)),
                   "illegal": illegal, "truncated": truncated,
                   "engine_bad": engine_bad,
                   "engine_diagnostics": {
                       key: int(sum(int(row["engine_diagnostics"].get(key, 0))
                                    for row in rows))
                       for key in ("truncated_combats", "truncated_recruit_contexts",
                                   "turn_limit_end", "engine_stopped")}},
        "games": rows,
    }


def _strength_table(lines: List[str], model: Dict[str, Any], *, label: str,
                    max_turn: int) -> None:
    lines.append(f"### 模型 {label}：逐回合战团战力（主参考 P50）")
    lines.append("")
    lines.append("| 回合 | frames | games | 存活 | **P50(Σatk+Σhp)** | P25 | P75 | P90 | 均值 | "
                 "atk P50* | hp P50* | 战团数 P50 | bot P50 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    reference = model.get("bot_reference_body_p50") or {}
    for turn in range(1, int(max_turn) + 1):
        row = model["turns"].get(str(turn)) or {}
        bot_p50 = (reference.get(str(turn)) or {}).get("body_p50")
        lines.append(
            f"| T{turn} | {row.get('frames', 0)} | {row.get('games_at_turn', 0)} | "
            f"{row.get('survivors', 0)} | **{fmt(row.get('body_p50'), 0)}** | "
            f"{fmt(row.get('body_p25'), 0)} | {fmt(row.get('body_p75'), 0)} | "
            f"{fmt(row.get('body_p90'), 0)} | {fmt(row.get('body_mean'), 0)} | "
            f"{fmt(row.get('atk_p50'), 0)} | {fmt(row.get('hp_p50'), 0)} | "
            f"{fmt(row.get('size_p50'), 1)} | {fmt(bot_p50, 0)} |")
    lines.append("")
    lines.append("\\* atk P50 / hp P50 是**辅助量**，不能相加当作总攻血 P50；主参考只有 "
                 "P50(Σatk+Σhp)。`frames == games`（每局每回合一个回合末帧），"
                 "随回合下降是因为该回合前已淘汰的对局不再产生帧（条件分布）。")
    lines.append("")


def _tier_table(lines: List[str], model: Dict[str, Any], *, label: str,
                max_turn: int) -> None:
    lines.append(f"### 模型 {label}：逐回合酒馆等级")
    lines.append("")
    lines.append("| 回合 | frames | 等级 P50 | 均值 | ≥4 | ≥5 | ≥6 | 金币 P50 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for turn in range(1, int(max_turn) + 1):
        row = model["turns"].get(str(turn)) or {}
        lines.append(f"| T{turn} | {row.get('frames', 0)} | {fmt(row.get('tier_p50'), 1)} | "
                     f"{fmt(row.get('tier_mean'), 2)} | {fmt(row.get('tier_ge4'), 1, True)} | "
                     f"{fmt(row.get('tier_ge5'), 1, True)} | {fmt(row.get('tier_ge6'), 1, True)} | "
                     f"{fmt(row.get('gold_p50'), 0)} |")
    reach = model.get("reach_tier") or {}
    lines.append("")
    lines.append("达到目标等级的回合（按对局）：")
    lines.append("")
    lines.append("| 目标 | 达到局数 | 占比 | 首个回合 P50 | P25 | P75 | 最早 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("4", "5", "6"):
        row = reach.get(key) or {}
        lines.append(f"| {key} 本 | {row.get('games_reached', 0)} | "
                     f"{fmt(row.get('share_reached'), 1, True)} | {fmt(row.get('turn_p50'), 1)} | "
                     f"{fmt(row.get('turn_p25'), 1)} | {fmt(row.get('turn_p75'), 1)} | "
                     f"{fmt(row.get('turn_min'), 0)} |")
    lines.append("")


def _board_table(lines: List[str], board: Sequence[Dict[str, Any]]) -> None:
    if not board:
        lines.append("_（该局没有记录到最终战团）_")
        lines.append("")
        return
    lines.append("| # | 卡牌 | 卡表等级 | 攻击 | 生命 | 金色 | 关键词 |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for minion in board:
        lines.append(f"| {int(minion['position']) + 1} | {minion['name']} "
                     f"(`{minion['card_id']}`) | {minion['card_tier']} | {minion['atk']} | "
                     f"{minion['hp']}/{minion['max_hp']} | "
                     f"{'是' if minion['golden'] else '—'} | "
                     f"{', '.join(minion['keywords']) or '—'} |")
    lines.append("")


def _interpretation_lines(report: Dict[str, Any]) -> List[str]:
    """把「这组数字能推出什么」写成两三句，避免读者把 bot 面板当强对手。"""
    res = report["results"]
    paired = report["paired"]
    lines: List[str] = []
    bot_rank = report["bot_field"]["rank"]["mean_rank"]
    a, b = res["A"]["rank"], res["B"]["rank"]
    saturated = (a["top4_rate"] or 0) >= 0.95 and (b["top4_rate"] or 0) >= 0.95
    if saturated:
        lines.append(f"- **对手强度**：两个模型的前四率都 ≥95%，6 个引擎 bot 合并平均名次 "
                     f"{fmt(bot_rank, 2)}（8 座位的均匀值是 5.5）。也就是说 bot 面板"
                     "几乎总是被压在第 3 名之后，前四率/冠军率接近饱和，**主判据是配对差分**，"
                     "不是前四率。")
    else:
        lines.append(f"- **对手强度**：6 个引擎 bot 合并平均名次 {fmt(bot_rank, 2)}；"
                     f"两模型前四率 {fmt(a['top4_rate'], 1, True)} / "
                     f"{fmt(b['top4_rate'], 1, True)}。")
    ci = paired.get("mean_delta_ci95") or [None, None]
    lo, hi = ci[0], ci[1]
    if paired.get("mean_delta") is not None:
        if lo is not None and hi is not None and (lo > 0 or hi < 0):
            better = "A（7357）" if paired["mean_delta"] < 0 else "B（10004）"
            lines.append(f"- **配对差分**：Δ = rank_A − rank_B 均值 {fmt(paired['mean_delta'], 3)}，"
                         f"95% CI [{fmt(lo, 3)}, {fmt(hi, 3)}] 不含 0；{paired['pairs']} 局"
                         f"（A 更好 {paired['a_better']} / B 更好 {paired['b_better']} / "
                         f"相同 {paired['ties']}）。在这套规则与 bot 面板下，{better} 的名次更好。")
        else:
            lines.append(f"- **配对差分**：Δ 均值 {fmt(paired['mean_delta'], 3)}，"
                         f"95% CI [{fmt(lo, 3)}, {fmt(hi, 3)}] **跨 0**；{paired['pairs']} 局下"
                         "测不出稳定名次差（这是分辨率不足，不等于两者相同）。")
    lines.append("- **等级与战力**：`tier_p50` 与 `P50(Σatk+Σhp)` 是随回合的**条件分布**"
                 "（frames 随对局淘汰下降），只能与同一 `protocol_sha256` 的报告比较。")
    return lines


def render_markdown(report: Dict[str, Any], *, max_turn: int = 15) -> str:
    lines: List[str] = []
    res = report["results"]
    counters = report.get("rule_counters") or {}
    totals = report["totals"]
    lines.append("# 2 模型 + 6 bot 配对评测报告")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 协议 SHA256：`{report['protocol_sha256']}`")
    lines.append(f"- 有效性：**{'通过' if report['valid'] else '不通过'}**"
                 + ("" if report["valid"] else f"（{'; '.join(report['reasons'])}）"))
    lines.append(f"- 局数：{totals['games']}（完成 {totals['ranked']}）；"
                 f"决策 {totals['decisions']}；非法动作 {totals['illegal']}；"
                 f"截断 {totals['truncated']}；引擎异常 {totals['engine_bad']}")
    lines.append(f"- 模型 A：`{res['A']['model']}`（`{res['A']['file_sha256'][:16]}…`）")
    lines.append(f"- 模型 B：`{res['B']['model']}`（`{res['B']['file_sha256'][:16]}…`）")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    a, b = res["A"]["rank"], res["B"]["rank"]
    lines.append("| 指标 | 模型 A | 模型 B |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 平均名次（越小越好） | {fmt(a['mean_rank'], 3)} | {fmt(b['mean_rank'], 3)} |")
    lines.append(f"| 平均名次 95% CI | "
                 f"[{fmt((a['mean_rank_ci95'] or [None, None])[0], 2)}, "
                 f"{fmt((a['mean_rank_ci95'] or [None, None])[1], 2)}] | "
                 f"[{fmt((b['mean_rank_ci95'] or [None, None])[0], 2)}, "
                 f"{fmt((b['mean_rank_ci95'] or [None, None])[1], 2)}] |")
    lines.append(f"| 中位名次 | {fmt(a['median_rank'], 1)} | {fmt(b['median_rank'], 1)} |")
    lines.append(f"| 前四率 | {fmt(a['top4_rate'], 1, True)} | {fmt(b['top4_rate'], 1, True)} |")
    lines.append(f"| 冠军率 | {fmt(a['win_rate'], 1, True)} | {fmt(b['win_rate'], 1, True)} |")
    lines.append(f"| 期望效用 | {fmt(a['mean_utility'], 3)} | {fmt(b['mean_utility'], 3)} |")
    lines.append(f"| 最终战团 P50 | {fmt(res['A']['final_state']['body_p50'], 0)} | "
                 f"{fmt(res['B']['final_state']['body_p50'], 0)} |")
    lines.append(f"| 最终等级 P50 | {fmt(res['A']['final_state']['tier_p50'], 1)} | "
                 f"{fmt(res['B']['final_state']['tier_p50'], 1)} |")
    lines.append("")
    paired = report["paired"]
    lines.append("配对差分（同局、同 seed、同 bot 面板）：")
    lines.append("")
    lines.append(f"- Δ = rank_A − rank_B：均值 {fmt(paired['mean_delta'], 3)}"
                 f"（95% CI [{fmt((paired['mean_delta_ci95'] or [None, None])[0], 3)}, "
                 f"{fmt((paired['mean_delta_ci95'] or [None, None])[1], 3)}]），"
                 f"中位 {fmt(paired['median_delta'], 1)}，n={paired['pairs']}。负值 = A 更强。")
    lines.append(f"- A 名次更好 {paired['a_better']} 局 / B 更好 {paired['b_better']} 局 / "
                 f"相同 {paired['ties']} 局。")
    lines.append("")
    lines.extend(_interpretation_lines(report))
    lines.append("")
    lines.append("## 2. 排名明细")
    lines.append("")
    lines.append("| 名次 | A 局数 | A 占比 | B 局数 | B 占比 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for rank in range(1, N_PLAYERS + 1):
        lines.append(f"| {rank} | {a['rank_counts'][str(rank)]} | "
                     f"{fmt(a['rank_distribution'][str(rank)], 1, True)} | "
                     f"{b['rank_counts'][str(rank)]} | "
                     f"{fmt(b['rank_distribution'][str(rank)], 1, True)} |")
    lines.append("")
    lines.append("| 座位 | A 平均名次 | B 平均名次 |")
    lines.append("|---:|---:|---:|")
    for seat in ("0", "1"):
        lines.append(f"| seat {seat} | {fmt(res['A']['per_seat'][seat]['mean_rank'], 3)} | "
                     f"{fmt(res['B']['per_seat'][seat]['mean_rank'], 3)} |")
    lines.append("")
    lines.append(f"bot 面板（6 座合并）平均名次 {fmt(report['bot_field']['rank']['mean_rank'], 3)}，"
                 f"前四率 {fmt(report['bot_field']['rank']['top4_rate'], 1, True)}；"
                 f"最后一个结束回合的 bot 帧已结算战斗，剔除 {report['bot_field']['eot_skipped_frames']} 帧。")
    lines.append("")
    lines.append("## 3. 酒馆等级")
    lines.append("")
    for label in LABELS:
        _tier_table(lines, res[label], label=label, max_turn=max_turn)
    lines.append("## 4. 阵容战力（回合末战团）")
    lines.append("")
    for label in LABELS:
        _strength_table(lines, res[label], label=label, max_turn=max_turn)
    lines.append("## 5. 表现最佳的几局（最终详细阵容）")
    lines.append("")
    for label in LABELS:
        lines.append(f"### 模型 {label}：名次最优的局")
        lines.append("")
        for row in res[label]["best_games"]:
            final = row["final"].get(label) or {}
            lines.append(f"**seed {row['seed']}｜seat {row['seats'][label]}｜"
                         f"名次 {row['rank']}｜最终等级 {final.get('tier')}｜"
                         f"血量 {final.get('health')}｜Σatk+Σhp={final.get('body')}"
                         f"（{final.get('size')} 个随从）**")
            lines.append("")
            _board_table(lines, row["final_board"].get(label) or [])
        lines.append(f"### 模型 {label}：最终战力最高的局")
        lines.append("")
        for row in res[label]["top_power_games"]:
            final = row["final"].get(label) or {}
            lines.append(f"**seed {row['seed']}｜名次 {row['rank']}｜"
                         f"等级 {final.get('tier')}｜Σatk+Σhp={final.get('body')}**")
            lines.append("")
            _board_table(lines, row["final_board"].get(label) or [])
    lines.append("## 6. 规则执行与异常计数")
    lines.append("")
    lines.append("| 项目 | A | B |")
    lines.append("|---|---:|---:|")
    guard_a, guard_b = res["A"]["guard"], res["B"]["guard"]
    lines.append(f"| 决策数 | {res['A']['decisions']} | {res['B']['decisions']} |")
    lines.append(f"| 换位选择次数 | {guard_a['moves']} | {guard_b['moves']} |")
    lines.append(f"| 被逆换位规则屏蔽的动作位 | {guard_a['blocked_inverse']} | "
                 f"{guard_b['blocked_inverse']} |")
    lines.append(f"| 被重复去抖屏蔽的动作位 | {guard_a['blocked_repeat']} | "
                 f"{guard_b['blocked_repeat']} |")
    lines.append(f"| 被换位上限屏蔽的动作位 | {guard_a['blocked_cap']} | {guard_b['blocked_cap']} |")
    lines.append(f"| 守卫强制 EndTurn 次数 | {guard_a['forced']} | {guard_b['forced']} |")
    lines.append(f"| 因 80 动作上限强制 EndTurn | "
                 f"{int((counters.get('A') or {}).get('forced_cap80', 0))} | "
                 f"{int((counters.get('B') or {}).get('forced_cap80', 0))} |")
    lines.append(f"| 换位上限让步（引擎只给换位） | "
                 f"{int((counters.get('A') or {}).get('move_cap_overflow', 0))} | "
                 f"{int((counters.get('B') or {}).get('move_cap_overflow', 0))} |")
    lines.append(f"| 空掩码回退 | "
                 f"{int((counters.get('A') or {}).get('empty_mask_fallback', 0))} | "
                 f"{int((counters.get('B') or {}).get('empty_mask_fallback', 0))} |")
    lines.append("")
    lines.append(f"引擎诊断合计：`{json.dumps(totals['engine_diagnostics'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## 7. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python tools/nus_submit.py --phase eval --check-only")
    lines.append("python workflows/eval_two_models_six_bots.py --out artifacts/runs/duel-eval-001")
    lines.append("```")
    lines.append("")
    lines.append("协议字段（决定 `protocol_sha256`）：")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["protocol"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="输出目录（写 report.json/report.md）")
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--games", type=int, default=None, help="覆盖配置的局数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置的起始 seed")
    parser.add_argument("--device", default=None, help="cuda/cpu；默认自动")
    parser.add_argument("--model-a", default=None, help="覆盖模型 A 路径")
    parser.add_argument("--model-b", default=None, help="覆盖模型 B 路径")
    parser.add_argument("--best-games", type=int, default=None)
    parser.add_argument("--top-power-games", type=int, default=None)
    parser.add_argument("--no-opponents", action="store_true",
                        help="不采集 bot 战团参照（默认采集）")
    parser.add_argument("--max-turn-report", type=int, default=0,
                        help="markdown 表格最多列到第几回合；0 = 实际观测到的最大回合（上限 30）")
    args = parser.parse_args(argv)

    import torch

    cfg = load_config(Path(args.config))
    if args.games is not None:
        cfg["games"] = int(args.games)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.model_a:
        cfg["models"]["A"]["path"] = str(args.model_a)
    if args.model_b:
        cfg["models"]["B"]["path"] = str(args.model_b)
    cfg["best_games"] = int(args.best_games if args.best_games is not None
                            else cfg.get("best_games", 5))
    cfg["top_power_games"] = int(args.top_power_games if args.top_power_games is not None
                                 else cfg.get("top_power_games", 3))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("指定了 cuda，但当前环境没有可用 GPU")
    threads = int(os.environ.get("HSRL_TORCH_THREADS") or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    elif str(device) == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

    import hsbg_engine as E
    from hsbg.az.train.personal import fingerprint, training_source_digest

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[duel] out={out} config={args.config} device={device} "
          f"games={cfg['games']} seed={cfg['seed']}", flush=True)

    audits: Dict[str, Any] = {}
    policies: Dict[str, DuelSeatPolicy] = {}
    for label in LABELS:
        model_cfg = cfg["models"][label]
        evaluator, audit = audit_model(Path(model_cfg["path"]), label,
                                       expected_sha=str(model_cfg.get("file_sha256") or ""),
                                       device=device, model_id=str(model_cfg["id"]))
        audits[label] = audit
        policies[label] = DuelSeatPolicy(label, evaluator,
                                         max_actions_per_turn=int(cfg["max_actions_per_turn"]),
                                         move_cap=int(cfg["move_cap"]),
                                         move_repeat_limit=int(cfg["move_repeat_limit"]))
        print(f"[duel] model {label}: {audit['id']} sha256={audit['file_sha256'][:16]}… "
              f"params={audit['params']}", flush=True)

    protocol = build_protocol(cfg, fingerprint=fingerprint(),
                              source_sha=training_source_digest(), model_audits=audits,
                              runner_sha=sha256_text(Path(__file__)))
    print(f"[duel] protocol_sha256={protocol_sha256(protocol)}", flush=True)

    run = run_games(engine=E, cards=CardNames(E), policies=policies,
                    games=int(cfg["games"]), seed0=int(cfg["seed"]),
                    max_steps=int(cfg.get("max_steps", 50000)),
                    collect_opponents=not args.no_opponents)
    report = build_report(cfg=cfg, protocol=protocol, audits=audits, run=run,
                          best_k=int(cfg["best_games"]),
                          power_k=int(cfg["top_power_games"]))
    write_json(out / "report.json", report)
    max_turn = int(args.max_turn_report)
    if max_turn <= 0:
        observed = max((int(frame["turn"]) for label in LABELS
                        for frame in run["frames"][label]), default=0)
        max_turn = max(1, min(int(observed), 30))
    (out / "report.md").write_text(render_markdown(report, max_turn=max_turn),
                                   encoding="utf-8")
    print(json.dumps({key: report[key]
                      for key in ("valid", "reasons", "protocol_sha256", "paired", "totals")},
                     ensure_ascii=False, indent=2), flush=True)
    for label in LABELS:
        rank = report["results"][label]["rank"]
        final_state = report["results"][label]["final_state"]
        print(f"[duel] {label} ({report['results'][label]['model']}): "
              f"mean_rank={fmt(rank['mean_rank'], 3)} top4={fmt(rank['top4_rate'], 3)} "
              f"win={fmt(rank['win_rate'], 3)} final_body_p50={fmt(final_state['body_p50'], 0)} "
              f"final_tier_p50={fmt(final_state['tier_p50'], 1)}", flush=True)
    print(f"[duel] report -> {out / 'report.json'}", flush=True)
    if not report["valid"]:
        raise RuntimeError("评测不通过（非法动作/截断/引擎异常/局数不足），拒绝入榜："
                           + "; ".join(report["reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
