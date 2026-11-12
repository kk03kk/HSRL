"""统一对局记录规范（模拟器 / 实机共用）。

设计目标
--------
1. **同一套文件与字段**：模拟器与实机写同样的 ``decisions.jsonl`` /
   ``actions.jsonl`` / ``waits.jsonl`` / ``events.jsonl`` / ``observations.npy`` /
   ``summary.json`` / ``audit.json``，字段语义一一对应，可以按 ``turn`` +
   ``decision_index_in_turn`` 直接对齐比较。
2. **流分离**：模型真实决策（decision）、脚本执行（execution）、等待轮询
   （wait）、内部事件（event）写在不同的流里。统计动作分布时只读
   ``decisions.jsonl``，不会被轮询或重试污染。
3. **不伪造**：观测不到的字段一律 ``None``，并在 ``approximation_fields`` 里
   列出原因；模拟器额外标注 ``visibility="owner_full"``（引擎真值），实机标注
   ``visibility="owner_public_projection"``（Power.log 公开投影）。这两类记录
   可以比较，但不能声称等价。
4. **低开销**：本模块只有纯函数与 dict 组装（numpy 仅用于向量摘要，惰性导入），
   写盘全部交给 :class:`hsrl_agent.live.trace.GameTrace` 的异步 writer 线程。

规范版本：``hsai.game-trace/v2``（v1 是只有 decisions.jsonl 的实机旧格式）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "hsai.game-trace/v2"
AUDIT_SCHEMA = "hsai.game-trace-audit/v2"

SIDES = ("sim", "live")

VISIBILITY = {
    "sim": "owner_full",              # 引擎真值：包含对手手牌、随机池等隐藏信息
    "live": "owner_public_projection"  # Power.log 公开投影：无隐藏信息
}

#: 记录流。每个流只承担一种语义，绝不混写。
STREAMS = {
    "decisions.jsonl": "模型真实决策；一行一次决策（复用/轮询不进这里）",
    "decisions-full.jsonl.gz": "每次决策的完整快照 + 模型原始输出（gzip JSONL）",
    "actions.jsonl": "每次执行尝试：手势计划、原生时序、执行前后状态与差值、因果结果",
    "waits.jsonl": "等待轮询：同一状态同一动作的重复读取，不算决策",
    "events.jsonl": "桥接/脚本内部事件：重试、分支、暂停、阻断、回执",
    "observations.npy": "真实模型输入向量 (N, 3227) float32，逐决策顺序",
    "observations-cards.npy": "真实模型卡片输入 (N, 124) int32，逐决策顺序",
    "decisions-resolved.jsonl": "决策 + 其执行 + 执行后状态/差值的合并视图（派生）",
    "summary.json": "按回合聚合的动作统计与完整性计数",
    "performance.json": "推理/执行耗时分布与动作成功率",
    "audit.json": "完整性校验：缺失执行、重复决策、回合回退、状态不可还原项",
    "decision-report.md": "人类可读摘要",
}

#: 动作大类。跨边统计必须按大类比较，不能把"停止招募"当成一次主动作。
ACTION_CLASS: Dict[str, str] = {
    "end": "stop_recruiting",
    "refresh": "tavern_control",
    "freeze": "tavern_control",
    "upgrade": "tavern_control",
    "hero_power": "hero_power",
    "dark_gift": "hero_power",
    "buy": "buy",
    "sell": "sell",
    "play": "play",
    "magnetize": "magnetize",
    "move": "reposition",
    "choose": "choice",
    "branch": "choice",
    "target": "choice",
    "activate": "choice",
    "cancel": "cancel",
}

#: 引擎原生 245 维动作的 (kind, 参数) 解码表；与实机 ``Binding`` 词表同名同义。
ENGINE_KIND = {
    0: "end", 1: "refresh", 2: "freeze", 3: "upgrade", 4: "hero_power",
    5: "hero_power", 6: "dark_gift", 7: "cancel",
}

#: 引擎 ``Kw`` 位域（engine/crates/hsbg-engine/src/types.rs::Kw::NAMES 同序）。
KW_BITS: Tuple[Tuple[str, int], ...] = (
    ("TAUNT", 1 << 0), ("DIVINE_SHIELD", 1 << 1), ("WINDFURY", 1 << 2),
    ("REBORN", 1 << 3), ("VENOMOUS", 1 << 4), ("POISONOUS", 1 << 5),
    ("STEALTH", 1 << 6), ("MAGNETIC", 1 << 7), ("MEGA_WINDFURY", 1 << 8),
    ("CLEAVE", 1 << 9), ("IMMUNE", 1 << 10), ("IMMUNE_ATTACKING", 1 << 11),
)

#: 跨边可比的字段白名单（实机投影里一定有，模拟器也能给出）。
OBSERVABLE_FIELDS = ("turn", "phase", "gold", "tavern_tier", "upgrade_cost", "health", "armor")

CARD_TAG_KEYS = (
    "ATK", "HEALTH", "DAMAGE", "TAUNT", "DIVINE_SHIELD", "REBORN", "WINDFURY",
    "MEGA_WINDFURY", "VENOMOUS", "POISONOUS", "STEALTH", "MAGNETIC", "FROZEN",
    "PREMIUM", "CARDRACE", "TECH_LEVEL",
)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def digest_bytes(data: Any) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(bytes(data)).hexdigest()


def digest_json(payload: Any) -> str:
    return digest_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, default=str))


def digest_vector(array: Any) -> Dict[str, Any]:
    """输入向量的指纹与摘要；不复制数据，向量由 trace 单独落 npy。"""
    import numpy as np

    arr = np.ascontiguousarray(array)
    flat = arr.reshape(-1)
    return {
        "sha256": digest_bytes(flat.tobytes()),
        "shape": [int(v) for v in arr.shape],
        "dtype": str(flat.dtype),
        "stats": {
            "nonzero": int(np.count_nonzero(flat)),
            "sum": round(float(flat.sum()), 6),
            "abs_sum": round(float(np.abs(flat).sum()), 6),
        },
    }


def kw_to_tags(kw: int, bonus_kw: int = 0, *, golden: bool = False) -> Dict[str, int]:
    bits = int(kw or 0) | int(bonus_kw or 0)
    tags = {name: 1 for name, bit in KW_BITS if bits & bit}
    if golden:
        tags["PREMIUM"] = 1
    return tags


def action_kind(action: Optional[int]) -> str:
    """引擎动作下标 -> 统一 kind（实机 Binding.kind 同一词表）。"""
    if action is None:
        return "unknown"
    a = int(action)
    if a in ENGINE_KIND:
        return ENGINE_KIND[a]
    if 8 <= a <= 15:
        return "buy"
    if 16 <= a <= 22:
        return "sell"
    if 23 <= a <= 29:
        return "activate"
    if 30 <= a <= 45:
        return "choose"
    if 46 <= a <= 125:
        return "play"
    if 126 <= a <= 195:
        return "magnetize"
    if 196 <= a <= 244:
        return "move"
    return "unknown"


def action_params(action: int) -> Dict[str, Optional[int]]:
    """引擎动作下标 -> 槽位参数（与实机 binding 的 position/target 对齐）。"""
    a = int(action)
    if 4 <= a <= 5:
        return {"slot": a - 4}
    if 8 <= a <= 15:
        return {"shop_slot": a - 8}
    if 16 <= a <= 22:
        return {"board_slot": a - 16}
    if 23 <= a <= 29:
        return {"board_slot": a - 23}
    if 30 <= a <= 45:
        return {"choice_index": a - 30}
    if 46 <= a <= 125:
        return {"hand_slot": (a - 46) // 8, "position": (a - 46) % 8}
    if 126 <= a <= 195:
        return {"hand_slot": (a - 126) // 7, "board_slot": (a - 126) % 7}
    if 196 <= a <= 244:
        return {"from": (a - 196) // 7, "to": (a - 196) % 7}
    return {}


def action_class(kind: Optional[str]) -> str:
    return ACTION_CLASS.get(str(kind), "other")


# --------------------------------------------------------------------------- #
# 视图规范化
# --------------------------------------------------------------------------- #
def card(row: Optional[dict], *, position: Optional[int] = None) -> Optional[dict]:
    if not row:
        return None
    tags = row.get("tags") or {}
    return {
        "entity_id": row.get("entity_id"),
        "card_id": row.get("card_id"),
        "name": row.get("name"),
        "card_type": row.get("card_type"),
        "position": row.get("position") if position is None else position,
        "attack": row.get("attack"),
        "health": row.get("health"),
        "tier": row.get("tier"),
        "cost": row.get("cost"),
        "golden": bool(row.get("golden", tags.get("PREMIUM") == 1)),
        "tags": {k: tags[k] for k in CARD_TAG_KEYS if k in tags},
    }


def _cards(rows: Optional[Sequence[dict]]) -> List[dict]:
    return [c for c in (card(r, position=i + 1) for i, r in enumerate(rows or [])) if c]


def view(client: Optional[dict]) -> Optional[dict]:
    """把任一端的"我方视图"规范化成同一形状。

    实机传入 ``snapshot['client']``（Power.log 投影）；模拟器传入
    :func:`sim_view` 的产物（引擎真值投影成同一形状）。
    """
    if not client:
        return None
    hero = client.get("hero") or {}
    hero_tags = hero.get("tags") or {}
    return {
        "turn": client.get("turn"),
        "phase": client.get("phase"),
        "gold": client.get("gold"),
        "tavern_tier": client.get("tavern_tier"),
        "upgrade_cost": client.get("upgrade_cost"),
        "health": client.get("health", hero.get("health")),
        "armor": hero_tags.get("ARMOR", (client.get("hero") or {}).get("armor")),
        "hero": {
            "entity_id": hero.get("entity_id"),
            "card_id": hero.get("card_id"),
            "name": hero.get("name"),
            "health": hero.get("health"),
            "armor": hero_tags.get("ARMOR"),
        },
        "board": _cards(client.get("board")),
        "hand": _cards(client.get("hand")),
        "shop": _cards(client.get("shop")),
        "choices": _cards(client.get("choices")),
        "controls": _cards(client.get("controls")),
        # 协议态：模拟器是引擎 pending 的类型名（HeroSelect/Target/Discover/…），
        # 实机是客户端 interaction.mode（choice/target/branch）。两者都表示
        # "当前有一个待选请求"，所以同名同义地记在这一列。
        "pending_kind": (client.get("pending_kind")
                         or (client.get("interaction") or {}).get("mode")),
        "targets": [int(v) for v in (client.get("targets") or [])],
        "counters": {k: v for k, v in (client.get("counters") or {}).items()
                     if isinstance(v, (int, float))},
    }


def state_key(client: Optional[dict]) -> Optional[str]:
    """同一端内"状态相同"的判据（跨端不可比：实体 id 体系不同）。"""
    canonical = view(client)
    if canonical is None:
        return None
    return digest_json(canonical)[:16]


def board_signature(client: Optional[dict]) -> List[str]:
    """跨端可比的战团指纹：``card_id:atk:hp:golden:kw``。"""
    canonical = view(client) or {}
    out = []
    for row in canonical.get("board") or []:
        tags = row.get("tags") or {}
        kw = ",".join(sorted(k for k in tags if k not in ("PREMIUM", "ATK", "HEALTH")))
        out.append(f"{row.get('card_id')}:{row.get('attack')}:{row.get('health')}:"
                   f"{int(bool(row.get('golden')))}:{kw}")
    return out


def board_multiset(client: Optional[dict]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for row in (view(client) or {}).get("board") or []:
        key = str(row.get("card_id"))
        result[key] = result.get(key, 0) + 1
    return result


def observable(client: Optional[dict]) -> Dict[str, Any]:
    """跨端比较用的可观测字段白名单。"""
    canonical = view(client) or {}
    return {k: canonical.get(k) for k in OBSERVABLE_FIELDS}


# --------------------------------------------------------------------------- #
# 状态差值
# --------------------------------------------------------------------------- #
def _bounded(values: List[Any], limit: int) -> Dict[str, Any]:
    return {"values": values[:limit], "omitted": max(0, len(values) - limit)}


def _zone_delta(before: List[dict], after: List[dict], limit: int) -> Dict[str, Any]:
    b = {(r.get("entity_id"), r.get("card_id")): r for r in before}
    a = {(r.get("entity_id"), r.get("card_id")): r for r in after}
    removed = [{"entity_id": k[0], "card_id": k[1]} for k in b if k not in a]
    added = [{"entity_id": k[0], "card_id": k[1]} for k in a if k not in b]
    stats = []
    order_changed = [r.get("entity_id") for r in before] != [r.get("entity_id") for r in after]
    for key in b:
        if key not in a:
            continue
        x, y = b[key], a[key]
        if (x.get("attack"), x.get("health"), x.get("tier")) != (y.get("attack"), y.get("health"), y.get("tier")):
            stats.append({"entity_id": key[0], "card_id": key[1],
                          "attack": [x.get("attack"), y.get("attack")],
                          "health": [x.get("health"), y.get("health")],
                          "tier": [x.get("tier"), y.get("tier")]})
    if any(x.get("tags") != y.get("tags") for key, x in b.items() for y in [a.get(key) or {}]):
        for key in b:
            if key in a and b[key].get("tags") != a[key].get("tags"):
                stats.append({"entity_id": key[0], "card_id": key[1],
                              "tags": [b[key].get("tags"), a[key].get("tags")]})
    return {
        "removed": _bounded(removed, limit),
        "added": _bounded(added, limit),
        "order_changed": order_changed,
        "stats": _bounded(stats, limit),
    }


def delta(before: Optional[dict], after: Optional[dict], *, limit: int = 12) -> Optional[dict]:
    """执行前后状态差值；两侧任何一侧缺失就返回 None（绝不猜测）。"""
    b, a = view(before), view(after)
    if b is None or a is None:
        return None
    scalars = {}
    for key in ("turn", "phase", "pending_kind", "gold", "tavern_tier", "upgrade_cost",
                "health", "armor"):
        if b.get(key) != a.get(key):
            scalars[key] = [b.get(key), a.get(key)]
    result = {
        "scalars": scalars,
        "board": _zone_delta(b.get("board") or [], a.get("board") or [], limit),
        "hand": _zone_delta(b.get("hand") or [], a.get("hand") or [], limit),
        "shop": _zone_delta(b.get("shop") or [], a.get("shop") or [], limit),
    }
    counters = {}
    for key, value in (a.get("counters") or {}).items():
        prior = (b.get("counters") or {}).get(key)
        if isinstance(value, (int, float)) and isinstance(prior, (int, float)) and value != prior:
            counters[key] = round(float(value) - float(prior), 6)
    if counters:
        result["counters"] = counters
    changed = bool(scalars or counters)
    for zone in ("board", "hand", "shop"):
        part = result[zone]
        if (part["removed"]["values"] or part["added"]["values"]
                or part["stats"]["values"] or part["order_changed"]):
            changed = True
            break
    result["changed"] = changed
    return result


# --------------------------------------------------------------------------- #
# 模拟器视图构造（引擎 state_json -> 与实机同形状的 client）
# --------------------------------------------------------------------------- #
def build_catalog(catalog_rows: Iterable[dict]) -> Dict[int, dict]:
    """``hsbg_engine.catalog_json()`` -> {cid: {id, kind, name, cost, tier}}。"""
    table: Dict[int, dict] = {}
    for row in catalog_rows:
        cid = row.get("cid", row.get("id_num"))
        if cid is None:
            continue
        table[int(cid)] = {
            "card_id": row.get("id"),
            "kind": row.get("kind"),
            "name": row.get("name"),
            "name_zh": row.get("name_zh"),
            "cost": row.get("cost"),
            "tier": row.get("tier"),
        }
    return table


_CARD_TYPE = {
    "minion": "MINION", "spell": "SPELL", "hero": "HERO", "hero_power": "HERO_POWER",
    "battleground_trinket": "BATTLEGROUND_TRINKET", "trinket": "BATTLEGROUND_TRINKET",
    "battleground_spell": "BATTLEGROUND_SPELL", "enchantment": "ENCHANTMENT",
    "game_mode_button": "GAME_MODE_BUTTON",
}


def _sim_item(row: Optional[dict], catalog: Dict[int, dict], position: int,
              zone: str) -> Optional[dict]:
    """引擎 zone 条目 -> 统一卡牌行。

    引擎把商店/手牌包成 ``{"sid"|"hid": n, "item": {"Minion"|"Spell": {...}}}``；
    随从自带 ``eid``，法术没有实体 id，就用容器 id（``sid``/``hid``）作为
    可追踪的身份，绝不编造不存在的 entity_id。
    """
    if not row:
        return None
    container = row.get("item") if isinstance(row.get("item"), dict) else None
    payload: dict = {}
    item_kind = None
    if container:
        item_kind = next(iter(container), None)
        payload = container.get(item_kind) or {}
    else:
        payload = row
    cid = payload.get("card", row.get("card"))
    if cid is None:
        return None
    cid = int(cid)
    meta = catalog.get(cid) or {}
    kind = str(item_kind or meta.get("kind") or "").lower()
    golden = bool(payload.get("golden"))
    tags = kw_to_tags(payload.get("kw"), payload.get("bonus_kw"), golden=golden)
    if row.get("frozen"):
        tags["FROZEN"] = 1
    entity_id = payload.get("eid")
    if entity_id is None:
        entity_id = row.get("eid", row.get("hid", row.get("sid")))
    price = row.get("price")
    return {
        "entity_id": entity_id,
        "card_id": meta.get("card_id") or f"cid:{cid}",
        "name": meta.get("name"),
        "card_type": _CARD_TYPE.get(kind, kind.upper() or None),
        "position": position,
        "attack": payload.get("atk"),
        "health": payload.get("hp"),
        "tier": payload.get("tier", meta.get("tier")),
        "cost": price if price is not None else meta.get("cost"),
        "golden": golden,
        "tags": tags,
        "attachments": len(payload.get("attached") or []),
        "zone": zone,
        "item_kind": item_kind or ("minion" if "atk" in payload else None),
    }


def _sim_minion(row: Optional[dict], catalog: Dict[int, dict], position: int) -> Optional[dict]:
    return _sim_item(row, catalog, position, "minion")


def _sim_shop_slot(row: Optional[dict], catalog: Dict[int, dict], position: int) -> Optional[dict]:
    out = _sim_item(row, catalog, position, "shop")
    if out is not None:
        out["shop_slot"] = position - 1
        out["kept"] = bool(row.get("kept"))
    return out


def _sim_hand_slot(row: Optional[dict], catalog: Dict[int, dict], position: int) -> Optional[dict]:
    out = _sim_item(row, catalog, position, "hand")
    if out is not None:
        out["hand_slot"] = position - 1
        out["locked"] = int(row.get("locked") or 0)
    return out


def sim_view(state: dict, seat: int, catalog: Dict[int, dict]) -> Dict[str, Any]:
    """引擎真值 -> 统一 ``client`` 形状。

    只做字段搬运与已公开的位域解码，不做任何补全；引擎里没有的概念
    （如 UI ``controls``）留空，由调用方写进 ``approximation_fields``。
    """
    players = state.get("players") or []
    player = players[seat] if 0 <= seat < len(players) else {}
    hero_cid = player.get("hero")
    hero_meta = catalog.get(int(hero_cid)) if hero_cid is not None else {}
    pending = player.get("pending") or []
    choices, targets, pending_kinds = [], [], []
    for entry in pending:
        if not isinstance(entry, dict):
            continue
        for kind, payload in entry.items():
            pending_kinds.append(str(kind))
            payload = payload or {}
            options = payload.get("options") or []
            if kind == "Target":
                # Target 的 options 是**实体 id**（可选中的目标），不是卡牌 id。
                targets.extend(int(v) for v in options if isinstance(v, int))
            elif kind == "Discover":
                for option in options:
                    if isinstance(option, dict):
                        inner = next(iter(option.values()), None)
                        choices.append(_sim_item({"item": option}, catalog,
                                                 len(choices) + 1, "choice"))
                    elif isinstance(option, int):
                        choices.append(_sim_minion({"card": option}, catalog,
                                                   len(choices) + 1))
            elif isinstance(options, list):
                for option in options:
                    cid = option.get("card") if isinstance(option, dict) else option
                    if isinstance(cid, int):
                        choices.append(_sim_minion({"card": cid}, catalog,
                                                   len(choices) + 1))
    choices = [c for c in choices if c]
    return {
        "turn": state.get("turn"),
        "phase": state.get("phase"),
        "local_player_id": player.get("id", seat),
        "gold": player.get("gold"),
        "tavern_tier": player.get("tier"),
        "upgrade_cost": player.get("upgrade_cost"),
        "health": player.get("health"),
        "hero": {
            "entity_id": None,
            "card_id": (hero_meta or {}).get("card_id"),
            "name": (hero_meta or {}).get("name"),
            "health": player.get("health"),
            "tags": {"ARMOR": player.get("armor")} if player.get("armor") is not None else {},
        },
        "board": [_sim_minion(m, catalog, i + 1) for i, m in enumerate(player.get("board") or [])],
        "hand": [_sim_hand_slot(m, catalog, i + 1) for i, m in enumerate(player.get("hand") or [])],
        "shop": [_sim_shop_slot(m, catalog, i + 1) for i, m in enumerate(player.get("shop") or [])],
        "choices": choices,
        "controls": [],
        "pending_kind": pending_kinds[0] if pending_kinds else None,
        "targets": targets,
        "counters": {k: v for k, v in (player.get("stats") or {}).items()
                     if isinstance(v, (int, float))}
                    | {"placement": player.get("placement"), "alive": player.get("alive")},
        "choices_pending": bool(pending),
        "turn_data": player.get("turn_data"),
    }


SIM_APPROXIMATION_FIELDS = {
    "controls": "模拟器没有客户端 UI 控件，实机记录里的 controls 在这里为空",
    "shop[].cost": "引擎商店条目 price 为空时给卡牌基础费用；折后价不可得",
    "hero.entity_id": "引擎英雄没有客户端 entity_id",
    "choices": "只记录 discover/hero-select 的候选卡牌 id 与槽位，没有实机的任务列表语义",
}


# --------------------------------------------------------------------------- #
# 决策 / 执行 关联与完整性校验
# --------------------------------------------------------------------------- #
def decision_id(*, side: str, game_id: Any, seat: Any, turn: Any,
                state: Optional[str], action: Any, occurrence: int = 0) -> str:
    """稳定 id：同一端内 (对局, 座位, 回合, 状态, 动作, 第几次) 唯一。

    ``occurrence`` 是"同状态同动作"的序号：实机重试、模拟器同一回合内
    连续两次相同动作都能区分，同时保持可重复计算。
    """
    key = {"side": side, "game": str(game_id), "seat": seat, "turn": turn,
           "state": state, "action": action, "n": int(occurrence)}
    return "d-" + digest_json(key)[:16]


def join_decisions(decisions: Sequence[dict], executions: Sequence[dict],
                   *, limit: int = 24) -> Tuple[List[dict], List[str]]:
    """把决策与执行合并成一条可读记录；返回 (rows, problems)。"""
    by_decision: Dict[str, List[dict]] = {}
    orphans: List[str] = []
    for row in executions:
        did = row.get("decision_id")
        if not did:
            continue
        by_decision.setdefault(did, []).append(row)
    known = {row.get("decision_id") for row in decisions}
    for did in by_decision:
        if did not in known:
            orphans.append(f"execution without decision: {did}")
    rows: List[dict] = []
    problems: List[str] = list(orphans)
    for row in decisions:
        did = row.get("decision_id")
        attempts = sorted(by_decision.get(did) or [], key=lambda r: r.get("attempt", 0))
        physical = [a for a in attempts if a.get("physical")]
        receipts = [a for a in attempts if a.get("status") == "receipt"]
        terminal = receipts[-1] if receipts else (physical[-1] if physical else (attempts[-1] if attempts else None))
        merged = dict(row)
        merged["execution"] = {
            "attempts": len(attempts),
            "physical": len(physical),
            "status": (terminal or {}).get("status"),
            "outcome": (terminal or {}).get("outcome"),
            "error": (terminal or {}).get("error"),
            "latency_ms": ((terminal or {}).get("timing") or {}).get("elapsed_ms") if terminal else None,
            "received_at_log_position": (terminal or {}).get("received_at_log_position"),
        } if terminal or attempts else None
        after = None
        for attempt in reversed(attempts):
            if attempt.get("after") is not None:
                after = attempt["after"]
                break
        merged["after"] = view(after)
        merged["delta"] = delta(row.get("pre"), after)
        if merged["execution"] is None and row.get("kind") != "noop":
            problems.append(f"decision without execution: {did}")
        rows.append(merged)
    return rows, problems


def audit(decisions: Sequence[dict], executions: Sequence[dict],
          waits: Sequence[dict], *, side: str) -> dict:
    """完整性校验：不重、不漏、顺序可还原。"""
    problems: List[str] = []
    ids = [row.get("decision_id") for row in decisions]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        problems.append(f"duplicate decision_id: {sorted(dupes)[:5]}")
    sequences = [row.get("sequence") for row in decisions]
    if sequences != sorted(sequences):
        problems.append("decision sequence is not monotonic")
    turn_order = [row.get("turn") for row in decisions]
    regressions = [(a, b) for a, b in zip(turn_order, turn_order[1:])
                   if isinstance(a, int) and isinstance(b, int) and b < a]
    if regressions:
        problems.append(f"turn regressions: {regressions[:5]}")
    executed = {row.get("decision_id") for row in executions if row.get("decision_id")}
    missing = [row.get("decision_id") for row in decisions if row.get("decision_id") not in executed
               and row.get("kind") not in (None, "noop")]
    unresolved = []
    for row in decisions:
        attempts = [e for e in executions if e.get("decision_id") == row.get("decision_id")]
        if not attempts:
            continue
        if not any(a.get("status") == "receipt" or a.get("outcome") for a in attempts):
            unresolved.append(row.get("decision_id"))
    if missing:
        problems.append(f"decisions without any execution row: {len(missing)} {missing[:5]}")
    if unresolved:
        problems.append(f"decisions without terminal outcome: {len(unresolved)} {unresolved[:5]}")
    seen = {(row.get("turn"), row.get("action")) for row in decisions if not row.get("reused")}
    repeated = [row.get("decision_id") for row in decisions
                if row.get("reused") and (row.get("turn"), row.get("action")) in seen]
    if repeated:
        problems.append(f"reused rows still present in decisions.jsonl: {len(repeated)}")
    per_turn: Dict[Any, int] = {}
    for row in decisions:
        per_turn[row.get("turn")] = per_turn.get(row.get("turn"), 0) + 1
    return {
        "schema": AUDIT_SCHEMA,
        "side": side,
        "visibility": VISIBILITY.get(side),
        "decisions": len(decisions),
        "executions": len(executions),
        "waits": len(waits),
        "decisions_per_turn": {str(k): v for k, v in sorted(per_turn.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "missing_execution": len(missing),
        "unresolved_execution": len(unresolved),
        "problems": problems,
        "complete": not problems,
    }


def turns_summary(decisions: Sequence[dict]) -> Dict[str, dict]:
    """逐回合动作类型统计（报告与跨边比较的主表）。"""
    out: Dict[str, dict] = {}
    for row in decisions:
        key = str(row.get("turn"))
        bucket = out.setdefault(key, {"decisions": 0, "distinct": 0, "classes": {}, "kinds": {},
                                      "phase": row.get("phase"), "gold": None,
                                      "tier": None, "_last": None})
        bucket["decisions"] += 1
        cls = action_class(row.get("kind"))
        bucket["classes"][cls] = bucket["classes"].get(cls, 0) + 1
        bucket["kinds"][str(row.get("kind"))] = bucket["kinds"].get(str(row.get("kind")), 0) + 1
        # "有意义的决策"：动作身份（类型+源+目标+槽位）与上一条不同的决策。
        # 同一意图在新状态上的重读/重试仍然记录，但不再冒充第二次操作。
        identity = (row.get("kind"), row.get("source"), row.get("target"), row.get("position"))
        if identity != bucket["_last"]:
            bucket["distinct"] += 1
            bucket["_last"] = identity
        pre = row.get("pre") or {}
        if bucket["decisions"] == 1:
            # 回合开始值：跨端比较金币/等级节奏时要用"回合第一决策前"的状态。
            bucket["gold"] = pre.get("gold")
            bucket["tier"] = pre.get("tavern_tier")
    for bucket in out.values():
        bucket.pop("_last", None)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10 ** 6))
