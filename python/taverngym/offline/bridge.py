"""纯数据离线桥：没有窗口/网络/OS 输入；缺字段拒收，回执确认后才提交记忆。"""
from dataclasses import dataclass, field
import copy
import uuid

import numpy as np
import taverngym as tg

from .actions import decode
from .trace import digest


class BridgeBlocked(ValueError):
    pass


@dataclass(frozen=True)
class ClientSnapshot:
    game_id: str
    revision: int
    build: str
    data: dict

    @classmethod
    def from_dict(cls, data):
        data = copy.deepcopy(data)
        if not isinstance(data.get("game_id"), str) or not data["game_id"]:
            raise BridgeBlocked("missing game_id")
        if type(data.get("revision")) is not int or data["revision"] < 0:
            raise BridgeBlocked("invalid semantic revision")
        if not data.get("build"):
            raise BridgeBlocked("missing client build")
        return cls(data["game_id"], data["revision"], str(data["build"]), data)

    @property
    def stamp(self):
        data = {k: v for k, v in self.data.items() if k not in ("captured_at", "source_updated_at", "log_event_at")}
        return digest(data)


@dataclass(frozen=True)
class PolicyInput:
    observation: dict
    provenance: dict


@dataclass(frozen=True)
class ActionIntent:
    intent_id: str
    game_id: str
    revision: int
    stamp: str
    action: int
    semantic: dict
    binding: dict


@dataclass(frozen=True)
class ExecutionReceipt:
    intent_id: str
    game_id: str
    before_revision: int
    after_revision: int
    status: str  # applied / rejected / not_sent / waiting / unknown


def blocks():
    l = dict(tg.layout())
    sizes = {"global": l["GLOBAL_F"], "quests": l["N_QUESTS"] * l["QUEST_F"],
             "pending": l["PENDING_KINDS"], "board": l["N_BOARD"] * l["MINION_F"],
             "hand": l["N_HAND"] * l["ITEM_F"], "shop": l["N_SHOP"] * l["ITEM_F"],
             "choices": l["N_CHOICE"] * l["ITEM_F"]}
    sizes["opponents"] = tg.OBS_FLOATS - sum(sizes.values())
    return sizes


def encode_snapshot(snapshot, expected_identity, *, allowed_builds, allow_fixture=False):
    """接受旧系统 JSON 的显式公共上下文扩展；旧候选观测不是模型输入。

    客户端采集器尚未提供的基础/效果字段必须先由历史重建并登记来源。
    本函数不从随机 Game 或特权世界补齐字段。
    """
    if snapshot.build not in allowed_builds:
        raise BridgeBlocked(f"unsupported client build: {snapshot.build}")
    data = snapshot.data
    if data.get("ready") is not True or data.get("options_pending", False):
        raise BridgeBlocked("snapshot not ready for decision")
    if data.get("phase") not in ("recruit", "choice", "hero_select"):
        raise BridgeBlocked("snapshot is outside a supported decision phase")
    context = data.get("policy_context")
    if not isinstance(context, dict):
        raise BridgeBlocked("missing policy_context: global/quests/pending/board/hand/shop/choices/opponents, cards, extras, legal bindings")
    if context.get("identity") != expected_identity:
        raise BridgeBlocked("policy context identity mismatch")
    if context.get("format") != "taverngym.public-context.v1":
        raise BridgeBlocked("unsupported public context format")
    provenance = context.get("provenance", {})
    features = []
    for name, size in blocks().items():
        source = provenance.get(name)
        if source not in ("client", "history", "native_fixture"):
            raise BridgeBlocked(f"missing or invalid provenance: {name}")
        if source == "native_fixture" and not allow_fixture:
            raise BridgeBlocked("native fixture cannot be presented as real client encoding")
        values = context.get("features", {}).get(name)
        if not isinstance(values, list) or len(values) != size:
            raise BridgeBlocked(f"missing/invalid feature block: {name} ({size})")
        features.extend(values)
    for name in ("cards", "extras", "mask"):
        if provenance.get(name) not in ("client", "history", *(('native_fixture',) if allow_fixture else ())):
            raise BridgeBlocked(f"missing or invalid provenance: {name}")
    vocab = {tg.card_info(i)[0]: i for i in range(1, tg.N_CARDS)}
    vocab[""] = 0
    try:
        cards = [vocab[c] for c in context["cards"]]
        extras = [(vocab[c], owner, values) for c, owner, values in context["extras"]]
        mask = context["mask"]
        if len(cards) != tg.OBS_CARDS or len(mask) != tg.N_ACTIONS or any(type(x) is not bool for x in mask) or not any(mask):
            raise BridgeBlocked("invalid cards or legal mask shape/type")
        if any(type(owner) is not int or not 0 <= owner <= tg.OBS_CARDS or len(values) != 8 for _, owner, values in extras):
            raise BridgeBlocked("invalid effect token")
        if not np.isfinite(np.asarray(features, dtype=np.float32)).all() or any(not np.isfinite(v).all() for _, _, v in extras):
            raise BridgeBlocked("nonfinite policy input")
    except (KeyError, TypeError) as exc:
        raise BridgeBlocked(f"unknown card or missing policy context field: {exc}") from exc
    # 已有客户端字段不得与预编码上下文矛盾；不可从缺失值猜测。
    checks = (("turn", 0, 20), ("gold", 7, 10), ("tavern_tier", 10, 6), ("upgrade_cost", 11, 10))
    for key, i, scale in checks:
        if data.get(key) is None or not np.isclose(features[i] * scale, data[key], atol=1e-5):
            raise BridgeBlocked(f"client/context mismatch or missing: {key}")
    if not extras or int(extras[0][2][0]) != 0:
        raise BridgeBlocked("missing observable action-budget token")
    return PolicyInput({"features": np.asarray(features, dtype=np.float32), "cards": np.asarray(cards, dtype=np.int32),
                        "mask": np.asarray(mask, dtype=bool), "extras": extras}, copy.deepcopy(provenance))


def bind_action(snapshot, action):
    semantic = decode(action)
    if "action_bindings" not in snapshot.data:
        from .client_actions import translate
        bindings, _ = translate(snapshot.data)
    else:
        bindings = snapshot.data["action_bindings"]
    # 一项原生动作只能对应一个当帧绑定，不猜测坐标或目标。
    matches = [a for a in bindings if a.get("flat_action") == action]
    if len(matches) != 1:
        raise BridgeBlocked("missing or ambiguous flat-action binding")
    binding = copy.deepcopy(matches[0])
    if binding.get("semantic") != semantic:
        raise BridgeBlocked("binding semantics mismatch")
    entities = {c["entity_id"] for zone in ("board", "hand", "shop", "choices", "controls")
                for c in snapshot.data.get(zone, []) if c.get("entity_id") is not None}
    for key in ("source", "target"):
        if binding.get(key) is not None and binding[key] not in entities:
            raise BridgeBlocked(f"binding {key} disappeared")
    kind = semantic["kind"]
    zone_index = {"buy": ("shop", semantic.get("slot")), "sell": ("board", semantic.get("slot")),
                  "activate": ("board", semantic.get("slot")), "play": ("hand", semantic.get("hand")),
                  "magnetize": ("hand", semantic.get("hand")), "move": ("board", semantic.get("source_slot"))}
    if kind in zone_index:
        zone, index = zone_index[kind]
        cards = snapshot.data.get(zone, [])
        if index >= len(cards) or binding.get("source") != cards[index]["entity_id"]:
            raise BridgeBlocked("source slot/entity mismatch")
    if kind == "magnetize":
        board = snapshot.data.get("board", [])
        if semantic["board"] >= len(board) or binding.get("target") != board[semantic["board"]]["entity_id"]:
            raise BridgeBlocked("magnetize target mismatch")
    if kind == "move":
        board = snapshot.data.get("board", [])
        if semantic["target_slot"] >= len(board):
            raise BridgeBlocked("move destination does not exist")
        original = binding.get("client_action")
        if original:
            order = [c["entity_id"] for c in board]
            order.insert(semantic["target_slot"], order.pop(semantic["source_slot"]))
            if order != list(original.get("order") or []):
                raise BridgeBlocked("reorder result mismatch")
    if kind == "choose" and not binding.get("choice_id"):
        raise BridgeBlocked("choice request identity missing")
    return ActionIntent(uuid.uuid4().hex, snapshot.game_id, snapshot.revision, snapshot.stamp, action, semantic, binding)


@dataclass
class MemoryLedger:
    """两阶段提交；只有与待决意图匹配的明确 applied 回执推进记忆。"""
    hidden: object = None
    game_id: str | None = None
    revision: int = -1
    count: int = 0
    pending: tuple | None = None
    seen: set = field(default_factory=set)

    def begin(self, snapshot, intent, proposed_hidden):
        if snapshot.game_id != self.game_id:
            self.game_id, self.hidden, self.revision, self.count, self.pending = snapshot.game_id, None, -1, 0, None
            self.seen.clear()
        if self.pending is not None or snapshot.revision < self.revision:
            raise BridgeBlocked("pending intent or stale observation")
        if intent.stamp != snapshot.stamp or intent.revision != snapshot.revision or intent.game_id != snapshot.game_id:
            raise BridgeBlocked("intent is not bound to snapshot")
        self.pending = (intent, proposed_hidden)

    def confirm(self, receipt):
        if receipt.status not in ("applied", "rejected", "not_sent", "waiting", "unknown"):
            raise BridgeBlocked("invalid receipt status")
        key = (receipt.game_id, receipt.intent_id)
        if key in self.seen:
            return False
        if self.pending is None:
            raise BridgeBlocked("receipt without pending intent")
        intent, proposed = self.pending
        if (receipt.game_id, receipt.intent_id, receipt.before_revision) != (intent.game_id, intent.intent_id, intent.revision):
            raise BridgeBlocked("receipt identity mismatch")
        if receipt.status in ("waiting", "unknown"):
            return False
        if receipt.status == "applied":
            if receipt.after_revision <= receipt.before_revision:
                raise BridgeBlocked("applied receipt needs newer semantic revision")
            self.hidden, self.revision, self.count = proposed, receipt.after_revision, self.count + 1
        self.pending = None
        self.seen.add(key)
        return receipt.status == "applied"
