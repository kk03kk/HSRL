"""有证据的历史计数重建；日志从中途开始或有缺口时返回未知。"""
from .bridge import BridgeBlocked


class PublicHistory:
    def __init__(self):
        self.game_id = None
        self.turn = None
        self.revision = -1
        self.complete = False
        self.actions = self.freezes = 0
        self.receipts = set()
        self.counters = {}

    def apply(self, event):
        kind = event.get("type")
        game = event.get("game_id")
        revision = event.get("revision")
        if not game or type(revision) is not int or revision < 0:
            raise BridgeBlocked("history event needs game/revision")
        if kind == "game_start":
            if self.game_id == game:
                raise BridgeBlocked("duplicate game start")
            self.__init__()
            self.game_id, self.revision, self.complete = game, revision, True
            return
        if self.game_id != game:
            raise BridgeBlocked("history has no matching game start")
        key = event.get("intent_id")
        if kind == "receipt" and key in self.receipts:
            return
        if revision < self.revision:
            raise BridgeBlocked("out of order history")
        self.revision = revision
        if kind == "gap":
            self.complete = False
            self.counters.clear()
        elif kind == "turn_start":
            turn = event.get("turn")
            if type(turn) is not int or turn < 1 or self.turn is not None and turn <= self.turn:
                raise BridgeBlocked("invalid turn boundary")
            if self.turn is not None and turn != self.turn + 1:
                self.complete = False
            self.turn, self.actions, self.freezes = turn, 0, 0
        elif kind == "receipt":
            if not key or event.get("status") not in ("applied", "rejected", "not_sent", "waiting", "unknown"):
                raise BridgeBlocked("invalid history receipt")
            if event["status"] == "applied":
                if self.turn is None or event.get("turn") != self.turn:
                    raise BridgeBlocked("receipt turn mismatch")
                self.receipts.add(key)
                self.actions += 1
                self.freezes += int(event.get("kind") == "freeze")
                # 不从“上场”猜测最终战吼计数；Brann/附件等可改变触发次数。
                if event.get("kind") == "play":
                    self.counters.pop("battlecries_triggered", None)
            elif event["status"] in ("rejected", "not_sent"):
                self.receipts.add(key)
        elif kind == "public_counter":
            name, value = event.get("name"), event.get("value")
            if not isinstance(name, str) or not name or type(value) is not int or value < 0:
                raise BridgeBlocked("invalid public counter")
            self.counters[name] = {"value": value, "revision": revision, "source": "client"}
        else:
            raise BridgeBlocked("unsupported history event")

    def values(self):
        known = self.complete and self.turn is not None
        return {"game_id": self.game_id, "turn": self.turn, "history_complete": self.complete,
                "actions_this_turn": self.actions if known else None,
                "freezes_this_turn": self.freezes if known else None,
                "counters": dict(self.counters)}
