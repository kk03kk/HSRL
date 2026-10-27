"""原生模拟器投影/执行夹具。显式标记，绝不能当作真实客户端编码证据。"""
import json

import taverngym as tg

from .actions import decode
from .bridge import ClientSnapshot, ExecutionReceipt, BridgeBlocked, blocks
from .trace import public_view


def snapshot(env, slot, experiment_identity, revision):
    batch, extras = env.observe(), env.training_extras()
    v = public_view(env.game(slot // 8), slot % 8)
    me = v["me"]
    features, offset = {}, 0
    for name, width in blocks().items():
        features[name] = batch["features"][slot][offset:offset + width].tolist()
        offset += width
    def card_id(i):
        return tg.card_info(int(i))[0] if i else ""
    context = {"format": "taverngym.public-context.v1", "identity": experiment_identity,
               "features": features, "cards": [card_id(i) for i in batch["cards"][slot]],
               "mask": batch["action_mask"][slot].tolist(),
               "extras": [[card_id(c), owner, vals] for c, owner, vals in extras[slot]],
               "provenance": {key: "native_fixture" for key in (*blocks(), "cards", "mask", "extras")}}
    data = {"game_id": f"sandbox-{slot//8}", "revision": revision, "build": tg.game_version(),
            "phase": "choice" if me["pending"] else "recruit", "ready": bool(batch["ready"][slot]),
            "turn": v["turn"], "gold": me["gold"], "tavern_tier": me["tier"],
            "upgrade_cost": float(batch["features"][slot][11]) * 10,
            "source": "native_fixture", "policy_context": context}
    for zone, eid in (("board", "eid"), ("hand", "hid"), ("shop", "sid")):
        data[zone] = [{"entity_id": c[eid], "card_id": c["id"], "position": i + 1} for i, c in enumerate(me[zone])]
    pending = me.get("pending") or {}
    data["choices"] = [{"entity_id": 1000000 + i, "card_id": c.get("id", ""), "position": i + 1}
                       for i, c in enumerate(pending.get("options", []))]
    bindings = []
    for a, legal in enumerate(context["mask"]):
        if not legal:
            continue
        sem = decode(a)
        binding = {"flat_action": a, "semantic": sem}
        kind = sem["kind"]
        if kind in ("buy", "sell", "activate"):
            zone = "shop" if kind == "buy" else "board"
            binding["source"] = data[zone][sem["slot"]]["entity_id"]
        elif kind in ("play", "magnetize"):
            binding["source"] = data["hand"][sem["hand"]]["entity_id"]
            if kind == "magnetize":
                binding["target"] = data["board"][sem["board"]]["entity_id"]
        elif kind == "move":
            binding["source"] = data["board"][sem["source_slot"]]["entity_id"]
        elif kind == "choose":
            binding["choice_id"] = f"sandbox-choice-{revision}"
            # 目标枚举与普通选牌都必须绑定当前请求；原生 Choose 下标不冒充客户端 Options ID。
            binding["option_id"] = sem["slot"]
        bindings.append(binding)
    data["action_bindings"] = bindings
    return ClientSnapshot.from_dict(data)


class SimulatorExecutor:
    offline_only = True

    def __init__(self, env, slot, identity):
        self.env, self.slot, self.identity = env, slot, identity
        self.revision = 0
        self.receipts = {}

    def observe(self):
        return snapshot(self.env, self.slot, self.identity, self.revision)

    def advance_waiting(self, max_ticks=12000):
        """等待时只推进其他席位；不为非ready的learner制造决策或记忆更新。"""
        for _ in range(max_ticks):
            if self.env.observe()["ready"][self.slot]:
                return True
            view = public_view(self.env.game(self.slot // 8), self.slot % 8)
            if not view["me"]["alive"] or view["phase"] == "game_over":
                return False
            self.env.step(self.env.bot_actions())
            self.revision += 1
            if any(a or b or c for a, b, c in self.env.diagnostics()):
                raise BridgeBlocked("simulator guard during waiting")
        raise BridgeBlocked("waiting tick limit")

    def execute(self, intent):
        if intent.intent_id in self.receipts:
            return self.receipts[intent.intent_id]
        current = self.observe()
        if intent.game_id != current.game_id or intent.revision != current.revision or intent.stamp != current.stamp:
            raise BridgeBlocked("stale simulator intent")
        actions = self.env.bot_actions()
        actions[self.slot] = intent.action
        _, _, info = self.env.step(actions)
        applied = bool(info["applied"][self.slot])
        old = self.revision
        # 即使我方延期，其他席位也可能改变上下文；捕获 revision 跟随全局推进。
        self.revision += 1
        receipt = ExecutionReceipt(intent.intent_id, intent.game_id, old, self.revision,
                                   "applied" if applied else "rejected" if info["rejected"][self.slot] else "not_sent")
        self.receipts[intent.intent_id] = receipt
        return receipt


def decide(snapshot, model, cfg, ledger, expected_identity, *, allowed_builds, fixture=False):
    import torch
    from taverngym.rl.model import pack
    from .bridge import encode_snapshot, bind_action
    if ledger.pending:
        raise BridgeBlocked("cannot infer while a result is unknown")
    if ledger.game_id is not None and snapshot.game_id != ledger.game_id:
        ledger.hidden, ledger.count = None, 0
    if snapshot.game_id == ledger.game_id and snapshot.revision < ledger.revision:
        raise BridgeBlocked("stale snapshot")
    obs = encode_snapshot(snapshot, expected_identity, allowed_builds=allowed_builds, allow_fixture=fixture)
    hidden = ledger.hidden
    if hidden is None or ledger.count % cfg.memory_horizon == 0:
        hidden = torch.zeros((1, cfg.hidden), device=cfg.device)
    model.eval()
    with torch.no_grad():
        logits, _, _, proposed = model(pack([obs.observation], cfg.device), hidden)
    if not torch.isfinite(logits).all():
        raise BridgeBlocked("nonfinite inference")
    action = int(logits.argmax(-1)[0])
    intent = bind_action(snapshot, action)
    ledger.begin(snapshot, intent, proposed.detach().clone())
    return intent
