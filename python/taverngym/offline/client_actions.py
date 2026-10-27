"""旧系统已序列化 Options → 原生动作绑定。只翻译明确语义，不读取驱动。"""
from .actions import decode


def translate(data):
    bindings, unsupported = [], []
    zones = {zone: {c["entity_id"]: i for i, c in enumerate(data.get(zone, []))}
             for zone in ("hand", "board", "shop", "choices")}
    for original in data.get("actions", []):
        a = dict(original)
        kind, source, target = a.get("kind"), a.get("source"), a.get("target")
        flat = None
        reason = None
        interaction = a.get("interaction") or {}
        if a.get("selected") or a.get("sub_option_id") is not None or len(interaction.get("stages", [])) > 1:
            reason = "multi-selection or multi-stage interaction needs explicit protocol"
        elif kind in ("end", "refresh", "freeze", "upgrade", "cancel") and target is None:
            flat = {"end": 0, "refresh": 1, "freeze": 2, "upgrade": 3, "cancel": 7}[kind]
        elif a.get("scope") == "dark_gift" and target is None:
            flat = 6
        elif kind in ("buy", "sell", "activate") and target is None:
            zone, base = {"buy": ("shop", 8), "sell": ("board", 16), "activate": ("board", 23)}[kind]
            if source in zones[zone]:
                flat = base + zones[zone][source]
        elif kind == "play" and source in zones["hand"]:
            h = zones["hand"][source]
            if interaction.get("placement") == "magnetize" and target in zones["board"]:
                flat = 126 + h * 7 + zones["board"][target]
            elif target is None:
                card = data["hand"][h]
                if card.get("card_type") in ("SPELL", "BATTLEGROUND_SPELL"):
                    flat = 46 + h * 8
                elif card.get("card_type") == "MINION" and type(a.get("position")) is int and 1 <= a["position"] <= 8:
                    flat = 46 + h * 8 + a["position"] - 1
        elif kind in ("choose", "target") and a.get("choice_id") is not None:
            chosen = target if kind == "target" else source
            if chosen in zones["choices"] and zones["choices"][chosen] < 16:
                flat = 30 + zones["choices"][chosen]
        elif kind == "hero_power" and target is None:
            controls = [c for c in data.get("controls", []) if c.get("card_type") == "HERO_POWER"]
            slots = [i for i, c in enumerate(controls[:2]) if c["entity_id"] == source]
            if len(slots) == 1:
                flat = 4 + slots[0]
        elif kind == "reorder":
            old = [c["entity_id"] for c in data.get("board", [])]
            order = list(a.get("order") or [])
            if sorted(order) == sorted(old) and order != old:
                candidates = []
                for i in range(len(old)):
                    for j in range(len(old)):
                        trial = old.copy()
                        trial.insert(j, trial.pop(i))
                        if trial == order:
                            candidates.append(196 + i * 7 + j)
                # 等价相邻移动可以映射为两个原生动作，各自带同一客户端意图。
                for action in candidates:
                    sem = decode(action)
                    bindings.append({"flat_action": action, "semantic": sem, "source": old[sem["source_slot"]],
                                     "client_action": a})
                if candidates:
                    continue
        if flat is None or not 0 <= flat < 245 or reason:
            unsupported.append({"action": a, "reason": reason or "not losslessly representable or missing entity/position"})
        else:
            bindings.append({"flat_action": flat, "semantic": decode(flat), "source": source, "target": target,
                             "choice_id": a.get("choice_id"), "client_action": a})
    return bindings, unsupported
