"""245 个原生动作的无损语义描述。位置是零基，实体身份由适配层绑定。"""

N_ACTIONS = 245


def decode(action):
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < N_ACTIONS:
        raise ValueError("action must be an integer in [0,245)")
    if action < 8:
        if action in (4, 5):
            return {"kind": "hero_power", "slot": action - 4}
        return {"kind": {0: "end_turn", 1: "refresh", 2: "freeze", 3: "upgrade",
                         6: "dark_gift", 7: "cancel"}[action]}
    for lo, hi, kind in ((8, 16, "buy"), (16, 23, "sell"), (23, 30, "activate"),
                         (30, 46, "choose")):
        if lo <= action < hi:
            return {"kind": kind, "slot": action - lo}
    if action < 126:
        hand, position = divmod(action - 46, 8)
        return {"kind": "play", "hand": hand, "position": position}
    if action < 196:
        hand, board = divmod(action - 126, 7)
        return {"kind": "magnetize", "hand": hand, "board": board}
    source, target = divmod(action - 196, 7)
    return {"kind": "move", "source_slot": source, "target_slot": target}


def encode(description):
    # 只有 245 项，显式相等避免忽略未知字段或混用语义。
    for action in range(N_ACTIONS):
        if decode(action) == description:
            return action
    raise ValueError("unrepresentable action description")
