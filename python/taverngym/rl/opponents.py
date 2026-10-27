"""可复现的脚本分布扰动；未声称强于native，也未声称现实校准。"""
import taverngym as tg


def profile_action(obs, action, profile, slot):
    if profile == "mixed":
        profile = ("native", "tempo", "economy")[slot % 3]
    if profile == "native":
        return action
    l = dict(tg.layout())
    f, mask = obs["features"], obs["mask"]
    pending = l["GLOBAL_F"] + l["N_QUESTS"] * l["QUEST_F"]
    if not f[pending]:
        return action
    turn, tier = round(float(f[0]) * 20), round(float(f[10]) * 6)
    if profile == "economy":
        target = 1 + sum(turn >= t for t in (3, 6, 9, 12, 15))
        if tier < target and mask[l["A_UPGRADE"]]:
            return l["A_UPGRADE"]
    if not l["A_BUY"] <= action < l["A_SELL"]:
        return action
    start = pending + l["PENDING_KINDS"] + l["N_BOARD"] * l["MINION_F"] + l["N_HAND"] * l["ITEM_F"]
    best, score = action, -float("inf")
    for a in range(l["A_BUY"], l["A_SELL"]):
        row = start + (a - l["A_BUY"]) * l["ITEM_F"]
        if mask[a] and f[row + 1]:
            atk, hp = f[row + 10:row + 12]
            value = float(atk + (2 if profile == "tempo" else 1) * hp)
            if value > score:
                best, score = a, value
    return best
