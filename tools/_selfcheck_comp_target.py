"""临时自检：comp_target 的纯逻辑（不依赖引擎/torch）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsbg.az.train.comp_target import (  # noqa: E402
    CompTargetTracker, aggregate_comp_target, load_comps, parse_tier_weights,
)

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_comps.json")
with open(TMP, "w", encoding="utf-8") as fh:
    json.dump({"comps": [
        {"name": "S-comp", "tier": "S", "core_card_ids": ["C1", "C2", "C3"]},
        {"name": "B-comp", "tier": "B", "core_card_ids": ["C9"]},
        {"name": "empty", "tier": "A", "core_card_ids": []},
    ]}, fh)

COMPS = load_comps(TMP)
assert len(COMPS) == 2, COMPS
assert COMPS[0]["denom"] == 3.0 and COMPS[0]["weight"] == 1.0
assert COMPS[1]["denom"] == 2.0 and COMPS[1]["weight"] == 0.3  # 1 张核心卡 → 分母仍为 2

assert parse_tier_weights("S:1,A:0.5") == {"S": 1.0, "A": 0.5}
assert parse_tier_weights(None)["S"] == 1.0


class FakeGame:
    def __init__(self):
        self.turn = 1
        self.boards = {}

    def player_json(self, seat):
        return json.dumps({"board": [{"card": c} for c in self.boards.get(self.turn, [])]})


# index -> card_id：1->C1, 2->C2, 3->C3, 9->C9（其余返回空）
# card_id 内部用 (raw & 0x0fff) + 1 查引擎卡表，所以 fake lookup 的键要 +1
RAW = {"C1": 1, "C2": 2, "C3": 3, "C9": 9}
LOOKUP = {r + 1: cid for cid, r in RAW.items()}
tr = CompTargetTracker(coef=1.0, min_turn=8, cards_path=TMP,
                       lookup=lambda i: LOOKUP.get(i, ""))
g = FakeGame()
# T7：低于 min_turn，不打分
g.turn = 7
g.boards[7] = [1, 2, 3]
assert tr.observe(g, 0) == 0.0
# T8：命中 1/3 → 0.333，增量 0.333
g.turn = 8
g.boards[8] = [1]
d8 = tr.observe(g, 0)
assert abs(d8 - 1.0 * (1 / 3)) < 1e-9, d8
# 同一回合重复调用不再算
assert tr.observe(g, 0) == 0.0
# T9：命中 2/3 → 0.667，增量 0.333
g.turn = 9
g.boards[9] = [1, 2]
d9 = tr.observe(g, 0)
assert abs(d9 - (2 / 3 - 1 / 3)) < 1e-9, d9
# T10：卖光换 B 级（1 张核心卡 → 0.5 但权重 0.3 → 0.15），低于历史最高 → 无奖励
g.turn = 10
g.boards[10] = [9]
assert tr.observe(g, 0) == 0.0
# T11：转型拼满 S（3/3 → 1.0），增量 = 1.0 - 0.667
g.turn = 11
g.boards[11] = [1, 2, 3]
d11 = tr.observe(g, 0)
assert abs(d11 - (1.0 - 2 / 3)) < 1e-9, d11
assert tr.best_name == "S-comp" and tr.best_tier == "S"
st = tr.stats()
assert st["comp_target_turns"] == 4
assert st["comp_target_s_turns"] == 3, st
assert abs(st["comp_target_reward_sum"] - 1.0) < 1e-9, st

# set_floor：入场局那一帧不算自己拼的
tr2 = CompTargetTracker(coef=1.0, min_turn=8, cards_path=TMP,
                        lookup=lambda i: LOOKUP.get(i, ""))
tr2.set_floor(9)
g.turn = 9
g.boards[9] = [1, 2, 3]
assert tr2.observe(g, 0) == 0.0
g.turn = 10
g.boards[10] = [1, 2]
assert tr2.observe(g, 0) == 0.0  # 0.667 < 水位 1.0（注入帧设的水位）

# queue/flush
class Rec(dict):
    pass


tr3 = CompTargetTracker(coef=1.0, min_turn=8, cards_path=TMP,
                        lookup=lambda i: LOOKUP.get(i, ""))
recs = [dict(game_id=7, reward=1.0), dict(game_id=8, reward=1.0)]
assert tr3.queue(0, 7, 0.25) is True
assert tr3.queue(5, 7, 0.5) is True   # queue 不校验上界（与 tier/board 一致），flush 丢弃
assert tr3.queue(1, 99, 0.5) is True  # 串局
assert tr3.flush(recs) == 1
assert abs(recs[0]["reward"] - 1.25) < 1e-12
assert tr3.dropped == 2, tr3.dropped

# 关闭状态：零影响
off = CompTargetTracker(coef=0.0, cards_path=TMP)
assert off.enabled is False and off.stats() == {}
assert off.observe(g, 0) == 0.0 and off.queue(0, 1, 1.0) is False

agg = aggregate_comp_target([st, st])
assert agg["comp_target_turns"] == 8
assert abs(agg["comp_target_reward_sum"] - 2.0) < 1e-9
assert aggregate_comp_target([{"x": 1}]) == {}

os.remove(TMP)
print("comp_target self-check OK")
