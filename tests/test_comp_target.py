"""``hsbg/az/train/comp_target.py`` 的纯逻辑单测（不依赖引擎/torch）。

覆盖点：

1. ``load_comps``：跳过无核心卡的流派；**只有 1 张核心卡的流派分母仍按 2**（否则
   "买到那一张"就能拿满分，会把奖励刷成噪声）；等级权重按 ``tier`` 取；
2. ``parse_tier_weights``：字符串/None/非法输入的解析口径；
3. ``observe``：``min_turn`` 之前不打分；命中数线性给分；同一回合幂等；
   **奖励是历史最高完成度的增量**（转型到更强的阵容照样拿分，中途卖光不扣分）；
4. ``set_floor``：入场局注入的那一帧只设水位、不发奖励；
5. ``queue``/``flush``：越界与串局丢弃，正常条目写进 ``record["reward"]``；
6. ``coef=0`` 时零影响（``stats()`` 为空、``observe``=0、``queue``=False）；
7. ``aggregate_comp_target``：跨 worker 按回合数加权求均值、计数求和；空输入返回 ``{}``。

注释里的 ``(raw & 0x0fff) + 1`` 是引擎口径：``player_json`` 的 ``Minion.card`` 是
0-based CardId（高位是金色标记），而 ``card_info`` 要 1-based。所以下面的 fake lookup
键是 raw+1。
"""

from __future__ import annotations

import json

import pytest

from hsbg.az.train.comp_target import (
    CompTargetTracker,
    aggregate_comp_target,
    format_tier_weights,
    load_comps,
    parse_tier_weights,
)

RAW = {"C1": 1, "C2": 2, "C3": 3, "C9": 9}
LOOKUP = {r + 1: cid for cid, r in RAW.items()}


def _lookup(idx: int) -> str:
    return LOOKUP.get(int(idx), "")


@pytest.fixture()
def comps_path(tmp_path):
    p = tmp_path / "comps.json"
    p.write_text(json.dumps({"comps": [
        {"name": "S-comp", "tier": "S", "core_card_ids": ["C1", "C2", "C3"]},
        {"name": "B-comp", "tier": "B", "core_card_ids": ["C9"]},
        {"name": "no-core", "tier": "A", "core_card_ids": []},
    ]}, ensure_ascii=False), encoding="utf-8")
    return str(p)


class FakeGame:
    def __init__(self) -> None:
        self.turn = 1
        self.boards = {}

    def player_json(self, seat: int) -> str:
        return json.dumps({"board": [{"card": c} for c in self.boards.get(self.turn, [])]})


def test_load_comps_and_weights(comps_path):
    comps = load_comps(comps_path)
    assert [c["name"] for c in comps] == ["S-comp", "B-comp"]     # 无核心卡的被跳过
    assert comps[0]["denom"] == 3.0 and comps[0]["weight"] == 1.0
    assert comps[1]["denom"] == 2.0 and comps[1]["weight"] == 0.3  # 1 张核心卡 → 分母 2
    assert parse_tier_weights("S:1,A:0.5") == {"S": 1.0, "A": 0.5}
    assert parse_tier_weights(None)["S"] == 1.0
    assert format_tier_weights({"B": 0.3, "S": 1.0}) == "B:0.3,S:1"
    with pytest.raises(ValueError):
        parse_tier_weights("nonsense")


def test_reward_is_increment_of_running_best(comps_path):
    tr = CompTargetTracker(coef=1.0, min_turn=8, cards_path=comps_path, lookup=_lookup)
    g = FakeGame()
    g.turn, g.boards[7] = 7, [1, 2, 3]
    assert tr.observe(g, 0) == 0.0                      # min_turn 之前不打分
    g.turn, g.boards[8] = 8, [1]
    assert tr.observe(g, 0) == pytest.approx(1 / 3)      # 命中 1/3
    g.turn, g.boards[9] = 9, [1, 2]
    assert tr.observe(g, 0) == pytest.approx(2 / 3 - 1 / 3)
    assert tr.observe(g, 0) == 0.0                       # 同回合幂等
    g.turn, g.boards[10] = 10, [9]                       # 卖光换 B 级：低于历史最高
    assert tr.observe(g, 0) == 0.0
    g.turn, g.boards[11] = 11, [1, 2, 3]                 # 转型拼满 S 级
    assert tr.observe(g, 0) == pytest.approx(1.0 - 2 / 3)
    assert tr.best_name == "S-comp" and tr.best_tier == "S"
    st = tr.stats()
    assert st["comp_target_turns"] == 4
    assert st["comp_target_s_turns"] == 3
    assert st["comp_target_reward_sum"] == pytest.approx(1.0)


def test_floor_sets_watermark_without_paying(comps_path):
    tr = CompTargetTracker(coef=1.0, min_turn=8, cards_path=comps_path, lookup=_lookup)
    tr.set_floor(9)
    g = FakeGame()
    g.turn, g.boards[9] = 9, [1, 2, 3]
    assert tr.observe(g, 0) == 0.0                       # 注入帧：只设水位
    g.turn, g.boards[10] = 10, [1, 2]
    assert tr.observe(g, 0) == 0.0                       # 低于水位 → 不给分


def test_queue_flush_and_off(comps_path):
    tr = CompTargetTracker(coef=1.0, min_turn=8, cards_path=comps_path, lookup=_lookup)
    recs = [dict(game_id=7, reward=1.0), dict(game_id=8, reward=1.0)]
    assert tr.queue(0, 7, 0.25) is True
    assert tr.queue(5, 7, 0.5) is True       # queue 不校验上界，flush 丢弃
    assert tr.queue(1, 99, 0.5) is True      # 串局
    assert tr.flush(recs) == 1
    assert recs[0]["reward"] == pytest.approx(1.25)
    assert tr.dropped == 2

    off = CompTargetTracker(coef=0.0, cards_path=comps_path)
    assert off.enabled is False and off.stats() == {}
    assert off.observe(FakeGame(), 0) == 0.0 and off.queue(0, 1, 1.0) is False


def test_aggregate(comps_path):
    tr = CompTargetTracker(coef=1.0, min_turn=8, cards_path=comps_path, lookup=_lookup)
    st = tr.stats()
    agg = aggregate_comp_target([st, st])
    assert agg["comp_target_turns"] == 0            # 没有 observe 过
    assert aggregate_comp_target([{"x": 1}]) == {}
    assert aggregate_comp_target([]) == {}
