"""强制升本课程的单测：计划解析 / 状态机 / PPO 接线（forced 语义）。

不依赖引擎的部分全部用假 game 对象；引擎相关的端到端验证在
``tests/test_forced_tier_engine.py``（有引擎才跑）。
"""
from __future__ import annotations

import numpy as np
import pytest

from hsbg.az.train.forced_tier import (
    DEFAULT_FORCED_TIER_CARRY,
    DEFAULT_FORCED_TIER_PLAN,
    DEFAULT_FORCED_TIER_PLAN_SPEC,
    ForcedTierTracker,
    UPGRADE_ACTION,
    aggregate_forced_tier,
    forced_tier_rates,
    parse_decisions,
    parse_forced_tier_plan,
    parse_prob_schedule,
    plan_spec_of,
    prob_at,
    summarize_forced_tier,
)

# ── 计划解析 ─────────────────────────────────────────────────────────────


def test_default_plan_matches_documented_schedule():
    assert parse_forced_tier_plan(DEFAULT_FORCED_TIER_PLAN_SPEC) == DEFAULT_FORCED_TIER_PLAN
    assert DEFAULT_FORCED_TIER_PLAN == {2: 2, 5: 3, 6: 4, 8: 5}
    assert plan_spec_of(DEFAULT_FORCED_TIER_PLAN) == DEFAULT_FORCED_TIER_PLAN_SPEC


@pytest.mark.parametrize("spec", ["", None, "none", "off", "0", False])
def test_empty_spec_means_disabled(spec):
    assert parse_forced_tier_plan(spec) == {}


def test_parse_accepts_dict_list_and_spaces():
    assert parse_forced_tier_plan({"7": "3"}) == {7: 3}
    assert parse_forced_tier_plan([(2, 2), (5, 3)]) == {2: 2, 5: 3}
    assert parse_forced_tier_plan(" 2 : 2 , 5:3 ") == {2: 2, 5: 3}
    assert parse_forced_tier_plan("2:2;5:3") == {2: 2, 5: 3}


@pytest.mark.parametrize("spec", [
    "2", "a:2", "2:b", "0:2", "2:0", "2:8", "2:3,2:4", "2:4,5:3", "-1:2",
])
def test_parse_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_forced_tier_plan(spec)


# ── 概率计划 ────────────────────────────────────────────────────────────


def test_prob_schedule_piecewise_constant():
    sched = parse_prob_schedule("0:1.0,40M:0.75,80M:0.4,120M:0")
    assert [p for _, p in sched] == [1.0, 0.75, 0.4, 0.0]
    assert prob_at(sched, 0) == 1.0
    assert prob_at(sched, 39_999_999) == 1.0
    assert prob_at(sched, 40_000_000) == 0.75
    assert prob_at(sched, 200_000_000) == 0.0


def test_prob_schedule_default_and_errors():
    assert parse_prob_schedule("", default=0.3) == [(0.0, 0.3)]
    assert parse_prob_schedule("50M:0.5")[0] == (0.0, 0.5)  # 自动补 0 起点
    with pytest.raises(ValueError):
        parse_prob_schedule("0:1.5")
    with pytest.raises(ValueError):
        parse_prob_schedule("0:0.5,0:0.7")
    with pytest.raises(ValueError):
        parse_prob_schedule("abc:0.5")


def test_parse_decisions_units():
    assert parse_decisions("40M") == 40e6
    assert parse_decisions("500K") == 500e3
    assert parse_decisions("1B") == 1e9
    assert parse_decisions(1234) == 1234.0
    with pytest.raises(ValueError):
        parse_decisions("abc")


# ── 状态机 ──────────────────────────────────────────────────────────────


def test_on_time_force_and_achievement():
    t = ForcedTierTracker("2:2,5:3", carry=1)
    d = t.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=True)
    assert d.force and d.target == 2 and d.deadline == 2 and not d.deferred
    # 升级后的下一次决策：tier 已变成 2 -> 结算为 on_time
    t.decide(game_id=1, seat=0, turn=2, tier=2, upgrade_legal=False)
    st = t.game_stats()
    assert st["targets"]["2"]["on_time"] == 1
    assert st["targets"]["2"]["forced_actions"] == 1


def test_no_force_outside_deadline_window():
    t = ForcedTierTracker("5:3", carry=1)
    assert not t.decide(game_id=1, seat=0, turn=4, tier=2, upgrade_legal=True).force
    assert t.decide(game_id=1, seat=0, turn=5, tier=2, upgrade_legal=True).force
    assert t.decide(game_id=1, seat=0, turn=6, tier=2, upgrade_legal=True).deferred


def test_self_achieved_before_deadline():
    t = ForcedTierTracker("5:3", carry=1)
    t.decide(game_id=1, seat=0, turn=3, tier=3, upgrade_legal=False)  # 模型自己升到 3
    st = t.game_stats()
    assert st["targets"]["3"]["self"] == 1
    assert not t.decide(game_id=1, seat=0, turn=5, tier=3, upgrade_legal=True).force


def test_blocked_when_upgrade_illegal_and_counted_once_per_turn():
    t = ForcedTierTracker("2:2", carry=0)
    d1 = t.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=False)
    d2 = t.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=False)
    assert d1.blocked and d2.blocked and not d1.force and not d1.teacher
    assert t.blocked == 1  # 同回合只记一次


def test_missed_after_carry_window():
    t = ForcedTierTracker("2:2", carry=0)
    t.decide(game_id=1, seat=0, turn=1, tier=1, upgrade_legal=True)
    t.decide(game_id=1, seat=0, turn=3, tier=1, upgrade_legal=True)
    st = t.game_stats()
    assert st["targets"]["2"]["missed"] == 1
    assert st["targets"]["2"]["on_time"] == 0


def test_probability_gate_writes_teacher_instead_of_force():
    rng = np.random.default_rng(0)
    t = ForcedTierTracker("2:2", carry=0)
    # prob=0 -> 永不强制，改为教师标签
    d = t.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=True,
                 force_prob=0.0, rng=rng)
    assert not d.force and d.teacher
    assert t.teacher_labels == 1 and t.prob_skips == 1
    assert t.forced_actions == 0


def test_probability_one_always_forces():
    rng = np.random.default_rng(0)
    t = ForcedTierTracker("2:2", carry=0)
    for gid in range(20):
        d = t.decide(game_id=gid, seat=0, turn=2, tier=1, upgrade_legal=True,
                     force_prob=1.0, rng=rng)
        assert d.force


def test_state_is_isolated_per_game_and_seat():
    t = ForcedTierTracker("2:2", carry=0)
    assert t.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=True).force
    # 另一局同一回合也应独立判定
    assert t.decide(game_id=2, seat=0, turn=2, tier=1, upgrade_legal=True).force
    # 同一局另一座位
    assert t.decide(game_id=1, seat=1, turn=2, tier=1, upgrade_legal=True).force


def test_reset_folds_stats_without_growth():
    t = ForcedTierTracker("2:2", carry=0)
    for gid in range(5):
        t.decide(game_id=gid, seat=0, turn=2, tier=1, upgrade_legal=True)
        t.decide(game_id=gid, seat=0, turn=2, tier=2, upgrade_legal=False)
        t.reset()
    st = t.stats()
    assert st["targets"]["2"]["on_time"] == 5
    assert t.forced_actions == 5
    assert t._state == {}  # 状态已清空，不会无限增长


def test_summarize_and_rates():
    a = {"plan": "2:2", "carry": 1, "forced_actions": 1, "teacher_labels": 0,
         "blocked": 0, "prob_skips": 0,
         "targets": {"2": {"self": 0, "on_time": 1, "deferred": 0, "missed": 0,
                           "forced_actions": 1, "blocked": 0}}}
    b = {"plan": "2:2", "carry": 1, "forced_actions": 0, "teacher_labels": 0,
         "blocked": 0, "prob_skips": 0,
         "targets": {"2": {"self": 0, "on_time": 0, "deferred": 1, "missed": 0,
                           "forced_actions": 0, "blocked": 0}}}
    agg = summarize_forced_tier([a, b])
    assert agg["forced_actions"] == 1
    assert agg["targets"]["2"]["on_time"] == 1
    assert agg["targets"]["2"]["deferred"] == 1
    rates = forced_tier_rates(agg)
    assert rates["2"]["on_time"] == pytest.approx(0.5)
    assert rates["2"]["deferred"] == pytest.approx(0.5)


def test_aggregate_reads_game_meta_key():
    games = [{"forced_tier_game": {"plan": "2:2", "carry": 0,
                                   "targets": {"2": {"on_time": 1}}}},
             {"rank": 3}]
    agg = aggregate_forced_tier(games)
    assert agg["targets"]["2"]["on_time"] == 1


def test_disabled_tracker_returns_empty():
    t = ForcedTierTracker("")
    assert t.game_stats() == {} and t.stats() == {}
    d = t.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=True)
    assert not d.force and d.target is None
    assert summarize_forced_tier([{}]) == {}


# ── PPO 接线：forced 语义 ───────────────────────────────────────────────


class _FakeGame:
    """最小 game 接口：观测/掩码/回合/tier/状态。"""

    def __init__(self, *, turn: int, tier: int, legal_upgrade: bool = True,
                 game_id: int = 0) -> None:
        from hsbg.az.train.common import N_ACTIONS, OBS_CARDS, OBS_FLOATS
        from hsbg.env.obs_std import GLOBAL_FIELD_INDEX

        self.turn = int(turn)
        self.tier = int(tier)
        self.game_id = int(game_id)
        self._obs = np.zeros(OBS_FLOATS, dtype=np.float32)
        self._obs[GLOBAL_FIELD_INDEX["tier"]] = float(tier) / 6.0
        self._cards = np.zeros(OBS_CARDS, dtype=np.int32)
        self._mask = np.zeros(N_ACTIONS, dtype=np.uint8)
        self._mask[0] = 1  # EndTurn 恒合法
        self._mask[8] = 1  # 一个 buy
        if legal_upgrade:
            self._mask[UPGRADE_ACTION] = 1
        self.stepped = []

    def observe(self, seat):  # noqa: D401 - 与引擎同签名
        return (self._obs.tobytes(), self._cards.tobytes())

    def legal_mask(self, seat):
        return self._mask.tobytes()

    def player_json(self, seat):
        import json
        return json.dumps({"tier": self.tier, "gold": 10})

    def step(self, seat, action):
        self.stepped.append(int(action))


def _make_policy(plan: str, prob: float = 1.0, teacher: bool = False):
    from hsbg.az.train.ppo import CollectConfig, PPOSelfPlayPolicy

    class _Eval:
        name = "fake"

        def act(self, f, c, m, temperature, seed):
            legal = np.flatnonzero(m)
            a = int(legal[-1])  # 故意选非 Upgrade，验证强制覆盖
            return a, float(np.log(1.0 / max(legal.size, 1))), 0.0

    class _Model:
        """假模型：只为满足 ``__init__`` 的 .to()/.eval()（fast_act 路径不做前向）。"""

        def to(self, device):
            return self

        def eval(self):
            return self

    cfg = CollectConfig(opponent="bot", behavior="net", fast_act=True,
                        forced_tier_plan=plan, forced_tier_prob=prob,
                        forced_tier_teacher=teacher, forced_tier_carry=0)
    return PPOSelfPlayPolicy(model=_Model(), config=cfg, eval_fn=_Eval(),
                             move_model=False)


def test_policy_forces_upgrade_with_zero_logprob_and_forced_flag():
    pol = _make_policy("2:2")
    g = _FakeGame(turn=2, tier=1)
    pol.game_id = 1
    dec = pol.decide(g, 0)
    assert int(dec.action) == UPGRADE_ACTION
    rec = pol.records[-1]
    assert rec["forced"] is True
    assert rec["action"] == UPGRADE_ACTION
    assert rec["old_logprob"] == 0.0
    assert rec["forced_tier_target"] == 2
    assert dec.info["forced_tier"]["force"] is True


def test_policy_not_forced_outside_schedule():
    pol = _make_policy("2:2")
    g = _FakeGame(turn=3, tier=1)
    pol.game_id = 1
    pol.decide(g, 0)
    rec = pol.records[-1]
    assert rec["forced"] is False
    assert rec["action"] != UPGRADE_ACTION
    assert "forced_tier_target" not in rec


def test_policy_teacher_label_when_prob_zero():
    pol = _make_policy("2:2", prob=0.0, teacher=True)
    g = _FakeGame(turn=2, tier=1)
    pol.game_id = 1
    pol.decide(g, 0)
    rec = pol.records[-1]
    assert rec["forced"] is False
    assert rec.get("teacher_action") == UPGRADE_ACTION
    assert pol.forced_tier_game_meta()["targets"]["2"]["forced_actions"] == 0


def test_policy_disabled_is_byte_identical():
    pol = _make_policy("")
    g = _FakeGame(turn=2, tier=1)
    pol.game_id = 1
    pol.decide(g, 0)
    rec = pol.records[-1]
    assert set(rec) == {"features", "cards", "mask", "action", "old_logprob",
                        "value", "target_pi", "seat", "turn", "game_id", "forced"}
    assert pol.forced_tier_meta() == {}
    assert pol.forced_tier_game_meta() == {}


def test_forced_records_excluded_from_policy_loss():
    """forced 样本必须被 PPO 排除出策略/辅助损失（只保留价值链）。"""
    from hsbg.az.train.ppo import build_batch_from_records

    pol = _make_policy("2:2")
    g = _FakeGame(turn=2, tier=1)
    pol.game_id = 1
    pol.decide(g, 0)
    rec = dict(pol.records[-1])
    rec["rank"] = 4
    # 再补一条同局普通决策，保证 episode 结构完整
    g2 = _FakeGame(turn=2, tier=2, legal_upgrade=False)
    pol.game_id = 1
    pol.decide(g2, 0)
    rec2 = dict(pol.records[-1])
    rec2["rank"] = 4
    batch = build_batch_from_records([rec, rec2], meta={})
    assert batch.forced.tolist() == [True, False]
    # forced 的奖励/回报链仍在（rank 已知），但策略损失用 valid 掩码排除
    assert np.isfinite(batch.advantages).all()
