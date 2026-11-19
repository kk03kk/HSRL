"""`forced_spell` 在 **PPO 收尾路径** 上的行为测试（不依赖引擎/网络/GPU）。

为什么这条最重要：接线测试只能证明"线连着"，而真正会静默破坏训练的是
**`_finish()` 里那几行**——动作有没有被改、`logprob` 是不是 0、`forced` 标记对不对、
record 里有没有多出键。2026-11-14 已经吃过一次"配了但没生效"（RUNS §5.79），
所以这里直接把 `PPOSelfPlayPolicy._finish` 拉出来单测。

做法：`object.__new__` 造一个裸实例，只塞 `_finish` 用到的属性（不跑 `__init__`，
因此不需要 net / 引擎 / torch 前向）。
"""
from __future__ import annotations

import numpy as np
import pytest

from hsbg.az.train.forced_spell import ForcedSpellTracker, legal_spell_buys
from hsbg.env.obs_std import ITEM_F, ITEM_PRESENT, ITEM_IS_MINION, OBS_FLOATS, SLICE_SHOP, SPELL_PRESENT
from hsbg.env.action_space_std import A_BUY, N_SHOP
from hsbg.az.train.ppo import CollectConfig, PPOSelfPlayPolicy

N_ACTIONS = 245
SPELL_BUY_ACTION = A_BUY + 3


class _Game:
    """最小 game 替身：`_finish` 只读 `.turn`。"""

    def __init__(self, turn: int) -> None:
        self.turn = int(turn)


def _mask(*legal: int) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    for a in legal:
        m[int(a)] = True
    return m


def _policy(*, per_game: int = 0, turn_min: int = 6, prob: float = 1.0,
            teacher: bool = False) -> PPOSelfPlayPolicy:
    p = object.__new__(PPOSelfPlayPolicy)
    p.config = CollectConfig(max_actions_per_turn=0,
                             forced_spell_per_game=per_game,
                             forced_spell_turn_min=turn_min,
                             forced_spell_prob=prob,
                             forced_spell_teacher=teacher)
    p.name = "test"
    p.game_id = 7
    p.records = []
    p.action_hist = {}
    p._turn_counts = {}
    p._guard = None
    p._abias = None
    p._mix = None
    p._forced = None
    p.norm_mean = None
    p.norm_std = None
    p.forced_actions = 0
    p._rng = np.random.default_rng(0)
    p._spell = (ForcedSpellTracker(per_game, turn_min=turn_min)
                if per_game > 0 else None)
    p._spell_prob = float(prob)
    p._spell_teacher = bool(teacher)
    return p


def _finish(policy: PPOSelfPlayPolicy, *, turn: int = 8, action: int = 3,
            mask: np.ndarray | None = None, logprob: float = -0.5):
    features = np.zeros(OBS_FLOATS, dtype=np.float32)
    shop = features[SLICE_SHOP].reshape(N_SHOP, ITEM_F)
    shop[3, [ITEM_PRESENT, SPELL_PRESENT]] = 1
    shop[7, [ITEM_PRESENT, ITEM_IS_MINION]] = 1
    return policy._finish(
        _Game(turn), 0, features, np.zeros(4, dtype=np.int32),
        _mask(0, 1, 2, SPELL_BUY_ACTION) if mask is None else mask,
        action=int(action), logprob=float(logprob), value=0.25,
        target_pi=None, out_pi=None)


def test_forced_spell_overrides_the_sampled_action():
    """配额内且合法：动作被改成当前法术槽、logprob=0、forced=True（不进策略梯度）。"""
    p = _policy(per_game=3, turn_min=6)
    d = _finish(p, turn=8, action=3)
    assert int(d.action) == SPELL_BUY_ACTION
    assert d.info["logprob"] == 0.0
    assert d.info["forced"] is True
    rec = p.records[-1]
    assert rec["action"] == SPELL_BUY_ACTION
    assert rec["old_logprob"] == 0.0
    assert rec["forced"] is True
    assert rec["forced_spell"]["force"] is True


def test_quota_exhaustion_returns_control_to_the_policy():
    """每局 3 次：第 4 次不再强制，动作与 logprob 保持采样值（= 正常 on-policy 样本）。"""
    p = _policy(per_game=3, turn_min=0)
    for _ in range(3):
        assert int(_finish(p, turn=8, action=3).action) == SPELL_BUY_ACTION
    d = _finish(p, turn=8, action=3, logprob=-0.25)
    assert int(d.action) == 3
    assert d.info["logprob"] == pytest.approx(-0.25)
    assert d.info["forced"] is False
    assert p.records[-1]["forced_spell"]["force"] is False


def test_before_turn_min_nothing_is_forced():
    p = _policy(per_game=3, turn_min=6)
    d = _finish(p, turn=5, action=3)
    assert int(d.action) == 3
    assert d.info["forced"] is False


def test_illegal_spell_slot_is_never_forced():
    """掩码里法术槽不合法 ⇒ 不强制（引擎口径是唯一权威），并记 blocked。"""
    p = _policy(per_game=3, turn_min=0)
    d = _finish(p, turn=8, action=3, mask=_mask(0, 1, 2))
    assert int(d.action) == 3
    assert d.info["forced"] is False
    assert p._spell.stats()["blocked"] == 1
    assert p._spell.stats()["forced_actions"] == 0


def test_disabled_adds_no_record_key_and_no_meta():
    """per_game=0：record 集合、动作、logprob 全部与旧路径一致（逐位等价）。"""
    p = _policy(per_game=0)
    d = _finish(p, turn=8, action=3, logprob=-0.5)
    assert int(d.action) == 3
    assert d.info["logprob"] == pytest.approx(-0.5)
    assert d.info["forced"] is False
    assert "forced_spell" not in p.records[-1]
    assert p.forced_spell_meta() == {}
    assert p.forced_spell_game_meta(7, 0) == {}


def test_forced_action_is_not_counted_as_a_policy_choice():
    """被强制的样本不能算进 "策略自己按了" —— 否则"学会没学会"就无法判断。"""
    p = _policy(per_game=1, turn_min=0)
    _finish(p, turn=8, action=3)                      # forced -> 不计
    assert p._spell.stats()["policy_spell_actions"] == 0
    _finish(p, turn=8, action=SPELL_BUY_ACTION)       # 配额用完后策略自己按 -> 计
    assert p._spell.stats()["policy_spell_actions"] == 1


def test_meta_reports_forced_and_legal_counts():
    p = _policy(per_game=2, turn_min=0)
    _finish(p, turn=8, action=3)
    _finish(p, turn=8, action=3)
    _finish(p, turn=8, action=3)
    stats = p.forced_spell_meta()
    assert stats["forced_actions"] == 2
    assert stats["legal_decisions"] == 3
    game = p.forced_spell_game_meta(7, 0)
    assert game["forced"] == 2
    assert game["legal_decisions"] == 3
    # 别的局/座位不受影响（lock-step 隔离）
    assert p.forced_spell_game_meta(999, 0) == {}


def test_turn_cap_guard_beats_forced_spell():
    """回合动作上限触发时 mask 只剩 EndTurn，强制买法术必须让路（否则就是非法动作）。"""
    p = _policy(per_game=3, turn_min=0)
    p.config.max_actions_per_turn = 1
    _finish(p, turn=8, action=0)                 # 第 1 次：还在额度内（可能被强制）
    before = p._spell.stats()["forced_actions"]
    d = _finish(p, turn=8, action=0)             # 第 2 次 -> forced_turn 接管
    assert int(d.action) == 0
    assert d.info["forced"] is True              # 是"回合上限"这个守卫在强制
    assert "forced_spell" not in p.records[-1]   # tracker 根本没被咨询
    assert p._spell.stats()["forced_actions"] == before

def test_minion_in_eighth_slot_is_neither_forced_nor_counted():
    p = _policy(per_game=1, turn_min=0)
    d = _finish(p, action=15, mask=_mask(0,15))
    assert d.action==15 and not d.info["forced"]
    assert p._spell.stats()["policy_spell_actions"]==0


def test_skipped_forcing_writes_real_teacher_action():
    p = _policy(per_game=2, turn_min=0, prob=0, teacher=True)
    _finish(p, action=1)
    assert p.records[-1]["teacher_action"] == SPELL_BUY_ACTION


def test_native_observation_matches_actual_spell_slots():
    import json
    engine = pytest.importorskip("hsbg_engine")
    game = engine.Game(31407)
    for _ in range(60):
        if json.loads(game.player_json(0))["shop"] and game.needs_action(0):
            break
        for seat in range(8):
            if game.needs_action(seat):
                game.step(seat,next(i for i,ok in enumerate(game.legal_mask(seat)) if ok))
    raw = game.observe(0)[0]
    f = np.frombuffer(raw, dtype="<f4")
    mask = np.frombuffer(game.legal_mask(0),dtype=np.uint8)
    items = json.loads(game.player_json(0))["shop"]
    expected = tuple(A_BUY+i for i,s in enumerate(items) if "Spell" in s["item"] and mask[A_BUY+i])
    assert expected and all(a!=15 for a in expected)
    assert legal_spell_buys(f,mask)==expected
