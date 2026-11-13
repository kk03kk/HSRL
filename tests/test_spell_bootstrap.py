"""法术自举补贴（spell bootstrap）的回归测试。

背景（RUNS §5.54/§5.57/§5.58）：champion 整局**买 0 次法术**，而每回合商店里
都有一个合法的法术槽（``A_BUY+7``），真人每回合买 0.46–1.51 个。终局名次是稀疏
信号 ⇒ 这类动作的梯度≈0，策略永远不探索。``spell_bootstrap_coef`` 用**动作条件**
小额奖励把这类动作点亮，配额上限（每局 K 次）防止刷分。

这组测试钉住四件事：
1. 动作 id 与引擎布局一致（``A_BUY + N_SHOP - 1 == 15``）；
2. ``coef=0``（默认）时完全不动 record（旧路径逐位一致）；
3. 只补贴「买法术」动作，且每局最多 K 次；
4. coef=0 时 ``stats()`` 不新增任何键。
"""
from __future__ import annotations

import pytest

from hsbg.az.train.ppo import CollectConfig
from hsbg.az.train.reward_shaping import SPELL_BUY_ACTION, RewardShaper
from hsbg.env.action_space_std import A_BUY, N_SHOP


def test_spell_buy_action_matches_engine_layout() -> None:
    """法术槽 = 商店最后一个槽：A_BUY + N_SHOP - 1。"""
    assert N_SHOP == 8
    assert A_BUY == 8
    assert SPELL_BUY_ACTION == A_BUY + N_SHOP - 1 == 15


def _records(n: int, game_id: int = 7) -> list:
    return [{"game_id": game_id, "action": -1} for _ in range(n)]


def test_disabled_by_default_touches_nothing() -> None:
    """默认配置（coef=0）：不写 reward、不写 spell_bootstrap、stats() 为空。"""
    cfg = CollectConfig()
    shaper = RewardShaper(cfg, gamma=1.0)
    assert not shaper.spell_enabled
    recs = _records(2)
    assert shaper.attach_spell(recs, 0, 7, SPELL_BUY_ACTION) == 0.0
    assert "reward" not in recs[0]
    assert "spell_bootstrap" not in recs[0]
    assert shaper.stats() == {}


def test_spell_only_mode_does_not_emit_shaping_keys() -> None:
    """只开补贴（shaping_coef=0）：stats 只报补贴字段，不报势函数字段。"""
    cfg = CollectConfig(spell_bootstrap_coef=0.05, spell_bootstrap_per_game=3)
    shaper = RewardShaper(cfg, gamma=1.0)
    assert shaper.spell_enabled and not shaper.enabled
    recs = _records(1)
    assert shaper.attach_spell(recs, 0, 7, SPELL_BUY_ACTION) == pytest.approx(0.05)
    st = shaper.stats()
    assert "shaping_coef" not in st
    assert st["spell_bootstrap_hits"] == 1
    assert st["spell_bootstrap_sum"] == pytest.approx(0.05)


def test_only_spell_buys_get_the_bonus() -> None:
    """非「买法术」动作一律不加钱（买随从、升本、刷新…）。"""
    cfg = CollectConfig(spell_bootstrap_coef=0.05, spell_bootstrap_per_game=3)
    shaper = RewardShaper(cfg, gamma=1.0)
    recs = _records(4)
    for action in (0, 3, 8, 9, SPELL_BUY_ACTION - 1):
        assert shaper.attach_spell(recs, 0, 7, action) == 0.0
    assert "reward" not in recs[0]
    assert shaper.spell_hits == 0


def test_quota_is_per_game() -> None:
    """每局最多 K 次；配额用完后不再加钱，另一局重新计数。"""
    cfg = CollectConfig(spell_bootstrap_coef=0.05, spell_bootstrap_per_game=2)
    shaper = RewardShaper(cfg, gamma=1.0)
    recs = _records(4)
    assert shaper.attach_spell(recs, 0, 7, SPELL_BUY_ACTION) == pytest.approx(0.05)
    assert shaper.attach_spell(recs, 1, 7, SPELL_BUY_ACTION) == pytest.approx(0.05)
    # 配额用完
    assert shaper.attach_spell(recs, 2, 7, SPELL_BUY_ACTION) == 0.0
    assert recs[2].get("reward") is None
    # 另一局重新计数
    other = _records(1, game_id=8)
    assert shaper.attach_spell(other, 0, 8, SPELL_BUY_ACTION) == pytest.approx(0.05)
    assert shaper.spell_hits == 3


def test_game_id_mismatch_is_ignored() -> None:
    """防串局：record 的 game_id 与传入不一致时不写。"""
    cfg = CollectConfig(spell_bootstrap_coef=0.05, spell_bootstrap_per_game=3)
    shaper = RewardShaper(cfg, gamma=1.0)
    recs = _records(1, game_id=7)
    assert shaper.attach_spell(recs, 0, 8, SPELL_BUY_ACTION) == 0.0
    assert "reward" not in recs[0]


def test_invalid_config_rejected() -> None:
    """负系数 / 负数配额在构造期就被拒（不静默跑完一整段）。"""
    with pytest.raises(ValueError):
        CollectConfig(spell_bootstrap_coef=-0.1)
    with pytest.raises(ValueError):
        CollectConfig(spell_bootstrap_per_game=-1)
