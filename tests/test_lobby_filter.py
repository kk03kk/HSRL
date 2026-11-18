"""Lobby 种族过滤（``hsbg.az.train.lobby_filter``）的测试。

用户 2026-11-17 的要求："把机械和龙 ban 了，让模型先学会玩别的阵容"。
引擎只支持"强制加入"（``lobby_types``），没有排除列表；本模块用**确定性种子过滤**
实现 ban，且**不改引擎**（不动 source_hash / 卡表 / 模型兼容锚点）。

需要 `hsbg_engine`（本地已安装 37.0.2.256589）。
"""

from __future__ import annotations

import time

import pytest

from hsbg.az.train import lobby_filter as LF

engine = pytest.importorskip("hsbg_engine")


# ────────────────────────── 解析与位解码 ──────────────────────────
def test_parse_race_list_normalises_aliases():
    assert LF.parse_race_list("MECH,DRAGON") == ("MECHANICAL", "DRAGON")
    assert LF.parse_race_list("mechanical; dragon") == ("MECHANICAL", "DRAGON")
    assert LF.parse_race_list(["Mech", "DRAGON", "MECHANICAL"]) == ("MECHANICAL", "DRAGON")
    assert LF.parse_race_list("") == ()
    assert LF.parse_race_list(None) == ()
    assert LF.parse_race_list("ALL") == ()          # 不是 lobby 类型


def test_parse_race_list_rejects_unknown():
    with pytest.raises(ValueError, match="未知种族名"):
        LF.parse_race_list("MECH,DRAGONSLAYER")


def test_race_order_matches_obs_std():
    from hsbg.env.obs_std import RACE_NAMES
    assert LF.RACE_ORDER == tuple(RACE_NAMES)


def test_bits_match_state_json_bitmask():
    """观测 global 的 11 个种族位必须与 ``state_json()["lobby_races"]`` 的 bitmask 一致。"""
    for seed in (930000, 930001, 930002, 424242):
        g = engine.Game(seed=seed)
        f, _c = g.observe(0)
        import json
        mask = int(json.loads(g.state_json())["lobby_races"])
        assert LF.race_bits_from_features(f) == mask, seed


def test_lobby_has_five_races_by_default():
    g = engine.Game(seed=777)
    f, _c = g.observe(0)
    names = LF.lobby_races_from_features(f)
    assert len(names) == 5
    assert "NAGA" not in names        # 引擎不把 NAGA 放进抽签池（只有双种族卡会带上）


def test_is_excluded_blank_list_is_noop():
    g = engine.Game(seed=777)
    f, _c = g.observe(0)
    assert LF.is_excluded(f, ()) is False


# ────────────────────────── 确定性种子过滤 ──────────────────────────
def test_derive_seed_is_deterministic_and_distinct():
    a = [LF.derive_seed(930000, k) for k in range(64)]
    b = [LF.derive_seed(930000, k) for k in range(64)]
    assert a == b
    assert len(set(a)) == len(a)
    assert all(1 <= x < (1 << 63) for x in a)
    assert a[0] != LF.derive_seed(930001, 0)


def test_filtered_game_really_excludes_races():
    exclude = ("MECHANICAL", "DRAGON")
    stats = LF.LobbyFilterStats()
    for seed in range(930000, 930020):
        game, used = LF.make_filtered_game(engine, seed, exclude, learner_seat=0,
                                           stats=stats)
        f, _c = game.observe(0)
        names = LF.lobby_races_from_features(f)
        assert len(names) == 5
        assert "MECHANICAL" not in names and "DRAGON" not in names, (seed, used, names)
    # 接受率理论值 = C(8,5)/C(10,5) = 0.2222
    assert stats.applied == 20
    assert 0.10 <= stats.skipped / float(stats.attempt) <= 0.95


def test_filtered_game_is_deterministic_across_calls():
    exclude = ("MECHANICAL", "DRAGON")
    a, seed_a = LF.make_filtered_game(engine, 930123, exclude)
    b, seed_b = LF.make_filtered_game(engine, 930123, exclude)
    assert seed_a == seed_b
    assert LF.lobby_races_from_features(a.observe(0)[0]) == \
        LF.lobby_races_from_features(b.observe(0)[0])


def test_no_exclusion_keeps_the_original_seed():
    game, used = LF.make_filtered_game(engine, 930000, ())
    assert used == 930000
    import json
    assert int(json.loads(game.state_json())["seed"]) == 930000


def test_too_many_banned_races_fails_loudly():
    """禁到抽不满 5 个（这里禁 6 个）时必须报错，而不是静默放行。"""
    exclude = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC")
    with pytest.raises(RuntimeError, match="lobby 过滤失败"):
        LF.make_filtered_game(engine, 930000, exclude, max_tries=8)


def test_filter_cost_is_negligible():
    """建局（不跑模拟）很便宜：200 个候选种子的过滤总耗时应 < 2 s。"""
    t0 = time.time()
    for seed in range(930000, 930050):
        LF.make_filtered_game(engine, seed, ("MECHANICAL", "DRAGON"))
    dt = time.time() - t0
    assert dt < 2.0, f"过滤 50 局用了 {dt:.2f}s"


def test_partial_practice_uses_the_same_game_as_its_assigned_condition():
    exclude = ("MECHANICAL", "DRAGON")
    assigned = set()
    for seed in range(43600, 43680):
        selected = LF.derive_seed(seed, 719) < 0.25 * (1 << 63)
        assigned.add(selected)
        mixed, actual = LF.make_filtered_game(engine, seed, exclude, probability=0.25)
        reference, expected = LF.make_filtered_game(engine, seed, exclude if selected else ())
        assert actual == expected and mixed.state_json() == reference.state_json()
        assert LF.make_filtered_game(engine, seed, exclude, probability=0)[1] == seed
    assert assigned == {False, True}
