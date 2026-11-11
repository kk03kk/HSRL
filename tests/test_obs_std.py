"""标准观测（``hsbg.env.obs_std``）：3227 floats + 124 cards 契约回归。"""

from __future__ import annotations

import numpy as np
import pytest

engine = pytest.importorskip("hsbg_engine")

from hsbg.env import obs_std as o  # noqa: E402


def test_layout_matches_engine():
    ours = tuple(o.layout())
    theirs = tuple(tuple(x) for x in engine.layout())
    assert ours == theirs
    assert o.OBS_FLOATS == engine.OBS_FLOATS == 3227
    assert o.OBS_CARDS == engine.OBS_CARDS == 124
    assert engine.N_ACTIONS == 245


def test_decode_blocks_shapes():
    features = np.zeros(o.OBS_FLOATS, dtype=np.float32)
    cards = np.zeros(o.OBS_CARDS, dtype=np.int32)
    blocks = o.decode_blocks(features, cards)
    assert blocks["global"].shape == (o.GLOBAL_F,)
    assert blocks["board_minions"].shape == (o.N_BOARD, o.MINION_F)
    assert blocks["hand_items"].shape == (o.N_HAND, o.ITEM_F)
    assert blocks["shop_items"].shape == (o.N_SHOP, o.ITEM_F)
    assert blocks["opponent_boards"].shape == (o.N_OPP, o.N_BOARD, o.MINION_F)
    with pytest.raises(ValueError):
        o.decode_blocks(np.zeros(3, dtype=np.float32), cards)


def test_obs_std_declares_auto_tavern_contract():
    names = [name for name, _ in o.layout()]
    assert len(names) == 32
    assert names[0] == "MINION_F"
    assert "N_BOARD" in names and "N_HAND" in names and "N_SHOP" in names


def test_global_fields_decode_to_engine_values():
    """Field-level guard: global names/scales must match the Rust observation.

    The old test only checked block widths, so a same-size field reorder or a
    changed divisor would silently corrupt both training and live inputs.
    """
    import json

    game = engine.Game(seed=12345)
    while game.turn < 2 and not game.is_over():
        for seat in range(8):
            if game.needs_action(seat):
                game.bot_play(seat)
    player = json.loads(game.player_json(0))
    player.update(health=37, armor=13, gold=9, max_gold_bonus=2,
                  gold_next_turn=3, tier=4, free_refreshes=2,
                  dark_gift_uses=1, chronum_saved=3, alive=True)
    player["stats"].update(won_last=True, lost_last=False, spell_atk=4,
                           spell_hp=6, gem_atk=2, gem_hp=8,
                           tavern_atk=7, tavern_hp=5)
    game.set_player_from_json(0, json.dumps(player))
    raw_f, raw_c = game.observe(0)
    features = np.frombuffer(bytes(raw_f), dtype=np.float32).copy()
    cards = np.frombuffer(bytes(raw_c), dtype=np.int32).copy()
    global_block = o.decode_blocks(features, cards)["global"]

    def raw(name):
        idx = o.GLOBAL_FIELD_INDEX[name]
        scale = o.GLOBAL_FIELD_SCALE.get(name)
        return float(global_block[idx]) * (scale or 1.0)

    health = [int(h) for h in game.health()]
    expected = {
        "health": 37, "armor": 13, "gold": 9, "max_gold_bonus": 2,
        "gold_next_turn": 3, "tier": 4, "free_refreshes": 2,
        "dark_gift_uses": 1, "chronum": 3, "won_last": 1, "lost_last": 0,
        "spell_atk": 4, "spell_hp": 6, "gem_atk": 2, "gem_hp": 8,
        "tavern_atk": 7, "tavern_hp": 5, "deity_present": 1,
        "deity_atk": int(player["deity"]["atk"]),
        "deity_hp": int(player["deity"]["hp"]),
        "board_len": len(player.get("board") or []),
        "hand_len": len(player.get("hand") or []),
        "trinket_count": len(player.get("trinkets") or []),
        "turn": int(game.turn), "alive": sum(1 for h in health if h > 0),
        "me_alive": 1,
    }
    for name, value in expected.items():
        assert raw(name) == pytest.approx(float(value), abs=1e-4), name
