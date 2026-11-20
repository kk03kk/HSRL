"""Public-input and native deadline regressions for the isolated player150 release."""
import json

import numpy as np
import pytest
import torch

from hsbg.az.train.move_guard import MoveGuard, encode_move
from hsbg.az.train import player_observation as public
from hsbg.env.action_space_std import A_PLAY


def test_third_reposition_and_play_placement():
    guard = MoveGuard(cap=3, repeat_limit=0, hard_cap=True)
    base = np.ones(245, dtype=bool)
    key = dict(game_id=42, seat=0, turn=8)
    for src in range(3):
        guard.note_action(A_PLAY + src, **key)
        moves, _ = guard.guarded_mask(base, **key)
        assert moves[encode_move(src, src+1)]
        guard.note_action(encode_move(src, src+1), **key)
    after, _ = guard.guarded_mask(base, **key)
    assert not after[196:].any() and after[A_PLAY]
    for other in (dict(key, seat=1), dict(key, turn=9)):
        assert guard.guarded_mask(base, **other)[0][197]


def test_public_projection_preserves_start_and_can_learn():
    sample = dict(remaining=1, growth=[0]*10, health_prices=[False], last_seen_turn=None,
                  faces=[[19, "card", "Reward in 4 Gold"]])
    a = public.encode(sample)
    b = public.encode(dict(sample, remaining=0, faces=[[19, "card", "Reward in 3 Gold"]]))
    assert a[0] > b[0] and not np.array_equal(a[40:], b[40:])
    f = torch.cat((torch.ones(1, 3227), torch.from_numpy(a)[None]), 1)
    model = public.PublicInputAdapter()
    projected = model(f)
    assert torch.equal(projected, f[:, :3227])
    projected.sum().backward()
    assert model.globals.weight.grad.abs().sum() and model.board.weight.grad.abs().sum()


def test_native_deadline_and_freeze():
    e = pytest.importorskip("hsbg_engine")
    if not hasattr(e.Game, "player_card_faces"):
        pytest.skip("requires player150 native build")
    from workflows.mirror_hard_gates import _drive_to_turn
    cfg = json.dumps(dict(player_view_150=True, freeze_presses_per_turn=2))
    choices = set()
    for seed in range(12):
        g = _drive_to_turn(e.Game(seed=seed + 70010, config_json=cfg), 1)
        assert json.loads(g.player_card_faces(0))["remaining"] == 150
        if not g.legal_mask(0)[2]:
            continue
        g.step(0, 2)
        assert g.legal_mask(0)[2]
        g.step(0, 2)
        assert not g.legal_mask(0)[2]
        s = json.loads(g.state_json())
        player = s["players"][0]
        player["actions_this_turn"] = 149
        items = [slot["item"] for slot in player["shop"][:3]]
        if len(items) != 3:
            continue
        g = e.Game.from_json(json.dumps(s))
        assert json.loads(g.player_card_faces(0))["remaining"] == 1
        s["players"][0]["actions_this_turn"] = 150
        s["players"][0]["pending"] = [{"Discover": {"source": 0, "options": items}}]
        g = e.Game.from_json(json.dumps(s))
        assert sum(g.legal_mask(0)) == 1 and g.legal_mask(0)[0]
        assert json.loads(g.player_card_faces(0))["remaining"] == 0
        copy = g.copy()
        g.step(0, 0)
        copy.step(0, 0)
        assert copy.player_json(0) == g.player_json(0)
        after = json.loads(g.player_json(0))
        assert after["actions_this_turn"] == 150 and not after["pending"]
        assert len(after["hand"]) == len(player["hand"]) + 1
        def card(item):
            key, value = next(iter(item.items()))
            return value["card"] if isinstance(value, dict) else value
        award = card(after["hand"][-1]["item"])
        available = [card(item) for item in items]
        assert award in available
        choices.add(available.index(award))
    assert len(choices) >= 2
