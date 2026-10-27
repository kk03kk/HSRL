"""Offline driver budget: visible to policy, causal on execution, no live client."""
import copy
import json

import numpy as np
import pytest
import torch
import taverngym as tg
from taverngym.rl.config import Config, TRAIN_ACTION_BUDGET, identity
from taverngym.rl.collect import observation
from taverngym.rl.model import Policy, pack


def recruit():
    cfg = Config()
    g = tg.Game(541, json.dumps(cfg.rules))
    for seat in range(8):
        g.step(seat, json.loads(g.view(seat))['actions']['CHOOSE'])
    for _ in range(2):
        g.debug(json.dumps({'cmd':'add_minion','p':0,'zone':'board','card':'BG25_008'}))
    return cfg, g


def batch_with(g):
    env = tg.BatchEnv(1, controlled=range(8), config=Config().rules)
    env.auto_reset = False
    state = json.loads(env.save(0)); state['game'] = json.loads(g.state_json())
    env.load(0, json.dumps(state))
    return env


def test_budget_two_hundred_ends_repeated_moves_and_rejection_is_atomic():
    _, g = recruit()
    state = json.loads(g.state_json())
    state['players'][0].update(actions_this_turn=198, moves_this_turn=30)
    g = tg.Game.from_json(json.dumps(state))
    move = json.loads(g.view(0))['actions']['MOVE']
    for action in (move+1, move+7):
        assert g.legal_mask(0)[action]
        g.step(0, action)
    mask = np.frombuffer(g.legal_mask(0), dtype=np.uint8)
    assert mask.sum() == 1 and mask[0]
    before = g.state_json()
    with pytest.raises(ValueError, match='action limit'):
        g.step(0, move+1)
    assert g.state_json() == before
    for seat in range(8):
        while json.loads(g.state_json())['players'][seat]['pending']:
            g.step(seat, json.loads(g.view(seat))['actions']['CHOOSE'])
        g.step(seat, 0)
    after = json.loads(g.state_json())
    assert after['turn'] == state['turn']+1
    assert after['players'][0]['actions_this_turn'] == 0


def test_pending_discover_can_finish_after_budget_without_reopening_operations():
    _, g = recruit()
    g.debug(json.dumps({'cmd':'add_minion','p':0,'zone':'hand','card':'BG25_008'}))
    state = json.loads(g.state_json())
    item = state['players'][0]['hand'].pop()['item']
    state['players'][0]['pending'] = [{'Discover':{'source':item['Minion']['card'],'options':[item]}}]
    state['players'][0]['actions_this_turn'] = 200
    g = tg.Game.from_json(json.dumps(state))
    choose = json.loads(g.view(0))['actions']['CHOOSE']
    assert g.legal_mask(0)[choose]
    g.step(0, choose)
    mask = np.frombuffer(g.legal_mask(0), dtype=np.uint8)
    assert mask.sum() == 1 and mask[0]


@pytest.mark.parametrize('kind', ['legacy_gru','compact_gru','compact_ff'])
def test_existing_policy_input_already_contains_consumed_and_total_budget(kind):
    cfg, g = recruit()
    env = batch_with(g)
    original = observation(env.observe(), env.training_extras(), 0)
    used = original['extras'][0][2][1]
    assert original['extras'][0][2][2] == TRAIN_ACTION_BUDGET == 200
    slot = json.loads(env.save(0));slot['game']['players'][0]['actions_this_turn'] += 1
    env.load(0, json.dumps(slot))
    changed = observation(env.observe(), env.training_extras(), 0)
    assert changed['extras'][0][2][1] == used+1
    torch.set_num_threads(1);torch.manual_seed(71)
    cfg.policy_kind=kind;cfg.dim=16;cfg.hidden=16;cfg.layers=1;cfg.heads=2
    model=Policy(cfg).eval()
    with torch.no_grad():
        a,_=model.encode(pack([original],'cpu'))
        b,_=model.encode(pack([changed],'cpu'))
    assert not torch.equal(a,b)


def test_budget_changes_experiment_identity_without_changing_native():
    cfg = Config(); other = copy.deepcopy(cfg)
    other.rules['turn_action_caps'] = []
    assert identity(cfg)['digest'] != identity(other)['digest']
    assert tg.build_is_current()
