import json
import numpy as np
import pytest
import torch
from hsbg.az.train.cycle_tasks import KINDS, make_task, solve, achieved, own
from hsbg.az.train.model import build_net
from workflows.learn_cycle_process import configure_parameters


@pytest.fixture(scope='module')
def source():
    engine = pytest.importorskip('hsbg_engine')
    seed = next(s for s in range(11807001, 11807100)
                if (int(json.loads(engine.Game(seed=s).state_json())['lobby_races']) & 40) == 40)
    game = engine.Game(seed=seed)
    while game.turn < 8 and not game.is_over():
        for seat in range(8):
            if game.needs_action(seat):
                game.bot_play(seat)
    return {'seed': seed, 'seat': 0, 'turn': 8, 'state': json.loads(game.state_json())}


@pytest.mark.parametrize('kind', KINDS)
def test_native_goal_is_executable_and_source_unchanged(source, kind):
    before = json.dumps(source, sort_keys=True)
    game, goal, edits = make_task(source, kind, 1)
    assert edits and not achieved(game, goal)
    actions = []
    while not achieved(game, goal):
        plan = solve(game, goal, max_depth=goal['max_depth'] - len(actions))
        assert plan['actions']
        assert all(bytes(game.legal_mask(0))[a] for a in plan['actions'])
        action = plan['actions'][0]
        actions.append(action)
        game.step(0, action)
        assert not any(game.diagnostic_counts())
    assert set(goal['protected']) <= {m['eid'] for m in own(game)['board']}
    assert json.dumps(source, sort_keys=True) == before
    if kind == 'activate_draw':
        # Native activation is once per turn; do not teach an infinite loop.
        from hsbg.az.train.cycle_tasks import card_ids, KELP
        slot = next(i for i, m in enumerate(own(game)['board']) if m['card'] == card_ids()[KELP])
        assert not bytes(game.legal_mask(0))[23 + slot]


def test_head_only_update_preserves_representation_and_rank():
    torch.set_num_threads(1)
    model = build_net(arch='entity', d_model=32, n_layers=1, n_heads=4,
                      dropout=0, activate_head_split=True).eval()
    selected = configure_parameters(model, 'heads')
    assert 'q_activate_delta.weight' in selected and 'trunk.layers.0.linear1.weight' not in selected
    f = torch.randn(3, 3227)
    c = torch.randint(model.n_cards, (3, 124))
    before = tuple(x.detach().clone() for x in model(f, c))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    loss = -model(f, c)[0][:, 23].mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = model(f, c)
    assert not torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
