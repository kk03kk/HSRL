import json
import pytest

from hsbg.az.train.economic_tasks import fixture,solve,own,wealth_remaining,score


def natural_source(seed=11800000):
    e=pytest.importorskip('hsbg_engine')
    game=e.Game(seed=seed)
    while game.turn<8 and not game.is_over():
        for seat in range(game.num_players):
            if game.needs_action(seat):game.bot_play(seat)
    while json.loads(game.player_json(0))['pending']:
        game.step(0,game.bot_action(0))
    return json.loads(game.state_json())


@pytest.mark.parametrize('kind',['order_privateer','buy_privateer','redundant_privateer'])
def test_complete_economic_plan_realizes_verified_profit(kind):
    source=natural_source();before=json.dumps(source,sort_keys=True)
    game,edits=fixture(source,kind=kind,bounties=2,gold=5)
    result=solve(game)
    start=game.state_json()
    assert edits and game.state_json()==start
    if kind=='order_privateer':assert 46<=result['plan'][0]<126
    if kind=='buy_privateer':assert 8<=result['plan'][0]<16
    if kind=='redundant_privateer':assert not any(8<=a<16 for a in result['optimal_actions'])
    for action in result['plan']:
        assert bytes(game.legal_mask(0))[action]
        game.step(0,action)
    assert wealth_remaining(game)==0
    assert score(game,set(result['original_eids']))==result['value']
    assert result['value']>=11 if kind=='buy_privateer' else result['value']>=13
    assert json.dumps(source,sort_keys=True)==before


def test_dataset_label_certificate_and_lobby_gate():
    from workflows.build_economic_dataset import build_task
    source=natural_source()
    result=build_task(({'state':source,'seat':0,'seed':11800000,'turn':8},'buy_privateer',3,0))
    assert result['accepted'],result
    assert result['rows'] and all(row[0].shape==(3282,) for row in result['rows'])
    assert all(row[1].shape==(124,) for row in result['rows'])
    source['lobby_races'] &= ~(1 << 6)
    with pytest.raises(ValueError,match='Pirate lobby'):
        fixture(source,kind='buy_privateer')
