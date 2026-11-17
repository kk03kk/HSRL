import copy,json
import pytest
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture,run_operator,metrics,VARIANTS
from hsbg.az.train.research_economy import Operator


def test_public_operator_rejects_full_state():
    with pytest.raises(ValueError,match='restricted public'):
        Operator().act({'rng':123,'players':[]})


@pytest.mark.parametrize('variant',VARIANTS)
def test_native_whole_turn_is_legal_preserves_input_and_replay(variant):
    import hsbg_engine as E
    source=natural_source();original=copy.deepcopy(source)
    state,edits=fixture(source,variant)
    assert source==original and edits
    result=run_operator(state);measure=metrics(result)
    assert result['initial']==state and measure['actions']<=128
    game=E.Game.from_json(json.dumps(state))
    for step in result['steps']:
        assert json.loads(game.state_json())==step['before']
        assert bytes(game.legal_mask(0))[step['action']]
        game.step(0,step['action'])
        assert json.loads(game.state_json())==step['after']
    assert json.loads(game.state_json())==result['final']
    assert result['end_intent']['action']==0 and not result['end_intent']['executed']
    assert not any(game.diagnostic_counts())
