import copy,json
import numpy as np
import pytest
from tests.test_economic_tasks import natural_source
from workflows.payoff_bridge import privateer_triple
from hsbg.az.train.economic_tasks import debug,own
from hsbg.az.train import mechanism_obs as M
from workflows.build_payoff_dataset import gate

def test_explicit_triple_fixture_uses_native_purchase_and_preserves_resources():
    import hsbg_engine as E
    # Explicit test setup, not evidence that a natural source reached tier five.
    game=E.Game.from_json(json.dumps(natural_source()))
    debug(game,'set_player',tier=5,gold=10)
    source=json.loads(game.state_json());unchanged=copy.deepcopy(source)
    state,edits,action=privateer_triple(source)
    game=E.Game.from_json(json.dumps(state))
    def enabled():
        f,c=M.observe(game,0,M.SCHEMA)
        return gate(np.frombuffer(f,dtype='<f4'),np.frombuffer(c,dtype='<i4'))
    assert not enabled()
    before=own(game);game.step(0,action)
    assert enabled() and not any(game.diagnostic_counts())
    after=own(game)
    assert after['gold']<before['gold'] and after['health']==before['health'] and after['tier']==5
    assert any(x['id']=='BG33_825_G' for x in json.loads(game.view(0))['me']['hand'])
    assert source==unchanged and edits
    low=copy.deepcopy(source);low['players'][0]['tier']=4
    with pytest.raises(ValueError,match='tier'):privateer_triple(low)
