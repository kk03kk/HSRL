import json
import numpy as np
from hsbg.az.train.research_economy_policy import EconomicOptionPolicy,active
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture,public_view

class Fallback:
    model=None
    def __init__(self):self.calls=0
    def __call__(self,f,c,m,seat):
        self.calls+=1;p=np.zeros(245,np.float32);p[np.flatnonzero(m)[0]]=1
        return p,0.

def test_option_uses_public_gate_and_same_action_budget():
    import hsbg_engine as E
    source=natural_source();state,_=fixture(source,'missing_privateer')
    game=E.Game.from_json(json.dumps(state));fallback=Fallback();policy=EconomicOptionPolicy(fallback)
    assert active(public_view(game))
    for _ in range(4):
        d=policy.decide(game,0)
        assert d.info['research_option_active'] and bytes(game.legal_mask(0))[d.action]
        assert d.action!=0
        game.step(0,d.action)
    assert fallback.calls==0
    assert policy.policy._turn_counts[(0,game.turn)]==4
    policy.policy._turn_counts[(0,game.turn)]=100
    d=policy.decide(game,0)
    assert d.action==0 and d.info['forced_reason']=='action_cap'
    assert set(policy.evaluator.view)=={'me','mask','turn','phase','lobby_races'}
    policy.reset();assert not policy.policy._turn_counts and not policy.evaluator.operator.cycled

def test_option_falls_back_without_components():
    import hsbg_engine as E
    game=E.Game.from_json(json.dumps(natural_source()));fallback=Fallback()
    policy=EconomicOptionPolicy(fallback);d=policy.decide(game,0)
    assert not d.info['research_option_active'] and fallback.calls==1
    assert d.action==0
