import json
import numpy as np
import hsbg_engine as E
from hsbg.az.train.ppo import PPOGreedyPolicy
from hsbg.az.train.shadow_economy import ShadowEconomyPolicy,gold_context
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.economic_tasks import debug
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture
from workflows.build_payoff_dataset import gate


def test_shadow_queries_preserve_native_actions_state_and_guard():
    def base(features,cards,mask,seat):
        score=np.arange(245,dtype=np.float32)%17+1
        score*=mask;return score/score.sum(),.25
    state=fixture(natural_source(),'complete_gold')[0]
    plain=E.Game.from_json(json.dumps(state));shadow=E.Game.from_json(json.dumps(state))
    original=PPOGreedyPolicy(base);queried=ShadowEconomyPolicy(base);queries=0
    for _ in range(105):
        before=shadow.state_json();a=original.decide(plain,0);b=queried.decide(shadow,0)
        assert shadow.state_json()==before
        assert a.action==b.action and np.array_equal(a.pi,b.pi)
        assert a.info['forced']==b.info['forced']
        if b.info['shadow_teacher_action'] is not None:
            assert queried.evaluator.last_mask[b.info['shadow_teacher_action']];queries+=1
        plain.step(0,a.action);shadow.step(0,b.action)
        assert plain.state_json()==shadow.state_json()
        if a.action==0:break
    assert queries>1
    assert original._turn_counts==queried.policy._turn_counts
    assert original._freeze_counts==queried.policy._freeze_counts
    assert not any(shadow.diagnostic_counts())


def test_shadow_gate_matches_dataset_and_health_sequence_abstains():
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    f,c=M.observe(game,0,M.SCHEMA);f=np.frombuffer(f,dtype='<f4');c=np.frombuffer(c,dtype='<i4')
    assert gold_context(f,c)==gate(f,c)==True
    debug(game,'clear_zone',zone='hand')
    debug(game,'add_minion',zone='hand',card='BG26_525_G')
    # Create room without removing any of the four gate components.
    state=json.loads(game.state_json());state['players'][0]['board']=state['players'][0]['board'][-4:]
    game=E.Game.from_json(json.dumps(state));game.step(0,46)
    def base(features,cards,mask,seat):
        prior=np.asarray(mask,dtype=np.float32);return prior/prior.sum(),0.
    policy=ShadowEconomyPolicy(base);decision=policy.decide(game,0)
    assert decision.info['shadow_gold_context']
    assert decision.info['shadow_teacher_action'] is None
    assert decision.info['shadow_teacher_abstention']=='health-risk pending sequence outside teacher domain'
