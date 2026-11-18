import json
import numpy as np
import pytest
import hsbg_engine as E
from hsbg.az.train.first_action_probe import FirstActionPolicy
from hsbg.az.train.ppo import PPOGreedyPolicy
from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
from hsbg.az.train.economic_tasks import debug
from tests.test_economic_tasks import natural_source
from workflows.payoff_bridge import privateer_triple


def fixture():
    g=E.Game.from_json(json.dumps(natural_source()));debug(g,'set_player',tier=5,gold=10)
    state,_,action=privateer_triple(json.loads(g.state_json()))
    return E.Game.from_json(json.dumps(state)),action


def end(f,c,m,s):
    p=np.zeros(245,dtype=np.float32);p[0]=1;return p,.123


def test_first_buy_is_native_counts_once_and_resets():
    game,action=fixture();policy=FirstActionPolicy(PPOGreedyPolicy(end));policy.configure(action);policy.reset()
    before=json.loads(game.state_json());decision=policy.decide(game,0)
    assert decision.action==action and decision.info['bridge_first_action']
    game.step(0,decision.action)
    assert any(h['id']=='BG33_825_G' for h in json.loads(game.view(0))['me']['hand'])
    assert not any(game.diagnostic_counts()) and json.loads(game.state_json())['players'][0]['gold']<before['players'][0]['gold']
    decision=policy.decide(game,0);assert decision.action==0 and not decision.info['bridge_first_action']
    assert policy.policy._turn_counts[(0,game.turn)]==2
    policy.reset();game,action=fixture();policy.configure(action)
    assert policy.decide(game,0).info['bridge_first_action']


def test_first_buy_teacher_keeps_actual_history_and_guard_failure_is_loud():
    game,action=fixture();teacher=EconomicOptionPolicy(end,operator_mode='growth')
    policy=FirstActionPolicy(teacher);policy.configure(action);policy.reset()
    decision=policy.decide(game,0);assert teacher.previous[1]==action
    game.step(0,action);decision=policy.decide(game,0)
    assert not decision.info['bridge_first_action'] and decision.info['research_option_active']
    assert decision.info['research_operator_reason']=='restore_core'
    game,action=fixture();policy.reset();policy.configure(action)
    teacher.policy._turn_counts[(0,game.turn)]=100
    with pytest.raises(ValueError,match='ordinary guard'):policy.decide(game,0)
