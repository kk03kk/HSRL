import json
import numpy as np
import hsbg_engine as E
from hsbg.az.train.shadow_economy import PayoffDeploymentPolicy
from hsbg.az.train.economic_tasks import debug
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture


def setup(variant='complete_gold'):
    game=E.Game.from_json(json.dumps(fixture(natural_source(),variant)[0]))
    debug(game,'clear_zone',zone='hand');debug(game,'clear_zone',zone='board')
    for card in (('TB_BaconUps_045','BG33_825_G','BG33_823','BG36_523') if variant=='complete_gold'
                 else ('BG_LOE_077','BG33_825','BG33_823','BG36_523')):
        debug(game,'add_minion',zone='board',card=card)
    debug(game,'add_minion',zone='hand',card='BG36_344')
    return game


def end(features,cards,mask,seat):
    p=np.zeros(245,dtype=np.float32);p[0]=1;return p,.123


def test_deployment_only_intervenes_in_declared_gate_and_respects_cap():
    game=setup();policy=PayoffDeploymentPolicy(end);before=game.state_json()
    decision=policy.decide(game,0)
    assert decision.info['deployment_override_executed'] and 46<=decision.action<54
    assert decision.info['original_greedy_action']==0 and decision.info['value']==.123
    assert game.state_json()==before
    game.step(0,decision.action)
    assert not policy.decide(game,0).info['deployment_override_executed']
    assert not any(game.diagnostic_counts())
    game=setup('complete');assert PayoffDeploymentPolicy(end).decide(game,0).action==0
    game=setup();policy=PayoffDeploymentPolicy(end);policy.policy._turn_counts[(0,game.turn)]=100
    decision=policy.decide(game,0)
    assert decision.action==0 and decision.info['forced'] and not decision.info['deployment_override_executed']


def test_already_deploying_student_keeps_its_own_position():
    def left(features,cards,mask,seat):
        p=np.zeros(245,dtype=np.float32);p[46]=1;return p,0.
    decision=PayoffDeploymentPolicy(left).decide(setup(),0)
    assert decision.action==46 and not decision.info['deployment_override_executed']
