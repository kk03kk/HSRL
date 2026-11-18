import json
import numpy as np
from hsbg.az.train.gate_retention_probe import GateRetentionPolicy,gate_breaking_sales
from hsbg.az.train.economic_tasks import debug
from tests.test_payoff_deployment_intervention import setup


def prefers_brann_sale(features,cards,mask,seat):
    from hsbg.az.train.conditional_economy import current_gate_spec
    spec=current_gate_spec('gold_engine');board=cards[slice(*spec['board_slice'])]
    a=16+int(np.flatnonzero(board==spec['embedding_ids'][0][0])[0])
    p=np.zeros(245,dtype=np.float32);p[0]=.25;p[a]=.75
    return p,.5


def test_constraint_only_blocks_direct_loss_of_last_owned_gate_component():
    from hsbg.az.train import mechanism_obs as M
    game=setup();before=game.state_json()
    policy=GateRetentionPolicy(prefers_brann_sale)
    decision=policy.decide(game,0)
    assert decision.action==0 and decision.info['gate_retention_override_executed']
    assert decision.info['value']==.5 and game.state_json()==before
    f,c=M.observe(game,0,M.SCHEMA);f=np.frombuffer(f,dtype='<f4');c=np.frombuffer(c,dtype='<i4')
    assert len(gate_breaking_sales(f,c,list(game.legal_mask(0))))==4
    debug(game,'add_minion',zone='hand',card='TB_BaconUps_045')
    decision=GateRetentionPolicy(prefers_brann_sale).decide(game,0)
    assert 16<=decision.action<23 and not decision.info['gate_retention_override_executed']
    game.step(0,decision.action)
    assert not any(game.diagnostic_counts())


def test_gate_constraint_respects_forced_end_and_is_inactive_outside_gold():
    policy=GateRetentionPolicy(prefers_brann_sale);game=setup()
    policy.policy._turn_counts[(0,game.turn)]=100
    decision=policy.decide(game,0)
    assert decision.action==0 and decision.info['forced'] and not decision.info['gate_retention_override_executed']
    ordinary=setup('complete');f,c=ordinary.observe(0)
    assert gate_breaking_sales(np.frombuffer(f,dtype='<f4'),np.frombuffer(c,dtype='<i4'),list(ordinary.legal_mask(0)))==[]
