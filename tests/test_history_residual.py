import json
import numpy as np
import pytest
import torch
import hsbg_engine as E
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.history_residual import (HistoryResidual,current_spec,save_adapter,load_adapter,
    HistoryResidualEvaluator,HistoryResidualPolicy)
from hsbg.az.train.conditional_economy import fork_conditional
from hsbg.az.train.ppo import PPOEvaluator
from tests.test_conditional_economy import small_parent
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture


def sample(variant='complete_gold'):
    game=E.Game.from_json(json.dumps(fixture(natural_source(),variant)[0]))
    f,c=M.observe(game,0,M.SCHEMA)
    return game,np.frombuffer(f,dtype='<f4').copy(),np.frombuffer(c,dtype='<i4').copy(),np.asarray(list(bytes(game.legal_mask(0))),bool)


def test_zero_residual_preserves_base_and_critic_and_gate(tmp_path):
    base,meta=small_parent();base=fork_conditional(base,meta,mechanisms=True,gate_mode='gold_engine').eval()
    evaluator=PPOEvaluator(base,device='cpu');adapter=HistoryResidual(current_spec(True)).eval()
    wrapped=HistoryResidualEvaluator(evaluator,adapter)
    _,f,c,m=sample();p,v=evaluator(f,c,m,0);q,w=wrapped(f,c,m,0)
    assert np.array_equal(p,q) and v==w and wrapped.active
    before={k:x.clone() for k,x in base.state_dict().items()}
    with torch.no_grad():adapter.output.bias[0]=2
    q,w=wrapped(f,c,m,0);assert not np.array_equal(p,q) and v==w
    _,f,c,m=sample('normal');p,v=evaluator(f,c,m,0);q,w=wrapped(f,c,m,0)
    assert np.array_equal(p,q) and v==w and not wrapped.active
    assert all(torch.equal(x,before[k]) for k,x in base.state_dict().items())
    identity={'sha256':'test','bytes':123};path=tmp_path/'model.pt'
    save_adapter(path,adapter,identity,{});restored=load_adapter(path,identity)
    assert all(torch.equal(x,restored.state_dict()[k]) for k,x in adapter.state_dict().items())
    with pytest.raises(ValueError,match='base identity'):load_adapter(path,{'sha256':'wrong','bytes':123})


def test_history_is_only_paired_difference_and_can_separate_aliases():
    torch.set_num_threads(1);torch.manual_seed(42)
    on=HistoryResidual(current_spec(True));off=HistoryResidual(current_spec(False));off.load_state_dict(on.state_dict())
    f=torch.zeros(2,3282);c=torch.zeros(2,124,dtype=torch.long);p=torch.zeros(2,245)
    h=torch.zeros(2,7);h[0,4]=1;h[1,5]=1
    # Fit only the declared observation alias. A feed-forward no-history arm
    # cannot distinguish these identical current observations.
    opt=torch.optim.Adam(on.parameters(),lr=.01);target=torch.tensor([20,21])
    for _ in range(30):
        loss=torch.nn.functional.cross_entropy(on(f,c,p,h),target);opt.zero_grad();loss.backward();opt.step()
    assert on(f,c,p,h).argmax(-1).tolist()==[20,21]
    off.load_state_dict(on.state_dict());result=off(f,c,p,h)
    assert torch.equal(result[0],result[1])
    assert float(on(f,c,p,h).detach().abs().max())<=6


def test_native_policy_requires_actual_action_feedback():
    base,meta=small_parent();base=fork_conditional(base,meta,mechanisms=True,gate_mode='gold_engine').eval()
    policy=HistoryResidualPolicy(PPOEvaluator(base,device='cpu'),HistoryResidual(current_spec(True)))
    game,_,_,_=sample();before=json.loads(game.state_json());decision=policy.decide(game,0)
    assert decision.info['public_history_actions']==0
    with pytest.raises(ValueError,match='actual execution feedback'):policy.decide(game,0)
    game.step(0,decision.action);policy.record_actual(before,json.loads(game.state_json()),0,decision.action)
    assert not policy.awaiting and policy.history.actions==1
    assert not any(game.diagnostic_counts())
    policy.reset();assert policy.history.actions==0
