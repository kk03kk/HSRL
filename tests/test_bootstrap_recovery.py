import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from taverngym.research.bootstrap import GIFT_PAIR_BANS_3701, agreement, readiness
from taverngym.rl.collect import collect
from taverngym.rl.config import Config
from taverngym.rl.model import Policy, pack
from taverngym.rl.train import update, train, initialize_weights
from taverngym.rl.config import identity


def config():
    torch.set_num_threads(1)
    return Config(device='cpu',policy_kind='compact_gru',n_games=1,learner_seats=1,threads=1,
                  torch_threads=1,dim=16,heads=2,layers=1,hidden=16,seq_len=8,memory_horizon=32,
                  chunks_per_batch=2,bc_epochs=1,epochs=1,iterations=1,bc_waves=0,eval_games=1,
                  eval_seeds=[900000000],opponent_mode='bots')


def test_readiness_requires_real_rollouts_and_teacher_agreement():
    a={'decisions':300,'top1':.8}
    panels={k:{'mean_rank':4.5,'truncated_games':0,'eligible_for_selection':True}
            for k in ['bots_greedy','bots_sample']}
    assert readiness(a,panels)['ready_for_ppo']
    p=copy.deepcopy(panels);p['bots_sample']['mean_rank']=8
    assert not readiness(a,p)['ready_for_ppo']
    p=copy.deepcopy(panels);p['bots_sample']['truncated_games']=1
    assert not readiness(a,p)['ready_for_ppo']
    assert not readiness({'decisions':3,'top1':1.},panels)['ready_for_ppo']


def test_vectorized_teacher_evaluation_matches_online_gru():
    c=config();torch.manual_seed(91);m=Policy(c).eval()
    trajectories,_=collect(c,None,73001,teacher=True)
    result=agreement(m,trajectories,c)
    total=hits=0
    with torch.no_grad():
        for tr in trajectories:
            h=torch.zeros((1,c.hidden))
            for i,s in enumerate(tr['samples']):
                if i%c.memory_horizon==0:h.zero_()
                logits,_,_,h=m(pack([s['obs']],'cpu'),h)
                if s['obs']['mask'].sum()>1:
                    total+=1;hits+=int(logits.argmax(-1))==s['action']
    assert result['decisions']==total and result['top1']==pytest.approx(hits/total)


def test_supervised_pipeline_can_fit_a_fixed_nonforced_native_trajectory():
    c=config();torch.manual_seed(123)
    trajectories,_=collect(c,None,73001,teacher=True)
    # Bounded functional overfit test, not a skill/generalization experiment.
    m=Policy(c);optimizer=torch.optim.Adam(m.parameters(),lr=.003)
    start=agreement(m,trajectories,c)['nll']
    fit=replace(c,bc_epochs=2,bc_rank_coef=0.)
    for _ in range(8):update(m,optimizer,trajectories,fit,bc=True)
    end=agreement(m,trajectories,c)['nll']
    assert end<start*.7


def test_bot_training_selects_best_against_bots_not_weak_anchor_mix(tmp_path):
    c=config()
    result={k:{'eligible_for_selection':True,'mean_rank':rank} for k,rank in
            [('bots_greedy',8.),('bots_sample',8.),('anchor_mix_greedy',1.),('anchor_mix_sample',1.)]}
    with patch('taverngym.rl.train.evaluate',return_value=result):train(c,tmp_path/'run')
    state=torch.load(tmp_path/'run/latest.pt',map_location='cpu',weights_only=True)
    assert state['progress']['best_rank']==8.


def test_weights_only_transfer_canonicalizes_one_ulp_constant_but_rejects_real_change(tmp_path):
    c=config();model=Policy(c);state=copy.deepcopy(model.state_dict());signature=identity(c)
    state['utility']=torch.nextafter(state['utility'],torch.full_like(state['utility'],float('inf')))
    path=tmp_path/'cuda-like.pt';torch.save({'model':state,'identity':signature},path)
    result=initialize_weights(model,path,c,signature)
    assert result['canonicalized_constant_buffers']==['utility']
    state['utility'][2]+=.01;torch.save({'model':state,'identity':signature},path)
    with pytest.raises(ValueError,match='incompatible buffer utility'):initialize_weights(model,path,c,signature)
