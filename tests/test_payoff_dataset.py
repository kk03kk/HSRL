import json
from types import SimpleNamespace
import numpy as np
import torch
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.conditional_economy import fork_conditional
from tests.test_conditional_economy import small_parent
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture
from workflows.build_payoff_dataset import gate,label

def test_label_gate_matches_model_and_excludes_forced_or_normal_context():
    import hsbg_engine as E
    base,meta=small_parent();model=fork_conditional(base,meta,mechanisms=True,gate_mode='gold_engine')
    for variant in ('normal','complete','complete_gold','missing_privateer'):
        initial,_=fixture(natural_source(),variant);game=E.Game.from_json(json.dumps(initial))
        ff,cc=M.observe(game,0,M.SCHEMA);f=np.frombuffer(ff,dtype='<f4').copy();c=np.frombuffer(cc,dtype='<i4').copy()
        mask=np.asarray(list(bytes(game.legal_mask(0))),dtype=bool)
        assert gate(f,c)==bool(model.gate(torch.tensor(f[None]),torch.tensor(c[None]))[0])
        action=int(np.flatnonzero(mask)[0]);decision=SimpleNamespace(action=action,
            info={'forced':False,'research_option_active':True,'research_operator_reason':'test'})
        row=label(f,c,mask,decision,10)
        assert (row is not None)==(variant=='complete_gold')
        if row is not None:
            assert row[3][action] and row[4]=='T10_12' and np.array_equal(row[2],mask)
        decision.info['forced']=True;assert label(f,c,mask,decision,10) is None
