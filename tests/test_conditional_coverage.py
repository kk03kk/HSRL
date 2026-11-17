import json
import numpy as np
import torch
from hsbg.az.train.conditional_economy import fork_conditional
from tests.test_conditional_economy import small_parent
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture
from tools.audit_conditional_coverage import public_gate

def test_trace_gate_matches_model_gate_for_actual_states():
    import hsbg_engine as E
    parent,meta=small_parent();model=fork_conditional(parent,meta)
    initial=natural_source()
    states=[initial]+[fixture(initial,k)[0] for k in ('normal','complete','complete_gold','missing_privateer')]
    for state in states:
        game=E.Game.from_json(json.dumps(state));f,c=game.observe(0)
        expected=model.gate(torch.tensor(np.frombuffer(f,dtype='<f4').copy()[None]),
                            torch.tensor(np.frombuffer(c,dtype='<i4').copy()[None],dtype=torch.long)).item()
        actual=public_gate(state,0,list(bytes(game.legal_mask(0))))
        assert actual['active']==expected
    assert not public_gate(initial,0)['active']
    assert all(public_gate(s,0)['active'] for s in states[1:])
