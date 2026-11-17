import numpy as np
import pytest
from workflows.learn_payoff_process import EVENTS,mixed_event_sampling

def data():
    reasons=[r for group in EVENTS.values() for r in group]
    rows=[(reason,phase,dev) for reason in reasons for phase in ('T8_9','T10_12','T13_plus') for dev in (False,True)]
    return {'reasons':np.array([r for r,p,d in rows]),'kinds':np.array([p for r,p,d in rows]),
        'dev':np.array([d for r,p,d in rows])}

def test_mixed_sampling_keeps_batch_budget_and_source_holdout():
    raw=data();groups,sizes=mixed_event_sampling(raw)
    assert sum(sizes.values())==96
    assert sum(n for k,n in sizes.items() if k.startswith('phase/'))==48
    assert sum(n for k,n in sizes.items() if k.startswith('event/'))==48
    for category in ('phase/','event/'):
        selected=np.concatenate([v for k,v in groups.items() if k.startswith(category)])
        assert not raw['dev'][selected].any()
        assert sorted(selected.tolist())==np.flatnonzero(~raw['dev']).tolist()

def test_mixed_sampling_rejects_unknown_or_missing_roles():
    raw=data();raw['reasons'][0]='unrecognized'
    with pytest.raises(ValueError,match='unknown'):mixed_event_sampling(raw)
    raw=data();raw['dev'][raw['reasons']=='search']=True
    with pytest.raises(ValueError,match='empty'):mixed_event_sampling(raw)
