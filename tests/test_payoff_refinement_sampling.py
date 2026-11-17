import numpy as np
import pytest
from workflows.learn_payoff_refinement import mixed_origin_sampling


def data():
    rows=[(phase,reason,origin) for phase in ('T8_9','T10_12','T13_plus')
        for reason in ('unlock_growth_payoff','wealth','cycle_hand','liquidate_cycle','search','end') for origin in (0,1)]
    return {'kinds':np.array([r[0] for r in rows]),'reasons':np.array([r[1] for r in rows]),
        'origin':np.array([r[2] for r in rows]),'dev':np.zeros(len(rows),dtype=bool)}


def test_query_source_and_event_mixture_has_exact_budget_and_no_development():
    raw=data();groups,sizes=mixed_origin_sampling(raw)
    assert sum(sizes.values())==96
    assert sum(n for k,n in sizes.items() if k.endswith('/0'))==48
    assert sum(n for k,n in sizes.items() if k.startswith('phase/'))==48
    assert set(np.concatenate(list(groups.values())).tolist())==set(range(len(raw['dev'])))
    assert all(np.all(raw['origin'][v]==int(k[-1])) for k,v in groups.items())
    raw['dev'][1]=True
    with pytest.raises(ValueError,match='leakage'):mixed_origin_sampling(raw)


def test_empty_origin_event_group_is_rejected_instead_of_changing_budget():
    raw=data();raw['origin'][raw['reasons']=='wealth']=0
    with pytest.raises(ValueError,match='empty predeclared'):mixed_origin_sampling(raw)
