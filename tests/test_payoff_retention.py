from tools.audit_payoff_retention import account


def action(index,a,before,after,*,override=False,alive=True):
    return {'seat':0,'turn':8,'action':a,'action_index':index,
        'state_before':{'players':[{'board':before,'alive':True}]},
        'state_after':{'players':[{'board':after,'alive':alive}]},
        'policy_info':{'deployment_override_executed':override,'original_greedy_action':1}}


def test_retention_counts_exact_entity_sale_separately_from_replacement():
    m={'eid':12,'card':1};gold={'eid':17,'card':2}
    rows=[action(0,46,[],[m],override=True),action(1,16,[m],[]),
        action(2,46,[],[gold]),action(3,0,[gold],[])]
    events,n=account(rows,{1,2})
    assert n==4 and len(events)==1
    assert events[0]['first_departure']['kind']=='sold'
    assert not events[0]['entity_at_end'] and events[0]['any_payoff_at_end']


def test_end_turn_uses_precombat_state_and_recruit_death_is_retained():
    m={'eid':12,'card':1}
    rows=[action(0,46,[],[m],override=True),action(1,0,[m],[])]
    events,_=account(rows,{1,2});assert events[0]['entity_at_end']
    rows[-1]=action(1,30,[m],[m],alive=False)
    events,_=account(rows,{1,2});assert events[0]['end']=='recruit_death'
    import pytest
    with pytest.raises(ValueError,match='unfinished'):account(rows[:1],{1,2})
