import copy
import pytest
from tools.analyze_whole_turn_student import compare

def data():
    rows=[{'seed':seed,'kind':'normal','student':{'eot_body':100+seed,'gold_spent':10,
        'core_count':3,'stopped_at_cap':False},'teacher':{'eot_body':200+seed,'gold_spent':20,
        'core_count':3,'stopped_at_cap':True}} for seed in range(4)]
    before={'closed_loop':{'rows':rows}};after=copy.deepcopy(before)
    for row in after['closed_loop']['rows']:row['student']['eot_body']+=50
    return before,after

def test_whole_turn_outcomes_are_paired_and_parent_caps_unknown():
    result=compare(*data())['normal']
    assert result['sources']==4
    assert result['differences']['student_minus_parent']['source_bootstrap_95ci']==[50,50]
    assert result['arms']['parent']['action_caps'] is None
    assert result['arms']['student']['eot_body_median']==151.5

def test_whole_turn_comparison_rejects_changed_teacher_or_pairing():
    before,after=data();after['closed_loop']['rows'][0]['teacher']['eot_body']+=1
    with pytest.raises(ValueError,match='teacher differs'):compare(before,after)
    before,after=data();after['closed_loop']['rows'].pop()
    with pytest.raises(ValueError,match='pairing differs'):compare(before,after)
