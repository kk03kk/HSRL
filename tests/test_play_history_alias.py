import json
from tests.test_economic_tasks import natural_source
from tools.prove_play_history_alias import construct


def test_two_native_histories_alias_current_inputs_but_not_teacher_memory():
    source=natural_source();before=json.dumps(source,sort_keys=True)
    result=construct(source)
    assert result['passed'] and all(result['input_equal'].values())
    a,b=result['histories']
    assert a['cycled_board_slots']==[4] and b['cycled_board_slots']==[5]
    assert (a['teacher_action'],b['teacher_action'])==(20,21)
    assert a['teacher_reason']==b['teacher_reason']=='liquidate_cycle'
    assert json.dumps(source,sort_keys=True)==before
