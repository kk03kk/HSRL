import pytest
from tools.analyze_capped_turns import summarize_turn


def test_cap_tail_excludes_forced_combat_transition_and_retains_signed_growth():
    rows=[dict(action=1,kind='Refresh',gold_before=10,gold_after=9,body_before=100,body_after=100,
               forced_reason=None,hand_before=2,board_before=7,pending_before=False) for _ in range(25)]
    rows[5]['body_before']=150
    rows.append(dict(action=0,kind='EndTurn',gold_before=7,gold_after=10,body_before=120,body_after=1,
                     forced_reason='action_cap',hand_before=4,board_before=6,pending_before=False))
    r=summarize_turn(rows)
    assert r['tail_n']==20 and r['actual_non_eot_actions']==25
    assert r['tail_gold_outflow']==20 and r['tail_body_delta']==-30
    assert r['gold_at_forced_eot']==7 and r['tail_kinds']=={'Refresh':20}
    rows[-1]['forced_reason']=None
    with pytest.raises(ValueError,match='forced EOT'):summarize_turn(rows)
