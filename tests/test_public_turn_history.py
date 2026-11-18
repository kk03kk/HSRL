import copy
import pytest
from hsbg.az.train.public_turn_history import PublicTurnHistory,project_own_record
from tests.test_economic_tasks import natural_source
from tools.prove_play_history_alias import construct


def test_actual_native_history_disambiguates_certificate_and_roundtrips():
    proof=construct(natural_source());marks=[]
    for trace in proof['histories']:
        history=PublicTurnHistory()
        for i,row in enumerate(trace['steps']):
            before=project_own_record(row['before'],0);after=project_own_record(row['after'],0)
            _,count=history.before(before);assert count==i
            history.after(before,after,row['action'])
            restored=PublicTurnHistory.restore(history.state_dict())
            assert restored.state_dict()==history.state_dict()
            history=restored
        marked,count=history.before(project_own_record(trace['final'],0))
        assert count==4;marks.append(marked.tolist())
        end=project_own_record(trace['final'],0);end['turn']+=1
        assert history.before(end)[0].sum()==0 and history.actions==0
    assert marks==[[0,0,0,0,1,0,0],[0,0,0,0,0,1,0]]


def test_history_rejects_wrong_schema_and_counts_actual_pending_decisions():
    history=PublicTurnHistory();view={'turn':8,'me':{'board':[],'hand':[]}}
    with pytest.raises(ValueError,match='matching'):history.after(view,view,30)
    history.before(view);history.after(view,view,30)
    assert history.actions==1
    state=history.state_dict();state['schema_sha256']='wrong'
    with pytest.raises(ValueError,match='schema'):PublicTurnHistory.restore(state)
    state=history.state_dict();state['entities']=[1,1]
    with pytest.raises(ValueError,match='invalid'):PublicTurnHistory.restore(state)
