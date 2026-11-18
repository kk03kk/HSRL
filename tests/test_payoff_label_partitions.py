import numpy as np
import pytest
from tools.analyze_payoff_labels import select_rows


def test_fit_diagnostics_keep_development_and_queries_separate():
    raw={'dev':np.array([False,True,False,False]),'origin':np.array([0,0,1,0])}
    ix,part=select_rows(raw,'all')
    assert ix.tolist()==[0,1,2,3]
    assert part.tolist()==['original_training','development','student_queries','original_training']
    assert select_rows(raw,'development')[0].tolist()==[1]
    assert select_rows(raw,'student_queries')[0].tolist()==[2]
    assert select_rows(raw,'original_training')[0].tolist()==[0,3]
    raw['dev'][2]=True
    with pytest.raises(ValueError,match='leakage'):select_rows(raw,'all')


def test_old_dataset_without_origins_is_supported_without_inventing_queries():
    raw={'dev':np.array([False,True])}
    assert select_rows(raw,'development')[0].tolist()==[1]
    with pytest.raises(ValueError,match='empty'):select_rows(raw,'student_queries')
