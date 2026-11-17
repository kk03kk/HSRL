import json
from types import SimpleNamespace

import pytest
from hsbg.az.train.common import engine_info
from hsbg.az.train.eval_policy import protocol_from_args
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.evaluation.trace import DecisionRecorder
from tests.test_decision_recording import EndTurnPolicy
from tools.validate_eval_evidence import validate_panel


def test_parallel_validation_matches_serial_with_real_steps(tmp_path):
    pytest.importorskip('hsbg_engine')
    protocol=protocol_from_args(SimpleNamespace(),engine=engine_info())
    payload={**protocol.as_dict(),'protocol_sha256':protocol.sha256()}
    models=[]
    for name in ('a','b'):
        identity={'name':name,'checkpoint_sha256':name*64,'protocol':payload}
        recorder=DecisionRecorder(tmp_path/f'{name}.jsonl.gz',identity=identity)
        result=evaluate_policy(EndTurnPolicy(),EvalConfig(games=1,seed=637201,decision_recorder=recorder))
        manifest=recorder.finish(result)
        models.append({'name':name,'checkpoint_sha256':name*64,'decision_trace':manifest,
                       'seed_sequence':result['seed_sequence'],'rank_sequence':result['rank_sequence'],
                       'outcome':result})
    panel=tmp_path/'panel.json'
    panel.write_text(json.dumps({'protocol':payload,'models':models}),encoding='utf-8')
    serial=validate_panel(panel,replay=3)
    assert validate_panel(panel,replay=3,workers=2) == serial
    models[1]['rank_sequence']=[1 if models[1]['rank_sequence'][0]!=1 else 2]
    panel.write_text(json.dumps({'protocol':payload,'models':models}),encoding='utf-8')
    with pytest.raises(ValueError,match='panel ranks differ'): validate_panel(panel,replay=0,workers=2)
    panel.write_text(json.dumps({'protocol':payload,'models':[]}),encoding='utf-8')
    with pytest.raises(ValueError,match='nonempty'): validate_panel(panel)
