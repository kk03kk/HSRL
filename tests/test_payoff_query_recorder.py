import json
import numpy as np
import hsbg_engine as E
from hsbg.az.train.shadow_economy import ShadowEconomyPolicy
from hsbg.az.train import mechanism_obs as M
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture
from workflows.build_payoff_queries import QueryRecorder


def test_query_labels_teacher_target_but_records_executed_student_action(tmp_path):
    def end(features,cards,mask,seat):
        prior=np.zeros(245,dtype=np.float32);prior[0]=1;return prior,0.
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    policy=ShadowEconomyPolicy(end)
    recorder=QueryRecorder(tmp_path/'trace.jsonl.gz',identity={'mechanism_schema':M.SCHEMA},policy=policy)
    recorder.begin_game(game,0,123,0);before=recorder.before(game,0);decision=policy.decide(game,0)
    assert decision.action==0 and decision.info['shadow_teacher_action']!=0
    game.step(0,decision.action);recorder.after(game,before,decision,decision.action);recorder.abort()
    assert len(recorder.labels)==1
    assert recorder.labels[0][3][decision.info['shadow_teacher_action']]
    assert not recorder.labels[0][3][0]
    assert recorder.counts['action']==1
