import gzip,json
import numpy as np
import hsbg_engine as E
from hsbg.runtime.artifacts import file_identity
from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture
from tools.audit_payoff_closed_loop import scan


def test_shadow_reconstructs_native_teacher_history_and_guards(tmp_path):
    def fallback(features,cards,mask,seat):
        prior=np.asarray(mask,dtype=np.float32);return prior/prior.sum(),0.
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    policy=EconomicOptionPolicy(fallback,operator_mode='growth');path=tmp_path/'trace.jsonl.gz';count=0
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for index in range(100):
            before=json.loads(game.state_json());turn=game.turn;decision=policy.decide(game,0)
            game.step(0,decision.action)
            stream.write(json.dumps({'event':'action','state_before':before,'state_after':json.loads(game.state_json()),
                'seat':0,'turn':turn,'action':decision.action,'action_index':index,'policy_info':decision.info})+'\n')
            count+=1
            if decision.action==0:break
    assert not any(game.diagnostic_counts())
    result=scan((str(path),{**file_identity(path),'actions':count},True))
    assert result['counts']['actions']==count
    assert result['counts']['eligible']>20
    assert result['counts']['agreement']==result['counts']['eligible']
    assert result['counts']['cycle_memory_changes_action']>0
    assert not result['recruit_deaths']
