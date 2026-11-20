import gzip,json
import numpy as np
import hsbg_engine as E
from hsbg.runtime.artifacts import file_identity
from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture
from tools.audit_payoff_closed_loop import scan,RecordedEvaluator


def test_cycle_memory_ablation_is_explicit_and_preserves_other_history():
    from hsbg.az.train.research_economy_policy import _Evaluator
    from hsbg.az.train.research_economy import Operator
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    def fallback(features,cards,mask,seat):raise AssertionError('unexpected fallback')
    regular=EconomicOptionPolicy(fallback,operator_mode='growth')
    found=False
    for _ in range(80):
        decision=regular.decide(game,0)
        operator=regular.evaluator.operator
        if decision.info['research_operator_reason'] in ('liquidate_cycle','keep_stronger_cycle'):
            import copy
            altered=_Evaluator(fallback,'growth_no_cycle_memory')
            altered.operator=copy.deepcopy(operator);altered.view=regular.evaluator.view
            last_play=altered.operator.last_play
            expected=copy.deepcopy(operator);expected.cycled.clear()
            action,reason=expected.act({**altered.view,'mask':regular.evaluator.last_mask.tolist()})
            prior,_=altered(None,None,regular.evaluator.last_mask,0)
            assert int(prior.argmax())==action and altered.reason==reason
            assert not altered.operator.cycled and altered.operator.last_play==last_play
            assert operator.cycled
            if action != decision.action:
                found=True;break
        game.step(0,decision.action)
        if decision.action==0:break
    assert found
    policy=EconomicOptionPolicy(fallback,operator_mode='growth_no_cycle_memory')
    policy.reset();assert policy.evaluator.operator.mode=='growth'
    assert not any(game.diagnostic_counts())


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


def test_student_only_unsafe_target_is_explicit_abstention(tmp_path):
    from hsbg.az.train.economic_tasks import debug
    from hsbg.az.train.ppo import PPOGreedyPolicy
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    debug(game,'clear_zone',zone='hand');debug(game,'clear_zone',zone='shop')
    # Retain exactly the protected four cores: no alternative safe target.
    debug(game,'clear_zone',zone='board')
    for card in ('TB_BaconUps_045','BG33_825_G','BG33_823','BG36_523'):
        debug(game,'add_minion',zone='board',card=card)
    debug(game,'add_spell',zone='hand',card='BG30_804')
    evaluator=RecordedEvaluator();policy=PPOGreedyPolicy(evaluator)
    path=tmp_path/'unsafe.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for i in range(2):
            evaluator.action=46 if i==0 else next(a for a in range(30,46) if bytes(game.legal_mask(0))[a])
            before=json.loads(game.state_json());turn=game.turn;decision=policy.decide(game,0)
            game.step(0,decision.action)
            stream.write(json.dumps({'event':'action','state_before':before,'state_after':json.loads(game.state_json()),
                'seat':0,'turn':turn,'action':decision.action,'action_index':i,'policy_info':decision.info})+'\n')
    result=scan((str(path),{**file_identity(path),'actions':2},False))
    assert result['counts']['teacher_unavailable']==1
    assert result['counts']['teacher_unavailable_gold_gate']==1
    assert result['teacher_abstention_examples'][0]['action_index']==1


def test_shadow_records_gold_gate_exit_after_a_core_sale(tmp_path):
    from hsbg.az.train.ppo import PPOGreedyPolicy
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    board=json.loads(game.view(0))['me']['board']
    sell=16+next(i for i,m in enumerate(board) if m['id']=='TB_BaconUps_045')
    evaluator=RecordedEvaluator();policy=PPOGreedyPolicy(evaluator)
    path=tmp_path/'gate-exit.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for i,action in enumerate((sell,0)):
            evaluator.action=action;before=json.loads(game.state_json());turn=game.turn
            decision=policy.decide(game,0);game.step(0,decision.action)
            stream.write(json.dumps({'event':'action','state_before':before,'state_after':json.loads(game.state_json()),
                'seat':0,'turn':turn,'action':decision.action,'action_index':i,'policy_info':decision.info})+'\n')
    result=scan((str(path),{**file_identity(path),'actions':2},False))
    assert result['counts']['gold_gate_actions']==1
    assert result['counts']['observed_gate_exits']==1
    assert result['counts']['gate_exits_following_sell']==1
    assert result['gate_transition_examples'][0]['missing_groups']==['TB_BaconUps_045']
