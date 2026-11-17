import copy,json
import pytest
from tests.test_economic_tasks import natural_source
from workflows.probe_whole_turn_economy import fixture,run_operator,metrics,VARIANTS
from hsbg.az.train.research_economy import Operator


def test_public_operator_rejects_full_state():
    with pytest.raises(ValueError,match='restricted public'):
        Operator().act({'rng':123,'players':[]})


def test_specialized_teacher_does_not_cycle_health_loss_battlecries():
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug
    from workflows.probe_whole_turn_economy import public_view
    game=E.Game.from_json(json.dumps(natural_source()))
    debug(game,'clear_zone',zone='board');debug(game,'clear_zone',zone='hand')
    debug(game,'add_minion',zone='board',card='TB_BaconUps_045')
    debug(game,'add_minion',zone='hand',card='BG26_525_G')
    view=public_view(game)
    assert any(view['mask'][46:54]),'risk card is native legal, not globally banned'
    action,reason=Operator().act(view)
    assert not 46<=action<54
    debug(game,'add_minion',zone='board',card='BG26_525')
    action,reason=Operator().act(public_view(game))
    assert action==17 and reason=='remove_health_risk'


def test_specialized_teacher_does_not_spend_health_to_buy():
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug,card_ids
    from workflows.probe_whole_turn_economy import public_view
    game=E.Game.from_json(json.dumps(natural_source()))
    debug(game,'clear_zone',zone='board');debug(game,'clear_zone',zone='hand')
    debug(game,'clear_zone',zone='shop');debug(game,'add_minion',zone='shop',card='BG25_520')
    state=json.loads(game.state_json())
    state['players'][0]['trinkets']=[card_ids()['BG32_MagicItem_821']]
    game=E.Game.from_json(json.dumps(state))
    view=public_view(game);before=copy.deepcopy(view)
    assert view['me']['shop'][0]['costs_health'] and view['mask'][8]
    action,_=Operator().act(view)
    assert action!=8 and view==before


def test_native_recruit_death_is_an_explicit_bad_outcome_not_silent_omission():
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug
    from hsbg.az.train.policies import Decision
    from tools.probe_economic_starts import run_turn,canonical_items
    game=E.Game.from_json(json.dumps(natural_source()))
    debug(game,'clear_zone',zone='board');debug(game,'clear_zone',zone='hand')
    debug(game,'add_minion',zone='board',card='TB_BaconUps_045')
    debug(game,'add_minion',zone='hand',card='BG26_525_G')
    state=json.loads(game.state_json());state['players'][0]['health']=1;state['players'][0]['armor']=0
    game=E.Game.from_json(json.dumps(state));game.step(0,46)
    state=json.loads(game.state_json());assert state['players'][0]['pending']
    class Choose:
        def reset(self):pass
        def decide(self,game,seat):
            return Decision(next(a for a in range(30,46) if bytes(game.legal_mask(seat))[a]))
    _,trace=run_turn(Choose(),state,canonical_items(),allow_recruit_death=True)
    outcome=metrics(trace)
    assert trace['end_intent']['reason']=='recruit_death' and len(trace['steps'])==1
    assert not outcome['alive'] and not outcome['eot_observed'] and outcome['eot_body']==0
    assert not any(E.Game.from_json(json.dumps(trace['final'])).diagnostic_counts())


def test_policy_cap_is_recorded_as_a_forced_boundary():
    import hsbg_engine as E
    import numpy as np
    from hsbg.az.train.ppo import PPOGreedyPolicy
    game=E.Game.from_json(json.dumps(natural_source()))
    def evaluator(features,cards,mask,seat):
        prior=np.asarray(mask,dtype=np.float32);return prior/prior.sum(),0.
    policy=PPOGreedyPolicy(evaluator)
    policy._turn_counts[(0,game.turn)]=policy.max_actions_per_turn
    decision=policy.decide(game,0)
    assert decision.action==0 and decision.info['forced']
    assert decision.info['forced_reason']=='action_cap'


@pytest.mark.parametrize('variant',VARIANTS)
def test_native_whole_turn_is_legal_preserves_input_and_replay(variant):
    import hsbg_engine as E
    source=natural_source();original=copy.deepcopy(source)
    state,edits=fixture(source,variant)
    assert source==original and edits
    result=run_operator(state);measure=metrics(result)
    assert result['initial']==state and measure['actions']<=128
    game=E.Game.from_json(json.dumps(state))
    for step in result['steps']:
        assert json.loads(game.state_json())==step['before']
        assert bytes(game.legal_mask(0))[step['action']]
        game.step(0,step['action'])
        assert json.loads(game.state_json())==step['after']
    assert json.loads(game.state_json())==result['final']
    assert result['end_intent']['action']==0 and not result['end_intent']['executed']
    assert not any(game.diagnostic_counts())


def test_whole_turn_labels_replay_and_include_only_genuine_end(tmp_path):
    from workflows.build_whole_turn_dataset import make_task
    (tmp_path/'traces').mkdir()
    source={'seed':11800000,'state':natural_source()}
    task,rows=make_task((source,'complete_gold',str(tmp_path)))
    assert task['split']=='dev' and task['labels']==len(rows)
    assert all(target.sum()==1 and (target<=mask).all() for f,c,mask,target,*rest in rows)
    if task['teacher']['stopped_at_cap']:
        assert rows[-1][-1]!='natural_end'
    else:
        assert rows[-1][-1]=='natural_end' and rows[-1][3][0]


def test_reused_baseline_requires_exact_state_and_teacher(tmp_path):
    import gzip
    from hsbg.runtime.artifacts import atomic_json,file_identity
    from workflows.learn_whole_turn_process import reuse_baseline
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'identity-only-test')
    state={'seed':123,'players':[]}
    metric={'actions':1,'gold_spent':3,'eot_body':100,'core_count':2}
    task={'seed':10,'kind':'normal','initial':state,'teacher':metric}
    trace=tmp_path/'10-normal.jsonl.gz'
    with gzip.open(trace,'wt') as stream:stream.write(json.dumps({'policy':'parent','initial':state})+'\n')
    atomic_json(tmp_path/'run.json',{'checkpoint':file_identity(checkpoint)})
    atomic_json(tmp_path/'complete.json',{'valid':True,'run':file_identity(tmp_path/'run.json'),
        'rows':[{'source_seed':10,'variant':'normal','trace':file_identity(trace),'parent':metric,'operator':metric}]})
    result=reuse_baseline(tmp_path,[task],checkpoint)
    assert result['all']['eot_body_mean']==100 and result['reused_baseline']
    task['teacher']={**metric,'eot_body':120}
    result=reuse_baseline(tmp_path,[task],checkpoint)
    assert result['changed_teacher_cases']==1
    assert result['all']['eot_body_mean']==100 and result['all']['teacher_eot_body_mean']==120
    task['initial']={'seed':124,'players':[]}
    with pytest.raises(ValueError,match='not identical'):reuse_baseline(tmp_path,[task],checkpoint)


def test_legacy_teacher_action_sequences_remain_unchanged():
    import hashlib
    expected={'normal':'12f7cdb83dffaae72fd3d510bf8a61b00fddd416913569d9c54195471abdec79',
        'complete':'9fca3e0613fc446bdf63755910adbd1c212e82e0aa5219ca16d3914c594a1706',
        'complete_gold':'f0d22d72c58b4e89c003991adf5d3a560c6ff8636d616903d2b044112ac33a61',
        'missing_privateer':'c827e8b528fcd92edd21627c69b93f209e6caa28f7f0303c768b3b5632b903c6'}
    for variant,want in expected.items():
        trace=run_operator(fixture(natural_source(),variant)[0])
        got=hashlib.sha256(json.dumps([s['action'] for s in trace['steps']]).encode()).hexdigest()
        assert got==want


def test_target_safe_teacher_preserves_privateer_when_evolving():
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug,card_ids
    from workflows.probe_whole_turn_economy import public_view
    state,_=fixture(natural_source(),'normal');game=E.Game.from_json(json.dumps(state))
    debug(game,'clear_zone',zone='board');debug(game,'clear_zone',zone='hand');debug(game,'clear_zone',zone='shop')
    for card in ('BG_LOE_077','BG33_823','BG33_825'):debug(game,'add_minion',zone='board',card=card)
    debug(game,'add_minion',zone='shop',card='BGS_115')
    debug(game,'add_spell',zone='hand',card='BG30_804')
    state=json.loads(game.state_json());state['players'][0]['board'][2].update(atk=50,hp=50,max_hp=50)
    game=E.Game.from_json(json.dumps(state));game.step(0,46);view=public_view(game)
    old,_=Operator().act(view);new,_=Operator('target_safe').act(view)
    assert old==32 and new==33
    game.step(0,new)
    assert json.loads(game.player_json(0))['board'][2]['card']==card_ids()['BG33_825']
    assert not any(game.diagnostic_counts())


def test_target_safe_teacher_does_not_cast_destructive_spell_on_only_cores():
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug
    from workflows.probe_whole_turn_economy import public_view
    game=E.Game.from_json(json.dumps(natural_source()))
    for zone in ('board','hand','shop'):debug(game,'clear_zone',zone=zone)
    for card in ('BG_LOE_077','BG33_823','BG33_825'):debug(game,'add_minion',zone='board',card=card)
    debug(game,'add_spell',zone='hand',card='EBG_Spell_032')
    view=public_view(game);assert any(view['mask'][46:54])
    action,_=Operator('target_safe').act(view)
    assert not 46<=action<54


def test_growth_teacher_acquires_payoff_and_reserves_upgrade_economy():
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug
    from workflows.probe_whole_turn_economy import public_view
    state,_=fixture(natural_source(),'normal');game=E.Game.from_json(json.dumps(state))
    debug(game,'clear_zone',zone='hand');debug(game,'clear_zone',zone='shop')
    debug(game,'add_minion',zone='shop',card='BG36_344')
    action,reason=Operator('growth').act(public_view(game))
    assert action==8 and reason=='complete_engine'
    debug(game,'clear_zone',zone='shop');state=json.loads(game.state_json())
    state['players'][0]['tier']=5;state['players'][0]['gold']=30
    game=E.Game.from_json(json.dumps(state));view=public_view(game)
    assert view['mask'][3]
    assert Operator('growth').act(view)==(3,'unlock_growth_payoff')
    assert Operator('target_safe').act(view)[0]!=3
