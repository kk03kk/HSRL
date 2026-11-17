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
