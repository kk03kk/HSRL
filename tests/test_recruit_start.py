import gzip,json
import numpy as np
import pytest
import hsbg_engine as engine
from hsbg.az.train.common import engine_info
from hsbg.az.train.recruit_start import choose_start,load_bank,validate_options
from hsbg.runtime.artifacts import atomic_json,file_identity
from tests.test_economic_tasks import natural_source


def write_bank(path):
    path.mkdir()
    source=natural_source(11800001)
    with gzip.open(path/'states.jsonl.gz','wt') as f:
        f.write(json.dumps({'source_seed':11800001,'state':source,'split':'train'})+'\n')
    atomic_json(path/'manifest.json',{'schema':'hsai.recruit-start-bank/v1','variant':'test',
        'engine':engine_info(),'states':1,'data':file_identity(path/'states.jsonl.gz')})
    return source,file_identity(path/'manifest.json')['sha256']


def test_disabled_start_is_verbatim_and_invalid_modes_fail():
    game=object()
    assert choose_start(game,bank='missing',probability=0,seed=0,engine=None)==(game,None)
    for bank,p in (('',1),('x',0),('x',float('nan')),('x',1.1)):
        with pytest.raises(ValueError):validate_options(bank,p)
    with pytest.raises(ValueError):validate_options('x',1,lockstep=False)
    with pytest.raises(ValueError):validate_options('x',1,rust_collector=True)


def test_start_rng_is_fresh_deterministic_and_source_unmodified(tmp_path):
    source,sha=write_bank(tmp_path/'bank')
    fresh=engine.Game(seed=11830001)
    before=json.dumps(load_bank(str(tmp_path/'bank'))[1],sort_keys=True)
    a,info=choose_start(fresh,bank=tmp_path/'bank',probability=1,seed=11830001,
                        engine=engine,expected_sha256=sha)
    b,_=choose_start(fresh,bank=tmp_path/'bank',probability=1,seed=11830001,
                     engine=engine,expected_sha256=sha)
    state=json.loads(a.state_json())
    assert a.state_json()==b.state_json()
    assert state['rng']==json.loads(fresh.state_json())['rng']
    assert state['players']==source['players'] and state['turn']==8
    assert info['source_seed']==11800001 and a.needs_action(0)
    assert json.dumps(load_bank(str(tmp_path/'bank'))[1],sort_keys=True)==before
    with pytest.raises(ValueError,match='identity changed'):
        choose_start(fresh,bank=tmp_path/'bank',probability=1,seed=1,engine=engine,expected_sha256='bad')


def test_endowment_preserves_gold_health_tier_and_is_explicit():
    from workflows.build_recruit_curriculum import endow,CORES
    from hsbg.az.train.economic_tasks import card_ids
    source=natural_source(11800001);before=json.dumps(source,sort_keys=True)
    state,edits=endow(source);p=state['players'][0]
    assert edits and len(p['board'])<=7 and len(p['hand'])==2
    assert {card_ids()[x] for x in CORES}<={m['card'] for m in p['board']}
    assert all(p[k]==source['players'][0][k] for k in ('gold','health','tier'))
    assert json.dumps(source,sort_keys=True)==before


def test_lockstep_episode_starts_at_bank_turn_without_fictitious_prefix(tmp_path):
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo import CollectConfig,build_batch_from_records
    from hsbg.az.train.parallel_collect import collect_ppo_lockstep
    _,sha=write_bank(tmp_path/'bank')
    class EndTurn:
        def __call__(self,features,cards,mask,seat):
            prior=np.zeros(245,np.float32);prior[int(np.flatnonzero(mask)[0])]=1
            return prior,0.
    model=build_net(8)
    cfg=CollectConfig(games=2,seed=11830001,lockstep=True,max_steps=5000,
        recruit_start_bank=str(tmp_path/'bank'),recruit_start_p=1,
        recruit_start_manifest_sha256=sha)
    records,meta=collect_ppo_lockstep(model,cfg,eval_fn=EndTurn(),device='cpu',seed=11830001,games=2)
    assert records and min(r['turn'] for r in records)==8
    assert meta['recruit_start']['games']==2 and meta['recruit_start']['decisions']==len(records)
    assert not meta['illegal_actions'] and not meta['truncated_games']
    assert all(1<=r['rank']<=8 for r in records)
    batch=build_batch_from_records(records,gamma=1.,lam=.995)
    assert batch.validate()==[] and np.isfinite(batch.advantages).all()
