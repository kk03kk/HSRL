import copy
import json

import numpy as np
import pytest
import torch

from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.model import build_net, load_checkpoint, save_checkpoint
from hsbg.az.train.ppo import build_batch_from_records, pack_records_arrays, build_batch_from_arrays
from workflows.upgrade_mechanism_checkpoint import enable_mechanisms


def fixture(counter=0):
    engine=pytest.importorskip('hsbg_engine')
    game=engine.Game(seed=20261118)
    for _ in range(2000):
        if game.turn>=2:
            break
        for seat in range(game.num_players):
            if game.needs_action(seat):
                game.step(seat,game.bot_action(seat))
    state=json.loads(game.state_json())
    own=state['players'][0]
    own['board']=own['board'][:1]
    cid=next(i for i in M._allowed_native_cards() if engine.card_info(i+1)[0]=='BG33_823')
    own['board'][0].update(card=cid,atk=6,hp=6,max_hp=6,tier=6,counter=counter)
    own.update(gold=10,hand=[],pending=[],ended_turn=False)
    return engine.Game.from_json(json.dumps(state))


def test_counter_pair_is_identified_without_hidden_state():
    a,b=fixture(0),fixture(8)
    assert a.observe(0)==b.observe(0)
    fa,ca=M.observe(a,0,M.SCHEMA);fb,cb=M.observe(b,0,M.SCHEMA)
    assert ca==cb and fa[:3227*4]==fb[:3227*4] and fa!=fb
    af=np.frombuffer(fa,dtype='<f4');bf=np.frombuffer(fb,dtype='<f4')
    assert np.flatnonzero(af!=bf).tolist()==[3227+M.N_STATS]
    assert bf[3227+M.N_STATS]==8
    # The new encoding does not inspect other players or the game RNG.
    class OwnOnly:
        def observe(self,seat):return a.observe(seat)
        def player_json(self,seat):
            assert seat==0
            return a.player_json(seat)
        def state_json(self):raise AssertionError('full hidden state must not be read')
        def view(self,seat):raise AssertionError('opponent view must not be read')
    assert M.observe(OwnOnly(),0,M.SCHEMA)==(fa,ca)
    a.step(0,1);b.step(0,1)
    assert len(json.loads(a.player_json(0))['hand'])==0
    assert len(json.loads(b.player_json(0))['hand'])==1


def test_zero_upgrade_gradient_and_roundtrip(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(101)
    model=build_net(arch='entity',d_model=32,n_layers=1,n_heads=4,dropout=0).eval()
    f=torch.randn(3,3227); c=torch.randint(model.n_cards,(3,124))
    with torch.no_grad():before=tuple(x.clone() for x in model(f,c))
    enable_mechanisms(model)
    extra=torch.randn(3,M.N_EXTRA)
    extended=torch.cat([f,extra],dim=1)
    with torch.no_grad():after=model(extended,c)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    model(extended,c)[0].square().mean().backward()
    assert model.mechanism_counter.weight.grad.abs().sum()>0
    assert model.mechanism_global.weight.grad.abs().sum()>0
    path=tmp_path/'model.pt';save_checkpoint(model,str(path))
    loaded,meta=load_checkpoint(str(path))
    assert meta['mechanism_schema_sha256']==M.SCHEMA_SHA256
    with torch.no_grad():restored=loaded(extended,c)
    assert all(torch.equal(a,b) for a,b in zip(after,restored))
    with pytest.raises(ValueError,match='features'):loaded(f,c)
    payload=torch.load(path,weights_only=True)
    payload['meta']['mechanism_schema_sha256']='invalid'
    torch.save(payload,path)
    with pytest.raises(ValueError,match='schema identity'):load_checkpoint(str(path))


def test_extended_samples_survive_both_ppo_batch_paths():
    records=[]
    for i in range(3):
        f=np.zeros(M.OBS_FLOATS,np.float32);f[-1]=i+1
        records.append(dict(features=f,cards=np.zeros(124,np.int16),mask=np.ones(245,bool),
            action=0,old_logprob=-1.,value=0.,seat=0,game_id=9,rank=1,turn=i+1))
    a=build_batch_from_records(records)
    b=build_batch_from_arrays(pack_records_arrays(records))
    assert not a.validate() and not b.validate()
    assert np.array_equal(a.features,b.features)
    assert a.features[:,-1].tolist()==[1,2,3]


def test_unknown_schema_and_nonfinite_fail_closed():
    game=fixture()
    with pytest.raises(ValueError,match='unknown'):M.observe(game,0,'new_unknown')
    player=json.loads(game.player_json(0));player['stats']['gold_spent_turn']=float('nan')
    with pytest.raises(ValueError,match='non-finite'):M.encode_player(player)


def native_target_source_pair():
    import hsbg_engine as E
    from tests.test_economic_tasks import natural_source
    from workflows.probe_whole_turn_economy import fixture as whole_fixture
    from hsbg.az.train.economic_tasks import debug
    state,_=whole_fixture(natural_source(),'normal');game=E.Game.from_json(json.dumps(state))
    debug(game,'clear_zone',zone='hand')
    debug(game,'add_spell',zone='hand',card='BG28_698')
    debug(game,'add_spell',zone='hand',card='BG30_804')
    initial=game.state_json();a=E.Game.from_json(initial);b=E.Game.from_json(initial)
    a.step(0,46);b.step(0,54)
    return a,b


def test_native_target_source_alias_is_ignored_by_legacy_entity():
    a,b=native_target_source_pair();fa,ca=a.observe(0);fb,cb=b.observe(0)
    assert fa==fb and bytes(a.legal_mask(0))==bytes(b.legal_mask(0))
    assert np.flatnonzero(np.frombuffer(ca,dtype='<i4')!=np.frombuffer(cb,dtype='<i4')).tolist()==[4]
    torch.set_num_threads(1)
    model=build_net(arch='entity',d_model=16,n_layers=1,n_heads=2).eval()
    f=torch.tensor(np.stack([np.frombuffer(fa,dtype='<f4'),np.frombuffer(fb,dtype='<f4')]))
    c=torch.tensor(np.stack([np.frombuffer(ca,dtype='<i4'),np.frombuffer(cb,dtype='<i4')]),dtype=torch.long)
    with torch.no_grad():outputs=model(f,c)
    assert all(torch.equal(o[0],o[1]) for o in outputs)
