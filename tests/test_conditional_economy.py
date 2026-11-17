import copy,json
import numpy as np
import pytest
import torch
from hsbg.az.train.conditional_economy import fork_conditional,current_gate_spec
from hsbg.az.train.model import build_net,save_checkpoint,load_checkpoint
from workflows.learn_cycle_process import configure_parameters
from hsbg.env import obs_std as O


def inputs():
    import hsbg_engine as E
    from tests.test_economic_tasks import natural_source
    from workflows.probe_whole_turn_economy import fixture
    initial=natural_source();endowed,_=fixture(initial,'missing_privateer')
    arrays=[]
    for state in (initial,endowed):
        game=E.Game.from_json(json.dumps(state));f,c=game.observe(0)
        arrays.append((np.frombuffer(f,dtype='<f4').copy(),np.frombuffer(c,dtype='<i4').copy(),
                       np.asarray(list(bytes(game.legal_mask(0))),bool)))
    return tuple(torch.as_tensor(np.stack([a[i] for a in arrays]),
        dtype=torch.float32 if i==0 else torch.long if i==1 else torch.bool) for i in range(3))


def small_parent():
    torch.manual_seed(513);torch.set_num_threads(1)
    model=build_net(arch='entity',d_model=16,n_layers=1,n_heads=2).eval()
    return model,{'arch':'entity','d_model':16,'n_layers':1,'n_heads':2}


def test_gate_uses_distinct_owned_core_types_and_public_lobby():
    base,meta=small_parent();model=fork_conditional(base,meta)
    f=torch.zeros((4,3227));c=torch.zeros((4,124),dtype=torch.long)
    spec=current_gate_spec();f[:,spec['lobby_feature']]=1
    b=O.SLICE_CARD_BOARD.start;h=O.SLICE_CARD_HAND.start;s=O.SLICE_CARD_SHOP.start
    c[:,b]=spec['embedding_ids'][0][0]
    c[0,b+1]=spec['embedding_ids'][0][1]  # two Branns still one type
    c[1,h]=spec['embedding_ids'][1][1]
    c[2,s]=spec['embedding_ids'][1][0]  # shop is not ownership
    c[3,h]=spec['embedding_ids'][2][0];f[3,spec['lobby_feature']]=0
    assert model.gate(f,c).tolist()==[False,True,False,False]


def test_neutral_fork_specialist_gradient_and_exact_fallback_critic(tmp_path):
    base,meta=small_parent();reference=copy.deepcopy(base);model=fork_conditional(base,meta).eval()
    f,c,mask=inputs();assert model.gate(f,c).tolist()==[False,True]
    with torch.no_grad():
        expected=reference(f,c,mask);initial=model(f,c,mask)
    assert all(torch.equal(x,y) for x,y in zip(initial,expected))
    trainable=configure_parameters(model,'all')
    assert trainable and all(n.startswith('specialist.') for n in trainable)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    action=int(torch.flatnonzero(mask[1])[0]) if hasattr(torch,'flatnonzero') else int(mask[1].nonzero()[0])
    loss=-model(f,c,mask)[0][1,action];loss.backward();opt.step()
    assert all(p.grad is None for p in model.base.parameters())
    with torch.no_grad():after=model(f,c,mask)
    assert torch.equal(after[0][0],expected[0][0])
    assert not torch.equal(after[0][1],expected[0][1])
    assert all(torch.equal(a,b) for a,b in zip(after[1:],expected[1:]))
    for key,value in model.base.state_dict().items():assert torch.equal(value,reference.state_dict()[key])
    path=tmp_path/'conditional.pt';save_checkpoint(model,str(path),meta=meta)
    restored,loaded=load_checkpoint(str(path))
    assert loaded['arch']=='conditional_economy_v1'
    with torch.no_grad():roundtrip=restored(f,c,mask)
    assert all(torch.equal(a,b) for a,b in zip(after,roundtrip))
    payload=torch.load(path,weights_only=True)
    payload['meta']['conditional_gate']['embedding_ids'][0][0]+=1
    bad=tmp_path/'wrong-gate.pt';torch.save(payload,bad)
    with pytest.raises(ValueError,match='identity differs'):load_checkpoint(str(bad))
    with pytest.raises(ValueError,match='external normalization'):
        save_checkpoint(model,str(tmp_path/'normalized.pt'),norm_mean=np.zeros(3227))


@pytest.mark.parametrize('section,prefix',[
    ('model','base.'),('model','specialist.'),
    ('anchor_model','base.'),('anchor_model','specialist.')])
def test_trainer_audit_rejects_each_corrupted_component_table(section,prefix):
    from types import SimpleNamespace
    from workflows.audit_full_trainer import audit_card_tables
    tables=SimpleNamespace(static=np.zeros((2,3),dtype=np.float32),
                           text=np.ones((2,4),dtype=np.float32))
    weights={p+n:torch.from_numpy(t.copy()) for p in ('base.','specialist.')
             for n,t in [('card_static',tables.static),('card_text',tables.text)]}
    state={'model':weights,'anchor_model':copy.deepcopy(weights)}
    assert len(audit_card_tables(state,'conditional_economy_v1',tables))==8
    state[section][prefix+'card_static'][0,0]=9
    with pytest.raises(ValueError,match='card table differs'):
        audit_card_tables(state,'conditional_economy_v1',tables)


def test_trainer_audit_still_checks_single_entity_tables():
    from types import SimpleNamespace
    from workflows.audit_full_trainer import audit_card_tables
    tables=SimpleNamespace(static=np.zeros((2,3),dtype=np.float32),
                           text=np.ones((2,4),dtype=np.float32))
    state={'model':{'card_static':torch.from_numpy(tables.static),
                    'card_text':torch.from_numpy(tables.text)},'anchor_model':None}
    assert audit_card_tables(state,'entity',tables)==['model.card_static','model.card_text']
    del state['model']['card_text']
    with pytest.raises(ValueError,match='card table differs'):
        audit_card_tables(state,'entity',tables)


def test_mechanism_conditional_fork_is_neutral_and_can_separate_native_source_alias(tmp_path):
    from hsbg.az.train import mechanism_obs as M
    from tests.test_mechanism_observation import native_target_source_pair
    base,meta=small_parent();model=fork_conditional(base,meta,mechanisms=True).eval()
    assert M.schema_of(base)=='' and model.input_floats==3282
    arrays=[]
    for game in native_target_source_pair():
        f,c=M.observe(game,0,M.SCHEMA)
        arrays.append((np.frombuffer(f,dtype='<f4').copy(),np.frombuffer(c,dtype='<i4').copy(),
                       np.asarray(list(bytes(game.legal_mask(0))),bool)))
    f,c,mask=(torch.as_tensor(np.stack([a[i] for a in arrays]),
        dtype=torch.float32 if i==0 else torch.long if i==1 else torch.bool) for i in range(3))
    assert model.gate(f,c).tolist()==[True,True]
    with torch.no_grad():before=base(f[:,:3227],c,mask);neutral=model(f,c,mask)
    assert all(torch.equal(x,y) for x,y in zip(before,neutral))
    configure_parameters(model,'all')
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    action=int(mask[0].nonzero()[0]);loss=-model(f,c,mask)[0][0,action]
    loss.backward()
    assert model.specialist.mechanism_pending.weight.grad.abs().sum()>0
    opt.step()
    with torch.no_grad():after=model(f,c,mask)
    assert not torch.equal(after[0][0],after[0][1])
    assert all(torch.equal(a,b) for a,b in zip(after[1:],before[1:]))
    assert not model.base.mechanism_pending.weight.any()
    path=tmp_path/'mechanism-conditional.pt';save_checkpoint(model,str(path),meta=meta)
    restored,loaded=load_checkpoint(str(path))
    assert loaded['mechanism_schema_sha256']==M.SCHEMA_SHA256
    with torch.no_grad():roundtrip=restored(f,c,mask)
    assert all(torch.equal(a,b) for a,b in zip(after,roundtrip))
    payload=torch.load(path,weights_only=True);payload['meta']['mechanism_schema_sha256']='wrong'
    torch.save(payload,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='wrapper observation schema'):
        load_checkpoint(str(tmp_path/'bad.pt'))
