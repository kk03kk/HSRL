import pytest
import torch

from hsbg.az.train.process_learning import process_loss,log_policy,forward_kl


def test_set_loss_is_probability_mass_and_has_finite_gradient():
    logits=torch.zeros((2,4),requires_grad=True)
    masks=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool)
    targets=torch.tensor([[1,1,0,0],[0,1,0,0]],dtype=torch.bool)
    loss=process_loss(logits,masks,targets)
    expected=-(torch.tensor(2/3).log()+torch.tensor(.5).log())/2
    assert torch.allclose(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,3]==0 and logits.grad[0,2]>0 and logits.grad[0,0]<0
    lp=log_policy(logits,masks)
    assert forward_kl(lp,lp).item()==0


def test_invalid_or_illegal_label_fails_closed():
    x=torch.zeros((1,3));mask=torch.tensor([[1,0,0]],dtype=torch.bool)
    with pytest.raises(ValueError):process_loss(x,mask,~mask)
    with pytest.raises(ValueError):process_loss(x,mask,torch.zeros_like(mask))


def test_short_process_run_publishes_checks_and_reload(tmp_path,monkeypatch):
    import argparse,gzip,json
    import numpy as np
    from hsbg.az.train.model import build_net,save_checkpoint
    from hsbg.runtime.artifacts import file_identity,atomic_json
    from tests.test_economic_tasks import natural_source
    from workflows.build_economic_dataset import build_task
    from workflows import learn_economic_process as learner
    torch.set_num_threads(1)
    bank=tmp_path/'data';bank.mkdir()
    source=natural_source()
    tasks=[build_task((dict(state=source,seat=0,seed=seed,turn=8),'buy_privateer',2,0))
           for seed in (11800000,11800001)]
    assert all(t['accepted'] for t in tasks)
    rows=[r for t in tasks for r in t.pop('rows')]
    f,c,m,t,seeds,kinds=zip(*rows)
    np.savez_compressed(bank/'labels.npz',features=f,cards=c,masks=m,targets=t,
                        seeds=seeds,dev=np.array([s%5==0 for s in seeds]),kinds=kinds)
    np.savez_compressed(bank/'anchors.npz',features=f,cards=c,masks=m)
    with gzip.open(bank/'tasks.jsonl.gz','wt') as stream:
        for task in tasks:
            task['split']='dev' if task['seed']%5==0 else 'train'
            stream.write(json.dumps(task)+'\n')
    atomic_json(bank/'manifest.json',{'files':{n:file_identity(bank/n)
        for n in ('labels.npz','anchors.npz','tasks.jsonl.gz')}})
    model=build_net(arch='entity',d_model=32,n_layers=1,n_heads=4)
    parent=tmp_path/'parent.pt';save_checkpoint(model,str(parent))
    monkeypatch.setattr(learner,'source_identity',lambda path:{'tracked_dirty':False})
    args=argparse.Namespace(checkpoint=str(parent),dataset=str(bank),out=str(tmp_path/'out'),
        device='cpu',updates=2,lr=3e-6,anchor_kl=3.,batch=4,seed=1181802)
    learner.run(args)
    result=json.loads((tmp_path/'out/result.json').read_text())
    assert result['roundtrip_exact']
    assert len(list((tmp_path/'out/checks').glob('*.json')))==2
    assert (tmp_path/'out/bc_state.pt').exists()


def test_full_board_probe_explicitly_preserves_original_and_credits_disposable_value():
    from tests.test_economic_tasks import natural_source
    from workflows.build_economic_dataset import build_task
    from tools.probe_full_board_process import fill_board
    task=build_task((dict(state=natural_source(),seat=0,seed=11800000,turn=8),'buy_privateer',2,0))
    full=fill_board(task)
    assert len(full['initial']['players'][0]['board'])==7
    assert len(full['protected_eids'])==len(task['initial']['players'][0]['board'])
    assert full['objective']==task['objective']+len(full['liquid_eids'])
    assert not set(full['protected_eids']) & set(full['liquid_eids'])
