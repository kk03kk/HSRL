"""新版本卡表必须同时迁移策略与anchor，完整训练状态不能被重置。"""
from types import SimpleNamespace

import numpy as np
import torch


def test_migration_changes_only_main_and_anchor_card_buffers(tmp_path, monkeypatch):
    from hsbg.az.train import entity_model
    from workflows.migrate_checkpoint_3702 import migrate
    static = np.ones((3,2), dtype=np.float32)
    text = np.full((3,4), 2, dtype=np.float32)
    monkeypatch.setattr(entity_model, 'load_card_tables',
                        lambda: SimpleNamespace(static=static, text=text))
    state = {'card_static':torch.zeros(3,2), 'card_text':torch.zeros(3,4),
             'weight':torch.tensor([1.23])}
    anchor = {k:v.clone() for k,v in state.items()}
    anchor['weight'] += 1
    full = dict(model=state, anchor_model=anchor, updates=4159,
                optimizer={'state':{0:{'step':torch.tensor(123), 'exp_avg':torch.tensor([.3])}}},
                rng=torch.get_rng_state(), scheduler={'last_epoch':4159})
    model, trainer = tmp_path/'model.pt', tmp_path/'trainer.pt'
    torch.save({'state_dict':state},model)
    torch.save(full,trainer)
    result = migrate(model,tmp_path/'migrated',trainer)
    saved = torch.load(tmp_path/'migrated/trainer.pt',weights_only=False)
    for name in ('model','anchor_model'):
        assert torch.equal(saved[name]['card_static'],torch.from_numpy(static))
        assert torch.equal(saved[name]['card_text'],torch.from_numpy(text))
        assert torch.equal(saved[name]['weight'],full[name]['weight'])
    assert saved['updates']==4159 and saved['scheduler']==full['scheduler']
    assert torch.equal(saved['rng'],full['rng'])
    for key,value in full['optimizer']['state'][0].items():
        assert torch.equal(saved['optimizer']['state'][0][key],value)
    assert result['inputs']['trainer.pt'] != result['outputs']['trainer.pt']
    assert torch.equal(torch.load(trainer,weights_only=False)['model']['card_static'],state['card_static'])
