import json

import pytest
import torch

from hsbg.az.train.checkpoint_io import publish_final_checkpoint
from hsbg.az.train.model import build_net
from hsbg.az.train.ppo import PPOConfig
from hsbg.az.train.ppo_trainer import PPOTrainer


def test_final_pointer_names_a_complete_same_generation(tmp_path):
    model = build_net(16)
    trainer = PPOTrainer(model, PPOConfig(device='cpu'))
    trainer.updates = 5
    trainer.extra['cum_decisions'] = 123
    result = publish_final_checkpoint(tmp_path, trainer, model,
                                      meta={'arch':'tiny','trainer_updates':5}, history_len=5)
    latest = json.loads((tmp_path/'latest.json').read_text(encoding='utf-8'))
    assert latest == result and latest['final'] and latest['version'] == 5
    state = torch.load(latest['trainer'], map_location='cpu', weights_only=False)
    weights = torch.load(latest['model'], map_location='cpu', weights_only=False)
    assert state['updates'] == weights['meta']['trainer_updates'] == 5
    for name, value in state['model'].items():
        assert torch.equal(value, weights['state_dict'][name])
    with pytest.raises(FileExistsError):
        publish_final_checkpoint(tmp_path, trainer, model, meta={'arch':'tiny'})


def test_failed_final_generation_keeps_previous_pointer(tmp_path, monkeypatch):
    model = build_net(16)
    trainer = PPOTrainer(model, PPOConfig(device='cpu'))
    (tmp_path/'latest.json').write_text('{"version": -1}')
    monkeypatch.setattr(trainer, 'save', lambda _: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        publish_final_checkpoint(tmp_path, trainer, model, meta={})
    assert json.loads((tmp_path/'latest.json').read_text()) == {'version':-1}
