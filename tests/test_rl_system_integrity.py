"""数据损坏不能进入 PPO；分段保存/恢复必须保留完整学习状态。"""
import copy
import numpy as np
import pytest


@pytest.mark.parametrize('field,value', [('advantages',np.nan),('returns',np.inf),('dones',.5)])
def test_invalid_targets_are_rejected_before_any_optimizer_step(field,value):
    import torch
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo import PPOConfig
    from hsbg.az.train.ppo_trainer import PPOTrainer
    from tests.test_az_ppo import _batch_from_model
    model=build_net(16)
    trainer=PPOTrainer(model,PPOConfig(device='cpu',epochs=1,minibatch=64))
    batch=_batch_from_model(model,n_games=1)
    getattr(batch,field)[0]=value
    before=copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError,match='PPO 批次校验失败'):
        trainer.update(batch)
    assert trainer.updates==0 and not trainer.optimizer.state
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_batch_rejects_shape_damage_and_conflicting_terminal_ranks():
    from hsbg.az.train.ppo import build_batch_from_records
    from tests.test_az_ppo import _synthetic_records
    batch=build_batch_from_records(_synthetic_records(n_games=1))
    good=copy.deepcopy(batch)
    batch.masks=batch.masks[:1]
    assert any('shape' in x for x in batch.validate())
    good.ranks[0]=8 if good.ranks[0]!=8 else 1
    assert any('最终名次不一致' in x for x in good.validate())


def test_cpu_trainer_segmented_resume_matches_uninterrupted_update(tmp_path):
    import torch
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo import PPOConfig
    from hsbg.az.train.ppo_trainer import PPOTrainer
    from tests.test_az_ppo import _batch_from_model
    torch.manual_seed(931415)
    config=PPOConfig(device='cpu',epochs=1,minibatch=8,anchor_coef=.1,kl_early_stop=False)
    full=PPOTrainer(build_net(16),config)
    full.set_anchor()
    initial=copy.deepcopy(full.state_dict())
    batch=_batch_from_model(full.model,n_games=1)
    for _ in range(4):full.update(batch)
    expected=copy.deepcopy(full.state_dict())
    part=PPOTrainer(build_net(16),copy.deepcopy(config),state=initial)
    for _ in range(2):part.update(batch)
    checkpoint=tmp_path/'trainer.pt'
    part.save(str(checkpoint))
    resumed=PPOTrainer.load(build_net(16),str(checkpoint),device='cpu')
    for _ in range(2):resumed.update(batch)
    actual=resumed.state_dict()
    for section in ('model','anchor_model'):
        assert all(torch.equal(v,actual[section][k]) for k,v in expected[section].items())
    for parameter,state in expected['optimizer']['state'].items():
        for name,value in state.items():
            assert torch.equal(value,actual['optimizer']['state'][parameter][name])
    assert actual['scheduler']==expected['scheduler']
    assert actual['rng']==expected['rng'] and actual['updates']==expected['updates']==4
    assert actual['python_rng']==expected['python_rng']
    assert torch.equal(actual['torch_rng'],expected['torch_rng'])


@pytest.mark.parametrize('field',['rewards','advantages','returns','dones','behavior_logprobs'])
def test_array_builder_does_not_silently_drop_corrupted_reward_columns(field):
    from hsbg.az.train.ppo import build_batch_from_arrays, pack_records_arrays
    from tests.test_az_ppo import _synthetic_records
    packed=pack_records_arrays(_synthetic_records(n_games=1))
    packed[field]=np.zeros(len(packed['ranks'])-1,dtype=np.float32)
    with pytest.raises(ValueError,match='核心字段长度损坏 '+field):
        build_batch_from_arrays(packed)


@pytest.mark.parametrize('field', ['truncated_combats','truncated_recruit_contexts','turn_limit_end'])
def test_native_engine_guard_diagnostics_cannot_be_silently_accepted(field,tmp_path,monkeypatch):
    import json
    import gzip
    import hsbg_engine as E
    from hsbg.az.train.engine_quality import require_clean_engine,diagnostics
    game=E.Game(seed=93)
    assert not any(require_clean_engine(game).values())
    state=json.loads(game.state_json())
    state[field]=True if field=='turn_limit_end' else 1
    broken=E.Game.from_json(json.dumps(state))
    assert diagnostics(broken)[field]==1
    monkeypatch.setenv('HSRL_FAILURE_DIR',str(tmp_path))
    with pytest.raises(RuntimeError,match='引擎非正常结束'):
        require_clean_engine(broken)
    files=list(tmp_path.glob('engine-quality-*.json.gz'))
    assert len(files)==1
    assert json.load(gzip.open(files[0],'rt'))['state'][field]==state[field]


def test_engineering_audit_rejects_corrupt_adam_moments_and_reset_steps(tmp_path):
    import torch
    from workflows.audit_3702_recovery import optimizer_audit,check_optimizer_progress
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo import PPOConfig
    from hsbg.az.train.ppo_trainer import PPOTrainer
    from tests.test_az_ppo import _batch_from_model
    trainer=PPOTrainer(build_net(16),PPOConfig(device='cpu',epochs=1,minibatch=64))
    trainer.update(_batch_from_model(trainer.model,n_games=1))
    path=str(tmp_path/'trainer.pt');trainer.save(path)
    before=optimizer_audit(path)
    with pytest.raises(ValueError,match='Adam计数'):
        check_optimizer_progress(before,before)
    next(iter(trainer.optimizer.state.values()))['exp_avg'].flatten()[0]=float('inf')
    trainer.save(path)
    with pytest.raises(ValueError,match='Adam非有限'):
        optimizer_audit(path)
