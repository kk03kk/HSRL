"""真实 Adam 续训设备回归，CUDA 版由同节点 A/B 作业内执行。"""
import pytest
import torch

from hsbg.az.train.model import build_net
from hsbg.az.train.ppo import PPOConfig
from hsbg.az.train.ppo_trainer import PPOTrainer
from tests.test_az_ppo import _batch_from_model


def test_restore_maps_checkpoint_to_host(monkeypatch, tmp_path):
    model = build_net(16)
    trainer = PPOTrainer(model, PPOConfig(device="cpu"))
    path = trainer.save(str(tmp_path / "trainer.pt"))
    load = torch.load
    locations = []

    def observe(*args, **kwargs):
        locations.append(kwargs["map_location"])
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe)
    # 记录反序列化目标即可，构造过程另由 CUDA 实测覆盖。
    monkeypatch.setattr(PPOTrainer, "__init__", lambda self, *args, **kwargs: None)
    PPOTrainer.load(build_net(16), path, device="cuda:0")
    assert locations == ["cpu"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA Adam设备回归")
def test_cuda_restore_adam_counters_and_fixed_batch_update(tmp_path):
    model = build_net(16)
    batch = _batch_from_model(model, n_games=1, decisions_per_seat=2)
    original = PPOTrainer(model, PPOConfig(device="cpu", epochs=1, minibatch=8, anchor_coef=.02))
    original.set_anchor()
    original.update(batch)
    path = original.save(str(tmp_path / "trainer.pt"))
    snapshots = []
    for legacy, expected in ((True, "cuda"), (False, "cpu")):
        cfg = PPOConfig(device="cuda:0", epochs=1, minibatch=8, anchor_coef=.02, amp="bf16")
        trainer = PPOTrainer.load(build_net(16), path, config=cfg, device="cuda:0",
                                  _benchmark_legacy_device_load=legacy)
        assert trainer.optimizer.state
        for state in trainer.optimizer.state.values():
            assert state["step"].device.type == expected
            assert state["exp_avg"].device.type == state["exp_avg_sq"].device.type == "cuda"
        stats = trainer.update(batch)
        snapshots.append((stats, trainer))
    (old, baseline), (new, fixed) = snapshots
    assert old["updates"] == new["updates"] > 0
    assert old["processed_samples"] == new["processed_samples"]
    assert old["stopped_early"] == new["stopped_early"]
    for name, tensor in baseline.model.state_dict().items():
        assert torch.equal(tensor, fixed.model.state_dict()[name])
    a, b = baseline.optimizer.state_dict()["state"], fixed.optimizer.state_dict()["state"]
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].keys() == b[key].keys()
        for field in a[key]:
            assert torch.equal(a[key][field].cpu(), b[key][field].cpu())
