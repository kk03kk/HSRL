import pytest
import torch
from hsbg.az.train.model import build_net, load_checkpoint, save_checkpoint
from workflows.upgrade_activation_checkpoint import enable_activation_head


def test_independent_activation_preserves_parent_and_can_break_fixed_odds(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11823)
    model = build_net(arch='entity', d_model=32, n_layers=1, n_heads=4, dropout=0).eval()
    f = torch.randn(6, 3227)
    c = torch.randint(model.n_cards, (6, 124))
    before = model(f, c)
    old_difference = before[0][:, 23:30] - before[0][:, 16:23]
    assert torch.allclose(old_difference, old_difference[:1].expand_as(old_difference), atol=1e-5)
    enable_activation_head(model)
    after = model(f, c)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    (-after[0][:, 23].mean()).backward()
    assert model.q_activate_delta.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.q_activate_delta.weight.copy_(torch.eye(32) * .1)
    changed = model(f, c)
    assert torch.equal(changed[0][:, :23], before[0][:, :23])
    assert torch.equal(changed[0][:, 30:], before[0][:, 30:])
    assert torch.equal(changed[1], before[1])
    difference = changed[0][:, 23:30] - changed[0][:, 16:23]
    assert not torch.allclose(difference, difference[:1].expand_as(difference))
    path = tmp_path / 'model.pt'
    save_checkpoint(model, str(path))
    restored, meta = load_checkpoint(str(path))
    assert meta['activate_head_split']
    assert all(torch.equal(a, b) for a, b in zip(changed, restored(f, c)))
    payload = torch.load(path, weights_only=True)
    del payload['meta']['activate_head_split']
    torch.save(payload, path)
    with pytest.raises(ValueError, match='metadata/state mismatch'):
        load_checkpoint(str(path))
