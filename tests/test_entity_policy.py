import numpy as np
import pytest
import taverngym


def test_entity_policy_masks_and_backpropagates():
    torch = pytest.importorskip("torch")
    from taverngym.entity_policy import EntityPolicy

    env = taverngym.BatchEnv(n_games=4, seed=815)
    obs = env.observe()
    model = EntityPolicy(width=32, layers=2, heads=4, feedforward=64)
    inputs = tuple(torch.from_numpy(obs[key]) for key in ("features", "cards", "action_mask"))
    out = model(*inputs)
    assert out.logits.shape == (4, taverngym.N_ACTIONS)
    assert out.value.shape == (4,) and out.placement_logits.shape == (4, 8)
    mask = inputs[-1]
    assert torch.isfinite(out.logits[mask]).all()
    assert torch.isneginf(out.logits[~mask]).all()
    distribution = torch.distributions.Categorical(logits=out.logits)
    actions = distribution.sample()
    assert obs["action_mask"][np.arange(4), actions.numpy()].all()
    loss = (
        -distribution.log_prob(actions).mean()
        + out.value.square().mean()
        + torch.nn.functional.cross_entropy(out.placement_logits, torch.zeros(4, dtype=torch.long))
    )
    loss.backward()
    assert model.card_id.weight.grad is not None
    assert torch.isfinite(model.card_id.weight.grad).all()
    _, _, info = env.step(actions.numpy())
    assert not info["rejected"].any()
