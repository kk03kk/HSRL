"""Training semantics, privacy, recurrence, probability, checkpoint and collector regression gates."""
from dataclasses import replace
import json

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="training tests require the train extra")

import taverngym as tg
from taverngym.rl.config import Config, identity
from taverngym.rl.collect import collect, gae, observation, InterruptedCollection
from taverngym.rl.model import Policy, pack
from taverngym.rl.train import atomic_save, load_policy, prefix_hidden, rng_state, restore_rng, run_lock, update


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return Config(device="cpu", n_games=1, learner_seats=1, threads=1, torch_threads=1,
                  dim=32, layers=1, heads=2, hidden=32, seq_len=4, chunks_per_batch=2,
                  epochs=1, bc_epochs=1)


def obs():
    e = tg.BatchEnv(1, controlled=range(8))
    return observation(e.observe(), e.training_extras(), 0)


def test_profile_rejects_naga_and_unimplemented_text(cfg):
    with pytest.raises(ValueError, match="Naga"):
        replace(cfg, rules={**cfg.rules, "lobby_types": ["NAGA"]}).validate()
    with pytest.raises(ValueError, match="AllVanilla"):
        replace(cfg, rules={**cfg.rules, "content": "AllVanilla"}).validate()


def test_teacher_query_does_not_consume_rng_or_change_state():
    e = tg.BatchEnv(2, seed=7, controlled=range(8), threads=2)
    before = [e.save(i) for i in range(2)]
    a = e.bot_actions()
    assert [e.save(i) for i in range(2)] == before
    mask = e.observe()["action_mask"]
    assert all(mask[i, a[i]] for i in range(16))


def test_finished_full_lobbies_do_not_autoreset():
    e = tg.BatchEnv(1, seed=91, controlled=range(8))
    e.auto_reset = False
    done = np.zeros(8, bool)
    for _ in range(2000):
        _, d, info = e.step(e.bot_actions())
        assert not np.any(d & done)
        done |= d
        if done.all(): break
    assert done.all() and e.episode_index(0) == 0
    assert not e.observe()["ready"].any()
    before = e.save(0)
    _, d, _ = e.step([0] * 8)
    assert not d.any() and e.save(0) == before


def test_gae_terminal_credit_has_no_waiting_tick_discount():
    samples = [{"value": 0.0} for _ in range(5)]
    adv, ret = gae(samples, 1, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(adv, np.ones(5))
    np.testing.assert_allclose(ret, adv)


def test_card_identity_and_memory_influence_policy(cfg):
    m = Policy(cfg).eval()
    o = obs()
    h = torch.zeros((1, cfg.hidden))
    with torch.no_grad():
        logits, _, _, nh = m(pack([o], "cpu"), h)
        other = {**o, "cards": o["cards"].copy()}
        other["cards"][0] = 1 if other["cards"][0] != 1 else 2
        changed, _, _, _ = m(pack([other], "cpu"), h)
        remembered, _, _, _ = m(pack([o], "cpu"), nh)
    legal = torch.as_tensor(o["mask"])
    assert not torch.allclose(logits[:, legal], changed[:, legal])
    assert not torch.allclose(logits[:, legal], remembered[:, legal])
    a = logits.argmax(-1).item()
    assert o["mask"][a]


def test_prefix_reconstruction_matches_sequential_inference(cfg):
    m = Policy(cfg).eval()
    o = obs()
    trajectory = {"samples": [{"obs": o} for _ in range(3)]}
    h = torch.zeros((1, cfg.hidden))
    with torch.no_grad():
        for _ in range(3): *_, h = m(pack([o], "cpu"), h)
    rebuilt = prefix_hidden(m, [(trajectory, 3)], "cpu")
    torch.testing.assert_close(rebuilt[0], h, atol=2e-6, rtol=2e-5)


def test_ppo_update_has_consistent_old_probability_and_changes_policy(cfg):
    m = Policy(cfg).eval()
    o, samples = obs(), []
    h = torch.zeros((1, cfg.hidden))
    with torch.no_grad():
        for _ in range(3):
            logits, value, _, h = m(pack([o], "cpu"), h)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            samples.append({"obs": o, "action": a.item(), "logp": dist.log_prob(a).item(), "value": value.item()})
    before = m.policy_head[0].weight.detach().clone()
    metrics = update(m, torch.optim.Adam(m.parameters(), lr=1e-3), [{"samples": samples, "rank": 1}], cfg)
    assert metrics["updates"] == 1 and abs(metrics["kl"]) < 1e-5
    assert not torch.equal(before, m.policy_head[0].weight)


def test_checkpoint_roundtrip_rng_and_semantic_rejection(cfg, tmp_path):
    m = Policy(cfg)
    signature = identity(cfg)
    path = tmp_path / "model.pt"
    atomic_save({"model": m.state_dict(), "identity": signature, "config": cfg.to_dict(), "rng": rng_state()}, path)
    loaded = load_policy(path, cfg, signature)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in m.state_dict().items())
    state = rng_state()
    expected = (torch.rand(4), np.random.rand(4))
    restore_rng(state)
    torch.testing.assert_close(torch.rand(4), expected[0])
    np.testing.assert_array_equal(np.random.rand(4), expected[1])
    with pytest.raises(ValueError, match="mismatch"):
        load_policy(path, cfg, {"digest": "wrong-schema"})


def test_exclusive_run_lock(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(OSError):
            with run_lock(tmp_path): pass


def test_complete_episode_collection_and_interrupt(cfg):
    tr, stats = collect(cfg, None, 123, teacher=True)
    assert len(tr) == 1 and tr[0]["samples"] and 1 <= tr[0]["rank"] <= 8
    assert stats["samples"] == len(tr[0]["samples"])
    assert all(s["obs"]["mask"][s["action"]] for s in tr[0]["samples"])
    with pytest.raises(InterruptedCollection):
        collect(cfg, None, 123, teacher=True, stop=lambda: True)
