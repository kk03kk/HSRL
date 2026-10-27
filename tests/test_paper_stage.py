from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import taverngym as tg

from taverngym.rl.config import Config, identity
from taverngym.rl.model import Policy, pack
from taverngym.rl.collect import collect, observation
from taverngym.rl.train import prefix_hidden, update, train, load_policy


def config(kind="compact_ff"):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    return Config(device="cpu", policy_kind=kind, n_games=1, learner_seats=1, threads=1,
                  torch_threads=1, dim=16, layers=1, heads=2, hidden=16, seq_len=8,
                  chunks_per_batch=2, bc_waves=1, bc_epochs=1, epochs=1, iterations=1,
                  eval_every=1, eval_games=1, eval_seeds=[900000000], opponent_mode="bots")


@pytest.mark.parametrize("kind", ["legacy_gru", "compact_gru", "compact_ff"])
def test_policy_contract_privacy_and_backward(kind):
    c = config(kind)
    model = Policy(c)
    env = tg.BatchEnv(1, controlled=range(8), config=c.rules)
    obs = observation(env.observe(), env.training_extras(), 0)
    tensors = pack([obs], "cpu")
    logits, value, rank, hidden = model(tensors, torch.zeros(1, c.hidden))
    assert logits.shape == (1, 245) and value.shape == (1,) and rank.shape == (1, 8)
    assert hidden.shape == (1, c.hidden) and torch.isfinite(logits).all()
    assert torch.all(logits[~tensors["mask"]] == -1e9)
    (logits[tensors["mask"]].mean() + value.mean() + rank.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    if kind.startswith("compact"):
        assert model.token_count == 49 and model.owner_map.min() == 0 and model.owner_map.max() == 48
        assert model.static_cards.shape[0] == tg.N_CARDS and model.rule_text.shape == (tg.N_CARDS, 64)
        assert torch.count_nonzero(model.rule_text[0]) == 0
        assert int(model.owner_map[-1]) == 48


def test_feedforward_ignores_hidden_and_prefix_does_no_encode():
    c = config()
    m = Policy(c).eval()
    e = tg.BatchEnv(1, controlled=range(8), config=c.rules)
    o = observation(e.observe(), e.training_extras(), 0)
    x = pack([o], "cpu")
    assert torch.equal(m(x, torch.zeros(1, c.hidden))[0], m(x, torch.ones(1, c.hidden) * 10)[0])
    def forbidden(*a, **k):
        raise AssertionError("feedforward prefix must not re-encode")
    m.encode = forbidden
    assert prefix_hidden(m, [(None, 32)], "cpu").shape == (1, 1, c.hidden)


def test_compact_effects_are_pooled_into_the_right_owner():
    c = config()
    m = Policy(c).eval()
    e = tg.BatchEnv(1, controlled=range(8), config=c.rules)
    o = observation(e.observe(), e.training_extras(), 0)
    mutated = copy.deepcopy(o)
    mutated["extras"][0] = (0, 0, [0., 10., 60., 0., 0., 0., 0., 0.])
    a, _ = m.encode(pack([o], "cpu"))
    b, _ = m.encode(pack([mutated], "cpu"))
    assert not torch.equal(a, b)


def test_stage_anchor_is_frozen_and_kl_measured():
    c = config()
    c.anchor_coef = .1
    tr, _ = collect(c, None, 73001, teacher=True)
    m = Policy(c)
    anchor = copy.deepcopy(m).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in anchor.state_dict().items()}
    update(m, torch.optim.Adam(m.parameters(), lr=1e-4), tr, c, bc=True)
    fresh, _ = collect(c, m, 73002)
    metrics = update(m, torch.optim.Adam(m.parameters(), lr=1e-4), fresh, c, anchor=anchor)
    assert metrics["anchor_kl"] >= -1e-6
    assert metrics["updates"] > 0
    assert all(torch.equal(v, anchor.state_dict()[k]) for k, v in before.items())


def test_budget_counts_unique_ppo_samples_and_rejects_architecture_mix(tmp_path):
    c = config()
    c.iterations = 3
    c.decision_budget = 1
    train(c, tmp_path / "run")
    s = torch.load(tmp_path / "run/latest.pt", map_location="cpu", weights_only=True)
    assert s["progress"]["iteration"] == 1
    assert s["progress"]["ppo_decisions"] > 1 and s["progress"]["bc_decisions"] > 0
    wrong = replace(c, policy_kind="compact_gru")
    with pytest.raises(ValueError, match="mismatch"):
        load_policy(tmp_path / "run/latest.pt", wrong, identity(wrong))


@pytest.mark.parametrize("profile", ["native", "tempo", "economy", "mixed"])
def test_script_profiles_are_legal_and_naturally_finish(profile):
    c = config()
    c.bot_profile = profile
    _, stats = collect(c, None, 73001, record=False, allow_truncated=True)
    assert stats["truncated_games"] == 0
    assert stats["bot_profile"] == profile and 1 <= stats["ranks"][0] <= 8


def test_bank_roots_are_separated_and_synthetic_provenance_cannot_be_hidden(tmp_path):
    from taverngym.research.bank import export_natural, summarize, check_splits
    train_bank = export_natural(tmp_path / "train.json", seed=73001, games=1, split="train")
    test_bank = export_natural(tmp_path / "test.json", seed=73002, games=1, split="test")
    report = summarize(train_bank)
    assert report["records"] > 0 and report["human_calibrated"] is False
    assert train_bank["training_import_allowed"] is False and train_bank["policy_input"] is False
    assert check_splits([train_bank, test_bank])["root_disjoint"]
    leaked = copy.deepcopy(train_bank)
    leaked["split"] = "test"
    with pytest.raises(ValueError, match="leaked"):
        check_splits([train_bank, leaked])


def test_client_bank_is_anonymized_and_cannot_be_mixed_as_native_training(tmp_path):
    from taverngym.research.bank import import_client, summarize
    record = json.loads(Path("tests/fixtures/offline/legacy-client.json").read_text(encoding="utf-8"))[0]["observation"]
    # 原真实样例处于第一回合买牌前；用其可见商品构造本单元测试的非空战团。
    # 真实文件的批量导入另有运行产物验证，不伪装此构造为实机证据。
    record["board"] = [copy.deepcopy(next(c for c in record["shop"] if c.get("attack") is not None and c.get("health") is not None))]
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"observation": record}, ensure_ascii=False) + "\n", encoding="utf-8")
    bank = import_client(path, tmp_path / "bank.json")
    assert bank["records"] and bank["training_import_allowed"] is False
    assert bank["records"][0]["root"] != record["game_id"]
    assert "game_id" not in bank["records"][0]
    assert summarize(bank)["source"] == "historical_automated_agent_client"


def test_client_bank_skips_broken_or_unknown_fields_without_inventing_values(tmp_path):
    from taverngym.research.bank import import_client
    board = [{"card_id": "UNMAPPED_CARD", "attack": 1, "health": 2}]
    base = {"phase": "recruit", "game_id": "private-id", "build": "unmapped", "board": board}
    rows = [[], {"sample": []}, {"observation": []}, base,
            {**base, "turn": 3, "board": [{**board[0], "health": float("nan")}]},
            {**base, "turn": 3}]
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    bank = import_client(path, tmp_path / "bank.json")
    assert bank["malformed_lines"] == 3 and bank["unknown_board_fields_skipped"] == 2
    assert len(bank["records"]) == 1 and not bank["training_import_allowed"]


def test_compact_policy_does_not_read_current_hidden_opponent_shop():
    c = config()
    env = tg.BatchEnv(1, seed=73001, controlled=range(8), config=c.rules)
    env.auto_reset = False
    env.step(env.bot_actions())
    before = observation(env.observe(), env.training_extras(), 0)
    saved = json.loads(env.save(0))
    assert saved["game"]["players"][1]["shop"]
    saved["game"]["players"][1]["shop"] = []
    env.load(0, json.dumps(saved))
    after = observation(env.observe(), env.training_extras(), 0)
    for key in ("features", "cards", "mask"):
        np.testing.assert_array_equal(before[key], after[key])
    assert before["extras"] == after["extras"]
    model = Policy(c).eval()
    with torch.no_grad():
        a = model(pack([before], "cpu"), torch.zeros(1, c.hidden))[0]
        b = model(pack([after], "cpu"), torch.zeros(1, c.hidden))[0]
    assert torch.equal(a, b)
