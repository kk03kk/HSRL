"""1022804：BC后第0波评测的自然回合上限，不能伪造名次或终止全部训练。"""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch

from taverngym.rl.collect import collect
from taverngym.rl.config import Config
from taverngym.rl.model import Policy
from taverngym.rl.train import evaluate, selection_score, train


def config():
    torch.set_num_threads(1)
    return Config(device="cpu", n_games=1, learner_seats=1, threads=1, torch_threads=1,
                  dim=16, layers=1, heads=2, hidden=16, seq_len=8, chunks_per_batch=2,
                  iterations=1, bc_waves=1, bc_epochs=1, epochs=1, eval_every=1,
                  eval_games=1, eval_seeds=[900000000])


def test_training_stays_strict_but_evaluation_excludes_capped_lobby(tmp_path):
    c = config()
    c.rules = {**c.rules, "max_turns": 1}
    with pytest.raises(RuntimeError, match="collection failure"):
        collect(c, None, 73001, teacher=True)
    with pytest.raises(ValueError, match="evaluation-only"):
        collect(c, None, 73001, allow_truncated=True)
    trajectories, stats = collect(c, None, 73001, record=False, allow_truncated=True, artifact_dir=tmp_path)
    assert trajectories == [] and stats["ranks"] == []
    assert stats["truncated_games"] == 1 and stats["games"] == 0
    assert stats["mean_rank"] is None and stats["win"] is None
    snapshots = list(tmp_path.glob("failure-*.json"))
    assert len(snapshots) == 1
    saved = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert saved["diagnostic"][0] and saved["actions"] and saved["initial"]


def test_empty_evaluation_is_json_safe_and_ineligible():
    c = config()
    c.rules = {**c.rules, "max_turns": 1}
    model = Policy(c)
    before = torch.get_rng_state().clone()
    result = evaluate(c, model, anchor=model)
    assert torch.equal(before, torch.get_rng_state())
    assert all(p["truncated_games"] == 1 and p["mean_rank"] is None and p["ranks"] == [] for p in result.values())
    assert selection_score(result) is None
    json.dumps(result, allow_nan=False)


def test_partial_evaluation_does_not_promote_survivor_biased_best():
    result = {"bots_sample": {"mean_rank": 1.0, "eligible_for_selection": False},
              "anchor_mix_sample": {"mean_rank": 2.0, "eligible_for_selection": True}}
    assert selection_score(result) is None
    result["bots_sample"]["eligible_for_selection"] = True
    assert selection_score(result) == 1.5


def test_invalid_validation_does_not_abort_ppo_or_create_best(tmp_path):
    c = config()
    invalid = {"bots_sample": {"mean_rank": None, "eligible_for_selection": False}}
    with patch("taverngym.rl.train.evaluate", return_value=invalid):
        train(c, tmp_path / "run")
    state = torch.load(tmp_path / "run/latest.pt", map_location="cpu", weights_only=True)
    assert state["progress"]["iteration"] == 1 and state["progress"]["last_eval"] == 1
    assert not (tmp_path / "run/best.pt").exists()


def test_one_capped_lobby_does_not_discard_other_natural_results():
    import taverngym as tg
    c = replace(config(), n_games=2)
    _, expected = collect(c, None, 73001, record=False, allow_truncated=True)
    original = tg.BatchEnv
    class FirstLobbyGuard(original):
        def diagnostics(self):
            result = super().diagnostics()
            if json.loads(self.game(0).view(0))["phase"] == "game_over":
                result[0] = (True, 0, None)
            return result
    with patch("taverngym.rl.collect.tg.BatchEnv", FirstLobbyGuard):
        _, stats = collect(c, None, 73001, record=False, allow_truncated=True)
    assert stats["truncated_game_indices"] == [0] and stats["games"] == 1
    assert stats["ranks"] == expected["ranks"][1:]
