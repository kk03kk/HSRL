import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import taverngym as tg

from taverngym.offline.actions import decode, encode
from taverngym.offline.bridge import (BridgeBlocked, ClientSnapshot, ExecutionReceipt,
                                     MemoryLedger, bind_action, encode_snapshot)
from taverngym.offline.capture import capture
from taverngym.offline.evaluate import compare, import_metrics
from taverngym.offline.sandbox import SimulatorExecutor, decide, snapshot
from taverngym.offline.trace import read, replay, statistics, validate
from taverngym.rl.config import Config, identity
from taverngym.rl.model import Policy


@pytest.fixture
def cfg():
    return Config(device="cpu", n_games=1, learner_seats=1, threads=1, torch_threads=1,
                  dim=16, hidden=16, layers=1, heads=2, max_ticks=12000)


@pytest.fixture
def world(cfg):
    env = tg.BatchEnv(1, seed=73001, controlled=range(8), config=cfg.rules)
    env.auto_reset = False
    env.step(env.bot_actions())
    return env


def test_all_actions_roundtrip():
    assert len({json.dumps(decode(a), sort_keys=True) for a in range(245)}) == 245
    for a in range(245):
        assert encode(decode(a)) == a
    for a in (-1, 245, True, 1.5):
        with pytest.raises(ValueError):
            decode(a)


def test_trace_reproduces_native_and_does_not_change_rng(cfg, tmp_path):
    path = tmp_path / "bot.jsonl"
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    plain = capture(cfg, seed=73001)
    logged = capture(cfg, seed=73001, path=path)
    assert torch.equal(before, torch.get_rng_state())
    for k in ("placements", "ticks", "applied", "deferred", "status"):
        assert plain[k] == logged[k]
    rows = read(path)
    assert replay(rows)["state_hashes_verified"]
    report = statistics(rows)
    assert report["effective_decisions"] > 0 and report["rejected"] == 0
    assert report["mean_entropy"] is None
    assert "board" not in rows[1]["decisions"][0]["before"]["opponents"][0]
    bad = copy.deepcopy(rows)
    bad[1]["actions"][0] = 1000
    with pytest.raises(ValueError):
        validate(bad)
    with pytest.raises(ValueError):
        validate(rows[:-1])
    bad = copy.deepcopy(rows)
    bad[1]["post_hashes"][0] = "wrong"
    with pytest.raises(ValueError, match="state mismatch"):
        replay(bad)


def test_neural_capture_and_memory_are_replayable(cfg, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = Policy(cfg)
    path = tmp_path / "model.jsonl"
    plain = capture(cfg, seed=73003, actor=model, greedy=False)
    logged = capture(cfg, seed=73003, actor=model, greedy=False, path=path)
    assert plain["status"] == logged["status"] == "completed"
    assert plain["placements"] == logged["placements"]
    rows = read(path)
    assert replay(rows)["state_hashes_verified"]
    d = rows[1]["decisions"][0]
    assert d["topk"] and d["entropy"] is not None


def test_native_projection_matches_every_tensor_and_rejects_missing(world, cfg):
    ident = identity(cfg)
    s = snapshot(world, 0, ident, 0)
    projected = encode_snapshot(s, ident, allowed_builds=[tg.game_version()], allow_fixture=True).observation
    native = world.observe()
    np.testing.assert_array_equal(projected["features"], native["features"][0])
    np.testing.assert_array_equal(projected["cards"], native["cards"][0])
    np.testing.assert_array_equal(projected["mask"], native["action_mask"][0])
    assert projected["extras"] == [(c, o, v) for c, o, v in world.training_extras()[0]]
    with pytest.raises(BridgeBlocked, match="fixture"):
        encode_snapshot(s, ident, allowed_builds=[tg.game_version()])
    with pytest.raises(BridgeBlocked, match="build"):
        encode_snapshot(s, ident, allowed_builds=["not-this-build"], allow_fixture=True)
    for key in ("global", "opponents", "pending"):
        data = copy.deepcopy(s.data)
        del data["policy_context"]["features"][key]
        with pytest.raises(BridgeBlocked):
            encode_snapshot(ClientSnapshot.from_dict(data), ident, allowed_builds=[tg.game_version()], allow_fixture=True)
    data = copy.deepcopy(s.data)
    data["policy_context"]["cards"][0] = "UNRECOGNIZED_CARD"
    with pytest.raises(BridgeBlocked, match="unknown card"):
        encode_snapshot(ClientSnapshot.from_dict(data), ident, allowed_builds=[tg.game_version()], allow_fixture=True)
    data = copy.deepcopy(s.data)
    data["policy_context"]["extras"] = []
    with pytest.raises(BridgeBlocked, match="budget"):
        encode_snapshot(ClientSnapshot.from_dict(data), ident, allowed_builds=[tg.game_version()], allow_fixture=True)


def test_real_historical_client_sample_is_not_falsely_supported(cfg):
    cases = json.loads(Path("tests/fixtures/offline/legacy-client.json").read_text(encoding="utf-8"))
    assert cases
    for case in cases:
        s = ClientSnapshot.from_dict(case["observation"])
        with pytest.raises(BridgeBlocked, match="policy_context"):
            encode_snapshot(s, identity(cfg), allowed_builds=[s.build])


def test_receipts_commit_once_and_unknown_does_not_advance(world, cfg):
    s = snapshot(world, 0, identity(cfg), 2)
    action = next(x["flat_action"] for x in s.data["action_bindings"])
    intent = bind_action(s, action)
    ledger = MemoryLedger()
    ledger.begin(s, intent, "new hidden")
    for status in ("unknown", "waiting"):
        assert not ledger.confirm(ExecutionReceipt(intent.intent_id, s.game_id, 2, 2, status))
        assert ledger.hidden is None and ledger.count == 0
    with pytest.raises(BridgeBlocked):
        ledger.confirm(ExecutionReceipt("wrong", s.game_id, 2, 3, "applied"))
    receipt = ExecutionReceipt(intent.intent_id, s.game_id, 2, 3, "applied")
    assert ledger.confirm(receipt)
    assert not ledger.confirm(receipt)
    assert ledger.hidden == "new hidden" and ledger.count == 1
    with pytest.raises(BridgeBlocked):
        ledger.begin(s, intent, "bad")


def test_simulator_model_intent_and_receipt_loop(world, cfg):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    actor = Policy(cfg)
    ledger = MemoryLedger()
    executor = SimulatorExecutor(world, 0, identity(cfg))
    for _ in range(8):
        s = executor.observe()
        if not s.data["ready"]:
            break
        intent = decide(s, actor, cfg, ledger, identity(cfg), allowed_builds=[tg.game_version()], fixture=True)
        receipt = executor.execute(intent)
        assert ledger.confirm(receipt) == (receipt.status == "applied")
        assert executor.execute(intent) == receipt
    assert ledger.count > 0


def test_disappeared_entity_is_blocked(world, cfg):
    s = snapshot(world, 0, identity(cfg), 1)
    buys = [a for a in s.data["action_bindings"] if a["semantic"]["kind"] == "buy"]
    assert buys
    data = copy.deepcopy(s.data)
    data["shop"] = []
    with pytest.raises(BridgeBlocked):
        bind_action(ClientSnapshot.from_dict(data), buys[0]["flat_action"])


def test_paired_same_policy_and_log_import(cfg, tmp_path):
    report = compare(cfg, {"bot": None, "same_bot": None}, [73001, 73002], out=tmp_path / "compare.json", modes=("greedy",))
    pair = report["paired_rank_differences"]["bot/greedy minus same_bot/greedy"]
    assert pair["mean"] == 0 and pair["ci95"] == [0, 0]
    p = tmp_path / "metrics.jsonl"
    p.write_text('noise\n{"phase":"validation","iteration":0}\n{bad\n', encoding="utf-8")
    imported = import_metrics(p)
    assert imported["phases"]["validation"][0]["iteration"] == 0
    assert imported["invalid_json_lines"] == [3]
    from taverngym.offline.evaluate import metrics_html
    assert "未记录" in metrics_html(imported)


def test_legacy_action_translation_preserves_slots_targets_and_multistage_blocks():
    from taverngym.offline.client_actions import translate
    data = {"board": [{"entity_id": 1}, {"entity_id": 2}],
            "hand": [{"entity_id": 10, "card_type": "MINION"}],
            "shop": [{"entity_id": 20}], "choices": [{"entity_id": 30}],
            "controls": [{"entity_id": 40, "card_type": "HERO_POWER"}],
            "actions": [{"kind": "buy", "source": 20}, {"kind": "play", "source": 10, "position": 2},
                        {"kind": "play", "source": 10, "target": 2, "interaction": {"placement": "magnetize"}},
                        {"kind": "target", "target": 30, "choice_id": 5},
                        {"kind": "reorder", "order": [2, 1]}, {"kind": "hero_power", "source": 40},
                        {"kind": "choose", "source": 30, "choice_id": 6, "selected": [30]}]}
    bindings, rejected = translate(data)
    assert {a["flat_action"] for a in bindings} == {8, 47, 127, 30, 197, 203, 4}
    assert len(rejected) == 1 and "multi" in rejected[0]["reason"]


@pytest.mark.parametrize("status", ["not_sent", "rejected"])
def test_unsuccessful_receipt_never_advances_memory(world, cfg, status):
    s = snapshot(world, 0, identity(cfg), 0)
    a = s.data["action_bindings"][0]["flat_action"]
    ledger = MemoryLedger()
    intent = bind_action(s, a)
    ledger.begin(s, intent, "forbidden")
    assert not ledger.confirm(ExecutionReceipt(intent.intent_id, s.game_id, 0, 1, status))
    assert ledger.hidden is None and ledger.count == 0 and ledger.pending is None


def test_import_failure_preserves_missing_probabilities_and_native_identity(cfg, tmp_path):
    from taverngym.offline.trace import import_failure
    env = tg.BatchEnv(1, seed=73001, controlled=range(8), config=cfg.rules)
    env.auto_reset = False
    initial = json.loads(env.save(0))
    actions = env.bot_actions().tolist()
    env.step(actions)
    payload = {"format": 2, "native": tg.build_info(), "initial": initial,
               "actions": [actions], "post": json.loads(env.save(0))}
    p, out = tmp_path / "failure.json", tmp_path / "converted.jsonl"
    p.write_text(json.dumps(payload), encoding="utf-8")
    import_failure(p, out)
    rows = read(out)
    assert all(d["topk"] is None for d in rows[1]["decisions"])
    assert replay(rows)["state_hashes_verified"]
    payload["native"]["source_hash"] = "wrong"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="original native"):
        import_failure(p, tmp_path / "bad.jsonl")


def test_new_game_memory_is_reset_and_context_contradictions_block(world, cfg):
    s = snapshot(world, 0, identity(cfg), 0)
    bad = copy.deepcopy(s.data)
    bad["gold"] += 1
    with pytest.raises(BridgeBlocked, match="gold"):
        encode_snapshot(ClientSnapshot.from_dict(bad), identity(cfg), allowed_builds=[tg.game_version()], allow_fixture=True)
    model = Policy(cfg)
    ledger = MemoryLedger(hidden=torch.full((1, cfg.hidden), float("nan")), game_id="old-game", count=10)
    intent = decide(s, model, cfg, ledger, identity(cfg), allowed_builds=[tg.game_version()], fixture=True)
    assert ledger.game_id == s.game_id and ledger.count == 0
    ledger.confirm(ExecutionReceipt(intent.intent_id, s.game_id, 0, 1, "applied"))
    assert torch.isfinite(ledger.hidden).all()


def test_public_history_deduplicates_and_never_invents_missing_counters():
    from taverngym.offline.history import PublicHistory
    h = PublicHistory()
    assert h.values()["actions_this_turn"] is None
    h.apply({"type": "game_start", "game_id": "g", "revision": 0})
    h.apply({"type": "turn_start", "game_id": "g", "revision": 1, "turn": 1})
    receipt = {"type": "receipt", "game_id": "g", "revision": 2, "turn": 1,
               "intent_id": "a", "status": "applied", "kind": "freeze"}
    h.apply(receipt)
    h.apply(receipt)
    assert h.values()["actions_this_turn"] == h.values()["freezes_this_turn"] == 1
    h.apply({"type": "public_counter", "game_id": "g", "revision": 2, "name": "battlecries_triggered", "value": 4})
    h.apply({**receipt, "intent_id": "b", "revision": 3, "kind": "play"})
    assert "battlecries_triggered" not in h.values()["counters"]
    h.apply({"type": "turn_start", "game_id": "g", "revision": 4, "turn": 2})
    assert h.values()["actions_this_turn"] == 0
    h.apply({"type": "gap", "game_id": "g", "revision": 5})
    assert h.values()["actions_this_turn"] is None
    with pytest.raises(BridgeBlocked):
        h.apply({**receipt, "intent_id": "c"})
