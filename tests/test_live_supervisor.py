"""D-phase regressions: batch supervision, unsent reconciliation, gesture guard,
final placement settle. All offline; no game, no window, no input."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client"))

from hsrl_agent.live.contracts import Binding, Decision
from hsrl_agent.live.controller import Controller
from hsrl_agent.live.runtime import settle_placement
from hsrl_agent.live.supervisor import FailureBudget, classify_run_end, signature


def _decision(action=46, kind="play", source=41, turn=3):
    binding = Binding(action, kind, source=source, position=0)
    return Decision("game-1", 7, "stamp", action, binding, "ckpt", "logits")


def _controller(tmp_path, *, practical=True, enable_input=False):
    class Policy:
        sha = "ckpt"
        vocabulary = {}

    class Geometry:
        pass

    class Backend:
        pass

    clock = [100.0]
    c = Controller(Policy(), Geometry(), Backend(), Path(tmp_path),
                   enable_input=enable_input, practical=practical, clock=lambda: clock[0])
    return c, clock


# ---------------------------------------------------------------- supervision

def test_classify_ok_match_with_result(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps(dict(placement="2")), encoding="utf-8")
    incident = classify_run_end(tmp_path, 0, dict(placement="2"))
    assert incident["severity"] == "ok"
    assert incident["action"] == "continue"


def test_classify_unknown_delivery_is_s0(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps(dict(
        reason="input outcome unknown after bounded recovery; automatic retry disabled",
        client=dict(phase="recruit"))), encoding="utf-8")
    incident = classify_run_end(tmp_path, 2, None)
    assert incident["severity"] == "S0"
    assert incident["action"] == "stop_batch"


def test_classify_lobby_block_is_s3(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps(dict(reason="BLOCKED: autoplay stopped before queue")), encoding="utf-8")
    incident = classify_run_end(tmp_path, 2, None)
    assert incident["severity"] == "S3"


def test_classify_queue_failure_before_run_is_s3(tmp_path):
    incident = classify_run_end(tmp_path, 2, None, reason="autoplay stopped before queue")
    assert incident["severity"] == "S3"


def test_result_banner_is_recognised_only_in_completed_context():
    from hsrl_agent.live.lobby import result_banner
    rows = [dict(text="第二名！", confidence=0.95, point=(0.50, 0.55)),
            dict(text="等级", confidence=0.95, point=(0.5, 0.7))]
    assert result_banner(rows) is not None
    assert result_banner([dict(text="酒馆战棋", confidence=0.95, point=(0.42, 0.02))]) is None


def test_result_banner_dismissal_requires_completed_gameplay():
    import pytest
    from hsrl_agent.live.contracts import Blocked
    from hsrl_agent.live.lobby import LobbyFlow
    flow = LobbyFlow()
    rows = [dict(text="第二名！", confidence=0.95, point=(0.50, 0.55))]
    with pytest.raises(Blocked):
        flow.inspect(rows, "tok", 100.0, pending_gameplay=False, completed_gameplay=False)
    with pytest.raises(Blocked):
        flow.inspect(rows, "tok", 100.0, pending_gameplay=True, completed_gameplay=True)
    assert flow.inspect(rows, "tok", 100.0, pending_gameplay=False, completed_gameplay=True) is None
    action = flow.inspect(rows, "tok", 100.4, pending_gameplay=False, completed_gameplay=True)
    assert action is not None and action.kind == "dismiss_result"


def test_classify_internal_fault_is_s1(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps(dict(
        reason="internal TypeError repeated; input paused, durable ledger retained for review",
        client=dict(phase="choice"))), encoding="utf-8")
    incident = classify_run_end(tmp_path, 2, None)
    assert incident["severity"] == "S1"
    assert incident["signature"].startswith("choice:")


def test_signature_collapses_turn_numbers():
    assert signature("waiting for complete visible choice row; turn=8") == \
        signature("waiting for complete visible choice row; turn=11")


def test_budget_stops_on_s0_and_on_repeated_signature():
    budget = FailureBudget(max_total=3, max_same=2)
    s0 = dict(severity="S0", signature="recruit:input outcome")
    budget.record(s0)
    assert budget.allow(s0) == (False, "severity S0 forbids another match")

    budget = FailureBudget(max_total=3, max_same=2)
    s1 = dict(severity="S1", signature="choice:internal typeerror")
    budget.record(s1)
    assert budget.allow(s1)[0] is True
    budget.record(s1)
    assert budget.allow(s1)[0] is True
    budget.record(s1)
    allowed, why = budget.allow(s1)
    assert allowed is False and "signature repeated" in why


def test_budget_allows_distinct_failures_until_total():
    budget = FailureBudget(max_total=3, max_same=2)
    for i in range(3):
        inc = dict(severity="S1", signature="sig-%d" % i)
        budget.record(inc)
        assert budget.allow(inc)[0] is True
    inc = dict(severity="S1", signature="sig-final")
    budget.record(inc)
    allowed, why = budget.allow(inc)
    assert allowed is False and "budget exhausted" in why


# ------------------------------------------------------- unsent reconciliation

def _before(log_position=100, turn=3, player=1):
    return dict(game_id="game-1", client=dict(local_player_id=player, log_position=log_position,
                                              turn=turn, phase="recruit", client_submission_position=90))


def test_never_submitted_input_is_retired_after_grace(tmp_path):
    controller, clock = _controller(tmp_path)
    controller.pending = (_decision(), _before(), clock[0] - 3.0)
    snapshot = dict(game_id="game-1", client=dict(local_player_id=1, log_position=140, turn=3,
                                                  client_submission_position=90,
                                                  client_playback_position=140))
    assert controller.unsent_reconciliation(snapshot) is True
    assert controller.pending is None
    archived = list((Path(tmp_path) / "expired-unsent").glob("*.json"))
    assert len(archived) == 1
    proof = json.loads(archived[0].read_text(encoding="utf-8"))
    assert proof["proof"]["kind"] == "client_never_submitted"
    events = [json.loads(line) for line in (Path(tmp_path) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "expired_unsent_input"


def test_submitted_input_is_never_retired_early(tmp_path):
    controller, clock = _controller(tmp_path)
    controller.pending = (_decision(), _before(), clock[0] - 30.0)
    snapshot = dict(game_id="game-1", client=dict(local_player_id=1, log_position=140, turn=3,
                                                  client_submission_position=120))
    assert controller.unsent_reconciliation(snapshot) is None
    assert controller.pending is not None


def test_unsent_grace_is_not_shortened(tmp_path):
    controller, clock = _controller(tmp_path)
    controller.pending = (_decision(), _before(), clock[0] - 1.0)
    snapshot = dict(game_id="game-1", client=dict(local_player_id=1, log_position=140, turn=3,
                                                  client_submission_position=90))
    assert controller.unsent_reconciliation(snapshot) is None


def _legal_snapshot(turn=3, submitted=120, extra_resolution=False):
    data = dict(local_player_id=1, log_position=140, turn=turn, phase="recruit",
                client_submission_position=submitted, board=[], hand=[], shop=[], controls=[],
                choices=[], interaction=None,
                resolutions=([dict(source=41, log_position=130, complete=True)] if extra_resolution else []))
    return dict(game_id="game-1", client=data, view=dict(pending=0))


def test_submitted_but_still_legal_option_is_retired(tmp_path, monkeypatch):
    import hsrl_agent.live.controller as controller_module
    controller, clock = _controller(tmp_path)
    decision = _decision(action=46, kind="play")
    controller.pending = (decision, _before(), clock[0] - 8.0)
    monkeypatch.setattr(controller_module, "bind_actions", lambda snap: (decision.binding,))
    assert controller.rejected_reconciliation(_legal_snapshot()) is True
    assert controller.pending is None
    archived = list((Path(tmp_path) / "expired-unsent").glob("*.json"))
    proof = json.loads(archived[0].read_text(encoding="utf-8"))["proof"]
    assert proof["kind"] == "server_did_not_take_still_legal"


def test_submitted_option_with_a_root_is_never_retired(tmp_path, monkeypatch):
    import hsrl_agent.live.controller as controller_module
    controller, clock = _controller(tmp_path)
    decision = _decision(action=46, kind="play")
    controller.pending = (decision, _before(), clock[0] - 8.0)
    monkeypatch.setattr(controller_module, "bind_actions", lambda snap: (decision.binding,))
    assert controller.rejected_reconciliation(_legal_snapshot(extra_resolution=True)) is None
    assert controller.pending is not None


def test_submitted_option_without_the_binding_is_never_retired(tmp_path, monkeypatch):
    import hsrl_agent.live.controller as controller_module
    controller, clock = _controller(tmp_path)
    controller.pending = (_decision(action=46, kind="play"), _before(), clock[0] - 8.0)
    monkeypatch.setattr(controller_module, "bind_actions", lambda snap: ())
    assert controller.rejected_reconciliation(_legal_snapshot()) is None


# ----------------------------------------------------------- gesture guard

def _snap(turn=3):
    return dict(game_id="game-1", client=dict(turn=turn, log_position=10))


def _confirm(controller, decision, plan, at=100.0):
    """Simulate the receipt that makes a gesture 'confirmed applied'."""
    controller.gesture_state[controller._gesture_key(decision, plan)] = dict(
        turn=3, at=at, count=1, confirmed_at=at)


def test_confirmed_move_repeat_is_suppressed_and_released_by_window(tmp_path):
    controller, clock = _controller(tmp_path)
    plan = [["drag", [0.2, 0.5], [0.6, 0.5]], ["move_pointer", [0.96, 0.91]]]
    move = _decision(action=201, kind="move")
    assert controller._gesture_allowed(move, plan, _snap()) is True
    _confirm(controller, move, plan, at=clock[0])
    clock[0] += 2.0
    assert controller._gesture_allowed(move, plan, _snap()) is False
    # A different destination is a different intent and stays allowed.
    other = [["drag", [0.2, 0.5], [0.4, 0.5]], ["move_pointer", [0.96, 0.91]]]
    assert controller._gesture_allowed(_decision(action=203, kind="move"), other, _snap()) is True
    # The window releases by itself instead of sliding forever.
    clock[0] += 5.0
    assert controller._gesture_allowed(move, plan, _snap()) is True


def test_unconfirmed_gesture_is_never_suppressed(tmp_path):
    """A swallowed first click must stay retryable (no starvation)."""
    controller, clock = _controller(tmp_path)
    plan = [["click", [0.5, 0.5]]]
    choose = _decision(action=32, kind="choose")
    assert controller._gesture_allowed(choose, plan, _snap()) is True
    clock[0] += 0.2
    assert controller._gesture_allowed(choose, plan, _snap()) is True


def test_confirmed_repeat_is_allowed_again_next_turn(tmp_path):
    controller, clock = _controller(tmp_path)
    plan = [["drag", [0.2, 0.5], [0.6, 0.5]]]
    move = _decision(action=201, kind="move")
    assert controller._gesture_allowed(move, plan, _snap(turn=3)) is True
    _confirm(controller, move, plan, at=clock[0])
    assert controller._gesture_allowed(move, plan, _snap(turn=4)) is True


def test_play_is_not_repeat_sensitive(tmp_path):
    controller, clock = _controller(tmp_path)
    plan = [["click", [0.5, 0.5]]]
    play = _decision(action=46, kind="play")
    assert controller._gesture_allowed(play, plan, _snap()) is True
    _confirm(controller, play, plan, at=clock[0])
    clock[0] += 0.3
    assert controller._gesture_allowed(play, plan, _snap()) is True


# -------------------------------------------------------- placement settling

class _Reader:
    def __init__(self, tags):
        self.tags = list(tags)

    def poll(self):
        tag = self.tags.pop(0) if self.tags else None
        return dict(game_id="game-1", hero=dict(tags=dict(PLAYER_LEADERBOARD_PLACE=tag)))


def test_placement_waits_for_the_final_batch():
    data = dict(game_id="game-1", hero=dict(tags=dict(PLAYER_LEADERBOARD_PLACE="4")))
    placement, source = settle_placement(_Reader(["3", "3"]), data, settle=1.0, interval=0.0,
                                         stable_for=0.0, sleep=lambda _s: None)
    assert (placement, source) == ("3", "stable")


def test_placement_falls_back_to_last_seen_on_timeout():
    data = dict(game_id="game-1", hero=dict(tags=dict(PLAYER_LEADERBOARD_PLACE=None)))
    placement, source = settle_placement(_Reader([None, None]), data, settle=0.0, interval=0.0,
                                         stable_for=0.0, sleep=lambda _s: None)
    assert source == "timeout"
    assert placement is None
