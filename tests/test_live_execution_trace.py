"""离线回归：执行语义、5 选一几何、训练动作规则与异步追踪。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client"))

from hsrl_agent.live.contracts import Binding, Decision, execution_stamp
from hsrl_agent.live.practical import PracticalGeometry
from hsrl_agent.live.trace import GameTrace, summarize
from tests.live_fixtures import snapshot


def _decision(action=46, kind="play", source=41, revision=1):
    binding = Binding(action, kind, source=source, position=0)
    return Decision("game-1", revision, "stamp", action, binding, "ckpt", "logits")


def test_execution_stamp_ignores_unrelated_public_opponent_update():
    s = snapshot()
    s["client"]["choices"] = [dict(entity_id=41, card_id="BG20_100", card_type="MINION", tags={})]
    s["client"]["interaction"] = dict(mode="choice", request_id=1, min_select=1, max_select=1, owner=1)
    s["view"].update(pending=2, choices=[dict(card_id="BG20_100", card_only=True)])
    before = execution_stamp(s)
    changed = json.loads(json.dumps(s))
    changed["view"]["opponents"][0]["tier"] += 1
    assert execution_stamp(changed) == before
    changed["client"]["choices"][0]["entity_id"] = 99
    assert execution_stamp(changed) != before


def test_execution_stamp_ignores_unrelated_opponent_health_in_recruit():
    s = snapshot()
    s["client"]["board"] = [dict(entity_id=41, card_id="BG20_100", card_type="MINION",
                                 position=1, controller=1, tags={})]
    before = execution_stamp(s)
    changed = json.loads(json.dumps(s))
    changed["view"]["opponents"][0]["health"] -= 3
    changed["view"]["global"]["alive"] = 7
    assert execution_stamp(changed) == before


def test_execution_stamp_tracks_authorization_identity_not_stat_text():
    s = snapshot()
    s["client"]["choices"] = [dict(entity_id=41, card_id="BG20_100", card_type="MINION", tags={})]
    s["client"]["interaction"] = dict(mode="target", request_id=1, min_select=1, max_select=1,
                                      owner=1, targets=[41])
    s["view"].update(pending=4, choices=[dict(card_id="BG20_100", card_only=True)])
    before = execution_stamp(s)
    changed = json.loads(json.dumps(s))
    changed["client"]["choices"][0]["health"] = 99
    # A stat change is model context; it does not change which object is
    # clicked. Authorization, zone identity and index are what matter.
    assert execution_stamp(changed) == before
    changed = json.loads(json.dumps(s))
    changed["client"]["choices"] = []
    from hsrl_agent.live.contracts import Blocked
    with pytest.raises(Blocked, match="choice list missing"):
        execution_stamp(changed)


def _choice_profile():
    return json.loads((ROOT / "config/live/autoplay-255790.json").read_text(encoding="utf-8-sig"))


def _choice_snapshot(count, visual=None):
    data = dict(turn=8, phase="choice", local_player_id=5, board=[], hand=[], shop=[], controls=[],
                interaction=dict(mode="choice", request_id=6, owner=5, min_select=1, max_select=1),
                choices=[dict(entity_id=4791 + i, card_id="BG34_Giant_205", card_type="MINION",
                              position=i + 1, tags={}) for i in range(count)],
                tavern_tier=5, log_position=1)
    return dict(game_id="g", revision=1, client=data, options=[],
                view=dict(pending=2, power_entities=[]),
                visual=visual or dict(window_token="t"))


def test_five_and_six_card_rows_are_derived_without_authorizing_blind_clicks():
    geometry = PracticalGeometry(_choice_profile())
    five = geometry.locations(_choice_snapshot(5))[0]
    xs = [five[f"entity:{4791 + i}"][0] for i in range(5)]
    assert xs == sorted(xs) and len(set(xs)) == 5
    six = geometry.locations(_choice_snapshot(6))[0]
    xs = [six[f"entity:{4791 + i}"][0] for i in range(6)]
    assert xs == sorted(xs) and len(set(xs)) == 6
    timewarp = PracticalGeometry(_choice_profile())
    snap = _choice_snapshot(5)
    snap["view"]["pending"] = 8
    points = timewarp.locations(snap)[0]
    assert all(f"entity:{4791 + i}" in points for i in range(5))


def test_measured_choice_boxes_override_provisional_row_geometry():
    geometry = PracticalGeometry(_choice_profile())
    snap = _choice_snapshot(5)
    entities = [c["entity_id"] for c in snap["client"]["choices"]]
    boxes = [(0.10 + i * 0.18, 0.30, 0.10, 0.30) for i in range(5)]
    snap["visual"]["choice_ui"] = dict(visible=True, entities=entities, boxes=boxes)
    points = geometry.locations(snap)[0]
    for eid, (x, y, w, h) in zip(entities, boxes):
        assert points[f"entity:{eid}"] == pytest.approx((x + w / 2, y + h / 2))


def test_move_and_insertion_use_destination_slot_not_gap_midpoint():
    """Real 255790 resolves a move at the final slot, not the row-edge gap."""
    from hsrl_agent.live.contracts import Decision, Binding
    from hsrl_agent.live.practical import PracticalGeometry

    profile = _choice_profile()
    geometry = PracticalGeometry(profile)
    board = [dict(entity_id=100 + i, card_id=f"BG20_10{i}", card_type="MINION",
                  position=i + 1, controller=1, attack=1, health=1, tags={})
             for i in range(7)]
    data = dict(turn=11, phase="recruit", local_player_id=1, board=board, hand=[], shop=[],
                choices=[], controls=[], interaction=None, tavern_tier=6, log_position=1)
    snap = dict(game_id="g", revision=1, client=data, options=[], view=dict(pending=0, power_entities=[]),
                visual=dict(window_token="t"))
    decision = Decision("g", 1, "s", 201, Binding(201, "move", source=100, position=6), "c", "l")
    plan = geometry.plan(decision, snap)
    x, y = plan[0][2]
    # Final 0-based position 6 is server slot 7. Use the calibrated immutable
    # slot; recomputing a midpoint after deleting the source shifted this one
    # slot to the right in the archived 00:45:44 move.
    slot_x = geometry.locations(snap)[0]["position:board:7"][0]
    assert x == pytest.approx(slot_x)


def test_move_from_left_edge_to_middle_uses_target_slot_before_source_removal():
    from hsrl_agent.live.contracts import Decision, Binding
    from hsrl_agent.live.practical import PracticalGeometry

    geometry = PracticalGeometry(_choice_profile())
    board = [dict(entity_id=8520 + i, card_id=f"BG20_10{i}", card_type="MINION",
                  position=i + 1, controller=1, attack=1, health=1, tags={})
             for i in range(7)]
    data = dict(turn=11, phase="recruit", local_player_id=1, board=board, hand=[], shop=[],
                choices=[], controls=[], interaction=None, tavern_tier=6, log_position=1)
    snap = dict(game_id="g", revision=1, client=data, options=[], view=dict(pending=0, power_entities=[]),
                visual=dict(window_token="t"))
    # Archived real action 201: source slot 0 -> final 0-based slot 5. Old code
    # dropped at 0.7041 and the server reported selectedPosition=7.
    decision = Decision("g", 1, "s", 201, Binding(201, "move", source=8520, position=5), "c", "l")
    plan = geometry.plan(decision, snap)
    points = geometry.locations(snap)[0]
    assert plan[0][2] == pytest.approx(list(points["position:board:6"]))
    assert plan[0][2][0] < 0.7041


def test_end_turn_clicks_calibrated_start_combat_control():
    from hsrl_agent.live.contracts import Decision, Binding
    from hsrl_agent.live.practical import PracticalGeometry

    geometry = PracticalGeometry(_choice_profile())
    data = dict(turn=9, phase="recruit", local_player_id=1, board=[], hand=[], shop=[],
                choices=[], controls=[], interaction=None, tavern_tier=6, log_position=1)
    snap = dict(game_id="g", revision=1, client=data, options=[],
                view=dict(pending=0, power_entities=[]), visual=dict(window_token="t"))
    decision = Decision("g", 1, "s", 0, Binding(0, "end"), "c", "l")
    plan = geometry.plan(decision, snap)
    assert plan[0] == ("click", list(geometry.locations(snap)[0]["control:end"]))
    assert plan[1] == ("move_pointer", [0.96, 0.91])


def test_already_gold_trinket_slots_are_not_a_new_highlight():
    from hsrl_agent.live.contracts import Binding
    from hsrl_agent.live.frame_gate import FrameGate

    class G(FrameGate):
        def __init__(self):
            pass

    gate = G()
    gate.admitted_binding = Binding(31, "choose", source=10)
    gate.recheck_min_support = .65
    gate.previous = ("same", None, "window", 0., None)
    gate.admitted = ("same", "window", 0.)
    gate.profile = _choice_profile()
    snapshot = {"client": {"choices": [{"entity_id": i} for i in (10, 11, 12, 13)]}}

    import hsrl_agent.live.frame_gate as fg
    old_layout = fg.layout_key
    fg.layout_key = lambda _: "same"
    import hsrl_agent.live.choice_ui as cu
    old = cu.slot_colors

    admitted = [dict(entity=i, base=[1.0, 1.0], gold=[0.0, 0.0]) for i in (10, 11, 12, 13)]
    same = [dict(entity=i, base=[1.0, 1.0], gold=[0.0, 0.0]) for i in (10, 11, 12, 13)]
    hover_brightened = [dict(entity=i, base=[1.0, 1.0], gold=[0.0, 0.0]) for i in (10, 11, 12, 13)]
    hover_brightened[1] = dict(entity=11, base=[0.8, 1.0], gold=[0.6, 0.8])
    new_gold = [dict(entity=i, base=[1.0, 1.0], gold=[0.0, 0.0]) for i in (10, 11, 12, 13)]
    new_gold[1] = dict(entity=11, base=[0.0, 0.0], gold=[1.0, 1.0])
    frames = iter((admitted, same, admitted, hover_brightened, admitted, new_gold))
    cu.slot_colors = lambda *_: next(frames)
    try:
        # The same all-gold trinket row must never look like a new highlight.
        assert gate.highlight_moved(snapshot, None) is False
        # Green-rail brightening from the hover animation is also not gold.
        assert gate.highlight_moved(snapshot, None) is False
        # A genuinely new gold-dominant slot outside the authorized source is
        # still rejected.
        assert gate.highlight_moved(snapshot, None) is True
    finally:
        cu.slot_colors = old
        fg.layout_key = old_layout


def test_freeze_is_confirmed_by_real_trigger_root_receipt():
    from dataclasses import replace
    from hsrl_agent.live.contracts import Binding, Decision
    from hsrl_agent.live.verification import verify
    from tests.live_fixtures import snapshot

    s = snapshot()
    s["client"].update(phase="recruit", turn=7, gold=5,
                       shop=[dict(entity_id=500, card_id="X", card_type="MINION",
                                  position=1, controller=13, tags={})],
                       controls=[dict(entity_id=42, card_id="TB_BaconShopLockAll_Button",
                                      card_type="GAME_MODE_BUTTON", position=0, controller=1, tags={})],
                       log_position=100)
    decision = Decision("game-1", 1, "s", 2,
                        Binding(2, "freeze", source=42, option_id=5), "c", "l")
    before = {"game_id": "game-1", "revision": 1, "client": json.loads(json.dumps(s["client"]))}
    after = json.loads(json.dumps(before))
    after["revision"] = 2
    after["client"]["log_position"] = 120
    after["client"]["resolutions"] = [dict(
        kind="freeze", source=42, complete=True, closed_by=None, block_type="PLAY",
        log_position=120, started_at_log_position=110,
        entity_changes=[(42, "TAG_SCRIPT_DATA_NUM_1", "2", "3")],
        submission=dict(log_position=105, option_id=5, source=42, owner=1,
                        sub_option_id=-1, target=0, position=0))]
    assert verify(decision, before, after, require_playback=True) == "applied"


def test_freeze_action_is_hidden_when_the_tavern_is_already_all_frozen():
    from hsrl_agent.live.actions import bind_actions
    from tests.live_fixtures import snapshot

    s = snapshot()
    s["client"].update(
        phase="recruit", turn=7, gold=5,
        shop=[dict(entity_id=500, card_id="X", card_type="MINION",
                   position=1, controller=13, tags={"FROZEN": "1"})],
        controls=[dict(entity_id=42, card_id="TB_BaconShopLockAll_Button",
                       card_type="GAME_MODE_BUTTON", position=0,
                       controller=1, tags={})],
        log_position=100,
    )
    s["options"] = [dict(id=5, entity=42, type="POWER", error="NONE",
                         has_targets=False, targets=[], sub_options=[])]
    assert not any(b.kind == "freeze" for b in bind_actions(s))
    s["client"]["shop"][0]["tags"]["FROZEN"] = "0"
    assert any(b.kind == "freeze" for b in bind_actions(s))


def test_move_retry_grace_preserves_timeout_for_non_move_actions():
    c = _controller()
    c.practical = True
    c.timeout = 12.
    c.recovery_timeout = 120.
    snap = snapshot()
    snap["game_id"] = "game-1"
    snap["client"].update(turn=11, phase="recruit", ready=True,
                           options_pending=False, client_task_list=None,
                           log_position=100,
                           board=[{"entity_id": 1}],
                           hand=[{"entity_id": 1, "controller": 1}],
                           choices=[], interaction=None,
                           client_submission_position=90, resolutions=[])
    before = json.loads(json.dumps(snap["client"]))
    before["board"] = [{"entity_id": 1}]
    before["hand"] = [{"entity_id": 1, "controller": 1}]
    started = 0.
    c.clock = lambda: 13.
    move = Decision("game-1", 1, "s", 200,
                    Binding(200, "move", source=1, position=5), "c", "l")
    play = Decision("game-1", 1, "s", 46,
                    Binding(46, "play", source=1, position=0), "c", "l")
    archived = {"game_id": "game-1", "client": before}
    assert c._retry_unsent(snap, move, archived, started) is False
    assert c._retry_unsent(snap, play, archived, started) is True


def test_move_receipt_is_not_retried_when_final_slot_differs():
    from hsrl_agent.live.contracts import Binding, Decision
    from hsrl_agent.live.verification import verify
    from tests.live_fixtures import snapshot

    s = snapshot()
    s["client"].update(
        phase="recruit", turn=11, log_position=100,
        board=[dict(entity_id=eid, card_id="X", card_type="MINION",
                    position=i + 1, controller=1, tags={})
               for i, eid in enumerate((1, 2, 3, 4, 5, 6, 7))],
    )
    decision = Decision("game-1", 1, "s", 200,
                        Binding(200, "move", source=1, position=4), "c", "l")
    before = {"game_id": "game-1", "revision": 1,
              "client": json.loads(json.dumps(s["client"]))}
    after = json.loads(json.dumps(before))
    after["revision"] = 2
    after["client"]["log_position"] = 120
    # Requested final 0-based slot 4 means [2,3,4,5,1,6,7]; make the server
    # result differ by one slot so the exact-order check cannot pass.
    after["client"]["board"] = [
        dict(entity_id=eid, card_id="X", card_type="MINION",
             position=i + 1, controller=1, tags={})
        for i, eid in enumerate((7, 2, 3, 4, 1, 5, 6))
    ]
    after["client"]["resolutions"] = [dict(
        kind="reorder", source=1, complete=True, log_position=120,
        started_at_log_position=110)]
    # A server root for the exact source proves delivery even when the
    # observed final order differs from the requested slot.  Treating it as
    # unsent caused the duplicate drag in the 03:02 game.
    assert verify(decision, before, after, require_playback=True) == "applied"


class _NoPolicy:
    sha = "ckpt"


class _NoGeometry:
    pass


class _NoBackend:
    pass


def _controller():
    from hsrl_agent.live.controller import Controller
    return Controller(_NoPolicy(), _NoGeometry(), _NoBackend(), ROOT / "artifacts/local/_trace-guard-test",
                      enable_input=False, practical=True)


def test_controller_state_persists_move_history_without_duplicate_turn_keyword(tmp_path):
    c = _controller()
    c.out = tmp_path
    c.enable_input = True
    c.practical = True
    c._note_action(("game-1", 3), 197)
    c._save_state()
    state = json.loads((tmp_path / "controller-state.json").read_text(encoding="utf-8"))
    assert state["move_history"][0]["turn"] == ["game-1", 3]
    assert state["move_history"][0]["last_action"] == 197


def test_live_move_guard_matches_frozen_training_rules():
    c = _controller()
    c.enable_input = False
    turn = ("game-1", 3)
    bindings = (
        Binding(0, "end"),
        Binding(196, "move", source=1, position=0),
        Binding(203, "move", source=2, position=0),
        Binding(197, "move", source=1, position=1),
    )
    snap = snapshot()
    snap["game_id"] = turn[0]
    snap["client"]["turn"] = turn[1]
    assert {b.action for b in c._guarded_bindings(snap, bindings)} == {0, 196, 203, 197}
    c._note_action(turn, 197)
    # 197 = from 0 to 2; its inverse is 210 (from 2 to 0), not 203.
    c._note_action(turn, 196)
    c._note_action(turn, 196)
    # Same action twice is the training repeat limit; a third copy is blocked.
    out = {b.action for b in c._guarded_bindings(snap, bindings)}
    assert 196 not in out
    for _ in range(2):
        c._note_action(turn, 197)
    # Five total moves have now been applied; all Move actions are removed.
    assert {b.action for b in c._guarded_bindings(snap, bindings)} == {0}


def test_trace_writes_async_bounded_and_reports_identities(tmp_path):
    out = tmp_path / "run"
    trace = GameTrace(out)
    snap = snapshot()
    decision = _decision()
    did = trace.decision(snap, decision, policy_ms=1.25, reused=False,
                         bindings=[decision.binding])
    trace.execution(did, snapshot=snap, plan=[["click", [0.5, 0.5]]],
                    status="sent", timing=dict(elapsed_ms=3.5), physical=True)
    trace.execution(did, snapshot=None, plan=[], status="receipt", outcome="applied",
                    after=snap["client"])
    from PIL import Image
    trace.frame(out / "frames" / "sample.jpg", Image.new("RGB", (8, 8), "black"),
                dict(reason="offline test"), quality=70)
    status = trace.close()
    assert status["events_dropped"] == 0 and not status["writer_alive"]
    assert not status["frame_writer_alive"] and status["frames_dropped"] == 0
    summary = summarize(out)
    assert summary["counts"]["decisions"] == 1
    assert summary["counts"]["executions"] == 1
    lines = (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["decision_id"] == did
    with gzip.open(out / "decisions-full.jsonl.gz", "rt", encoding="utf-8") as stream:
        full = [json.loads(line) for line in stream]
    assert full[0]["snapshot"]["game_id"] == "game-1"
    performance = json.loads((out / "performance.json").read_text(encoding="utf-8"))
    assert performance["inference_ms"]["count"] == 1
    assert (out / "decision-report.md").is_file()
    assert (out / "frames" / "sample.jpg").is_file()
    assert (out / "frames" / "sample.json").is_file()


def test_five_card_choice_trace_keeps_full_model_snapshot(tmp_path):
    trace = GameTrace(tmp_path)
    snap = _choice_snapshot(5)
    snap["approximation"] = dict(mode="public-approx-v1", fields={})
    decision = _decision(action=31, kind="choose", source=4791)
    did = trace.decision(snap, decision, policy_ms=2.0, reused=False,
                         bindings=[decision.binding])
    trace.close()
    with gzip.open(tmp_path / "decisions-full.jsonl.gz", "rt", encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    assert row["decision_id"] == did
    assert len(row["snapshot"]["client"]["choices"]) == 5
