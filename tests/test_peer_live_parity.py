"""Offline parity regressions for the peer action and observation interfaces."""
from types import SimpleNamespace

import pytest

from hsrl_agent.live.actions import bind_actions
from hsrl_agent.live.adaptive import AdaptiveProjector
from hsrl_agent.live.contracts import Binding, Blocked, Decision, state_stamp
from hsrl_agent.live.controller import Controller
from hsrl_agent.live.projection import card_minion, rogers_remaining
from tests.live_fixtures import option, snapshot


def test_full_board_magnetic_targets_are_card_specific_and_authorized():
    s = snapshot()
    hand = dict(card_id="BG_DEEP_015", entity_id=100, card_type="MINION", tags={"MAGNETIC":"1", "MAGNETIC_TO_RACE":"11"})
    board = [dict(card_id=f"host-{i}", entity_id=i + 1, card_type="MINION", tags={"CARDRACE":race})
             for i, race in enumerate(("11", "17", "15", "26", "18", "20", "11"))]
    s["client"].update(hand=[hand], board=board)
    s["options"] = [option(100, targets=[1, 2, 3, 4, 5, 6])]
    def targets():
        return {(b.target, b.action) for b in bind_actions(s) if b.kind == "magnetize"}
    assert targets() == {(1,126),(2,127),(4,129)}
    assert not any(b.kind == "play" for b in bind_actions(s))
    del hand["tags"]["MAGNETIC_TO_RACE"]
    hand["card_id"] = "BG_DEEP_015_G"
    assert targets() == {(1,126),(2,127),(4,129)}
    hand["card_id"] = "BG31_859"
    assert targets() == {(2,127),(4,129),(5,130)}
    hand["card_id"] = "BG25_807t2"
    assert targets() == {(2,127),(3,128),(4,129)}
    hand["card_id"] = "ordinary"
    assert targets() == {(2,127),(4,129)}
    hand["tags"]["MAGNETIC"] = "0"
    assert not targets()


def test_public_dual_race_and_rogers_remaining():
    c = dict(card_id="BG_DEEP_015", tier=3, tags={"ATK":"3","HEALTH":"1","CARDRACE":"17"})
    assert set(card_minion(c, ("UNDEAD","MECHANICAL"))["races"]) == {"UNDEAD","MECHANICAL"}
    c.update(card_id="BG33_823", tags={"TAG_SCRIPT_DATA_NUM_1":"7","TAG_SCRIPT_DATA_NUM_3":"9"})
    assert rogers_remaining(c) == 7
    del c["tags"]["TAG_SCRIPT_DATA_NUM_1"]
    assert rogers_remaining(c) is None


def test_lobby_evidence_identity_and_conflict():
    p = AdaptiveProjector({}, {})
    first = ("BEAST","DEMON","MECHANICAL","PIRATE","UNDEAD")
    p.observe_lobby("game-1", first)
    with pytest.raises(Blocked):
        p.observe_lobby("game-1", first[:-1]+("DRAGON",))
    p.observe_lobby("game-2", first[:-1]+("DRAGON",))
    assert p.lobby_game == "game-2" and "UNDEAD" not in p.lobby_races


def test_peer_end_waits_once_and_never_dispatches(tmp_path):
    class Peer:
        peer_round = True
        calls = 0
        def decide(self, s, bindings):
            self.calls += 1
            b = next(b for b in bindings if b.kind == "end")
            return Decision(s["game_id"],s["revision"],state_stamp(s),0,b,"peer","logits")
    p = Peer()
    c = Controller(p, None, None, tmp_path, practical=False, enable_input=False)
    s = snapshot()
    assert c.tick(s) == c.tick(s) == "waiting_next_turn"
    assert p.calls == 1
    s["client"]["turn"] += 1
    assert c.tick(s) == "waiting_next_turn" and p.calls == 2


def test_peer_recruit_budget_and_toggle(tmp_path):
    c = Controller(SimpleNamespace(peer_round=True), None, None, tmp_path)
    s = snapshot()
    turn = (s["game_id"],s["client"]["turn"])
    bindings = [Binding(0,"end"),Binding(2,"freeze"),Binding(197,"move"),Binding(203,"move")]
    for _ in range(3):
        c._note_action(turn,2)
    assert 2 not in {b.action for b in c._guarded_bindings(s, bindings)}
    c._note_action(turn,197)
    assert 203 not in {b.action for b in c._guarded_bindings(s, bindings)}
    for _ in range(9):
        c._note_action(turn,197)
    assert {b.action for b in c._guarded_bindings(s, bindings)} == {0}
    c.turn_counts[turn].update(actions=299,moves=0)
    assert 197 in {b.action for b in c._guarded_bindings(s,bindings)}
    c._note_action(turn,1)
    assert c._guarded_bindings(s,bindings) == (bindings[0],)

def test_toggle_requires_a_server_option_on_both_profiles():
    s = snapshot()
    s["client"].update(shop=[dict(entity_id=10, card_id="m", tags={"FROZEN":"1"})],
                       controls=[dict(entity_id=11, card_id="TB_BaconShopLockAll_Button", tags={})])
    s["options"] = [option(11)]
    assert not any(b.action == 2 for b in bind_actions(s))
    s["action_profile"] = "peer-r1"
    assert any(b.action == 2 for b in bind_actions(s))
    s["options"] = []
    assert not any(b.action == 2 for b in bind_actions(s))
