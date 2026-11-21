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

def test_hero_header_requires_exactly_one_clean_five_type_subtitle():
    from hsrl_agent.live.vision import races_from_hero_banner
    title = dict(text="选择一个英雄", confidence=.99, point=(.50,.12))
    subtitle = dict(text="亡灵，恶魔，海盗，畸变怪，野猪人", confidence=.94, point=(.50,.168))
    assert races_from_hero_banner([title, subtitle]) == ["ABERRATION","DEMON","PIRATE","QUILBOAR","UNDEAD"]
    assert races_from_hero_banner([subtitle]) is None
    for change in (dict(text="禁用：亡灵，恶魔，海盗，畸变怪，野猪人"),dict(confidence=.83),
                   dict(text="亡灵，恶魔，海盗，海盗，野猪人"),dict(point=(.5,.31))):
        assert races_from_hero_banner([title,{**subtitle,**change}]) is None

def test_positional_magnetic_play_uses_only_a_witnessed_play_option():
    s = snapshot()
    s["action_profile"] = "peer-r1"
    s["client"].update(hand=[dict(entity_id=10,card_id="BG_DEEP_015",card_type="MINION",
                                 tags={"MAGNETIC":"1","MAGNETIC_TO_RACE":"11"})],
                       board=[dict(entity_id=11,card_id="BG25_010",tags={"CARDRACE":"UNDEAD"})])
    s["options"] = [option(10)]
    by = {b.action:b for b in bind_actions(s)}
    assert by[126].target == 11 and by[126].position == 0
    assert 46 not in by and 47 in by
    s["options"] = [{**option(10), "has_targets":True}]
    assert all(b.kind != "magnetize" for b in bind_actions(s))
    s["options"] = []
    assert [b.kind for b in bind_actions(s)] == ["end"]

def test_peer_refuses_a_missing_lobby_at_decision_time():
    from hsrl_agent.live.policy import require_peer_lobby
    with pytest.raises(Blocked, match="five observed"):
        require_peer_lobby({"lobby_races":[0]*11})
    require_peer_lobby({"lobby_races":[1]*5+[0]*6})


def test_pinned_numeric_dictionary_reads_from_lf_checkout():
    from hsrl_agent.live.tags import symbolic_line
    assert symbolic_line('tag=3919 value=1', 255790) == 'tag=ADDITIONAL_HERO_POWER_INDEX value=1'
    assert symbolic_line('tag=3919 value=1', 256589) == 'tag=3919 value=1'


def test_recruit_tooltip_reads_wrapped_available_list_only():
    from hsrl_agent.live.vision import races_from_recruit_tooltip
    rows = [dict(text=text, confidence=.99, point=(.5, y)) for y, text in enumerate((
        '本局对战中有：', '恶魔，机械，海盗，畸', '变怪，野兽',
        '本局对战中没有：', '亡灵，元素，野猪人，', '鱼人，龙'))]
    assert races_from_recruit_tooltip(rows) == ['ABERRATION','BEAST','DEMON','MECHANICAL','PIRATE']
    assert races_from_recruit_tooltip(rows[:3]) is None
    assert races_from_recruit_tooltip(rows[:1]+rows[2:]) is None


def test_secondary_power_uses_hero_link_not_entity_order():
    from hsrl_agent.live.projection import ordered_hero_powers
    primary = dict(entity_id=2806, card_type='HERO_POWER', tags={})
    secondary = dict(entity_id=226, card_type='HERO_POWER', tags={'ADDITIONAL_HERO_POWER_INDEX':'1'})
    data = dict(controls=[secondary, primary], hero={'tags':{'ADDITIONAL_HERO_POWER_ENTITY_1':'226'}})
    assert ordered_hero_powers(data) == [primary, secondary]
    data['hero']['tags']['ADDITIONAL_HERO_POWER_ENTITY_1'] = '2806'
    with pytest.raises(Blocked, match='explicit slot identity'):
        ordered_hero_powers(data)
