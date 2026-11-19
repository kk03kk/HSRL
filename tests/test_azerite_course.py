import pytest

pytest.importorskip("hsbg_engine")
from tools.probe_four_growth_curves import AZ, KELP, SURF, card_ids, item_card, own, step
from workflows.azerite_course import actionable, measure, opening


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_element_demonstration_follows_entities_after_permutation(seed):
    game = opening(seed, "elem_offer")
    buy, = actionable(game, game.legal_mask(0))
    assert item_card(own(game)["shop"][buy - 8]) == card_ids()[AZ]
    game = opening(seed, "elem_budget")
    activate, = actionable(game, game.legal_mask(0))
    assert own(game)["board"][activate-23]["card"] == card_ids()[KELP]
    step(game,activate)
    choice, = actionable(game,game.legal_mask(0))
    player = own(game)
    eid = player["pending"][-1]["Target"]["options"][choice-30]
    assert any(m["eid"] == eid and m["card"] == card_ids()[SURF] for m in player["board"])


def test_sustained_element_requires_distinct_late_turns_and_real_payment():
    turns = [dict(turn=t,strength=350,absorptions=[dict(shop=0,gain=800)],
                  choices=[]) for t in range(9,19)]
    for t in [13,14,15]:
        turns[t-9]["absorptions"].extend([dict(shop=6,gain=100)]*3)
    assert not measure(turns,True)["sustained"]
    turns[16-9]["absorptions"].append(dict(shop=6,gain=0))
    assert not measure(turns,True)["sustained"]
    turns[16-9]["absorptions"].append(dict(shop=6,gain=50))
    assert measure(turns,True)["sustained"]
