import json
from math import log1p
from types import SimpleNamespace

import numpy as np
import pytest

from workflows.adaptive_counterfactual import count_golden, golden_versions, select
from workflows.adaptive_triple_opportunity import offers
from workflows.seal_adaptive_futures import grouped, values


def test_replicated_growth_counts_only_common_survivors():
    def result(rank, body9, body12=None):
        boards = {"9": body9}
        if body12 is not None:
            boards["12"] = body12
        return {"rank": rank, "boards": boards}

    gain = values(result(4, 2, 9), result(2, 3, 29))
    loss = values(result(2, 5, 10), result(5, 5))
    neither = values(result(7, 3), result(7, 3))
    draws = np.array([[0, 1], [0, 0], [1, 1]])
    report = grouped([[gain, loss], [neither, neither]], draws)
    expected_growth = log1p(29) - log1p(9) - log1p(3) + log1p(2)
    assert report["pairs"] == 4 and report["common"] == 1
    assert report["effects"]["common_growth"]["mean"] == pytest.approx(expected_growth)
    assert report["effects"]["common_growth"]["ci95"] == pytest.approx([expected_growth] * 2)
    assert report["effects"]["rank_gain"]["mean"] == -.25
    assert report["effects"]["reach12"]["mean"] == -.25


def test_refresh_alternative_requires_two_ordinary_accessible_copies_and_a_gold_purchase():
    card = next(iter(golden_versions()))
    unit = {"card":card, "golden":False}
    shop = [{"owned":2, "triple_need":3, "costs_health":False, "price":3}]
    player = {"board":[unit], "hand":[{"locked":0,"item":{"Minion":unit}}],
              "shop":[{"item":{"Minion":unit}}]}
    legal = np.ones(245,dtype=bool)
    game = SimpleNamespace(player_json=lambda _:json.dumps(player),legal_mask=lambda _:legal,
                           view=lambda _:json.dumps({"me":{"shop":shop}}))
    p = np.zeros(245)
    p[1],p[8] = .9,.1
    offer = select("complete_triple",9,1,player,{},p,game,0,None)
    assert offer[0] == 8 and offer[1]["triple_card"] == card
    assert offer[1]["golden_before"] == count_golden(player,golden_versions()[card]) == 0
    assert select("complete_triple",9,8,player,{},p,game,0,None) is None
    shop[0]["costs_health"] = True
    assert offers(game,0) == []
    shop[0]["costs_health"] = False
    player["hand"][0]["locked"] = 1
    assert offers(game,0) == []
    player["hand"][0]["locked"] = 0
    legal[8] = False
    assert offers(game,0) == []
