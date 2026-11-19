import json
from types import SimpleNamespace

import numpy as np
import pytest

from hsbg.az.train.sale_intent_policy import FEATURES, SaleIntentPolicy


class Evaluator:
    model = None

    def __call__(self, f, c, mask, seat):
        p = np.zeros(245)
        p[[0, 1, 16, 17]] = [1, 2, 4, 3]
        p[~mask] = 0
        return p / p.sum(), 0


class Parent:
    def __init__(self):
        self.evaluator = Evaluator()

    def reset(self):
        pass

    def decide(self, game, seat):
        mask = np.ones(245, bool)
        p, _ = self.evaluator(None, None, mask, seat)
        return SimpleNamespace(action=int(p.argmax()), info={})


def model():
    n = len(FEATURES)
    return dict(features=FEATURES, bias=2, center=[0]*n, scale=[1]*n, coef=[0]*n)


def test_once_per_eligible_turn_and_seat():
    actor = SaleIntentPolicy(Parent(), model(), {"cards":[]})
    state = dict(health=25,armor=0,gold=0,board=[dict(atk=6,hp=6), dict(atk=4,hp=4)],hand=[],shop=[],tier=4)
    game = SimpleNamespace(turn=7,player_json=lambda seat:json.dumps(state))
    assert actor.decide(game,0).action == 1
    assert actor.decide(game,1).action == 1
    assert actor.decide(game,0).action == 16
    game.turn += 1
    state["gold"] = 1
    assert actor.decide(game,0).action == 16
    state["gold"] = 0
    assert actor.decide(game,0).action == 1
    actor.reset()
    assert actor.overrides == []
    assert actor.decide(game,0).action == 1


@pytest.mark.parametrize("changes", [{"scale":[0]*9}, {"coef":[0]*8}, {"bias":float("nan")},
                                     {"features":FEATURES[::-1]}])
def test_invalid_classifier_rejected(changes):
    with pytest.raises(ValueError):
        SaleIntentPolicy(Parent(),{**model(), **changes},{"cards":[]})
