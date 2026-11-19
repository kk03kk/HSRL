"""Greedy research policy combining a verified parent with a public pre-sale classifier."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np

from hsbg.runtime.artifacts import file_identity


FEATURES = ("turn", "effective_health", "board_count", "sold_share", "hand_count",
            "playable_hand_count", "tier", "one_gold_offers", "sale_log_margin")
CARDS = Path(__file__).resolve().parents[3] / "engine/crates/hsbg-engine/data/cards.json"


class SaleIntentPolicy:
    name = "peer_sale_intent_v1"
    supports_pi = False

    def __init__(self, parent, classifier, card_table):
        if tuple(classifier["features"]) != FEATURES:
            raise ValueError("incompatible sale-intent feature order")
        self.center, self.scale, self.coef = (
            np.asarray(classifier[key], dtype=float) for key in ("center", "scale", "coef"))
        self.bias = float(classifier["bias"])
        if (any(x.shape != (len(FEATURES),) or not np.isfinite(x).all()
                for x in (self.center, self.scale, self.coef))
                or np.any(self.scale <= 0) or not math.isfinite(self.bias)):
            raise ValueError("invalid sale-intent coefficients")
        self.costs = [entry["cost"] for entry in card_table["cards"]]
        self.parent = parent
        self.evaluate = parent.evaluator
        self.model = self.evaluate.model
        parent.evaluator = self
        self.reset()

    def reset(self):
        self.parent.reset()
        self.used_turn = {}
        self.overrides = []

    def decide(self, game, seat):
        self.game = game
        self.proposal = None
        choice = self.parent.decide(game, seat)
        if self.proposal is not None and not choice.info.get("forced"):
            original, executed = self.proposal
            if choice.action != executed:
                raise ValueError("sale-intent evaluator and executed action differ")
            self.used_turn[seat] = game.turn
            if original != executed:
                self.overrides.append((game.turn, original, executed))
        return choice

    def __call__(self, floats, cards, mask, seat):
        probabilities, value = self.evaluate(floats, cards, mask, seat)
        sale = int(probabilities.argmax())
        turn = self.game.turn
        if turn < 7 or self.used_turn.get(seat) == turn or not 16 <= sale < 23:
            return probabilities, value
        player = json.loads(self.game.player_json(seat))
        board = player["board"]
        health = player["health"] + player["armor"]
        if player["gold"] or ((health <= 15) == (len(board) <= 4)):
            return probabilities, value
        sizes = [unit["atk"] + unit["hp"] for unit in board]
        total = max(1, sum(sizes))
        if sizes[sale - 16] * 100 < 15 * total:
            return probabilities, value
        alternatives = mask.copy()
        for i, size in enumerate(sizes):
            if size * 100 >= 15 * total:
                alternatives[16 + i] = False
        if not alternatives.any():
            return probabilities, value
        offers = sum("Spell" in slot["item"] and 0 < self.costs[slot["item"]["Spell"]["card"]] <= 1
                     for slot in player["shop"])
        playable = int(mask[46:126].reshape(10, 8).any(1).sum())
        margin = math.log(max(1e-12, float(probabilities[sale])) /
                          max(1e-12, float(probabilities[alternatives].max())))
        features = [turn, health, len(board), sizes[sale - 16] / total, len(player["hand"]),
                    playable, player["tier"], offers, min(12, margin)]
        logit = self.bias + ((np.asarray(features) - self.center) / self.scale) @ self.coef
        score = 1 / (1 + math.exp(-float(np.clip(logit, -30, 30))))
        if score >= .8:
            probabilities, value = self.evaluate(floats, cards, alternatives, seat)
        self.proposal = (sale, int(probabilities.argmax()))
        return probabilities, value


def load_bundle(directory, protocol, load_parent):
    from .common import engine_info

    directory = Path(directory).resolve()
    spec = json.loads((directory / "policy.json").read_bytes())
    runtime = engine_info()
    table = CARDS.read_bytes().replace(b"\r\n", b"\n")
    if (spec["schema"] != "hsai.sale_intent/v1" or spec["threshold"] != .8
            or spec["action_profile"] != os.environ.get("HSAI_ACTION_PROFILE")
            or sha256(table).hexdigest() != spec["feature_cards_sha256"]
            or protocol.engine.get("source_hash") != runtime["source_hash"]
            or runtime["source_hash"] not in spec["engine_sources"]):
        raise ValueError("sale-intent bundle is incompatible with this runtime or field")
    protocol.assert_training_contract()
    parent = directory / spec["parent"]["path"]
    classifier = directory / "classifier.json"
    if (file_identity(parent) != spec["parent"]["identity"]
            or file_identity(classifier) != spec["classifier"]):
        raise ValueError("sale-intent bundle assets differ")
    result = load_parent(str(parent))
    result["policy"] = SaleIntentPolicy(result["policy"], json.loads(classifier.read_bytes()), json.loads(table))
    result.update(kind="research-policy", real_path=str(directory),
                  meta={**result["meta"], "research_policy_sha256": file_identity(directory/"policy.json")["sha256"]})
    return result
