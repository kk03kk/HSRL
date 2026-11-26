"""An entity Transformer policy for TavernGym observations, based on TavernAgent."""

import hashlib
import json
import math
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from taverngym import N_CARDS, card_static, catalog_json, layout


class PolicyOutput(NamedTuple):
    logits: Tensor
    value: Tensor
    placement_logits: Tensor


def _rule_embeddings(width=128):
    table = np.zeros((N_CARDS, width), dtype=np.float32)
    for card in json.loads(catalog_json()):
        text = " ".join(card["text"].lower().split())
        row = table[card["cid"] + 1]
        for i in range(len(text) - 2):
            code = hashlib.blake2b(text[i : i + 3].encode(), digest_size=4).digest()
            n = int.from_bytes(code, "little")
            row[(n >> 1) % width] += 1 if n & 1 else -1
        norm = np.linalg.norm(row)
        if norm:
            row /= norm
    return torch.from_numpy(table)


def _sym_log(x):
    return x.sign() * x.abs().log1p()


def _masked_mean(x, present):
    return (x * present.unsqueeze(-1)).sum(dim=-2) / present.sum(dim=-1, keepdim=True).clamp_min(1)


class EntityPolicy(nn.Module):
    """Batched observations in, masked action logits, value and placement logits out.

    Row zero of a supplied rule-embedding table is reserved for padding. Pass only
    ready seats to a Categorical distribution. The action logits of inactive seats
    can all be masked.
    """

    def __init__(self, width=384, layers=8, heads=8, feedforward=1664, rule_embeddings=None):
        super().__init__()
        self.layout = sizes = dict(layout())
        global_size = (
            sizes["GLOBAL_F"] + sizes["N_QUESTS"] * sizes["QUEST_F"] + sizes["PENDING_KINDS"]
        )
        card_prefix = 3 + sizes["N_POWERS"] + 3 * sizes["N_QUESTS"] + sizes["N_SECRETS"]
        feature_offset = global_size
        card_offset = card_prefix
        self.groups = {}
        for name, count, size in (
            ("board", sizes["N_BOARD"], sizes["MINION_F"]),
            ("hand", sizes["N_HAND"], sizes["ITEM_F"]),
            ("shop", sizes["N_SHOP"], sizes["ITEM_F"]),
            ("choices", sizes["N_CHOICE"], sizes["ITEM_F"]),
        ):
            self.groups[name] = (
                slice(feature_offset, feature_offset + count * size),
                slice(card_offset, card_offset + count),
                count,
                size,
            )
            feature_offset += count * size
            card_offset += count
        self.global_features = slice(0, global_size)
        self.global_cards = (slice(0, card_prefix), slice(card_offset, card_offset + 8))
        opponent_count = sizes["N_OPP"]
        opponent_size = sizes["OPP_F"] * opponent_count
        self.opponent_features = slice(feature_offset, feature_offset + opponent_size)
        self.opponent_boards = slice(feature_offset + opponent_size, -sizes["ACTION_BUDGET_F"])
        self.opponent_cards = slice(card_offset + 8, card_offset + 8 + opponent_count)
        self.opponent_board_cards = slice(self.opponent_cards.stop, None)
        self.budget_features = slice(-sizes["ACTION_BUDGET_F"], None)

        static_width, data = card_static()
        static = np.frombuffer(data, dtype="<f4").reshape(N_CARDS, static_width).copy()
        self.register_buffer("static_cards", torch.from_numpy(static))
        rules = _rule_embeddings() if rule_embeddings is None else torch.as_tensor(rule_embeddings)
        if rules.ndim != 2 or rules.shape[0] != N_CARDS:
            raise ValueError("rule_embeddings must have one row per card index, including padding")
        rules = rules.detach().clone().float()
        rules[0].zero_()
        self.register_buffer("rule_cards", rules)

        self.card_id = nn.Embedding(N_CARDS, width, padding_idx=0)
        self.card_static = nn.Linear(static_width, width, bias=False)
        self.card_rules = nn.Linear(rules.shape[1], width, bias=False)
        self.global_projection = nn.Linear(global_size + sizes["ACTION_BUDGET_F"], width)
        self.minion_projection = nn.Linear(sizes["MINION_F"], width)
        self.item_projection = nn.Linear(sizes["ITEM_F"], width)
        self.opponent_projection = nn.Linear(sizes["OPP_F"], width)
        token_count = 1 + opponent_count + sum(group[2] for group in self.groups.values())
        self.slot_embedding = nn.Parameter(torch.empty(token_count, width))
        self.append_slot = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.append_slot, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.scalar_actions = nn.Linear(width, sizes["A_BUY"])
        self.pointers = nn.ModuleDict(
            {
                name: nn.Linear(width, width, bias=False)
                for name in ("buy", "sell", "activate", "choose", "play", "magnetize", "move")
            }
        )
        self.pair_context = nn.ModuleDict(
            {name: nn.Linear(width, width, bias=False) for name in ("play", "magnetize", "move")}
        )
        self.value = nn.Linear(width, 1)
        self.placement = nn.Linear(width, sizes["N_OPP"] + 1)
        self.pointer_scale = 1 / math.sqrt(width)

    def _embed_cards(self, ids):
        ids = ids.long()
        return (
            self.card_id(ids)
            + self.card_static(_sym_log(self.static_cards[ids]))
            + self.card_rules(self.rule_cards[ids])
        )

    def _tokens(self, features, cards):
        sizes = self.layout
        global_features = torch.cat(
            (features[:, self.global_features], features[:, self.budget_features]),
            dim=-1,
        )
        global_ids = torch.cat(tuple(cards[:, part] for part in self.global_cards), dim=-1)
        global_token = self.global_projection(_sym_log(global_features)) + _masked_mean(
            self._embed_cards(global_ids),
            global_ids.ne(0),
        )
        tokens = [global_token.unsqueeze(1)]
        present = [torch.ones_like(features[:, :1], dtype=torch.bool)]
        for name, (f_span, c_span, count, size) in self.groups.items():
            raw = features[:, f_span].reshape(-1, count, size)
            project = self.minion_projection if name == "board" else self.item_projection
            tokens.append(project(_sym_log(raw)) + self._embed_cards(cards[:, c_span]))
            # Some menu actions have no card ID, but still have a present flag.
            present.append(raw[:, :, 0].ne(0))

        opponent_count = sizes["N_OPP"]
        opponent = features[:, self.opponent_features].reshape(-1, opponent_count, sizes["OPP_F"])
        boards = features[:, self.opponent_boards].reshape(
            -1,
            opponent_count,
            sizes["N_BOARD"],
            sizes["MINION_F"],
        )
        board_ids = cards[:, self.opponent_board_cards].reshape(
            -1, opponent_count, sizes["N_BOARD"]
        )
        board_tokens = self.minion_projection(_sym_log(boards)) + self._embed_cards(board_ids)
        tokens.append(
            self.opponent_projection(_sym_log(opponent))
            + self._embed_cards(cards[:, self.opponent_cards])
            + _masked_mean(board_tokens, boards[..., 0].ne(0))
        )
        present.append(opponent[..., 0].ne(0))
        return torch.cat(tokens, dim=1), torch.cat(present, dim=1)

    def forward(self, features, cards, action_mask):
        tokens, present = self._tokens(features, cards)
        encoded = self.encoder(tokens + self.slot_embedding, src_key_padding_mask=~present)
        global_token = encoded[:, 0]
        board, hand, shop, choices, _ = encoded[:, 1:].split(
            [group[2] for group in self.groups.values()] + [self.layout["N_OPP"]],
            dim=1,
        )
        positions = torch.cat((board, global_token.unsqueeze(1) + self.append_slot), dim=1)

        def pointer(kind, slots):
            query = self.pointers[kind](global_token).unsqueeze(1)
            return (slots * query).sum(dim=-1) * self.pointer_scale

        def pairs(kind, source, target):
            query = self.pointers[kind](source) + self.pair_context[kind](global_token).unsqueeze(1)
            return ((query @ target.transpose(1, 2)) * self.pointer_scale).flatten(1)

        logits = torch.cat(
            (
                self.scalar_actions(global_token),
                pointer("buy", shop),
                pointer("sell", board),
                pointer("activate", board),
                pointer("choose", choices),
                pairs("play", hand, positions),
                pairs("magnetize", hand, board),
                pairs("move", board, board),
            ),
            dim=-1,
        )
        return PolicyOutput(
            logits.masked_fill(~action_mask.bool(), -torch.inf),
            self.value(global_token).squeeze(-1),
            self.placement(global_token),
        )
