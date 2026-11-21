"""Standalone inference architecture and loader for ``obs2-policy-v1`` checkpoints.

Takes patched AutoTavern ``observe2`` inputs (4017 float32 features, 447 int32 card IDs)
and the native legal action mask (first 196 actions, excluding Move) and predicts one
legal native action index via a two-stage (source group -> conditional target) recurrent
policy.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
from . import runtime as tg
import torch
from torch import nn

from .recurrent import HiddenState, ProjectedLSTMCell

FORMAT = "obs2-policy-v1"
# Numeric columns per card slot: own minion 31 + 9 (board_x), item 40, opponent 9 + 35 (opp_sum).
WIDTH = 44


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 precision and return float32."""
    u = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
    return (rounded << 16).view(np.float32)


def log_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


class Layout:
    """Maps AutoTavern ``observe2`` features, card slots, and native actions to network indices."""

    def __init__(self) -> None:
        if not hasattr(tg, "obs2_blocks"):
            raise RuntimeError(
                "Installed taverngym does not expose observe2 / obs2_blocks(). "
                "Apply patches/autotavern-observe2.patch to AutoTavern commit fb61201 and rebuild."
            )
        d = self.d = dict(tg.layout())
        float_blocks, card_blocks = tg.obs2_blocks()
        self.floats, self.cards = d["OBS2_FLOATS"], d["OBS2_CARDS"]
        fo: dict[str, tuple[int, int]] = {}
        co: dict[str, tuple[int, int]] = {}

        offset = d["OBS_FLOATS"]
        for name, n in float_blocks:
            fo[name] = (offset, n)
            offset += n
        assert offset == self.floats

        offset = d["OBS_CARDS"]
        for name, n in card_blocks:
            co[name] = (offset, n)
            offset += n
        assert offset == self.cards

        powers = d["N_POWERS"]
        board = 1 + powers + 2 + 3 * d["N_QUESTS"] + d["N_SECRETS"]
        hand = board + d["N_BOARD"]
        shop = hand + d["N_HAND"]
        choice = shop + d["N_SHOP"]
        trinket = choice + d["N_CHOICE"]
        opponent = trinket + 8
        opp_board = opponent + d["N_OPP"]
        assert opp_board + d["N_OPP"] * d["N_BOARD"] == d["OBS_CARDS"]
        self.board, self.hand, self.shop, self.choice, self.opponent = (
            board,
            hand,
            shop,
            choice,
            opponent,
        )

        zones = [
            (0, 1),
            (1, powers),
            (1 + powers, 1),
            (2 + powers, 1),
            (3 + powers, 3 * d["N_QUESTS"]),
            (3 + powers + 3 * d["N_QUESTS"], d["N_SECRETS"]),
            (board, d["N_BOARD"]),
            (hand, d["N_HAND"]),
            (shop, d["N_SHOP"]),
            (choice, d["N_CHOICE"]),
            (trinket, 8),
            (opponent, d["N_OPP"]),
            (opp_board, d["N_OPP"] * d["N_BOARD"]),
        ] + list(co.values())
        self.zones = len(zones)

        slot = np.zeros((self.cards, self.zones + 1), np.float32)
        for z, (start, count) in enumerate(zones):
            slot[start : start + count, z] = 1.0
            slot[start : start + count, -1] = np.arange(count) / max(count, 1)
        assert slot[:, :-1].sum() == self.cards
        self.slot = slot

        zero = self.floats
        gather = np.full((self.cards, WIDTH), zero, np.int64)
        offset = d["GLOBAL_F"] + d["N_QUESTS"] * d["QUEST_F"] + d["PENDING_KINDS"]
        controller = list(range(offset))
        for start, count, width in (
            (board, d["N_BOARD"], d["MINION_F"]),
            (hand, d["N_HAND"], d["ITEM_F"]),
            (shop, d["N_SHOP"], d["ITEM_F"]),
            (choice, d["N_CHOICE"], d["ITEM_F"]),
            (opponent, d["N_OPP"], d["OPP_F"]),
            (opp_board, d["N_OPP"] * d["N_BOARD"], d["MINION_F"]),
        ):
            for i in range(count):
                gather[start + i, :width] = offset + i * width + np.arange(width)
            offset += count * width
        assert offset == d["OBS_FLOATS"]

        x, start = d["X_MINION"], fo["board_x"][0]
        for i in range(d["N_BOARD"]):
            gather[board + i, d["MINION_F"] : d["MINION_F"] + x] = (
                start + i * x + np.arange(x)
            )
        width, start = d["OPP_SUM_F"], fo["opp_sum"][0]
        for i in range(d["N_OPP"]):
            gather[opponent + i, d["OPP_F"] : d["OPP_F"] + width] = (
                start + i * width + np.arange(width)
            )
        for cards_key, floats_key, per in (
            ("board_att", "board_att_n", 1),
            ("hand_att", "hand_att_n", 1),
            ("shop_att", "shop_att_n", 1),
            ("choice_att", "choice_att_n", 1),
            ("oppb_att", "oppb_att_n", 1),
            ("effects", "effects_n", 1),
            ("counters", "counters_n", 1),
            ("levels", "levels_n", 1),
            ("sched", "sched_f", 3),
        ):
            (cs, cn), (fs, fn) = co[cards_key], fo[floats_key]
            assert fn == cn * per
            for i in range(cn):
                gather[cs + i, :per] = fs + i * per + np.arange(per)
        self.gather = gather

        for name in ("g2", "quest_data", "last_combat"):
            start, n = fo[name]
            controller += list(range(start, start + n))
        self.controller = np.array(controller)

        attached: list[int] = []
        parent: list[int] = []
        for cards_key, start, count, k in (
            ("board_att", board, d["N_BOARD"], d["K_ATT_BOARD"]),
            ("hand_att", hand, d["N_HAND"], d["K_ATT_ITEM"]),
            ("shop_att", shop, d["N_SHOP"], d["K_ATT_ITEM"]),
            ("choice_att", choice, d["N_CHOICE"], d["K_ATT_ITEM"]),
            ("oppb_att", opp_board, d["N_OPP"] * d["N_BOARD"], d["K_ATT_OPP"]),
        ):
            cs, cn = co[cards_key]
            assert cn == count * k
            attached += range(cs, cs + cn)
            parent += [start + i for i in range(count) for _ in range(k)]
        self.attached = np.array(attached)
        self.parent = np.array(parent)

        # Fixed native action layout excluding Move (actions 0..A_MOVE-1).
        self.actions = n = d["A_MOVE"]
        kind = np.zeros(n, np.int64)
        source = np.full(n, -1)
        target = np.full(n, -1)
        position = np.full(n, -1.0)
        chosen = np.zeros(n)
        for a in range(n):
            if a < d["A_BUY"]:
                kind[a] = [7, 1, 2, 3, 4, 4, 5, 8][a]
                if d["A_HERO_POWER"] <= a < d["A_DARK_GIFT"]:
                    source[a] = 1 + a - d["A_HERO_POWER"]
            elif a < d["A_SELL"]:
                kind[a], source[a] = 6, shop + a - d["A_BUY"]
            elif a < d["A_ACTIVATE"]:
                kind[a], source[a] = 9, board + a - d["A_SELL"]
            elif a < d["A_CHOOSE"]:
                kind[a], source[a] = 10, board + a - d["A_ACTIVATE"]
            elif a < d["A_PLAY"]:
                kind[a], source[a], target[a] = (
                    0,
                    1 + powers + 1,
                    choice + a - d["A_CHOOSE"],
                )
                chosen[a] = a - d["A_CHOOSE"] + 1
            elif a < d["A_MAGNETIZE"]:
                h, p = divmod(a - d["A_PLAY"], d["N_POS"])
                kind[a], source[a], position[a] = 11, hand + h, p
            else:
                h, b = divmod(a - d["A_MAGNETIZE"], d["N_BOARD"])
                kind[a], source[a], target[a] = 12, hand + h, board + b

        keys: dict[tuple[Any, ...], int] = {}
        group = np.zeros(n, np.int64)
        for a in range(n):
            if kind[a] == 0:
                key: tuple[Any, ...] = ("choice",)
            elif kind[a] in (7, 8):
                key = ("kind", int(kind[a]))
            else:
                key = (int(kind[a]), int(source[a]))
            group[a] = keys.setdefault(key, len(keys))
        self.groups = len(keys)

        members = [[a for a in range(n) if group[a] == g] for g in range(self.groups)]
        self.width = max(map(len, members))
        self.members = np.full((self.groups, self.width), -1)
        self.member = np.zeros(n, np.int64)
        for g, m in enumerate(members):
            self.members[g, : len(m)] = m
            self.member[m] = np.arange(len(m))
        self.group_kind = np.array([kind[m[0]] for m in members])
        self.group_source = np.array([source[m[0]] for m in members])
        self.kind, self.source, self.target, self.group = kind, source, target, group
        self.numeric = np.stack(
            [
                position / 10.0,
                np.zeros(n),
                chosen / 4.0,
                np.full(n, -0.25),
                np.zeros(n),
                np.zeros(n),
                (target >= 0).astype(float),
                np.ones(n),
            ],
            axis=1,
        ).astype(np.float32)


class Obs2Network(nn.Module):
    """Two-stage (source -> target) recurrent policy network over AutoTavern ``observe2``."""

    def __init__(
        self,
        layout: Layout | None = None,
        card: int = 32,
        state: int = 512,
        layers: int = 0,
        heads: int = 4,
        dropout: float = 0.0,
        flat: int | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(
            card=card,
            state=state,
            layers=layers,
            heads=heads,
            dropout=dropout,
            flat=flat,
        )
        L = self.layout = layout or Layout()
        width, data = tg.card_static()
        static = np.frombuffer(data, "<f4").reshape(tg.N_CARDS, width)
        self.register_buffer(
            "static",
            torch.tensor(np.sign(static) * np.log1p(np.abs(static))),
            persistent=False,
        )
        self.register_buffer("slot", torch.tensor(L.slot), persistent=False)
        for name in (
            "gather",
            "controller",
            "attached",
            "parent",
            "members",
            "member",
            "group_kind",
            "group_source",
            "kind",
            "source",
            "target",
            "group",
        ):
            self.register_buffer(
                name + "_index",
                torch.as_tensor(getattr(L, name), dtype=torch.long),
                persistent=False,
            )
        self.register_buffer("numeric", torch.tensor(L.numeric), persistent=False)
        membership = np.zeros((L.actions, L.groups), np.float32)
        membership[np.arange(L.actions), L.group] = 1.0
        self.register_buffer("membership", torch.tensor(membership), persistent=False)

        self.hidden_size = 256
        self.slot_width = L.zones + 1
        self.entity_width = WIDTH + self.slot_width + width

        self.card_id_embedding = nn.Embedding(tg.N_CARDS, card)
        self.card_encoder = nn.Sequential(
            nn.Linear(card + self.entity_width, card),
            nn.ReLU(),
            nn.Linear(card, card),
            nn.ReLU(),
        )
        self.attachment = nn.Linear(card, card, bias=False)
        self.controller_encoder = nn.Sequential(
            nn.Linear(len(L.controller), 32), nn.ReLU()
        )
        self.hero_embedding = nn.Embedding(3, 32)
        self.player_context = nn.Sequential(nn.Linear(96, 32), nn.ReLU())
        self.mixer = None
        if layers:
            self.context_token = nn.Linear(32, card)
            self.mixer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    card,
                    heads,
                    4 * card,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                ),
                layers,
                enable_nested_tensor=False,
            )
        self.flatten = nn.Linear(card, flat) if flat else None
        self.drop = nn.Dropout(dropout)
        self.state_encoder = nn.Sequential(
            nn.Linear(L.cards * (flat or card) + 32, state),
            nn.ReLU(),
            nn.Linear(state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.kind_embedding = nn.Embedding(14, 256)
        self.none_card = nn.Parameter(torch.zeros(card))
        self.source_fusion = nn.Linear(card + 256, 256)
        self.target_fusion = nn.Linear(card * 3 + 256, 256)
        self.action_numeric = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256)
        )
        self.recurrent = ProjectedLSTMCell(256, 256, 256, forget_bias=1.0)
        self.source_query = nn.Linear(256, 256, bias=False)
        self.target_query = nn.Linear(256, 256, bias=False)
        self.value = nn.Linear(256, 1)
        self.tables: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return self.none_card.device

    def train(self, mode: bool = True):
        self.tables = None
        return super().train(mode)

    def initial_state(self, batch_size: int = 1) -> HiddenState:
        zero = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return zero, zero.clone()

    def encode(
        self,
        floats: torch.Tensor,
        cards: torch.Tensor,
        width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.cat((floats, floats.new_zeros(len(floats), 1)), 1)
        numbers = log_scale(raw[:, self.gather_index])
        weight, bias = self.card_encoder[0].weight, self.card_encoder[0].bias
        card = self.config["card"]
        a, b, c = card, card + WIDTH, card + WIDTH + self.slot_width
        tables = self.tables
        if tables is None:
            tables = (
                self.card_id_embedding.weight @ weight[:, :a].T
                + self.static @ weight[:, c:].T,
                self.slot @ weight[:, b:c].T + bias,
            )
            if not self.training and not torch.is_grad_enabled():
                self.tables = tables
        per_card, per_slot = tables
        hidden = torch.relu(
            numbers @ weight[:, a:b].T
            + nn.functional.embedding(cards, per_card)
            + per_slot
        )
        entities = self.card_encoder[3](self.card_encoder[2](hidden))
        attached = entities[:, self.attached_index] * (
            cards[:, self.attached_index] > 0
        ).unsqueeze(-1)
        folded = torch.zeros_like(entities).index_add(1, self.parent_index, attached)
        entities = entities + self.attachment(folded)

        control = self.controller_encoder(log_scale(raw[:, self.controller_index]))
        hero = self.hero_embedding(
            torch.zeros(len(raw), dtype=torch.long, device=raw.device)
        )
        context = self.player_context(torch.cat((control, hero, hero), -1))
        if self.mixer is not None:
            entities = self.mix(entities, cards, context, width)
        flat = entities if self.flatten is None else self.flatten(entities)
        return (
            self.state_encoder(self.drop(torch.cat((flat.flatten(1), context), -1))),
            entities,
        )

    def mix(
        self,
        entities: torch.Tensor,
        cards: torch.Tensor,
        context: torch.Tensor,
        width: int | None = None,
    ) -> torch.Tensor:
        present = cards > 0
        counts = present.sum(1)
        max_w = max(int(counts.max()), 1) if width is None else width
        order = torch.argsort((~present).to(torch.int8), dim=1, stable=True)[:, :max_w]
        valid = torch.arange(order.shape[1], device=cards.device)[None] < counts[:, None]
        index = order[..., None].expand(-1, -1, entities.shape[-1])
        tokens = torch.cat(
            (self.context_token(context)[:, None], entities.gather(1, index)), 1
        )
        padding = torch.cat((valid.new_zeros(len(valid), 1), ~valid), 1)
        mixed = self.mixer(tokens, src_key_padding_mask=padding)
        change = torch.where(valid[..., None], mixed[:, 1:] - tokens[:, 1:], 0)
        return entities.scatter_add(1, index, change)

    def sources(self, entities: torch.Tensor) -> torch.Tensor:
        slot = self.group_source_index
        card = entities[:, slot.clamp_min(0)]
        card = torch.where((slot >= 0)[None, :, None], card, self.none_card)
        kinds = self.kind_embedding(self.group_kind_index).expand(len(entities), -1, -1)
        return torch.tanh(self.source_fusion(torch.cat((card, kinds), -1)))

    def targets(
        self, entities: torch.Tensor, groups: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = self.members_index[groups]
        valid = actions >= 0
        actions = actions.clamp_min(0)
        rows = torch.arange(len(entities), device=entities.device)[:, None]

        def card(slot: torch.Tensor) -> torch.Tensor:
            found = entities[rows, slot.clamp_min(0)]
            return torch.where((slot >= 0)[..., None], found, self.none_card)

        joined = torch.cat(
            (
                card(self.source_index[actions]),
                card(self.target_index[actions]),
                self.kind_embedding(self.kind_index[actions]),
                self.none_card.expand(*actions.shape, -1),
            ),
            -1,
        )
        embedded = torch.tanh(
            self.target_fusion(joined) + self.action_numeric(self.numeric[actions])
        )
        return embedded, actions, valid

    def group_legal(self, legal: torch.Tensor) -> torch.Tensor:
        return (legal.float() @ self.membership) > 0

    @torch.no_grad()
    def step(
        self,
        floats: torch.Tensor,
        cards: torch.Tensor,
        legal: torch.Tensor,
        hidden: HiddenState,
        forced: int | None = None,
        sample: bool = False,
    ) -> tuple[int, int, HiddenState]:
        """Single decision step.

        Returns ``(action, greedy_action, next_hidden)``.
        """
        states, entities = self.encode(floats[None], cards[None])
        after_state = self.recurrent(states, hidden)
        sources = self.sources(entities)[0]
        logits = (sources @ self.source_query(after_state[0])[0]).masked_fill(
            ~self.group_legal(legal[None])[0], -torch.inf
        )
        greedy_group = int(logits.argmax())
        if forced is not None:
            group = int(self.group_index[forced])
        elif sample:
            group = int(torch.multinomial(logits.softmax(-1), 1).item())
        else:
            group = greedy_group

        after_source = self.recurrent(sources[group][None], after_state)
        embedded, actions, valid = self.targets(
            entities, torch.tensor([group], device=entities.device)
        )
        allowed = valid[0] & legal[actions[0]]
        target = (embedded[0] @ self.target_query(after_source[0])[0]).masked_fill(
            ~allowed, -torch.inf
        )
        greedy = int(actions[0, int(target.argmax())]) if group == greedy_group else -1
        if forced is not None:
            chosen = int(forced)
        elif sample:
            chosen = int(actions[0, int(torch.multinomial(target.softmax(-1), 1).item())])
        else:
            chosen = greedy
        return chosen, greedy, after_source

    @torch.no_grad()
    def act_batch(
        self,
        floats: torch.Tensor,
        cards: torch.Tensor,
        legal: torch.Tensor,
        hidden: HiddenState,
        forced: torch.Tensor | None = None,
        sample: torch.Tensor | bool = False,
        width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, HiddenState]:
        """Batched decision step supporting greedy argmax and stochastic sampling."""
        states, entities = self.encode(floats, cards, width=width)
        after_state = self.recurrent(states, hidden)
        sources = self.sources(entities)
        logits = (sources * self.source_query(after_state[0])[:, None]).sum(-1)
        slogp = logits.masked_fill(~self.group_legal(legal), -torch.inf).log_softmax(-1)
        rows = torch.arange(len(slogp), device=slogp.device)

        if isinstance(sample, bool):
            sample_mask = torch.full(
                (len(slogp),), sample, dtype=torch.bool, device=slogp.device
            )
        else:
            sample_mask = sample.to(dtype=torch.bool, device=slogp.device)

        if forced is None:
            forced_tensor = torch.full(
                (len(slogp),), -1, dtype=torch.long, device=slogp.device
            )
        else:
            forced_tensor = forced.to(dtype=torch.long, device=slogp.device)

        drawn_group = torch.multinomial(slogp.exp(), 1)[:, 0]
        group = torch.where(
            forced_tensor >= 0,
            self.group_index[forced_tensor.clamp_min(0)],
            torch.where(sample_mask, drawn_group, slogp.argmax(-1)),
        )
        after_source = self.recurrent(sources[rows, group], after_state)
        embedded, actions, valid = self.targets(entities, group)
        allowed = valid & legal.gather(1, actions)
        tlogits = (embedded * self.target_query(after_source[0])[:, None]).sum(-1)
        tlogp = tlogits.masked_fill(~allowed, -torch.inf).log_softmax(-1)
        drawn_member = torch.multinomial(tlogp.exp(), 1)[:, 0]
        member = torch.where(
            forced_tensor >= 0,
            self.member_index[forced_tensor.clamp_min(0)],
            torch.where(sample_mask, drawn_member, tlogp.argmax(-1)),
        )
        chosen_actions = actions[rows, member]
        log_probs = slogp[rows, group] + tlogp[rows, member]
        return chosen_actions, log_probs, after_source


def load_policy(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Obs2Network, dict[str, Any]]:
    """Safely load an ``obs2-policy-v1`` checkpoint using ``weights_only=True``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format')!r}; expected {FORMAT!r}"
        )
    net = Obs2Network(**payload.get("config", {}))
    net.load_state_dict(payload["model"])
    net.to(device).eval()
    net.tables = None
    return net, payload


class Obs2Policy:
    """High-level stateful policy wrapper for a single seat.

    Call ``new_game()`` at the start of every lobby to reset the recurrent LSTM state.
    Uses only AutoTavern's native legal action mask (``action_mask[:196]``).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
        sample: bool = False,
        bf16_obs: bool = False,
    ) -> None:
        self.net, self.payload = load_policy(checkpoint, device=device)
        self.sample = sample
        self.bf16_obs = bf16_obs
        self.hidden: HiddenState = self.net.initial_state()

    def new_game(self) -> None:
        self.hidden = self.net.initial_state()

    def _decode_inputs(
        self,
        features: np.ndarray | bytes | bytearray | memoryview | str,
        cards: np.ndarray | bytes | bytearray | memoryview | str,
        action_mask: np.ndarray | bytes | bytearray | memoryview | str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(features, str):
            features = np.frombuffer(base64.b64decode(features), "<f4")
            cards = np.frombuffer(base64.b64decode(cards), "<i4")
            action_mask = np.frombuffer(base64.b64decode(action_mask), np.uint8)
        else:
            if isinstance(features, (bytes, bytearray, memoryview)):
                features = np.frombuffer(features, "<f4")
            if isinstance(cards, (bytes, bytearray, memoryview)):
                cards = np.frombuffer(cards, "<i4")
            if isinstance(action_mask, (bytes, bytearray, memoryview)):
                action_mask = np.frombuffer(action_mask, np.uint8)
        f_arr = np.asarray(features, dtype=np.float32)
        if not f_arr.flags.writeable:
            f_arr = f_arr.copy()
        if self.bf16_obs:
            f_arr = bf16_round(f_arr)
        c_arr = np.asarray(cards, dtype=np.int64)
        if not c_arr.flags.writeable:
            c_arr = c_arr.copy()
        m_arr = np.asarray(action_mask)[: self.net.layout.actions].astype(bool, copy=True)
        return f_arr, c_arr, m_arr

    def act(
        self,
        features: np.ndarray | bytes | bytearray | memoryview | str,
        cards: np.ndarray | bytes | bytearray | memoryview | str,
        action_mask: np.ndarray | bytes | bytearray | memoryview | str,
        sample: bool | None = None,
    ) -> int:
        f_arr, c_arr, m_arr = self._decode_inputs(features, cards, action_mask)
        device = self.net.device
        floats = torch.as_tensor(f_arr, device=device)
        ids = torch.as_tensor(c_arr, device=device)
        legal = torch.as_tensor(m_arr, device=device)
        use_sample = self.sample if sample is None else sample
        action, _, self.hidden = self.net.step(
            floats, ids, legal, self.hidden, sample=use_sample
        )
        return action

    def act_game(self, game: Any, seat: int, sample: bool | None = None) -> int:
        """Read ``observe2`` and native ``legal_mask`` directly from a ``taverngym.Game``."""
        features, cards = game.observe2(seat)
        mask = game.legal_mask(seat)
        return self.act(features, cards, mask, sample=sample)
