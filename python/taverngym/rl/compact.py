"""49-token owner pooling with static card features and deterministic lexical rule features.

规则文本hash是词形特征，不是预训练语义embedding；本模型不是论文私有实现的复刻。
"""
from functools import lru_cache
import hashlib
import json
import re

import numpy as np
import torch
from torch import nn
import taverngym as tg
from .model import LegacyPolicy, signed_log


@lru_cache(maxsize=1)
def card_assets():
    width, raw = tg.card_static()
    static = np.frombuffer(raw, dtype="<f4").reshape(tg.N_CARDS, width).copy()
    catalog = {c["id"]: c for c in json.loads(tg.catalog_json())}
    text = np.zeros((tg.N_CARDS, 64), dtype=np.float32)
    for i in range(1, tg.N_CARDS):
        card = catalog[tg.card_info(i)[0]]
        words = re.findall(r"[a-z]+|\d+", card["text"].lower())
        for token in words + [a + ":" + b for a, b in zip(words, words[1:])]:
            h = hashlib.blake2s(token.encode(), digest_size=4).digest()
            text[i, int.from_bytes(h[:2], "little") % 64] += 1 if h[2] & 1 else -1
        norm = np.linalg.norm(text[i])
        if norm:
            text[i] /= norm
    return torch.from_numpy(static), torch.from_numpy(text)


class NoMemory(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.hidden = hidden

    def forward(self, x, hidden=None):
        return self.projection(x), x.new_zeros((1, x.shape[0], self.hidden))


class CompactPolicy(LegacyPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_recurrent = cfg.policy_kind == "compact_gru"
        if not self.is_recurrent:
            self.memory = NoMemory(cfg.dim, cfg.hidden)
        static, text = card_assets()
        if cfg.policy_kind == "entity_ff":
            from .rule_assets import load_assets
            text = load_assets(cfg.rule_asset_path, cfg.rule_asset_sha256)
        self.register_buffer("static_cards", static.clone())
        self.register_buffer("rule_text", text.clone())
        self.static_projection = nn.Linear(static.shape[1], cfg.dim, bias=False)
        self.text_projection = nn.Linear(text.shape[1], cfg.dim, bias=False)
        self.compact_position = nn.Embedding(49, cfg.dim)
        self.critic = nn.Linear(cfg.hidden, 1)
        l = dict(tg.layout())
        base = 1 + l["N_POWERS"] + 2 + 3 * l["N_QUESTS"] + l["N_SECRETS"]
        owners = torch.zeros(tg.OBS_CARDS + 1, dtype=torch.long)
        for j in range(41):
            owners[base + 1 + j] = j + 1
        opp_hero = base + 41 + 8
        for j in range(7):
            owners[opp_hero + 1 + j] = 42 + j
            for k in range(7):
                owners[opp_hero + 8 + j * 7 + k] = 42 + j
        self.register_buffer("owner_map", owners)
        self.token_count = 49

    def encode(self, obs):
        f = signed_log(obs["features"].float())
        z = torch.cat([f, torch.zeros_like(f[:, :1])], 1)[:, self.numeric_map]
        def card_repr(ids):
            return self.card(ids) + self.static_projection(signed_log(self.static_cards[ids])) + self.text_projection(self.rule_text[ids])
        base = card_repr(obs["cards"]) + self.position(self.positions) + self.numeric(z)
        effects = card_repr(obs["extra_ids"]) + self.extra(signed_log(obs["numbers"]))
        batch, dim = len(f), base.shape[-1]
        pooled = base.new_zeros((batch, 49, dim))
        counts = base.new_zeros((batch, 49, 1))
        def add(values, owner, valid):
            index = owner[..., None].expand(-1, -1, dim)
            pooled.scatter_add_(1, index, values * valid[..., None])
            counts.scatter_add_(1, owner[..., None], valid[..., None].to(base.dtype))
        add(base, self.owner_map[1:][None].expand(batch, -1), obs["cards"] != 0)
        add(effects, self.owner_map[obs["owners"]], obs["present"])
        pooled = pooled / counts.clamp_min(1).sqrt()
        pooled[:, 0] += self.global_encoder(f)
        pooled += self.compact_position.weight[None]
        padding = counts.squeeze(-1) == 0
        padding[:, 0] = False
        encoded = self.entities(pooled, src_key_padding_mask=padding)
        # pointer仍引用原始125槽，映射回其宿主；不改245动作语义。
        return encoded[:, 0], encoded[:, self.owner_map]

    def heads(self, memory, entities, mask):
        logits, _, rank = super().heads(memory, entities, mask)
        return logits, self.critic(memory).tanh().squeeze(-1), rank
