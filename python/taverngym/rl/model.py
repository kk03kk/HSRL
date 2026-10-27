"""Entity encoder + GRU + source/target-aware masked policy and rank distribution."""
import numpy as np
import torch
from torch import nn
import taverngym as tg


def signed_log(x):
    return x.sign() * x.abs().log1p()


def pack(observations, device):
    if not observations:
        raise ValueError("empty observation batch")
    n = len(observations)
    width = max(len(o["extras"]) for o in observations)
    ids = np.zeros((n, width), dtype=np.int64)
    owners = np.zeros_like(ids)
    numbers = np.zeros((n, width, 8), dtype=np.float32)
    present = np.zeros((n, width), dtype=bool)
    for i, o in enumerate(observations):
        if len(o["extras"]) > 512:
            raise ValueError("effects schema capacity exceeded, refusing silent truncation")
        for j, (card, owner, values) in enumerate(o["extras"]):
            ids[i, j], owners[i, j], numbers[i, j], present[i, j] = card, owner, values, True
    arrays = {"features": np.stack([o["features"] for o in observations]),
              "cards": np.stack([o["cards"] for o in observations]).astype(np.int64),
              "mask": np.stack([o["mask"] for o in observations]),
              "extra_ids": ids, "owners": owners, "numbers": numbers, "present": present}
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(numbers).all():
        raise ValueError("nonfinite observation")
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


class LegacyPolicy(nn.Module):
    is_recurrent = True
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden
        self.card = nn.Embedding(tg.N_CARDS, cfg.dim, padding_idx=0)
        self.position = nn.Embedding(tg.OBS_CARDS + 1, cfg.dim)
        self.extra = nn.Sequential(nn.Linear(8, cfg.dim), nn.Tanh())
        self.global_encoder = nn.Sequential(nn.Linear(tg.OBS_FLOATS, cfg.dim), nn.LayerNorm(cfg.dim))
        l = dict(tg.layout())
        self.numeric = nn.Linear(l["ITEM_F"], cfg.dim)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * cfg.ff_multiplier,
                                          dropout=0.0, batch_first=True, norm_first=True)
        self.entities = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.memory = nn.GRU(cfg.dim, cfg.hidden, batch_first=True)
        self.action = nn.Embedding(tg.N_ACTIONS, cfg.dim)
        self.policy_head = nn.Sequential(nn.Linear(cfg.hidden + cfg.dim * 3, cfg.dim),
                                         nn.Tanh(), nn.Linear(cfg.dim, 1))
        self.rank_head = nn.Linear(cfg.hidden, 8)
        self.register_buffer("utility", torch.linspace(1, -1, 8))
        self.register_buffer("positions", torch.arange(1, tg.OBS_CARDS + 1))
        # Map base numeric blocks to the matching card slot (never concatenate IDs as floats).
        mapping = np.full((tg.OBS_CARDS, l["ITEM_F"]), tg.OBS_FLOATS, dtype=np.int64)
        c = 1 + l["N_POWERS"] + 2 + 3 * l["N_QUESTS"] + l["N_SECRETS"]
        f = l["GLOBAL_F"] + l["N_QUESTS"] * l["QUEST_F"] + l["PENDING_KINDS"]
        for count, size in ((l["N_BOARD"], l["MINION_F"]), (l["N_HAND"], l["ITEM_F"]),
                            (l["N_SHOP"], l["ITEM_F"]), (l["N_CHOICE"], l["ITEM_F"])):
            for _ in range(count):
                mapping[c, :size] = np.arange(f, f + size)
                c, f = c + 1, f + size
        c += 8 + l["N_OPP"]
        f += l["N_OPP"] * l["OPP_F"]
        for _ in range(l["N_OPP"] * l["N_BOARD"]):
            mapping[c, :l["MINION_F"]] = np.arange(f, f + l["MINION_F"])
            c, f = c + 1, f + l["MINION_F"]
        assert c == tg.OBS_CARDS and f == tg.OBS_FLOATS
        self.register_buffer("numeric_map", torch.from_numpy(mapping))
        # Binding Action strings are compiled from the authoritative flat action layout.
        import re
        base = 1 + l["N_POWERS"] + 2 + 3 * l["N_QUESTS"] + l["N_SECRETS"]
        zones = {"board": base + 1, "hand": base + l["N_BOARD"] + 1,
                 "shop": base + l["N_BOARD"] + l["N_HAND"] + 1,
                 "choice": base + l["N_BOARD"] + l["N_HAND"] + l["N_SHOP"] + 1}
        src, dst = np.zeros(tg.N_ACTIONS, np.int64), np.zeros(tg.N_ACTIONS, np.int64)
        for a in range(tg.N_ACTIONS):
            name = tg.action_name(a)
            ix = [int(x) for x in re.findall(r"\d+", name)]
            if name.startswith("Buy"): src[a] = zones["shop"] + ix[0]
            elif name.startswith(("Sell", "Activate")): src[a] = zones["board"] + ix[0]
            elif name.startswith("Choose"): src[a] = zones["choice"] + ix[0]
            elif name.startswith("HeroPower"): src[a] = 2 + ix[0]
            elif name.startswith(("Play", "Magnetize")):
                src[a] = zones["hand"] + ix[0]
                if len(ix) > 1: dst[a] = zones["board"] + min(ix[1], l["N_BOARD"] - 1)
            elif name.startswith("Move"):
                src[a], dst[a] = zones["board"] + ix[0], zones["board"] + ix[1]
        self.register_buffer("src", torch.from_numpy(src))
        self.register_buffer("dst", torch.from_numpy(dst))

    def encode(self, obs):
        f = signed_log(obs["features"].float())
        z = torch.cat([f, torch.zeros_like(f[:, :1])], dim=1)[:, self.numeric_map]
        base = self.card(obs["cards"]) + self.position(self.positions) + self.numeric(z)
        effects = self.card(obs["extra_ids"]) + self.position(obs["owners"]) + self.extra(signed_log(obs["numbers"]))
        cls = self.global_encoder(f).unsqueeze(1)
        tokens = torch.cat([cls, base, effects], dim=1)
        padding = torch.cat([torch.zeros((len(f), 1), device=f.device, dtype=torch.bool),
                             obs["cards"] == 0, ~obs["present"]], dim=1)
        encoded = self.entities(tokens, src_key_padding_mask=padding)
        return encoded[:, 0], encoded[:, :tg.OBS_CARDS + 1]

    def heads(self, memory, entities, mask):
        n = len(memory)
        x = torch.cat([memory[:, None].expand(-1, tg.N_ACTIONS, -1), entities[:, self.src],
                       entities[:, self.dst], self.action.weight[None].expand(n, -1, -1)], dim=-1)
        logits = self.policy_head(x).squeeze(-1).masked_fill(~mask, -1e9)
        rank = self.rank_head(memory)
        value = (rank.softmax(-1) * self.utility).sum(-1)
        return logits, value, rank

    def forward(self, obs, hidden):
        summary, entities = self.encode(obs)
        memory, next_hidden = self.memory(summary[:, None], hidden[None])
        return (*self.heads(memory[:, 0], entities, obs["mask"]), next_hidden[0])


def Policy(cfg):
    if cfg.policy_kind == "legacy_gru":
        return LegacyPolicy(cfg)
    from .compact import CompactPolicy
    return CompactPolicy(cfg)
