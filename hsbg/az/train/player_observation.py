"""Player-visible inputs for the 37.0.3 fixed-budget peer policy.

Each named card retains its slot. Numbers come from rendered public card text,
not arbitrary game counters. The identity and words distinguish named effects;
the ordered numeric channels distinguish costs and progress in the same text.
"""
import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCHEMA = "hsai.player150/v1"
SCHEMA_SHA = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
N_SLOTS = 68
N_NUMBERS = 8
N_WORDS = 16
FACE_WIDTH = 1 + N_NUMBERS + N_WORDS
N_GLOBAL = 2 + 10 + 8
WIDTH = 3227 + 2 * N_GLOBAL + N_SLOTS * FACE_WIDTH
TOKEN = re.compile(r"[+-]?\d+|[^\W\d_]+", re.UNICODE)


def scaled(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("non-finite public card value")
    return math.copysign(math.log1p(abs(x)), x)


@lru_cache(maxsize=32768)
def word_feature(word):
    digest = hashlib.blake2s(word.encode("utf-8"), digest_size=2).digest()
    return digest[0] % N_WORDS, (1 if digest[1] & 1 else -1)


@lru_cache(maxsize=16384)
def face_features(identity, text):
    numbers = []
    words = np.zeros(N_WORDS, dtype=np.float32)
    for term in [identity, *TOKEN.findall(text.casefold())]:
        if re.fullmatch(r"[+-]?\d+", term):
            numbers.append(scaled(int(term)))
            term = "#" + str(len(numbers)) + "=" + term
        i, sign = word_feature(term)
        words[i] += sign
    words /= max(1., np.linalg.norm(words))
    return numbers, words


def encode(facts):
    """Unknown global values and card faces retain their own presence flag."""
    remaining = facts["remaining"]
    if not 0 <= remaining <= 150:
        raise ValueError("invalid remaining recruitment budget")
    prices = facts["health_prices"]
    growth = facts["growth"]
    if len(prices) > 8 or len(growth) != 10:
        raise ValueError("invalid player resource layout")
    values = [remaining / 150, facts["last_seen_turn"], *growth, *prices, *([None] * (8-len(prices)))]
    extra = np.zeros(WIDTH-3227, dtype="<f4")
    for i, value in enumerate(values):
        if value is not None:
            extra[i] = value if i == 0 else scaled(value)
            extra[N_GLOBAL+i] = 1
    faces = extra[2*N_GLOBAL:].reshape(N_SLOTS, FACE_WIDTH)
    for slot, identity, text in facts["faces"]:
        if not 0 <= slot < N_SLOTS or not isinstance(text, str) or not identity:
            raise ValueError("invalid public face")
        numbers, words = face_features(identity, text)
        row = faces[slot]
        if row[0] == 0:
            n = min(N_NUMBERS, len(numbers))
            row[1:1+n] = numbers[:n]
        row[0] += 1
        row[1+N_NUMBERS:] += words
    return extra


class PublicInputAdapter(nn.Module):
    """Residual projections keep the original policy exactly at initialization."""
    def __init__(self):
        super().__init__()
        self.globals = nn.Linear(2*N_GLOBAL + 27*FACE_WIDTH, 68, bias=False)
        self.board = nn.Linear(FACE_WIDTH, 31, bias=False)
        self.item = nn.Linear(FACE_WIDTH, 40, bias=False)
        for layer in (self.globals, self.board, self.item):
            nn.init.zeros_(layer.weight)

    def forward(self, features):
        if features.shape[-1] != WIDTH:
            raise ValueError("public player schema width differs")
        extras = features[:, 3227:]
        faces = extras[:, 2*N_GLOBAL:].reshape(-1, N_SLOTS, FACE_WIDTH)
        misc = torch.cat((extras[:, :2*N_GLOBAL], faces[:, :19].flatten(1),
                          faces[:, 60:68].flatten(1)), 1)
        return torch.cat((features[:, :68] + self.globals(misc),
                          features[:, 68:285] + self.board(faces[:, 19:26]).flatten(1),
                          features[:, 285:1645] + self.item(faces[:, 26:60]).flatten(1),
                          features[:, 1645:3227]), 1)
