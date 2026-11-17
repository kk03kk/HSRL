"""Versioned, own-seat-only mechanism features, appended to legacy observations.

No opponent state, RNG, future shop or unrevealed choice is read. Counter access
is explicitly limited to Rogers and Hopebringer, whose progress is visible in
their dynamic rules text. Other internal Minion.counter uses are not assumed
public. Stats below are own action history or publicly displayed persistent buffs.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

LEGACY_FLOATS = 3227
SCHEMA = 'hsai.mechanisms/v1'
COUNTER_CARDS = ('BG33_823', 'BG33_823_G', 'BG36_364', 'BG36_364_G')
STAT_FIELDS = (
    'spells_cast', 'tavern_spells_cast', 'battlecries_triggered',
    'elementals_played', 'cards_played', 'golden_played',
    'minions_bought', 'minions_sold', 'gold_spent_total', 'gold_spent_turn',
    'refreshes', 'undead_atk', 'undead_hp', 'beetle_atk', 'beetle_hp',
    'eternal_knight_deaths', 'knight_bonus', 'volumizer_atk', 'volumizer_hp',
    'magnetized', 'elem_give_atk', 'elem_give_hp', 'spell_atk', 'spell_hp',
    'tavern_atk', 'tavern_hp', 'tavern_elemental_atk', 'tavern_elemental_hp',
    'elementals_played_turn', 'cards_played_turn',
)
N_STATS = len(STAT_FIELDS)
N_EXTRA = N_STATS + 7 + 10 + 8
OBS_FLOATS = LEGACY_FLOATS + N_EXTRA
SPEC = dict(schema=SCHEMA, stats=STAT_FIELDS, counters=COUNTER_CARDS,
            tokens=('board:7', 'hand:10', 'shop:8'), scaling='raw then model symlog',
            pending_source='legacy cards[4] via zero-initialized projection')
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def _allowed_native_cards():
    import hsbg_engine as engine
    _ncols, raw = engine.card_static()
    n = len(raw) // (4 * _ncols)
    return frozenset(i - 1 for i in range(1, n)
                     if engine.card_info(i)[0] in COUNTER_CARDS)


def _minion(value):
    if not isinstance(value, dict):
        return None
    if 'item' in value:
        return _minion(value['item'])
    if 'Minion' in value:
        return value['Minion']
    return value if 'card' in value and 'atk' in value else None


def encode_player(player):
    """Encode a single own player record; never accepts the full game object."""
    stats = player['stats']
    out = np.zeros(N_EXTRA, dtype=np.float32)
    for i, key in enumerate(STAT_FIELDS):
        out[i] = float(stats.get(key, 0))
    offset = N_STATS
    allowed = _allowed_native_cards()
    for name, width in (('board', 7), ('hand', 10), ('shop', 8)):
        entries = player[name]
        if len(entries) > width:
            raise ValueError(f'{name} exceeds mechanism schema width')
        for i, entry in enumerate(entries):
            m = _minion(entry)
            if m is not None and int(m['card']) in allowed:
                out[offset+i] = float(m.get('counter', 0))
        offset += width
    if not np.isfinite(out).all():
        raise ValueError('non-finite own mechanism feature')
    return out


def observe(game, seat, schema=''):
    """Return native bytes for legacy models, or explicitly versioned f32 bytes."""
    pair = game.observe(int(seat))
    if not schema:
        return pair
    if schema != SCHEMA:
        raise ValueError(f'unknown mechanism observation schema: {schema}')
    features, cards = pair
    if len(features) != LEGACY_FLOATS * 4:
        raise ValueError('legacy observation shape changed')
    own = json.loads(game.player_json(int(seat)))
    extra = encode_player(own).astype('<f4', copy=False).tobytes()
    return bytes(features) + extra, cards


def schema_of(model):
    return str(getattr(model, 'mechanism_schema', '') or '')


def valid_width(width):
    return int(width) in (LEGACY_FLOATS, OBS_FLOATS)
