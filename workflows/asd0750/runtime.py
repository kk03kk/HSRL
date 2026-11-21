"""Only the authored observation layout and card-table APIs used by the network."""

from hashlib import sha256
from pathlib import Path

import _asd_obs2
import hsbg_engine


N_CARDS = hsbg_engine.N_CARDS
STATIC_WIDTH = 66
AUTHOR_STATIC_SHA256 = "bd806c54b42f447af345ab87225a118cf09871dd5d652ab51ca313c0b190a359"
_static = None


def use_author_static(path: Path):
    global _static
    data = path.read_bytes()
    if len(data) != N_CARDS * STATIC_WIDTH * 4 or sha256(data).hexdigest() != AUTHOR_STATIC_SHA256:
        raise ValueError("unexpected 0750 static card table")
    _static = data


def card_static():
    if _static is None:
        raise RuntimeError("0750's static table must be pinned before constructing the policy")
    return STATIC_WIDTH, _static


def layout():
    dimensions = dict(hsbg_engine.layout())
    for key in ("OBS2_FLOATS", "OBS2_CARDS", "X_MINION", "OPP_SUM_F",
                "K_ATT_BOARD", "K_ATT_ITEM", "K_ATT_OPP"):
        dimensions[key] = getattr(_asd_obs2, key)
    return dimensions


def obs2_blocks():
    return _asd_obs2.FLOAT_BLOCKS, _asd_obs2.CARD_BLOCKS
