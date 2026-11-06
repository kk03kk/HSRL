"""Contract checks between the HSRL engine and the HSBRSIM observation/action encoding.

Shapes being equal is *not* enough (the two engines carry different card data), so this module
checks, at runtime and against the loaded native binding:

1. the fixed constants ``N_ACTIONS / OBS_FLOATS / OBS_CARDS / N_CARDS``;
2. every field of ``taverngym.layout()`` against HSBRSIM's frozen ``obs_std`` / ``common``
   constants (including action segment boundaries);
3. the card vocabulary: ``card_info(i)[0]`` for ``1..N_CARDS-1`` against the id list of the
   HSBRSIM data file the imported checkpoints were trained with;
4. the static card feature matrix: shape and per-column drift against the checkpoint buffer
   (drift is *reported*, not fatal — imported models keep their own buffers);
5. the action-name table at every segment boundary.

Everything is returned as a JSON-serialisable report; ``ok`` is False on any hard mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import hsbg_engine as E
from hsbg.az.train import common as HC
from hsbg.env import action_space_std as AS
from hsbg.env import obs_std as OS

# Frozen independently from the reference checkout, not from the running engine.
# Ship it with the Python package: deployment must not depend on a developer's D: drive.
VOCABULARY_REFERENCE = Path(__file__).parent / "data/hsbrsim_vocabulary_ec2d84d.json"
EXPECTED_VOCABULARY_SHA256 = "3b6fa2aeebe0955cb57438a9d8d5b487d61d8beb614f0c94c360dea3e9232270"

_FIELDS = (
    ("N_BOARD", OS.N_BOARD, 7),
    ("N_HAND", OS.N_HAND, 10),
    ("N_SHOP", OS.N_SHOP, 8),
    ("N_CHOICE", OS.N_CHOICE, 16),
    ("N_OPP", OS.N_OPP, 7),
    ("MINION_F", OS.MINION_F, 31),
    ("ITEM_F", OS.ITEM_F, 40),
    ("GLOBAL_F", OS.GLOBAL_F, 53),
    ("OPP_F", OS.OPP_F, 9),
    ("QUEST_F", OS.QUEST_F, 2),
    ("N_QUESTS", OS.N_QUESTS, 3),
    ("N_SECRETS", OS.N_SECRETS, 5),
    ("PENDING_KINDS", OS.PENDING_KINDS, 9),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_identity() -> Dict[str, Any]:
    """Identity of the loaded HSRL simulator (what every imported model is bound to)."""
    return {
        "engine": "hsrl-taverngym",
        "game_version": str(E.game_version()),
        "server_rules_version": str(E.server_rules_version()),
        "build_info": E.build_info(),
        "source_hash": E.source_hash(),
        "data_info": E.data_info(),
    }


def card_vocabulary(reference_path=None) -> Dict[str, Any]:
    """Compare every card index with a pinned, portable reference vocabulary."""
    reference = Path(reference_path) if reference_path is not None else VOCABULARY_REFERENCE
    ids = [E.card_info(i)[0] for i in range(1, E.N_CARDS)]
    out: Dict[str, Any] = {
        "n_cards": int(E.N_CARDS),
        "n_ids": len(ids),
        "reference": str(reference),
        "reference_present": reference.is_file(),
        "reference_kind": "bundled-frozen-vocabulary",
        "expected_ids_sha256": EXPECTED_VOCABULARY_SHA256,
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "ok": True,
        "mismatches": [],
    }
    if not reference.is_file():
        out["ok"] = False
        out["mismatches"] = ["bundled vocabulary missing; cannot prove vocabulary equality"]
        return out
    try:
        ref = json.loads(reference.read_text(encoding="utf-8"))
        ref_ids = ref["ids"]
        if ref.get("format") != "hsrl.card-vocabulary-reference.v1" or not isinstance(ref_ids, list):
            raise ValueError("invalid vocabulary reference format")
        if not all(isinstance(cid, str) and cid for cid in ref_ids) or len(set(ref_ids)) != len(ref_ids):
            raise ValueError("invalid or duplicate card IDs")
        digest = hashlib.sha256("\n".join(ref_ids).encode()).hexdigest()
        if digest != EXPECTED_VOCABULARY_SHA256 or ref.get("ids_sha256") != digest or ref.get("n_ids") != len(ref_ids):
            raise ValueError("frozen vocabulary checksum/count mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out.update(ok=False, mismatches=[f"invalid bundled vocabulary: {exc}"])
        return out
    out["reference_source"] = ref["source"]
    if len(ref_ids) != len(ids):
        out["ok"] = False
        out["mismatches"].append(f"length {len(ids)} != reference {len(ref_ids)}")
        return out
    bad = [i for i, (a, b) in enumerate(zip(ids, ref_ids)) if a != b]
    if bad:
        out["ok"] = False
        out["mismatches"] = [f"index {i}: {ids[i]} != {ref_ids[i]}" for i in bad[:20]]
    out["reference_sha256"] = _sha256(reference)
    return out


def static_drift(checkpoint_static: Optional[np.ndarray] = None,
                 checkpoint_text: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compare the engine's ``card_static`` with a checkpoint's buffer (content, not shape)."""
    width, raw = E.card_static()
    mine = np.frombuffer(raw, dtype="<f4").reshape(-1, width).copy()
    out: Dict[str, Any] = {
        "engine_shape": list(mine.shape),
        "engine_cols": int(width),
        "engine_rows": int(mine.shape[0]),
        "engine_sha256": hashlib.sha256(mine.tobytes()).hexdigest(),
    }
    if checkpoint_static is not None:
        his = np.asarray(checkpoint_static, dtype=np.float32)
        out["checkpoint_shape"] = list(his.shape)
        if his.shape != mine.shape:
            out.update(ok=False, note="shape mismatch")
            return out
        diff = np.abs(mine - his)
        rows = np.flatnonzero(diff.max(axis=1) > 1e-6)
        cols = np.flatnonzero(diff.max(axis=0) > 1e-6)
        out.update(
            ok=True,
            differing_rows=int(rows.size),
            differing_cols=[int(c) for c in cols],
            differing_cols_count=int(cols.size),
            rows_by_col={int(c): int((diff[:, c] > 1e-6).sum()) for c in cols},
            checkpoint_sha256=hashlib.sha256(his.tobytes()).hexdigest(),
            policy="checkpoint buffers are kept (the model was trained with them)",
        )
    if checkpoint_text is not None:
        text = np.asarray(checkpoint_text, dtype=np.float32)
        out["checkpoint_text_shape"] = list(text.shape)
        out["checkpoint_text_nonzero_rows"] = int((text != 0).any(axis=1).sum())
    return out


def action_table() -> Dict[str, Any]:
    """Action segment boundaries and the names printed by the engine at those boundaries."""
    seg = {
        "end_turn": (0, 0), "refresh": (1, 1), "freeze": (2, 2), "upgrade": (3, 3),
        "hero_power": (4, 5), "dark_gift": (6, 6), "cancel": (7, 7),
        "buy": (8, 15), "sell": (16, 22), "activate": (23, 29), "choose": (30, 45),
        "play": (46, 125), "magnetize": (126, 195), "move": (196, 244),
    }
    names: List[str] = []
    ok = True
    for name, (lo, hi) in seg.items():
        want = HC.ACTION_SEGMENTS[name]
        if (lo, hi) != tuple(want):
            ok = False
        names.append(f"{name}[{lo}]={E.action_name(lo)}")
        names.append(f"{name}[{hi}]={E.action_name(hi)}")
    return {"segments": seg, "hsbg_segments": {k: list(v) for k, v in HC.ACTION_SEGMENTS.items()},
            "ok": ok, "boundary_names": names}


def verify(checkpoint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full contract report. ``checkpoint`` may carry ``card_static`` / ``card_text`` arrays."""
    report: Dict[str, Any] = {"engine": engine_identity(), "hard_failures": []}
    consts = {
        "N_ACTIONS": (int(E.N_ACTIONS), int(HC.N_ACTIONS)),
        "OBS_FLOATS": (int(E.OBS_FLOATS), int(HC.OBS_FLOATS)),
        "OBS_CARDS": (int(E.OBS_CARDS), int(HC.OBS_CARDS)),
        "N_PLAYERS": (8, int(HC.N_PLAYERS)),
        "N_RANKS": (8, int(HC.N_RANKS)),
    }
    bad = [k for k, (a, b) in consts.items() if a != b]
    report["constants"] = {k: {"engine": a, "hsbg": b, "ok": a == b} for k, (a, b) in consts.items()}
    if bad:
        report["hard_failures"] += [f"constant mismatch {k}" for k in bad]

    layout = dict(E.layout())
    fields = {}
    for name, want, legacy in _FIELDS:
        have = layout.get(name)
        fields[name] = {"engine": have, "hsbg": want, "ok": have == want}
        if have != want:
            report["hard_failures"].append(f"layout {name}: {have} != {want}")
    report["layout"] = fields

    report["cards"] = card_vocabulary()
    if not report["cards"]["ok"]:
        report["hard_failures"].append("card vocabulary mismatch")

    report["actions"] = action_table()
    if not report["actions"]["ok"]:
        report["hard_failures"].append("action segment mismatch")

    static = checkpoint.get("card_static") if checkpoint else None
    text = checkpoint.get("card_text") if checkpoint else None
    report["static"] = static_drift(static, text)
    if static is not None and not report["static"].get("ok", True):
        report["hard_failures"].append("card_static shape mismatch")

    def _span(s):
        return [int(s.start), int(s.stop)] if isinstance(s, slice) else [int(s[0]), int(s[1])]

    report["obs_slices"] = {
        "global": _span(OS.SLICE_GLOBAL), "board": _span(OS.SLICE_BOARD),
        "hand": _span(OS.SLICE_HAND), "shop": _span(OS.SLICE_SHOP),
        "choices": _span(OS.SLICE_CHOICES), "opponents": _span(OS.SLICE_OPPONENTS),
        "opp_boards": _span(OS.SLICE_OPP_BOARDS),
        "last_slice_end": _span(OS.SLICE_OPP_BOARDS)[1],
    }
    if _span(OS.SLICE_OPP_BOARDS)[1] != int(E.OBS_FLOATS):
        report["hard_failures"].append("obs slice end != OBS_FLOATS")

    report["ok"] = not report["hard_failures"]
    return report


__all__ = ["verify", "engine_identity", "card_vocabulary", "static_drift", "action_table"]
