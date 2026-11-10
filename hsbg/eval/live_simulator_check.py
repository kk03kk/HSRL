"""Compare verified live transitions with the Rust simulator.

This is deliberately a *structural* comparator, not a claim that the live
projection is the complete engine state.  The live client exposes public
entities/tags and an action authorization; it does not expose hidden pool
contents, opponent hands or RNG.  The module therefore:

1. maps one live transition (before -> after) and the exact action identity;
2. replays the same action on a simulator state only when that state can be
   reconstructed without guessing;
3. compares a whitelist of observable quantities;
4. emits every mismatch with its field, real value, simulated value and a
   reason category so RNG/hidden-information effects are not reported as bugs.

The CLI is offline: it reads JSONL files produced by ``hsrl_agent.live.trace``
and never launches the game or creates native input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCHEMA = "hsai.live-sim-check/v1"


def board_signature(board: Sequence[dict]) -> List[dict]:
    return [
        {
            "card_id": card.get("card_id") or card.get("id"),
            "attack": card.get("attack", card.get("atk")),
            "health": card.get("health", card.get("hp")),
            "position": index,
        }
        for index, card in enumerate(board or [])
    ]


def card_multiset(board: Sequence[dict]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for card in board or []:
        key = str(card.get("card_id") or card.get("id"))
        result[key] = result.get(key, 0) + 1
    return result


def compare_observable(
    before: dict,
    after: dict,
    *,
    simulator_before: Optional[dict] = None,
    simulator_after: Optional[dict] = None,
    action: Optional[dict] = None,
    action_had_hidden_randomness: bool = False,
) -> dict:
    """Strict on stable fields; classify every difference.

    ``before``/``after`` are live trace summaries. ``simulator_*`` are optional
    reconstructed simulator summaries of the same shape. Without a simulator
    transition only observable live invariants are checked (money conservation
    is not asserted because effects/prices can legitimately change it).
    """
    differences: List[dict] = []
    checks: List[dict] = []

    def add(kind: str, field: str, real: Any, sim: Any = None, status: str = "mismatch",
            reason: str = "") -> None:
        (checks if status == "ok" else differences).append({
            "kind": kind,
            "field": field,
            "real": real,
            "simulator": sim,
            "status": status,
            "reason": reason,
        })

    # Always-verifiable identities. These are not effects and must agree even
    # when a card's RNG outcome differs.
    ra, ha, sa = after.get("board") or [], after.get("hand") or [], after.get("shop") or []
    kind = (action or {}).get("kind")

    if kind == "buy":
        source = (action or {}).get("source")
        if source in {c.get("entity_id") for c in sa}:
            add("invariant", "buy.source_departed_shop", False,
                reason="model authorization says the shop card moved; live after still has it")
        else:
            add("invariant", "buy.source_departed_shop", True, status="ok")
    if kind == "sell":
        source = (action or {}).get("source")
        if source in {c.get("entity_id") for c in ra}:
            add("invariant", "sell.source_departed_board", False,
                reason="requested sell; live after still has the entity on board")
        else:
            add("invariant", "sell.source_departed_board", True, status="ok")
    if kind == "play":
        source = (action or {}).get("source")
        if source in {c.get("entity_id") for c in ha}:
            add("invariant", "play.source_departed_hand", False,
                reason="requested play; live after still has the card in hand")
        else:
            add("invariant", "play.source_departed_hand", True, status="ok")

    if simulator_after is None:
        return {
            "schema": SCHEMA,
            "mode": "live-invariants",
            "action": action,
            "checks": checks,
            "differences": differences,
            "strict_equivalent": None,
            "interpretation": "No reconstructed simulator transition was supplied. "
                              "Only live-side invariants were checked.",
        }

    # With a simulator transition, compare only public, stable quantities.
    simb = simulator_after.get("board") or []
    if card_multiset([{"card_id": c.get("card_id") or c.get("id")} for c in ra]) != \
            card_multiset([{"card_id": c.get("card_id") or c.get("id")} for c in simb]):
        add("transition", "board.card_multiset",
            card_multiset([{"card_id": c.get("card_id") or c.get("id")} for c in ra]),
            card_multiset([{"card_id": c.get("card_id") or c.get("id")} for c in simb]),
            status="unverifiable" if action_had_hidden_randomness else "mismatch",
            reason="random/hidden effect" if action_had_hidden_randomness
            else "same action produced a different observable board")
    else:
        add("transition", "board.card_multiset", True, status="ok")
    for field in ("gold", "tavern_tier", "health", "armor"):
        real = after.get(field)
        sim = simulator_after.get(field)
        if real is None or sim is None:
            add("transition", field, real, sim, status="unverifiable",
                reason="field not present in this observation")
        elif int(real) == int(sim):
            add("transition", field, real, sim, status="ok")
        else:
            add("transition", field, real, sim,
                status="unverifiable" if action_had_hidden_randomness else "mismatch",
                reason="random/hidden effect" if action_had_hidden_randomness
                else "same action produced a different value")
    return {
        "schema": SCHEMA,
        "mode": "transition",
        "action": action,
        "checks": checks,
        "differences": differences,
        "strict_equivalent": not any(d["status"] == "mismatch" for d in differences),
        "interpretation": "Only fields present in both public observations were compared. "
                          "Random and hidden effects are reported as unverifiable, not as simulator bugs.",
    }


def load_transitions(actions_path: Path, full_path: Optional[Path] = None) -> List[dict]:
    """Resolve action executions against the nearest decision snapshot."""
    decisions = {}
    if full_path and full_path.exists():
        import gzip
        with gzip.open(full_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "decision":
                    decisions[row.get("decision_id")] = row
    rows = []
    with Path(actions_path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "execution":
                continue
            did = row.get("decision_id")
            full = decisions.get(did) or {}
            decision = full.get("decision") or {}
            snapshot = full.get("snapshot") or {}
            rows.append({
                "decision_id": did,
                "action": decision.get("binding") or {},
                "before": row.get("before"),
                "after": row.get("after"),
                "snapshot": snapshot,
                "status": row.get("status"),
                "outcome": row.get("outcome"),
            })
    return rows


def audit_file(actions_path: Path, *, full_path: Optional[Path] = None,
               out: Optional[Path] = None) -> dict:
    transitions = load_transitions(actions_path, full_path)
    reports = []
    for row in transitions:
        if row.get("before") is None or row.get("after") is None:
            continue
        reports.append(compare_observable(row["before"], row["after"], action=row.get("action")))
    result = {
        "schema": SCHEMA,
        "source": str(actions_path),
        "transitions": len(reports),
        "strict_equivalent": sum(1 for r in reports if r.get("strict_equivalent") is True),
        "with_differences": sum(1 for r in reports if r.get("differences")),
        "per_transition": reports,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = audit_file(args.actions, full_path=args.full, out=args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "per_transition"},
                     ensure_ascii=False, indent=2))
    return 0 if not result["with_differences"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
