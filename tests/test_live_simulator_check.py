"""Offline transition comparator tests (no live game, no engine launch)."""
from __future__ import annotations

from pathlib import Path

from hsbg.eval.live_simulator_check import compare_observable, load_transitions, audit_file


def _board(*pairs):
    return [dict(entity_id=100 + i, card_id=card, attack=atk, health=hp)
            for i, (card, atk, hp) in enumerate(pairs)]


def test_play_invariant_requires_source_to_leave_hand():
    before = dict(board=_board(("A", 1, 1)), hand=[dict(entity_id=7, card_id="SPELL")],
                  shop=[], gold=5, tavern_tier=2)
    after = dict(board=_board(("A", 1, 1)), hand=[dict(entity_id=7, card_id="SPELL")],
                 shop=[], gold=5, tavern_tier=2)
    report = compare_observable(before, after, action=dict(kind="play", source=7))
    assert report["strict_equivalent"] is None
    assert any(d["field"] == "play.source_departed_hand" for d in report["differences"])


def test_buy_invariant_accepts_removed_shop_card():
    before = dict(board=[], hand=[], shop=[dict(entity_id=9, card_id="MINION")], gold=3, tavern_tier=1)
    after = dict(board=[], hand=[dict(entity_id=9, card_id="MINION")], shop=[], gold=0, tavern_tier=1)
    report = compare_observable(before, after, action=dict(kind="buy", source=9))
    assert not report["differences"]


def test_simulator_compare_marks_random_effect_unverifiable():
    live = dict(board=_board(("A", 2, 2)), hand=[], shop=[], gold=3, tavern_tier=2)
    sim = dict(board=_board(("B", 1, 1)), hand=[], shop=[], gold=3, tavern_tier=2)
    report = compare_observable(dict(board=[], hand=[], shop=[], gold=3),
                                live, simulator_before={}, simulator_after=sim,
                                action=dict(kind="play", source=1),
                                action_had_hidden_randomness=True)
    assert report["strict_equivalent"] is True
    assert any(d["status"] == "unverifiable" for d in report["differences"])


def test_audit_file_reads_trace_rows(tmp_path):
    actions = tmp_path / "actions.jsonl"
    # The action row has no binding object here, so the comparator checks only
    # live invariants; the malformed line must not abort the audit.
    actions.write_text("\n".join([
        '{"event":"execution","decision_id":"d-1","before":{"board":[],"hand":[],"shop":[{"entity_id":9,"card_id":"X"}]},"after":{"board":[],"hand":[{"entity_id":9,"card_id":"X"}],"shop":[]},"status":"receipt","outcome":"applied"}',
        "not-json",
    ]) + "\n", encoding="utf-8")
    result = audit_file(actions, out=tmp_path / "report.json")
    assert result["transitions"] == 1
    assert result["with_differences"] == 0
    assert (tmp_path / "report.json").is_file()
