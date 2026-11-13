"""`HumanBoardPool(min_hand=...)` 单测：**"资源在手"的入场帧**。

背景（`docs/analysis/RUNS_20261113.md` §5.39）：真人在后期跑**循环经济**
（T12 每回合 `play` 22.0 / `sell` 10.4），而 champion 只有 8.1 / 1.2 ——
它不是"过度换血"，是**几乎不循环**。要教这件事，入场帧应当选**手里有资源**的时刻。

硬要求：
1. `min_hand=0/None` → 不过滤；
2. `min_hand=k` → 只保留观测手牌数 ≥ k 的帧；
3. 与 `min_tier` / `require_any_cards` 可以叠加，且各自的过滤计数分开记；
4. 手牌数取 payload（`hand_n`），没有 payload 时视为 0（fail-closed）。
"""

from __future__ import annotations

import json

from hsbg.az.train.human_board_pool import HumanBoardPool

ENGINE = "BG_LOE_077"
OTHER = "BG36_345"


def _row(turn: int, cards, hand_n: int = 0, tier=5) -> str:
    board = [[1000 + i, 10, 10] for i in range(len(cards))]
    detail = [{"atk": 10, "bonus_kw": 0, "card": 1000 + i, "card_id": cid,
               "golden": False, "hp": 10, "kw": 0, "position": i + 1,
               "shield_max": 0} for i, cid in enumerate(cards)]
    row = {
        "turn": turn, "board": board, "strength": 20 * len(cards),
        "game_id": f"g{turn}-{'_'.join(cards)}-{hand_n}", "tier": tier,
        "board_detail": detail, "hand_available": True,
        "hand": [{"card": 9000 + i, "atk": 1, "hp": 1, "position": i + 1}
                 for i in range(hand_n)],
    }
    return json.dumps(row, ensure_ascii=False)


def _write(tmp_path, rows) -> str:
    p = tmp_path / "bank.jsonl"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_min_hand_filters(tmp_path):
    bank = _write(tmp_path, [
        _row(9, [ENGINE], hand_n=2),
        _row(9, [ENGINE], hand_n=5),
        _row(9, [OTHER], hand_n=6),
    ])
    pool = HumanBoardPool(bank, sample="p50", min_hand=4)
    assert pool.kept == 2
    assert pool.filtered.get("hand_below_min") == 1


def test_min_hand_zero_is_off(tmp_path):
    bank = _write(tmp_path, [_row(9, [ENGINE], hand_n=0), _row(9, [ENGINE], hand_n=5)])
    assert HumanBoardPool(bank, sample="p50", min_hand=0).kept == 2


def test_min_hand_combined_with_engine_and_tier(tmp_path):
    bank = _write(tmp_path, [
        _row(9, [ENGINE], hand_n=5, tier=5),   # 保留
        _row(9, [ENGINE], hand_n=2, tier=5),   # 手牌不足
        _row(9, [OTHER], hand_n=5, tier=5),    # 无引擎
        _row(9, [ENGINE], hand_n=5, tier=4),   # 等级不足
    ])
    pool = HumanBoardPool(bank, sample="p50", min_tier=5,
                          require_any_cards={ENGINE}, min_hand=4)
    assert pool.kept == 1
    assert pool.filtered.get("hand_below_min") == 1
    assert pool.filtered.get("no_engine_card") == 1
    assert pool.filtered.get("tier_below_min") == 1


def test_kept_frame_carries_hand_payload(tmp_path):
    """留下来的帧必须仍然带着手牌（full 档注入时要把手牌写进引擎座位）。"""
    bank = _write(tmp_path, [_row(9, [ENGINE], hand_n=5)])
    pool = HumanBoardPool(bank, sample="p50", min_hand=4)
    assert pool.kept == 1
    pick = pool.sample_payload(9)
    assert pick is not None
    assert len(pick.get("hand") or []) == 5
