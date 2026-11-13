"""`HumanBoardPool(min_tier=...)` 的单测：入场课程按**观测等级**筛帧。

背景（`docs/analysis/EXPERT_PRIORS_20261113.md` §5.A）：专家先验是「T8 左右到五本」，
所以我们只在**五本以上**的真人帧上开课，让模型第一次经历「五本位置怎么运营」。

硬要求：
1. `min_tier=None/0` → **完全不过滤**（逐位等价旧行为）；
2. `min_tier=k` → 只保留 `tier>=k` 的帧；
3. 等级**未知**（没有 payload 的 board-only 行）→ **丢弃**（fail-closed，不假装知道等级）；
4. 过滤计数落在 `pool.filtered`，便于在训练日志里核对。
"""

from __future__ import annotations

import json

from hsbg.az.train.human_board_pool import HumanBoardPool, load_pool


def _detail(card: int, atk: int, hp: int) -> dict:
    return {"atk": atk, "bonus_kw": 0, "card": card, "card_id": f"T{card}",
            "golden": False, "hp": hp, "kw": 0, "position": 1, "shield_max": 0}


def _row(turn: int, card: int, atk: int, hp: int, tier, with_detail: bool = True) -> str:
    row = {
        "turn": turn,
        "board": [[card, atk, hp]],
        "strength": atk + hp,
        "game_id": f"g{turn}-{card}",
        "tier": tier,
        "perspective": "self",
    }
    if with_detail:
        row["board_detail"] = [_detail(card, atk, hp)]
    return json.dumps(row, ensure_ascii=False)


def _write(tmp_path, rows) -> str:
    p = tmp_path / "bank.jsonl"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_no_min_tier_keeps_everything(tmp_path):
    """默认（min_tier 不传）= 旧行为：等级 1/4/5/6 与无 payload 的行全部保留。"""
    bank = _write(tmp_path, [
        _row(8, 1001, 5, 5, 1),
        _row(8, 1002, 10, 10, 4),
        _row(8, 1003, 20, 20, 5),
        _row(8, 1004, 40, 40, 6),
        _row(8, 1005, 8, 8, None, with_detail=False),
    ])
    pool = HumanBoardPool(bank, sample="p50")
    assert pool.kept == 5
    assert pool.min_tier is None
    assert not pool.filtered


def test_min_tier_filters_and_counts(tmp_path):
    """min_tier=5 → 只留 tier 5/6；tier 1/4 计入 tier_below_min，未知计入 tier_unknown。"""
    bank = _write(tmp_path, [
        _row(8, 1001, 5, 5, 1),
        _row(8, 1002, 10, 10, 4),
        _row(8, 1003, 20, 20, 5),
        _row(8, 1004, 40, 40, 6),
        _row(8, 1005, 8, 8, None, with_detail=False),
    ])
    pool = HumanBoardPool(bank, sample="p50", min_tier=5)
    assert pool.min_tier == 5
    assert pool.kept == 2
    assert pool.filtered.get("tier_below_min") == 2
    assert pool.filtered.get("tier_unknown") == 1
    kept_cards = sorted(b.board[0][0] for b in pool.index.get(8, []))
    assert kept_cards == [1003, 1004]
    # 采样只能取到留下的帧
    for _ in range(20):
        pick = pool.sample(8)
        assert pick is not None
        assert pick[0][0] in (1003, 1004)


def test_min_tier_zero_is_off(tmp_path):
    """0 与 None 同义（配置文件里写 0 表示关闭）。"""
    bank = _write(tmp_path, [_row(8, 1001, 5, 5, 1), _row(8, 1002, 20, 20, 5)])
    assert HumanBoardPool(bank, sample="p50", min_tier=0).kept == 2


def test_load_pool_returns_none_when_everything_filtered(tmp_path):
    """全部被过滤 → 返回 None，让调用方退回纯 bot，而不是拿空池子跑。"""
    bank = _write(tmp_path, [_row(8, 1001, 5, 5, 1), _row(9, 1002, 6, 6, 4)])
    assert load_pool(bank, sample="p50", min_tier=5) is None


def test_payload_tier_survives_filter(tmp_path):
    """保留下来的帧仍带 payload.tier，注入端才能把真实等级写进引擎。"""
    bank = _write(tmp_path, [_row(8, 1003, 20, 20, 5)])
    pool = HumanBoardPool(bank, sample="p50", min_tier=5)
    pick = pool.sample_payload(8)
    assert pick is not None
    assert pick.get("tier") == 5
