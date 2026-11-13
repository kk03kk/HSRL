"""`HumanBoardPool(require_any_cards=...)` 单测：引擎入场课程的过滤器。

背景（`docs/analysis/RUNS_20261113.md` §5.35）：真人 T12 有 **64%** 的战团含引擎卡
（铜须/巴琳达/罗杰斯…），而 champion 只有 **9%**。要让模型学会"把资源变成属性"，
先得让它从"已经装好引擎"的位置开始，因此入场帧要能按**引擎卡 id**筛。

硬要求：
1. 不传 → 完全不过滤（逐位等价旧行为）；
2. 传 card_id 集合 → 只保留战团里至少含 1 张的帧；
3. 金卡后缀 `_G` 必须被剥掉再匹配（库里金卡是 `XXX_G`）；
4. 支持逗号/分号分隔的字符串（配置层直接传）；
5. 过滤计数落在 `pool.filtered["no_engine_card"]`，训练日志可核对。
"""

from __future__ import annotations

import json

from hsbg.az.train.human_board_pool import HumanBoardPool, load_pool

ENGINE = "BG_LOE_077"
OTHER = "BG36_345"


def _row(turn: int, cards, tier=5) -> str:
    board = [[1000 + i, 10, 10] for i in range(len(cards))]
    detail = [{"atk": 10, "bonus_kw": 0, "card": 1000 + i, "card_id": cid,
               "golden": cid.endswith("_G"), "hp": 10, "kw": 0, "position": i + 1,
               "shield_max": 0} for i, cid in enumerate(cards)]
    row = {"turn": turn, "board": board, "strength": 20 * len(cards),
           "game_id": f"g{turn}-{'_'.join(cards)}", "tier": tier,
           "board_detail": detail, "perspective": "self"}
    return json.dumps(row, ensure_ascii=False)


def _write(tmp_path, rows) -> str:
    p = tmp_path / "bank.jsonl"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_no_require_cards_keeps_all(tmp_path):
    bank = _write(tmp_path, [
        _row(8, [ENGINE]),
        _row(8, [OTHER, OTHER]),
        _row(8, [OTHER, ENGINE]),
    ])
    pool = HumanBoardPool(bank, sample="p50")
    assert pool.kept == 3
    assert not pool.require_cards
    assert not pool.filtered


def test_require_cards_filters(tmp_path):
    bank = _write(tmp_path, [
        _row(8, [ENGINE]),
        _row(9, [OTHER, OTHER]),
        _row(9, [OTHER, ENGINE]),
    ])
    pool = HumanBoardPool(bank, sample="p50", require_any_cards={ENGINE})
    assert pool.kept == 2
    assert pool.filtered.get("no_engine_card") == 1
    kept = sorted(b.board[0][0] for t in pool.index for b in pool.index[t])
    assert len(kept) == 2


def test_golden_suffix_is_stripped(tmp_path):
    """库里金卡写作 `XXX_G`；只有金卡版本在场时也必须算命中。"""
    bank = _write(tmp_path, [_row(8, [ENGINE + "_G"])])
    pool = HumanBoardPool(bank, sample="p50", require_any_cards=[ENGINE])
    assert pool.kept == 1


def test_string_spec_with_both_separators(tmp_path):
    """配置层传字符串：逗号与分号都要能解析。"""
    bank = _write(tmp_path, [_row(8, [OTHER]), _row(8, [ENGINE])])
    pool = HumanBoardPool(bank, sample="p50",
                          require_any_cards=f"{OTHER};{ENGINE}")
    assert pool.kept == 2
    assert pool.require_cards == frozenset({OTHER, ENGINE})


def test_combined_with_min_tier(tmp_path):
    bank = _write(tmp_path, [
        _row(8, [ENGINE], tier=4),
        _row(8, [ENGINE], tier=5),
        _row(8, [OTHER], tier=6),
    ])
    pool = HumanBoardPool(bank, sample="p50", min_tier=5, require_any_cards={ENGINE})
    assert pool.kept == 1
    assert pool.filtered.get("tier_below_min") == 1
    assert pool.filtered.get("no_engine_card") == 1


def test_load_pool_none_when_all_filtered(tmp_path):
    bank = _write(tmp_path, [_row(8, [OTHER])])
    assert load_pool(bank, sample="p50", require_any_cards={ENGINE}) is None
