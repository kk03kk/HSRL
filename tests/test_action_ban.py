"""加重护手（BG36_MagicItem_414）的待选屏蔽：真实 pytest 断言。

**用引擎实测出来的真实 schema**（list + 外部标记枚举，options 是整数卡索引）：
    "pending": [{"Trinket": {"free": false, "options": [3101, 3037, 3208], "replace": None}}]
同时兼容旧假设的 dict 形态。

2026-11-17：本文件原来在**模块级** ``raise SystemExit``，导致 ``pytest tests`` 直接
``INTERNALERROR``（整套测试跑不起来，审计 P1-5）。现在拆成真正的 ``test_*``，
命令行运行仍打印同样的诊断（``python tests/test_action_ban.py``）。
"""
from __future__ import annotations

import json

import numpy as np

from hsbg.az.train.action_ban import (BANNED_CHOICE_CARDS, BANNED_CHOICE_INDICES,
                                      CHOICE_HI, CHOICE_LO, ban_choice_options,
                                      has_pending_choice, trinket_options)
from hsbg.env.action_space_std import N_TOTAL_ACTIONS

#: 引擎真实形态：list + `{"Trinket": {...}}`，options 是整数卡索引
PJ_REAL = {"pending": [{"Trinket": {"free": False, "replace": None,
                                    "options": [3101, 3037, 3208]}}]}


def mk_mask(*legal):
    m = np.zeros(N_TOTAL_ACTIONS, dtype=bool)
    for a in legal:
        m[a] = True
    return m


def test_ban_constants_point_at_weighted_gauntlet():
    assert BANNED_CHOICE_CARDS == {"BG36_MagicItem_414"}
    assert BANNED_CHOICE_INDICES == {3037}
    assert (CHOICE_LO, CHOICE_HI) == (30, 46)


def test_trinket_options_reads_engine_schema():
    assert trinket_options(PJ_REAL) == [3101, 3037, 3208]


def test_bans_exactly_the_banned_option():
    m = mk_mask(0, CHOICE_LO + 0, CHOICE_LO + 1, CHOICE_LO + 2)
    n = ban_choice_options(PJ_REAL, m)
    assert n == 1
    assert not m[CHOICE_LO + 1]          # 加重护手那格已非法
    assert m[CHOICE_LO + 0] and m[CHOICE_LO + 2]
    assert m[0]                          # EndTurn 不受影响


def test_dict_shape_with_kind_is_banned():
    pj = {"pending": {"kind": "trinket", "options": [{"id": 3101}, {"id": 3037}]}}
    m = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
    assert ban_choice_options(pj, m) == 1
    assert not m[CHOICE_LO + 1]


def test_single_key_dict_is_banned():
    pj = {"pending": {"Trinket": {"options": [3037, 3101]}}}
    m = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
    assert ban_choice_options(pj, m) == 1
    assert not m[CHOICE_LO]


def test_no_ban_when_banned_card_absent():
    pj = {"pending": [{"Trinket": {"options": [3101, 3208]}}]}
    m = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
    assert ban_choice_options(pj, m) == 0
    assert m[CHOICE_LO]


def test_all_options_banned_keeps_a_legal_option():
    """全被禁时不能遮死（否则玩家无路可走）。"""
    pj = {"pending": [{"Trinket": {"options": [3037, 3037]}}]}
    m = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
    assert ban_choice_options(pj, m) == 0
    assert m[CHOICE_LO]


def test_non_trinket_pending_untouched():
    m6 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
    assert ban_choice_options({"pending": [{"HeroSelect": {"options": [188, 3679]}}]}, m6) == 0
    assert m6[CHOICE_LO]
    m7 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
    assert ban_choice_options({"pending": [{"Discover": {"options": [3037, 1]}}]}, m7) == 0


def test_accepts_serialised_json_string():
    raw = json.dumps({"pending": [{"Trinket": {"options": [3037, 3101]}}]})
    assert ban_choice_options(raw, mk_mask(0, CHOICE_LO, CHOICE_LO + 1)) == 1


def test_has_pending_choice_guard():
    assert not has_pending_choice(mk_mask(0, 1, 3))
    assert has_pending_choice(mk_mask(0, CHOICE_LO))


def main() -> int:                      # pragma: no cover - 只给人跑
    print("BANNED 名称 =", sorted(BANNED_CHOICE_CARDS),
          " 数字索引 =", sorted(BANNED_CHOICE_INDICES))
    for fn in (test_ban_constants_point_at_weighted_gauntlet,
               test_trinket_options_reads_engine_schema,
               test_bans_exactly_the_banned_option,
               test_dict_shape_with_kind_is_banned,
               test_single_key_dict_is_banned,
               test_no_ban_when_banned_card_absent,
               test_all_options_banned_keeps_a_legal_option,
               test_non_trinket_pending_untouched,
               test_accepts_serialised_json_string,
               test_has_pending_choice_guard):
        fn()
        print(f"  [OK] {fn.__name__}")
    print("\nBAN_STATIC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
