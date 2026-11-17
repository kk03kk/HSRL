"""静态验证：加重护手在待选里被遮掉，且不会把整手牌遮死。

**用引擎实测出来的真实 schema**（list + 外部标记枚举，options 是整数卡索引）：
    "pending": [{"Trinket": {"free": false, "options": [3101, 3037, 3208], "replace": None}}]
同时兼容旧假设的 dict 形态。
"""
import numpy as np

from hsbg.az.train.action_ban import (BANNED_CHOICE_CARDS, BANNED_CHOICE_INDICES,
                                      CHOICE_HI, CHOICE_LO, ban_choice_options,
                                      has_pending_choice, trinket_options)
from hsbg.env.action_space_std import N_TOTAL_ACTIONS

OK = True


def mk_mask(*legal):
    m = np.zeros(N_TOTAL_ACTIONS, dtype=bool)
    for a in legal:
        m[a] = True
    return m


def check(name, cond):
    global OK
    print("  [%s] %s" % ("OK" if cond else "FAIL", name))
    OK = OK and cond


print("BANNED 名称 =", sorted(BANNED_CHOICE_CARDS), " 数字索引 =", sorted(BANNED_CHOICE_INDICES))

# 1) 引擎真实形态：list + {"Trinket": {...}}，options 是整数
pj_real = {"pending": [{"Trinket": {"free": False, "replace": None,
                                    "options": [3101, 3037, 3208]}}]}
check("trinket_options 能取出 3 个选项", trinket_options(pj_real) == [3101, 3037, 3208])
m = mk_mask(0, CHOICE_LO + 0, CHOICE_LO + 1, CHOICE_LO + 2)
n = ban_choice_options(pj_real, m)
check("遮掉 1 个选项", n == 1)
check("加重护手那格已非法", not m[CHOICE_LO + 1])
check("其余两格仍合法", m[CHOICE_LO + 0] and m[CHOICE_LO + 2])
check("EndTurn 不受影响", m[0])

# 2) 兼容 dict 形态 {"kind": "trinket", options: [{id: 3037}, ...]}
pj_dict = {"pending": {"kind": "trinket", "options": [{"id": 3101}, {"id": 3037}]}}
m2 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("dict 形态也能遮", ban_choice_options(pj_dict, m2) == 1 and not m2[CHOICE_LO + 1])

# 3) 兼容 {"Trinket": ...} 单键 dict
pj_k = {"pending": {"Trinket": {"options": [3037, 3101]}}}
m3 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("单键 dict 也能遮", ban_choice_options(pj_k, m3) == 1 and not m3[CHOICE_LO])

# 4) 没有加重护手 → 不动
pj_ok = {"pending": [{"Trinket": {"options": [3101, 3208]}}]}
m4 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("无禁用项时不动", ban_choice_options(pj_ok, m4) == 0 and m4[CHOICE_LO])

# 5) 全是加重护手 → 不遮死
pj_all = {"pending": [{"Trinket": {"options": [3037, 3037]}}]}
m5 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("全禁用时不遮死", ban_choice_options(pj_all, m5) == 0 and m5[CHOICE_LO])

# 6) 非饰品待选（英雄选择 / 发现）→ 不动
pj_hero = {"pending": [{"HeroSelect": {"options": [188, 3679]}}]}
m6 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("HeroSelect 不动", ban_choice_options(pj_hero, m6) == 0 and m6[CHOICE_LO])
pj_disc = {"pending": [{"Discover": {"options": [3037, 1]}}]}
m7 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("Discover 不动", ban_choice_options(pj_disc, m7) == 0)

# 7) 也接受已序列化的 JSON 字符串
check("接受 JSON 字符串", ban_choice_options(
    '{"pending": [{"Trinket": {"options": [3037, 3101]}}]}',
    mk_mask(0, CHOICE_LO, CHOICE_LO + 1)) == 1)

# 8) 廉价守卫
check("has_pending_choice 正确识别无待选", not has_pending_choice(mk_mask(0, 1, 3)))
check("has_pending_choice 正确识别有待选", has_pending_choice(mk_mask(0, CHOICE_LO)))
check("CHOICE 区间是 30..46", (CHOICE_LO, CHOICE_HI) == (30, 46))

print()
print("BAN_STATIC_OK" if OK else "BAN_STATIC_FAIL")
raise SystemExit(0 if OK else 1)
