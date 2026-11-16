"""静态验证：加重护手（BG36_MagicItem_414）在待选里被遮掉，且不会把整手牌遮死。

不需要 GPU / 引擎：直接构造 pending JSON 与 245 维掩码。
"""
import numpy as np

from hsbg.az.train.action_ban import (BANNED_CHOICE_CARDS, CHOICE_HI, CHOICE_LO,
                                      ban_choice_options, has_pending_choice)
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


print("BANNED =", sorted(BANNED_CHOICE_CARDS))

# 1) 正常三选一，其中一项是加重护手 → 只遮那一项
pj = {"pending": {"kind": "trinket", "options": [
    {"id": "BG36_MagicItem_414"}, {"id": "BG32_MagicItem_271"}, {"id": "BG30_MagicItem_707"}]}}
m = mk_mask(0, CHOICE_LO + 0, CHOICE_LO + 1, CHOICE_LO + 2)
n = ban_choice_options(pj, m)
check("遮掉 1 个选项", n == 1)
check("加重护手那格已非法", not m[CHOICE_LO + 0])
check("其余两格仍合法", m[CHOICE_LO + 1] and m[CHOICE_LO + 2])
check("EndTurn 不受影响", m[0])

# 2) 没有加重护手 → 一个都不遮
pj2 = {"pending": {"kind": "trinket", "options": [{"id": "A"}, {"id": "B"}]}}
m2 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("无禁用项时不动", ban_choice_options(pj2, m2) == 0 and m2[CHOICE_LO])

# 3) 全是加重护手 → 不遮（防遮死）
pj3 = {"pending": {"kind": "trinket", "options": [
    {"id": "BG36_MagicItem_414"}, {"id": "BG36_MagicItem_414"}]}}
m3 = mk_mask(0, CHOICE_LO, CHOICE_LO + 1)
check("全禁用时不遮死", ban_choice_options(pj3, m3) == 0 and m3[CHOICE_LO])

# 4) 非 trinket 的待选（发现/时光扭曲）→ 不动
pj4 = {"pending": {"kind": "discover", "options": [{"id": "BG36_MagicItem_414"}]}}
m4 = mk_mask(0, CHOICE_LO)
check("非饰品待选不动", ban_choice_options(pj4, m4) == 0 and m4[CHOICE_LO])

# 5) 廉价守卫：没有待选动作时不该解析
check("has_pending_choice 正确识别无待选", not has_pending_choice(mk_mask(0, 1, 3)))
check("has_pending_choice 正确识别有待选", has_pending_choice(mk_mask(0, CHOICE_LO)))
check("CHOICE 区间是 30..45", (CHOICE_LO, CHOICE_HI) == (30, 46))

print()
print("BAN_STATIC_OK" if OK else "BAN_STATIC_FAIL")
raise SystemExit(0 if OK else 1)
