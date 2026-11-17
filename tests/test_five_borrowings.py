"""5 项借鉴里**与模型结构无关**的两项：④ 名次效用表、⑤ no_cancel_choose。

①②③（z-token / 卡表属性 / 磁力查询）的结构与 checkpoint 兼容性由
``tests/test_borrowed_arch_compat.py`` 覆盖（它能在**没有引擎**的机器上跑）。

2026-11-17 审计修了两处本文件自身的缺陷：
* 原来在模块级 ``raise SystemExit`` ⇒ ``pytest tests`` 直接 INTERNALERROR；
* ⑤ 段以前**根本没有调用** ``no_cancel_choose``，只是对手工构造的两个数组自证
  ⇒ 把这个函数删掉那三条断言依然全绿。现在真的调用它。
"""
from __future__ import annotations

import numpy as np

from hsbg.az.train.common import (rank_utility_table, set_rank_utility_table,
                                  set_win_bonus)
from hsbg.az.train.pending_guard import no_cancel_choose, without_cancel
from hsbg.env.action_space_std import A_CANCEL, A_CHOOSE, A_END, N_TOTAL_ACTIONS

#: 2026-11-17 用户口径：只奖励前二（2→3 名断崖 1.35）
RANK_TABLE = [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2]


def _mk(*legal):
    m = np.zeros(N_TOTAL_ACTIONS, dtype=bool)
    for a in legal:
        m[a] = True
    return m


# ────────────────────────── ④ 名次效用表 ──────────────────────────
def test_rank_utility_table_roundtrip():
    set_win_bonus(0.0)
    set_rank_utility_table(RANK_TABLE)
    got = [round(float(x), 3) for x in rank_utility_table()]
    assert got == RANK_TABLE
    # 冠军 vs 亚军 与 亚军 vs 季军 的断崖比例（这是"只奖励前二"的核心）
    assert round(got[0] - got[1], 3) == 0.8
    assert round(got[1] - got[2], 3) == 0.2
    assert got[0] == max(got) and got[-1] == min(got)     # u(1) 最大、u(8) 最小


def test_win_bonus_only_touches_first_place():
    set_win_bonus(0.0)
    base = [round(float(x), 4) for x in rank_utility_table()]
    set_win_bonus(0.7)
    bumped = [round(float(x), 4) for x in rank_utility_table()]
    assert bumped[0] == round(base[0] + 0.7, 4)
    assert bumped[1:] == base[1:]                          # 其余名次完全不动
    set_win_bonus(0.0)                                     # 还原全局状态


# ────────────────────────── ⑤ no_cancel_choose ──────────────────────────
def test_no_cancel_choose_bans_cancel_only_while_pending():
    mask = _mk(A_END, A_CANCEL, A_CHOOSE)
    out = no_cancel_choose(mask)
    assert out[A_CANCEL] is np.False_ or not out[A_CANCEL]
    assert out[A_END]                       # EndTurn 必须保留
    assert out[A_CHOOSE]                    # 待选项保留


def test_no_cancel_choose_keeps_cancel_without_pending():
    mask = _mk(A_END, A_CANCEL, 1)          # 没有待选
    out = no_cancel_choose(mask)
    assert out[A_CANCEL]                    # 与"全局禁 Cancel"不同


def test_no_cancel_choose_does_not_mutate_input():
    mask = _mk(A_END, A_CANCEL, A_CHOOSE)
    before = mask.copy()
    no_cancel_choose(mask)
    assert np.array_equal(mask, before)


def test_without_cancel_is_the_global_ban():
    """训练契约用的是全局禁（`without_cancel`），语义必须与 ⑤ 明确区分。"""
    mask = _mk(A_END, A_CANCEL, 1)
    assert not without_cancel(mask)[A_CANCEL]


def main() -> int:                          # pragma: no cover - 只给人跑
    set_win_bonus(0.0)
    set_rank_utility_table(RANK_TABLE)
    t = [round(float(x), 3) for x in rank_utility_table()]
    print("=== ④ 奖励表 ===")
    print("  生效表 =", t)
    print("    1→2 差 %.2f   2→3 崖 %.2f   3–6 全 0   7/8 负"
          % (t[0] - t[1], t[1] - t[2]))
    print("\n=== ⑤ no_cancel_choose ===")
    m = _mk(A_END, A_CANCEL, A_CHOOSE)
    print("  有待选时 Cancel 被禁 =", not no_cancel_choose(m)[A_CANCEL])
    print("  无待选时 Cancel 保留 =",
          bool(no_cancel_choose(_mk(A_END, A_CANCEL, 1))[A_CANCEL]))
    print("\nALL5_OK（①②③ 见 tests/test_borrowed_arch_compat.py）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
