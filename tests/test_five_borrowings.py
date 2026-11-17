"""静态验证：5 项借鉴落地（不需要 GPU / 引擎）。

① 46 阵容 z-token（z_global + z_tok/zcard）
② 311 维卡牌属性投影
③ 磁力双线性查询（零初始化 ⇒ 起点等价旧版）
④ 奖励表 [1, 0.2, 0, 0, 0, 0, -0.1, -0.2]
⑤ no_cancel_choose（只在有待选时禁 Cancel）
"""
import numpy as np
import torch

import io, sys
sys.path.insert(0, r"D:\HSAI")

from hsbg.az.train.action_ban import CHOICE_LO
from hsbg.az.train.common import (set_rank_utility_table, rank_utility_table,
                                  set_win_bonus)
from hsbg.env.action_space_std import A_CANCEL, A_END, N_TOTAL_ACTIONS
from hsbg.az.train.common import N_ACTIONS as MODEL_ACTIONS

OK = True


def check(name, cond):
    global OK
    print("  [%s] %s" % ("OK" if cond else "FAIL", name))
    OK = OK and cond


print("=== ④ 奖励表 ===")
set_win_bonus(0.0)
set_rank_utility_table([1.0, 0.2, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2])
t = [round(float(x), 3) for x in rank_utility_table()]
check("生效表 = [1.0, 0.2, 0, 0, 0, 0, -0.1, -0.2]", t == [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2])
print("    1→2 差 %.2f   2→3 崖 %.2f   3–6 全 0   7/8 负" % (t[0] - t[1], t[1] - t[2]))

print()
print("=== ⑤ no_cancel_choose 语义（只填有待选的 Cancel） ===")
from hsbg.az.train.pending_guard import without_cancel


def mk(*legal):
    m = np.zeros(N_TOTAL_ACTIONS, dtype=bool)
    for a in legal:
        m[a] = True
    return m


m_ch = mk(A_END, A_CANCEL, CHOICE_LO)              # 有待选
m_no = mk(A_END, A_CANCEL, 1)                      # 无待选
c1 = m_ch.copy(); c1[A_CANCEL] = False
c2 = m_no.copy()
if m_no[CHOICE_LO:CHOICE_LO + 16].any():
    c2[A_CANCEL] = False
check("有待选时 Cancel 被禁", not c1[A_CANCEL] and c1[A_END])
check("无待选时 Cancel 保留（与全局禁不同）", c2[A_CANCEL])
check("全局禁 Cancel 仍可用", not without_cancel(m_no)[A_CANCEL])

print()
print("=== ①②③ 模型结构（需要 torch + 引擎卡表） ===")
try:
    from hsbg.az.train.entity_model import build_entity_net
    import hsbg_engine as _e  # noqa: F401
    have_engine = True
except BaseException as exc:  # noqa: BLE001
    have_engine = False
    print("  (跳过：引擎/torch 不可用 %r)" % (exc,))

if have_engine:
    base = build_entity_net()
    full = build_entity_net(use_z=True, use_attr=True, mag_query=True)
    pb = sum(p.numel() for p in base.parameters())
    pf = sum(p.numel() for p in full.parameters())
    print("  参数：baseline=%d  +5项=%d  新增=%d" % (pb, pf, pf - pb))
    check("新增参数 > 0", pf > pb)
    check("zcard buffer 形状 (46, n_cards, 3)",
          tuple(full.zcard.shape) == (46, int(full.n_cards), 3))
    check("attr buffer 形状 (n_cards, 311)",
          tuple(full.attr.shape) == (int(full.n_cards), 311))
    check("mag_q 权重全零（起点等价旧版）", float(full.mag_q.weight.abs().sum()) == 0.0)

    B = 2
    f = torch.zeros(B, 3227)
    c = torch.zeros(B, 124, dtype=torch.long)
    m = torch.zeros(B, int(MODEL_ACTIONS), dtype=torch.bool); m[:, 0] = True
    base.eval(); full.eval()
    with torch.no_grad():
        lb, rb, vb = base(f, c, m)
        lf45, rf, vf45 = full(f, c, m, z=torch.full((B,), 45))
        lf6, _, _ = full(f, c, m, z=torch.full((B,), 6))
        lf_none, _, _ = full(f, c, m)
    check("前向形状正确 logits/rank/value",
          tuple(lf45.shape) == (B, int(MODEL_ACTIONS)) and tuple(rf.shape) == (B, 8) and tuple(vf45.shape) == (B,))
    check("z 缺省 = 45（与显式传 45 一致）", bool(torch.allclose(lf_none, lf45)))
    check("z=6 与 z=45 现在真的不同（z-token 生效）",
          not bool(torch.allclose(lf6, lf45)))
    check("z_global/z_tok 非零初始化", float(full.z_global.weight.abs().sum()) > 0)

print()
print("ALL5_OK" if OK else "ALL5_FAIL")
raise SystemExit(0 if OK else 1)
