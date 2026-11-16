"""动作层黑名单：把指定卡从「待选（PendingChoice）选项」里遮掉。

**为什么在策略层做、不改模拟器**（2026-11-17 用户定）：
「加重护手」（`BG36_MagicItem_414`，数字 id 3037）会让模型后期无脑变大、学不到操作，
所以训练与评测都不许再选它。但模拟器必须保持 37.0.2 原样（引擎指纹、卡表、
其他人/评测都依赖它），因此在**策略取掩码之后、采样之前**把那个选项置为非法。

为什么不会伪造 on-policy：被遮掉的是**动作掩码**本身，采样、logprob、PPO 损失
三条路径都用同一份 mask，所以策略分布与重要性比完全自洽，`illegal_actions` 仍为 0。

覆盖面：
- 训练（`PPOSelfPlayPolicy._observe`）与评测（`PPOGreedyPolicy.decide`）共用本模块；
- 只作用于 **learner 自己**；镜像注入的 7 个对手不受影响（它们本来就不过策略）。
- 只能拦「**摆在你面前让你选**」的饰品（T6 小饰品 / T9 大饰品 / 替换饰品）。
  随机白给类（如「未知宝珠」随机给一件）无法拦，属于引擎随机性，不改引擎就拦不住。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set

from hsbg.env.action_space_std import A_CHOOSE, N_CHOICE

#: 禁止模型主动选择的卡（**只写卡牌 id 字符串**）。加重护手 = BG36_MagicItem_414。
BANNED_CHOICE_CARDS: Set[str] = {"BG36_MagicItem_414"}

#: 待选动作在 245 维里的区间 [lo, hi)
CHOICE_LO = int(A_CHOOSE)
CHOICE_HI = int(A_CHOOSE) + int(N_CHOICE)


def has_pending_choice(mask: Any) -> bool:
    """掩码里有没有合法的待选动作（廉价守卫：多数决策在这里就返回 False，不解析 JSON）。"""
    try:
        return bool(mask[CHOICE_LO:CHOICE_HI].any())
    except Exception:  # noqa: BLE001 - 掩码类型异常时按"没有"处理，不影响主流程
        return False


def ban_choice_options(player_json: Optional[Dict[str, Any]], mask: Any,
                       banned: Optional[Sequence[str]] = None) -> int:
    """把 ``banned`` 里的卡从待选选项里遮掉，**原地改 mask**，返回被遮掉的选项数。

    仅当 ``pending.kind == "trinket"`` 时生效（饰品三选一 / 替换饰品）。
    为防"全被遮掉导致无路可走"，至少保留一个合法选项；一个都保不住就整体不动。
    """
    names = set(banned) if banned is not None else BANNED_CHOICE_CARDS
    if not names or not isinstance(player_json, dict):
        return 0
    pend = player_json.get("pending")
    if not isinstance(pend, dict) or str(pend.get("kind") or "") != "trinket":
        return 0
    opts = pend.get("options") or []
    if not isinstance(opts, list) or not opts:
        return 0

    hit = []
    for i, o in enumerate(opts):
        cid = (o or {}).get("id") if isinstance(o, dict) else None
        if cid in names:
            hit.append(i)
    if not hit:
        return 0

    # 至少留一个合法选项（防把整手牌遮死）
    keep = [i for i in range(len(opts))
            if i not in hit and CHOICE_LO + i < len(mask) and bool(mask[CHOICE_LO + i])]
    if not keep:
        return 0

    n = 0
    for i in hit:
        a = CHOICE_LO + i
        if a < len(mask) and bool(mask[a]):
            mask[a] = False
            n += 1
    return n
