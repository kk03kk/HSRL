"""动作层黑名单：把指定卡从「待选（PendingChoice）选项」里遮掉。

**为什么在策略层做、不改模拟器**（2026-11-17 用户定）：
「加重护手」（`BG36_MagicItem_414`，引擎数字 id **3037**）会让模型后期无脑变大、学不到操作，
所以训练与评测都不许再选它。但模拟器必须保持 37.0.2 原样，因此在**策略取掩码之后、采样之前**
把那个选项置为非法。

⚠️ 2026-11-17 修正（**v1.3 三轮的封禁因此完全没生效**，实测 r3 里加重护手仍占 3.81%）：
`player_json` 的 ``pending`` 是 **list + 外部标记枚举**，不是 dict：

    "pending": [{"HeroSelect": {"options": [188, 3679]}}]
    "pending": [{"Trinket": {"free": false, "options": [3037, ...], "replace": null}}]

旧代码用 ``isinstance(pend, dict) and pend["kind"] == "trinket"`` 判定 ⇒ 恒为 False ⇒ 静默失效。
现在同时兼容两种形态，并且 ``options`` 里的元素既可能是**整数卡索引**也可能是 ``{"id": ...}``。

为什么不会伪造 on-policy：被遮掉的是**动作掩码**本身，采样、logprob、PPO 损失三条路径都用同一份
mask，所以策略分布与重要性比完全自洽，``illegal_actions`` 仍为 0。
"""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional, Sequence, Set

from hsbg.env.action_space_std import A_CHOOSE, N_CHOICE

#: 禁止模型主动选择的卡（卡牌 id 字符串）。加重护手 = BG36_MagicItem_414（数字 3037）。
BANNED_CHOICE_CARDS: Set[str] = {"BG36_MagicItem_414"}
#: 同上，但用**引擎数字索引**（options 里就是这种）。3037 = BG36_MagicItem_414。
BANNED_CHOICE_INDICES: Set[int] = {3037}

#: 待选动作在 245 维里的区间 [lo, hi)
CHOICE_LO = int(A_CHOOSE)
CHOICE_HI = int(A_CHOOSE) + int(N_CHOICE)

_TRINKET_KEYS = ("Trinket", "trinket")


def has_pending_choice(mask: Any) -> bool:
    """掩码里有没有合法的待选动作（廉价守卫：多数决策在这里就返回 False，不解析 JSON）。"""
    try:
        return bool(mask[CHOICE_LO:CHOICE_HI].any())
    except Exception:  # noqa: BLE001 - 掩码类型异常时按"没有"处理，不影响主流程
        return False


def _pending_variants(pend: Any) -> List[dict]:
    """把 ``pending`` 归一化成 ``[{Variant: payload}, ...]``。

    兼容三种写法：list（**引擎实际形态**）、dict 带 ``kind``（旧假设/其它投影）、dict 单键。
    """
    out: List[dict] = []
    if isinstance(pend, list):
        for x in pend:
            if isinstance(x, dict):
                out.append(x)
    elif isinstance(pend, dict):
        if "kind" in pend:                     # {kind: "trinket", options: [...]}
            out.append({str(pend.get("kind")): pend})
        elif len(pend) == 1:                   # {Trinket: {...}}
            out.append(pend)
    return out


def trinket_options(player_json: Any) -> Optional[List[Any]]:
    """若当前待选是**饰品三选一/替换**，返回 options 列表（元素可能是 int 或 dict）；否则 None。"""
    pj = player_json
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except (ValueError, TypeError):
            return None
    if not isinstance(pj, dict):
        return None
    for var in _pending_variants(pj.get("pending")):
        for key in _TRINKET_KEYS:
            if key in var:
                payload = var[key]
                opts = (payload or {}).get("options") if isinstance(payload, dict) else None
                if isinstance(opts, list) and opts:
                    return opts
    return None


def _opt_index(o: Any) -> Optional[int]:
    """从选项里取引擎卡索引（int 或 {"id": n}）。"""
    if isinstance(o, bool):
        return None
    if isinstance(o, int):
        return o
    if isinstance(o, dict):
        v = o.get("index")
        if isinstance(v, int):
            return v
        v = o.get("id")
        if isinstance(v, int):
            return v
    return None


def _opt_name(o: Any) -> Optional[str]:
    if isinstance(o, dict):
        for k in ("id_str", "card_id", "code"):
            v = o.get(k)
            if isinstance(v, str):
                return v
    return None


def ban_choice_options(player_json: Any, mask: Any,
                       banned: Optional[Sequence[str]] = None,
                       banned_indices: Optional[Iterable[int]] = None) -> int:
    """把黑名单卡从待选选项里遮掉，**原地改 mask**，返回被遮掉的选项数。

    仅当待选里含 ``Trinket`` 时生效。为防"全被遮掉导致无路可走"，至少保留一个合法选项；
    一个都保不住就整体不动（EndTurn / Cancel 永远合法，不会卡死）。
    """
    names = set(banned) if banned is not None else BANNED_CHOICE_CARDS
    idxs = set(banned_indices) if banned_indices is not None else BANNED_CHOICE_INDICES
    if not names and not idxs:
        return 0
    opts = trinket_options(player_json)
    if not opts:
        return 0

    hit: List[int] = []
    for i, o in enumerate(opts):
        if _opt_index(o) in idxs or _opt_name(o) in names:
            hit.append(i)
    if not hit:
        return 0

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
