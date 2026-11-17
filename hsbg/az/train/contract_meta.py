"""动作契约的**唯一** meta 导出点（2026-11-17）。

背景：`freeze_cap` / `ban_trinkets` / `forbid_unfreeze` / `lobby_exclude` /
`max_actions_per_turn` 这些限制以前只存在于命令行，训练产物里查不到 ——
跑完之后无法自证"当时用的是哪套限制"。更麻烦的是 meta 有**至少四个**组装点
（串行、parallel 逐局、lockstep、process_pool 汇总），逐个手写必然漂移。

所以：一处定义，四处调用。
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["CONTRACT_META_KEYS", "contract_meta"]

#: 必须出现在每轮 collect meta 里的字段（测试锁死）。
CONTRACT_META_KEYS: tuple = (
    "max_actions_per_turn", "move_guard", "move_cap", "move_repeat_limit",
    "freeze_cap", "forbid_cancel", "forbid_unfreeze", "ban_trinkets",
    "lobby_exclude", "sims",
)


def contract_meta(cfg: Any) -> Dict[str, Any]:
    """从 ``CollectConfig`` 抽出"这一轮实际生效的动作契约/限制"。

    缺字段时给安全默认（不抛），因为历史配置对象可能没有新加的属性。
    """
    get = lambda name, default: getattr(cfg, name, default)  # noqa: E731
    return {
        "max_actions_per_turn": int(get("max_actions_per_turn", 0) or 0),
        "move_guard": bool(get("move_guard", False)),
        "move_cap": int(get("move_cap", 0) or 0),
        "move_repeat_limit": int(get("move_repeat_limit", 0) or 0),
        "freeze_cap": int(get("freeze_cap", 0) or 0),
        "forbid_cancel": bool(get("forbid_cancel", False)),
        "forbid_unfreeze": bool(get("forbid_unfreeze", False)),
        "ban_trinkets": bool(get("ban_trinkets", False)),
        "lobby_exclude": list(get("lobby_exclude", ()) or ()),
        "sims": int(get("sims", 0) or 0),
    }


def merge_contract_meta(metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """pool 路径：各 worker 的 meta 逐 shard 相同，取第一个带契约的即可。"""
    for m in metas:
        if isinstance(m, dict) and "max_actions_per_turn" in m:
            return {k: m[k] for k in CONTRACT_META_KEYS if k in m}
    return {}
