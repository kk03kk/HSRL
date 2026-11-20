"""Lobby 种族过滤（**不改引擎**的确定性"禁种族"开关）。

背景（2026-11-17，用户提出"把机械和龙 ban 了，让模型先学会玩别的阵容"）
--------------------------------------------------------------------------
引擎每局从 **10 个类型里均匀抽 5 个**（`flow.rs`：optional = 9 个部族 + ABERRATION，
`cfg.lobby_races = 5`），并且 `RulesConfig` **只有"强制加入"（`lobby_types`），没有"排除"**。
真正的引擎级 ban 需要给 `RulesConfig` 加字段并重编 Rust——那会改变
`source_hash`（= 两个 crate 全部 `.rs` + cards.json 的 FNV 指纹，**当前 `c4c71014e3d41760`**），
而所有模型 manifest 都记着这个指纹 ⇒ 等于宣布"换引擎"。

所以这里走**确定性种子过滤**：
  1. 用 `Game(seed)` 建一局（很便宜，不跑模拟）；
  2. 从**观测**的 global 块最后 11 位读 lobby 种族位（`f[42:53]`，与
     `hsbg/env/obs_std.py::RACES_COUNT` 同源，也和 `state_json()["lobby_races"]` 的 bitmask 一致）；
  3. 若含被禁种族，就用**确定性派生**的下一个候选种子重来。

性质：
* **零引擎改动**（不重编、不改指纹、不动卡表）；
* **确定性**：候选种子只有 `(base_seed, k)` 参与，与调用顺序/局数无关 ⇒ 同一 base 下
  所有模型面对**同一批局**（配对统计仍然成立）；
* **不与其他分片碰撞**：候选种子用 64-bit 混合，不是 `seed+k` 的连续偏移；
* **代价**：禁 2 个种族时接受率 = C(8,5)/C(10,5) = 56/252 ≈ **22.2%**，即平均建 4.5 局
  才接受 1 局；建局本身不跑模拟，实测可忽略（见 `tests/test_lobby_filter.py`）。

与真实游戏的差异（必须记住）：真实酒馆是 10 个部族（含 Naga、无 Aberration）抽 5；
本引擎的 10 个是 9 部族 + Aberration（**NAGA 不入抽签**，只会经双种族卡出现）。
禁种族后 lobby 变成"8 抽 5"，商店池变浓（同族卡更容易凑齐），属**刻意课程**，不是真实分布。
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .strict_mirror import BOUNDED_ACTION_PROFILES, OPEN_ACTION_PROFILE, PLAYER150_PROFILE

__all__ = [
    "RACE_ORDER", "parse_race_list", "race_bits_from_features",
    "lobby_races_from_features", "is_excluded", "derive_seed",
    "make_filtered_game", "LobbyFilterStats",
]

#: 与 `hsbg/env/obs_std.py::RACE_NAMES` / 引擎 `Races::NAMES` **逐位一致**的顺序。
RACE_ORDER: Tuple[str, ...] = (
    "BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC",
    "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION", "NAGA",
)

#: 观测 global 块里 lobby 种族位的起点（42..52，共 11 位；见 obs_std.GLOBAL_LOBBY_RACES_START）。
GLOBAL_LOBBY_RACES_START = 42

#: 常见别名 -> 规范名（容忍用户写 MECH / MECHANICAL、法文/简写等）。
_ALIASES = {
    "MECH": "MECHANICAL",
    "MECHANICALS": "MECHANICAL",
    "PIRATES": "PIRATE",
    "DRAGONS": "DRAGON",
    "BEASTS": "BEAST",
    "DEMONS": "DEMON",
    "ELEMENTALS": "ELEMENTAL",
    "MURLOCS": "MURLOC",
    "QUILBOARS": "QUILBOAR",
    "UNDEADS": "UNDEAD",
    "ABERRATIONS": "ABERRATION",
    "NAGAS": "NAGA",
    "ALL": "",           # "ALL" 不是 lobby 类型，忽略
    "NONE": "",
}


def parse_race_list(spec: Any) -> Tuple[str, ...]:
    """``"MECH,DRAGON"`` / ``["MECHANICAL"]`` → 规范名元组（去重保序）。

    未知名字直接抛 ``ValueError``（宁可起不来，也不要静默不生效——这是本仓库的既有教训）。
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        items = [x.strip() for x in spec.replace(";", ",").split(",")]
    else:
        items = [str(x).strip() for x in spec]
    out: List[str] = []
    for raw in items:
        if not raw:
            continue
        name = raw.upper().replace(" ", "_").replace("-", "_")
        name = _ALIASES.get(name, name)
        if not name:
            continue
        if name not in RACE_ORDER:
            raise ValueError(
                f"未知种族名 {raw!r}；可选：{', '.join(RACE_ORDER)}"
                "（别名：MECH=MECHANICAL）")
        if name not in out:
            out.append(name)
    return tuple(out)


def race_bits_from_features(features: Any) -> int:
    """从观测 features 的 global 块解出 lobby 种族 bitmask（与引擎的 `Races` bit 一致）。

    接受 ``Game.observe()`` 原样返回的 **bytes**（little-endian f32），也接受 ndarray/list。
    """
    if isinstance(features, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(features), dtype="<f4")
    else:
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
    start = GLOBAL_LOBBY_RACES_START
    if arr.size < start + len(RACE_ORDER):
        raise ValueError(f"features 太短（{arr.size}）：不是标准 3227 观测")
    mask = 0
    for i in range(len(RACE_ORDER)):
        if float(arr[start + i]) > 0.5:
            mask |= (1 << i)
    return mask


def lobby_races_from_features(features: Any) -> Tuple[str, ...]:
    """观测 → 本局 lobby 的种族名元组（按 ``RACE_ORDER`` 顺序）。"""
    mask = race_bits_from_features(features)
    return tuple(n for i, n in enumerate(RACE_ORDER) if mask >> i & 1)


def _index_of(name: str) -> int:
    try:
        return RACE_ORDER.index(name)
    except ValueError:  # pragma: no cover - parse_race_list 已挡住
        raise ValueError(f"未知种族名 {name!r}") from None


def is_excluded(features: Any, exclude: Sequence[str]) -> bool:
    """该局 lobby 是否含被禁种族（``exclude`` 为空时恒 False = 不过滤）。"""
    if not exclude:
        return False
    mask = race_bits_from_features(features)
    for name in exclude:
        if mask >> _index_of(name) & 1:
            return True
    return False


#: 64-bit 混合常数（splitmix64 的乘法器）；只用来把 ``(base, k)`` 打散成互不相同的种子。
_MIX_MUL = 0x9E3779B97F4A7C15


def derive_seed(base: int, k: int) -> int:
    """``(base, k)`` → 确定性候选种子（k 不改变时结果稳定；k 递增时互不相同）。"""
    x = (int(base) & 0xFFFFFFFFFFFFFFFF) + (int(k) & 0xFFFFFFFFFFFFFFFF) * _MIX_MUL
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    # 引擎 seed 是 u64；保持非 0 且落在 63 位内，便于与其它 seed 体系共存
    return int(x & 0x7FFFFFFFFFFFFFFF) or 1


class LobbyFilterStats:
    """过滤计数（写进 collect meta，便于确认"开关真的生效"）。"""

    __slots__ = ("attempt", "skipped", "applied")

    def __init__(self) -> None:
        self.attempt = 0
        self.skipped = 0
        self.applied = 0

    def to_meta(self) -> dict:
        return {"lobby_filter_attempt": int(self.attempt),
                "lobby_filter_skipped": int(self.skipped),
                "lobby_filter_applied": int(self.applied)}

    def add(self, other: "LobbyFilterStats") -> None:
        self.attempt += int(other.attempt)
        self.skipped += int(other.skipped)
        self.applied += int(other.applied)


def engine_rule_overrides() -> dict[str, int]:
    if os.environ.get("HSAI_ACTION_PROFILE") == PLAYER150_PROFILE:
        return {"freeze_presses_per_turn": 2, "player_view_150": True}
    if os.environ.get("HSAI_ACTION_PROFILE") == OPEN_ACTION_PROFILE:
        return {"freeze_presses_per_turn": 0}
    if os.environ.get("HSAI_ACTION_PROFILE") in BOUNDED_ACTION_PROFILES:
        return {"freeze_presses_per_turn": 3}
    return {}


def make_filtered_game(engine: Any, seed: int, exclude: Sequence[str] = (), *,
                       learner_seat: int = 0, max_tries: int = 96,
                       probability: float = 1.0,
                       stats: Optional[LobbyFilterStats] = None
                       ) -> Tuple[Any, int]:
    """建一局**lobby 不含被禁种族**的游戏，返回 ``(game, 实际 seed)``。

    ``exclude`` 为空时直接使用原 seed 建局，并应用动作配置对应的引擎规则。
    ``max_tries`` 用完仍找不到就抛错（不要静默用一局含禁族的对局顶上去）。
    """
    rules = engine_rule_overrides()
    game_kwargs = {"config_json": json.dumps(rules)} if rules else {}
    if not 0 <= probability <= 1:
        raise ValueError("lobby filter probability must be between zero and one")
    # A separate hash stream assigns the practice condition before seeing the lobby.
    if probability < 1 and derive_seed(seed, 719) >= probability * (1 << 63):
        exclude = ()
    if not exclude:
        if stats is not None:
            stats.applied += 1
        return engine.Game(seed=int(seed), **game_kwargs), int(seed)
    for k in range(int(max_tries)):
        cand = int(seed) if k == 0 else derive_seed(int(seed), k)
        if stats is not None:
            stats.attempt += 1
        game = engine.Game(seed=cand, **game_kwargs)
        f, _c = game.observe(int(learner_seat))
        if not is_excluded(f, exclude):
            if stats is not None:
                stats.applied += 1
            return game, cand
        if stats is not None:
            stats.skipped += 1
    raise RuntimeError(
        f"lobby 过滤失败：{max_tries} 个候选种子都含被禁种族 {list(exclude)}"
        "（禁得太多？接受率 = C(10-禁数,5)/C(10,5)）")
