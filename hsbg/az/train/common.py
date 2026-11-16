"""AZ 训练脚手架公共常量与工具。

约束（与任务板 task-3 一致）：
- 只复用 `hsbg_engine`（Rust 主引擎）的公开 API；本包是唯一的训练/搜索栈。
- 固定形状：245 动作 / 3227 features / 124 cards / 8 座；
- 本模块在**没有 hsbg_engine** 的开发机上也能 import（引擎相关入口惰性加载），
  这样数据/模型/报告相关的单测不依赖引擎。

动作分段与 docs/ACTION_SPACE_STD.md、docs/AZ_MCTS_SELFPLAY_DESIGN.md §2.3 一致，
与 AutoTavern 245 动作逐位对齐。
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np

# ── 固定形状 ──────────────────────────────────────────────────────────────
N_ACTIONS = 245
OBS_FLOATS = 3227
OBS_CARDS = 124
N_PLAYERS = 8
N_RANKS = 8
DATA_VERSION = "37.0.2.256589"

# ── 245 动作分段（闭区间，含端点）────────────────────────────────────────
# EndTurn=0, Refresh=1, Freeze=2, Upgrade=3, HeroPower=4..5, DarkGift=6,
# Cancel=7, Buy=8..15, Sell=16..22, Activate=23..29, Choose=30..45,
# Play=46..125, Magnetize=126..195, Move=196..244
ACTION_SEGMENTS: Dict[str, tuple[int, int]] = {
    "end_turn": (0, 0),
    "refresh": (1, 1),
    "freeze": (2, 2),
    "upgrade": (3, 3),
    "hero_power": (4, 5),
    "dark_gift": (6, 6),
    "cancel": (7, 7),
    "buy": (8, 15),
    "sell": (16, 22),
    "activate": (23, 29),
    "choose": (30, 45),
    "play": (46, 125),
    "magnetize": (126, 195),
    "move": (196, 244),
}

#: 名次效用 u(r)=1-2*(r-1)/(N-1)，u(1)=+1 … u(8)=-1（设计 §2.5）。
#: 2026-11-16 用户要求「提高吃鸡奖励」：新增**冠军奖金** b，u(1) 变成 1+b，
#: 其余名次**完全不动**（保住已经学到的"稳前四"结构）。
#: 关键：只用同一个数组同时喂 ①终局奖励 ②V(s)=Σp·u —— 两边必须一致，
#: 所以这里用**原地改写**（in-place）而不是换对象：model/ppo/train_loop 都是
#: ``from .common import RANK_UTILITY`` 持有同一个 ndarray 引用，改元素即可全局生效。
_BASE_RANK_UTILITY = np.array(
    [1.0 - 2.0 * (r - 1) / (N_PLAYERS - 1) for r in range(1, N_RANKS + 1)],
    dtype=np.float32,
)
RANK_UTILITY = _BASE_RANK_UTILITY.copy()

#: 当前生效的冠军奖金（0 = 论文原值；写进 run.env/checkpoint 便于复现）。
WIN_BONUS = 0.0


def set_win_bonus(bonus: float) -> float:
    """设置冠军奖金 b：u(1) ← 1+b，其余名次不变。返回实际生效值。

    b=0 时与论文原值**逐位相同**（回归安全）。传负数会被夹到 0。
    """
    global WIN_BONUS
    b = max(0.0, float(bonus or 0.0))
    RANK_UTILITY[:] = _BASE_RANK_UTILITY
    if b:
        RANK_UTILITY[0] = 1.0 + b
    WIN_BONUS = b
    return b


def rank_utility_table() -> "np.ndarray":
    """当前生效的名次效用表（只读副本，便于写日志/核对）。"""
    return RANK_UTILITY.copy()


def set_rank_utility_table(values) -> "np.ndarray":
    """**直接指定**整张名次效用表（2026-11-17 用户定：只奖励前二）。

    例：``[1.00, 0.35, -1.00, -1.15, -1.30, -1.45, -1.60, -1.75]``
    ⇒ 2 名→3 名断崖 1.35（原设计 0.286 的 4.7 倍），3–8 名之间只留 0.15 的梯度。

    与 ``set_win_bonus`` 的关系：两者都写同一个 ``RANK_UTILITY``；本函数**后写生效**，
    因此 ``--rank-utility-table`` 与 ``--win-bonus`` 同时给时以整表为准（训练侧会告警）。
    """
    vals = [float(v) for v in values]
    if len(vals) != int(N_RANKS):
        raise ValueError(f"rank utility 需要 {int(N_RANKS)} 个数，得到 {len(vals)} 个")
    RANK_UTILITY[:] = np.asarray(vals, dtype=np.float32)
    return rank_utility_table()


# ── 2026-11-16：把冠军奖金传进 spawn 子进程 ────────────────────────────────
# 奖励是在**采集子进程**里算的（``process_pool`` 用 ``mp.get_context("spawn")``，
# 子进程会重新 import 本模块 ⇒ 父进程的原地改写传不过去，奖金会**静默失效**）。
# 环境变量是 spawn 子进程唯一可靠继承的通道，所以在 import 时按它重新应用一遍。
# 父进程（train_entity_sync）在 set_win_bonus 之后会同步写 HSAI_WIN_BONUS。
def _apply_env_win_bonus() -> None:
    raw = os.environ.get("HSAI_WIN_BONUS")
    if raw:
        try:
            set_win_bonus(float(raw))
        except (TypeError, ValueError):
            pass
    # 整表优先：环境里有 HSAI_RANK_UTILITY 就覆盖掉上面的 win_bonus 结果。
    raw_tbl = os.environ.get("HSAI_RANK_UTILITY")
    if raw_tbl:
        try:
            set_rank_utility_table([float(x) for x in raw_tbl.split(",") if x.strip()])
        except (TypeError, ValueError):
            pass


_apply_env_win_bonus()


def segment_indices(name: str) -> np.ndarray:
    """某动作分组在 245 维动作空间中的下标数组。"""
    lo, hi = ACTION_SEGMENTS[name]
    return np.arange(lo, hi + 1, dtype=np.int64)


def group_of(action: int) -> str:
    """动作下标 -> 分组名（越界返回 ``"unknown"``）。"""
    a = int(action)
    for name, (lo, hi) in ACTION_SEGMENTS.items():
        if lo <= a <= hi:
            return name
    return "unknown"


def utility_from_rank(rank: Any) -> np.ndarray:
    """名次 1..8 -> 效用 [-1,1]；rank<=0（未知/被守卫截断）-> 0.0。"""
    r = np.asarray(rank, dtype=np.int32)
    out = np.zeros(r.shape, dtype=np.float32)
    ok = (r >= 1) & (r <= N_RANKS)
    out[ok] = RANK_UTILITY[r[ok] - 1]
    return out


# ── 引擎惰性加载（引擎缺失时给出可读错误，而不是 import 期崩溃）──────────
_ENGINE: Optional[Any] = None


def engine_available() -> bool:
    """当前环境是否可 import ``hsbg_engine``（Rust 主引擎已构建安装）。"""
    try:
        import hsbg_engine  # noqa: F401
    except Exception:  # noqa: BLE001 - 缺构建/缺包都视为不可用
        return False
    return True


def load_engine():
    """返回 ``hsbg_engine`` 模块；不可用时抛 RuntimeError（附构建提示）。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        import hsbg_engine as E
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "hsbg_engine 不可用：需要 Rust 主引擎构建安装 "
            "(`cd engine && python -m pip install -e .`, 见 docs/RUST_ENGINE.md)"
        ) from exc
    _ENGINE = E
    return E


def engine_info() -> Dict[str, Any]:
    """引擎/数据版本摘要（写进报告，保证可复现与跨版本可比性）。"""
    E = load_engine()
    try:
        data = E.data_info()
    except Exception:  # noqa: BLE001
        data = {}
    try:
        binfo = E.build_info()
        profile = str(binfo.get("profile"))
        source_hash = str(binfo.get("source_hash"))
    except Exception:  # noqa: BLE001
        profile, source_hash = None, None
    return {
        "engine_version": str(E.version()),
        "game_version": str(E.game_version()),
        "server_rules_version": str(E.server_rules_version()),
        "cards": int(data.get("cards", -1)),
        "build_profile": profile,
        "source_hash": source_hash,
        "n_actions": int(E.N_ACTIONS),
        "obs_floats": int(E.OBS_FLOATS),
        "obs_cards": int(E.OBS_CARDS),
        "data_version": str(E.game_version()),
    }


# ── 观测/动作数组工具 ─────────────────────────────────────────────────────
def obs_arrays(pair: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    """``Game.observe(seat) -> (features_bytes, cards_bytes)`` -> ndarray 拷贝。"""
    f_raw, c_raw = pair
    f = np.frombuffer(f_raw, dtype="<f4").copy().astype(np.float32, copy=False)
    c = np.frombuffer(c_raw, dtype="<i4").copy().astype(np.int32, copy=False)
    return f, c


def mask_from_bytes(raw: Any) -> np.ndarray:
    """``Game.legal_mask(seat)`` 的 u8[245] 字节 -> bool[245]。"""
    return np.frombuffer(raw, dtype=np.uint8).astype(bool)


def masked_normalize(policy: Any, mask: Any, eps: float = 1e-8) -> np.ndarray:
    """按合法掩码清零并归一；全 0 时退化为合法动作上的均匀分布。"""
    p = np.asarray(policy, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool)
    p[~m] = 0.0
    s = float(p.sum())
    if s > eps:
        return p / s
    legal = m.astype(np.float32)
    n = float(legal.sum())
    if n <= 0:
        return np.zeros(N_ACTIONS, dtype=np.float32)
    return legal / n


def uniform_mask_policy(mask: Any) -> np.ndarray:
    """合法动作上的均匀分布（非法动作 0）。"""
    m = np.asarray(mask, dtype=bool)
    legal = m.astype(np.float32)
    n = float(legal.sum())
    if n <= 0:
        return np.zeros(N_ACTIONS, dtype=np.float32)
    return legal / n


def one_hot_policy(action: int, mask: Optional[Any] = None) -> np.ndarray:
    """动作下标 -> 245 维 one-hot（可选按 mask 校验，非法则退化为均匀合法）。"""
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    a = int(action)
    if mask is None:
        if 0 <= a < N_ACTIONS:
            pi[a] = 1.0
        return pi
    m = np.asarray(mask, dtype=bool)
    if 0 <= a < N_ACTIONS and bool(m[a]):
        pi[a] = 1.0
        return pi
    return uniform_mask_policy(m)


def masked_argmax(logits: Any, mask: Any) -> int:
    """在合法动作上取 argmax（非法位视为 -inf）；无合法动作返回 0。"""
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if not bool(m.any()):
        return 0
    x = np.where(m, x, -np.inf)
    return int(np.argmax(x))


def masked_softmax(logits: Any, mask: Any, eps: float = 0.0) -> np.ndarray:
    """数值稳定的 masked softmax（NaN/Inf 兜底为均匀合法分布）。"""
    x = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if not bool(m.any()):
        return np.zeros(N_ACTIONS, dtype=np.float32)
    x[~m] = -np.inf
    x = x - np.max(x[m])
    e = np.exp(x)
    s = float(e.sum())
    if not math.isfinite(s) or s <= 0:
        return uniform_mask_policy(m)
    p = (e / s).astype(np.float32)
    if eps > 0.0:
        p = (1.0 - eps) * p + eps * uniform_mask_policy(m)
    return p


def policy_entropy(policy: Any, mask: Optional[Any] = None) -> float:
    """策略熵（nats）；全 0 策略返回 0。"""
    p = np.asarray(policy, dtype=np.float64).reshape(-1)
    if mask is not None:
        p = p * np.asarray(mask, dtype=bool)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


# ── JSON 友好 ─────────────────────────────────────────────────────────────
def to_jsonable(obj: Any) -> Any:
    """递归把 numpy 类型/数组转换为可 ``json.dump`` 的原生类型。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


__all__ = [
    "N_ACTIONS",
    "OBS_FLOATS",
    "OBS_CARDS",
    "N_PLAYERS",
    "N_RANKS",
    "DATA_VERSION",
    "ACTION_SEGMENTS",
    "RANK_UTILITY",
    "segment_indices",
    "group_of",
    "utility_from_rank",
    "engine_available",
    "load_engine",
    "engine_info",
    "obs_arrays",
    "mask_from_bytes",
    "masked_normalize",
    "uniform_mask_policy",
    "one_hot_policy",
    "masked_argmax",
    "masked_softmax",
    "policy_entropy",
    "to_jsonable",
]
