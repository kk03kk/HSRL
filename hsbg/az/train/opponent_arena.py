"""闭环对手难度控制器（只用 Python 标准库，可单独 import）。

训练时每回合把对手战团替换成「真人战团库」里的一条真实战团；控制器按学习者**最近一轮**
的实战胜率升降对手强度，把对手一直顶在「略高于自身水平」的位置：

- ``mid`` / ``late``：目标是该 band 的**战斗胜率**贴近 ``target_win``（默认 0.60 / 0.50）。
  胜率远高于目标 = 对手太弱（加难度），远低于目标 = 被压死（降难度）。终盘目标定在
  0.50 而不是 0.8：真实对局里学习者不可能长期压着同水平对手打，胜率略高于 50% 才是
  「有挑战但打得过」。
- ``early``：前期战团还没成型，单场胜负基本是运气，胜率不是好信号；改用**「活到
  ``reach_turn`` 的局占比」**当难度代理——活得太少说明前期就被打死，就该降强度。
  这个符号（太难 = err < 0 = 降强度）与胜率方向自然一致，所以 deadband / 步长 / 夹紧
  可以整套复用。
- **棘轮**：``mid`` / ``late`` 的胜率同时超过目标 + deadband、且 reach 也达标，说明学习者
  已经吃透当前难度，于是把 ``reach_target`` 每次收紧 0.05（最多收到 ``reach_floor``），
  把前期压力加回来；否则前期段会永远停在「很松」的档位（``reach_target`` 只会因棘轮下降）。

刻度：``strength`` 是**绝对刻度** ``Σ(atk+hp)``（与 ``human_board_pool`` 的战团强度同
口径）。调用方从战团库分位算出每个 band 的上下限后经 ``spec.bounds`` 传入；本模块不读
数据、不碰引擎，只做纯数值闭环。

实现约束（别顺手破坏）：

- **只 import 标准库**（``math`` / ``json`` / ``os`` / ``dataclasses`` / ``typing``）：
  不 import torch / numpy / hsbg_engine / 任何本仓库模块，因此能被单独 import，也不受
  包 ``__init__`` 副作用影响；
- ``update`` **绝不抛异常**：缺键 / 非数值 / NaN / inf 只跳过该 band 并把原因追加到
  ``notes``，``strength`` 永远是有限正数（不会把写坏的状态带进下一步采样）；
- 续训从磁盘恢复强度：每段训练都接着上次难度，不能每段重头再来。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

#: 回合分段：(band 名, 起始回合, 结束回合)，两端都是**闭区间**。
BANDS: Tuple[Tuple[str, int, int], ...] = (("early", 1, 7), ("mid", 8, 11), ("late", 12, 999))

#: 全部 band 名（顺序 = 强度由低到高）。
BAND_NAMES: Tuple[str, ...] = tuple(b[0] for b in BANDS)

#: 每个 band 的**代表回合**（band 中心）：强度代表值放在这里，相邻 band 之间按回合线性
#: 插值，回合之间才平滑、不跳变。early 中心 4；mid 中心 9.5（8..11 的中点）；late 取 13
#: （12 之后的第一档，再往后恒等于 late 强度）。
BAND_CENTER: Dict[str, float] = {"early": 4.0, "mid": 9.5, "late": 13.0}

#: 扁平摘要里要报的 reach 回合：「活到第 N 回合」的局占比（0..1）。
REACH_TURNS: Tuple[int, int] = (9, 12)

#: 强度下限保护。``strength`` 是 Σ(atk+hp)，真值远大于 1，这里只防写坏的状态变成 0/负数。
MIN_STRENGTH = 1e-6

#: ``notes`` 上限：控制器每回合都会被调用，说明不能无限涨（否则日志与内存都会爆）。
MAX_NOTES = 256


def band_of(turn: int) -> str:
    """回合 -> band 名；越界夹到最近 band（不抛异常，采集口径变了也不炸）。"""
    t = _count(turn)
    if t is None:  # 非数值（含负数、NaN、字符串）一律当成第 1 回合
        return BAND_NAMES[0]
    t = max(1, t)  # 负数/0 视作开局
    for name, lo, hi in BANDS:
        if lo <= t <= hi:
            return name
    return BAND_NAMES[-1]  # 只会落到 > 999 的情况，夹到最后一个 band


# ── 数值工具（全部容忍脏数据：失败返回 None 而不是抛异常）────────────────────
def _finite(x: Any) -> Optional[float]:
    """尽力转成有限 float；bool 与数字字符串可用，NaN/inf/非数字返回 None。"""
    if isinstance(x, bool):
        return float(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _count(x: Any) -> Optional[int]:
    """非负整数计数；整数值的 float 可接受，NaN/inf/负数/带小数/非数字 -> None。"""
    v = _finite(x)
    if v is None or v < 0.0:
        return None
    i = int(v)
    return i if float(i) == v else None


def _number(x: Any) -> Optional[float]:
    """**严格**数值：只接受 int/float（bool 除外）。

    读取持久化文件时用它：文件是我们自己写的，出现字符串/布尔只可能是被写坏或被人改过，
    这时该按「字段非法」处理（回退 init_strength），而不是替它做容错转换。
    采集侧 stats 走 :func:`_finite`（宽松，数字字符串也认），两处口径不同是有意的。
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return _finite(x)


def _clamp01(v: float) -> float:
    """夹到 [0, 1]（胜率/占比都该在这个范围；越界只可能是统计口径写错）。"""
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _add_note(notes: List[str], msg: str, seen: Optional[set] = None) -> None:
    """追加一条中文说明；``seen`` 用于去重（同类问题每轮都会出现），并限量防爆。"""
    if seen is not None:
        if msg in seen:
            return
        seen.add(msg)
    if len(notes) < MAX_NOTES:
        notes.append(msg)


def _step_of(err: float, eta: float, max_step: float) -> float:
    """一次迭代的**相对**步长：``eta*err`` 夹到 ±max_step。

    用相对（乘性）步长而不是绝对刻度：强度刻度是 Σ(atk+hp)，不同 band 的量级差很多，
    按比例移动才能让 eta / max_step 在所有 band 上有同样含义。
    """
    raw = eta * err
    if not math.isfinite(raw):
        return 0.0
    # 步长再大也不能把强度推到 <= 0（负强度没有意义，也会让后续 log/比值为空）。
    cap = min(max(0.0, max_step), 0.99)
    return max(-cap, min(cap, raw))


def _combat_stats(entry: Any) -> Optional[Tuple[float, int]]:
    """解析一个 band 的 ``(胜率, 战斗数)``；``combats``/``losses`` 缺失或非数值 -> None。"""
    if not isinstance(entry, dict):
        return None
    combats = _count(entry.get("combats"))
    losses = _count(entry.get("losses"))
    if combats is None or losses is None:
        return None
    win = 0.0 if combats == 0 else 1.0 - (losses / combats)
    return _clamp01(win), combats


def _reach_rate(reach: Any, turn: int, episodes: int) -> Optional[float]:
    """「活到第 ``turn`` 回合」的局占比；无样本 / 键缺失 / 非数值 -> None。"""
    if episodes <= 0 or not isinstance(reach, dict):
        return None
    n = _count(reach.get(str(turn)))
    if n is None:
        return None
    return _clamp01(n / episodes)


@dataclass
class ArenaSpec:
    """控制器超参。``enabled=False``（默认）时 ``target_for_turn`` 恒返回 ``None``。"""

    enabled: bool = False
    #: 中期/后期的目标战斗胜率；``early`` 这一项不参与控制（early 用 reach 事件）。
    target_win: Dict[str, float] = field(
        default_factory=lambda: {"early": 0.70, "mid": 0.60, "late": 0.50}
    )
    deadband: float = 0.05  # |实测-目标| <= deadband 不动（避免每轮抖动）
    eta: float = 0.6  # 每次迭代步长系数（相对/乘性刻度）
    max_step: float = 0.08  # 每次迭代最多 ±8% 强度（单轮不让难度跳崖）
    min_combats: int = 12  # 该 band 样本不足则不动（小样本胜率噪声太大）
    reach_turn: int = 9  # 「活到第 9 回合」是前期段的目标事件
    reach_target: float = 0.90  # 前期段目标：到达 reach_turn 的局占比
    reach_deadband: float = 0.03  # reach 事件的 deadband（比胜率略小：局数少但信号稳）
    reach_floor: float = 0.70  # reach_target 收紧的下限（不能把前期压到必死）
    ratchet_k: int = 3  # 连续 k 次达标 -> 收紧一次
    #: 每个 band 的强度上下限，调用方从战团库分位算好传入；缺失/非法时只做正数保护。
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ArenaState:
    """控制器的**持久**状态：这几项要随训练续跑，不能每段重置。"""

    strength: Dict[str, float]  # 当前每个 band 的「目标对手强度」（绝对刻度 Σ(atk+hp)）
    reach_target: float  # 当前前期段目标（会被棘轮收紧，所以与 spec 分开存）
    iters: int  # 已执行多少轮闭环更新
    streak: int  # 连续「全面达标」轮数（棘轮计数）
    ratchet_level: int  # 棘轮收紧次数（history 里画成台阶）
    last: Dict[str, Any] = field(default_factory=dict)  # 上一轮的 win/step/strength/reach_rate


#: ``ArenaSpec`` 的默认实例：超参被写坏时用它的值兜底（只读，不修改）。
_DEFAULTS = ArenaSpec()

#: 字段 -> 允许的 kv 解析类型；``target_win`` / ``bounds`` 是 dict，**不从 kv 解析**。
_DICT_FIELDS: Tuple[str, ...] = ("target_win", "bounds")
_BOOL_FIELDS: Tuple[str, ...] = ("enabled",)
_INT_FIELDS: Tuple[str, ...] = ("min_combats", "reach_turn", "ratchet_k")
_FLOAT_FIELDS: Tuple[str, ...] = (
    "deadband",
    "eta",
    "max_step",
    "reach_target",
    "reach_deadband",
    "reach_floor",
)
_SPEC_FIELDS: Tuple[str, ...] = tuple(f.name for f in fields(ArenaSpec))


class ArenaController:
    """把「每个 band 的目标对手强度」按实测胜率/到达率闭环推到位，并支持续训恢复。"""

    def __init__(
        self,
        spec: ArenaSpec,
        init_strength: Dict[str, float],
        path: Optional[str] = None,
        note: Optional[list] = None,
    ) -> None:
        # note 由调用方传入（通常是训练侧的 note 列表），这样说明能进同一条日志流。
        self.notes: List[str] = note if isinstance(note, list) else []
        self._seen: set = set()  # 去重键，避免每回合重复记同一条说明
        self.path: Optional[str] = str(path) if path else None
        if not isinstance(spec, ArenaSpec):
            _add_note(self.notes, f"spec 不是 ArenaSpec（{type(spec).__name__}），用默认超参",
                      self._seen)
            spec = ArenaSpec()
        self.spec: ArenaSpec = spec
        self.state = ArenaState(
            strength=self._sanitize_strength(init_strength, source="init_strength"),
            reach_target=self._sanitize_reach_target(self.spec.reach_target,
                                                    source="spec.reach_target"),
            iters=0,
            streak=0,
            ratchet_level=0,
            last={},
        )
        if self.path:
            self._load()  # 续训：能恢复就接着上次难度，恢复不了保留 init_strength

    # ── 对外 API ───────────────────────────────────────────────────────
    def target_for_turn(self, turn: int) -> Optional[float]:
        """该回合的目标对手强度（绝对刻度）；``enabled=False`` 时恒 ``None``。"""
        if not self.spec.enabled:
            return None
        try:
            band = band_of(turn)
            value = self._interp(_finite(turn) if _finite(turn) is not None else 1.0)
            return float(self._clamp_band(band, value))
        except Exception as exc:  # 兜底：采样侧每回合都调，绝不能让训练因为这里挂掉
            _add_note(self.notes, f"target_for_turn({turn!r}) 异常：{exc.__class__.__name__}: {exc}",
                      self._seen)
            return None

    def update(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """吃一轮统计，升降强度，返回可直接写进训练 history 的扁平摘要；绝不抛异常。"""
        try:
            summary = self._update_inner(stats)
        except Exception as exc:  # 不变量兜底：任何意外都只降级成本轮「不动」
            _add_note(self.notes, f"update 异常（{exc.__class__.__name__}: {exc}），本轮强度不变")
            self.state.iters += 1
            summary = self._history({}, {})
        self._save()  # 原子写；内部已兜底，不会抛
        return summary

    def summary(self) -> Dict[str, Any]:
        """当前控制状态快照（日志/调试用，返回副本）。"""
        try:
            return {
                "enabled": bool(self.spec.enabled),
                "strength": {n: float(self._strength_of(n)) for n in BAND_NAMES},
                "reach_target": float(self._reach_target()),
                "iters": int(self.state.iters),
                "streak": int(self.state.streak),
                "ratchet_level": int(self.state.ratchet_level),
                "last": dict(self.state.last),
                "notes": list(self.notes),
            }
        except Exception as exc:
            _add_note(self.notes, f"summary 异常（{exc.__class__.__name__}: {exc}）")
            return {"enabled": bool(getattr(self.spec, "enabled", False)), "strength": {},
                    "reach_target": None, "iters": 0, "streak": 0, "ratchet_level": 0,
                    "last": {}, "notes": list(self.notes)}

    # ── 强度插值与夹紧 ─────────────────────────────────────────────────
    def _interp(self, turn: float) -> float:
        """把各 band 的代表强度放在 band 中心，按回合线性插值；两端外推为常数。"""
        centers = [(BAND_CENTER[n], n) for n in BAND_NAMES]
        if turn <= centers[0][0]:
            return self._strength_of(centers[0][1])
        if turn >= centers[-1][0]:
            return self._strength_of(centers[-1][1])
        for (c0, n0), (c1, n1) in zip(centers, centers[1:]):
            if c0 <= turn <= c1:
                w = (turn - c0) / (c1 - c0)
                return self._strength_of(n0) * (1.0 - w) + self._strength_of(n1) * w
        return self._strength_of(centers[-1][1])

    def _clamp_band(self, band: str, value: float) -> float:
        """夹到 ``spec.bounds[band]``；bounds 缺失/非法时只做有限正数保护。"""
        v = _finite(value)
        if v is None or v <= 0.0:
            v = self._fallback_strength()
        raw = self.spec.bounds.get(band) if isinstance(self.spec.bounds, dict) else None
        lo = hi = None
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            lo, hi = _finite(raw[0]), _finite(raw[1])
            if lo is None or hi is None or hi < lo:
                _add_note(self.notes, f"spec.bounds[{band!r}]={raw!r} 非法，忽略上下限",
                          self._seen)
                lo = hi = None
        elif raw is not None:
            _add_note(self.notes, f"spec.bounds[{band!r}]={raw!r} 非法，忽略上下限", self._seen)
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return max(MIN_STRENGTH, v)

    def _strength_of(self, band: str) -> float:
        """取某 band 的当前强度；被写坏时用其它 band 兜底并回写（保持有限正数）。"""
        raw = self.state.strength.get(band) if isinstance(self.state.strength, dict) else None
        v = _finite(raw)
        if v is None or v <= 0.0:
            v = self._fallback_strength()
            if isinstance(self.state.strength, dict):
                self.state.strength[band] = v
            _add_note(self.notes, f"strength[{band!r}] 非法，用 {v:.6g} 兜底", self._seen)
        return float(v)

    def _fallback_strength(self) -> float:
        """兜底强度：优先取任何一个合法 band（保持难度量级），否则 1.0。"""
        if isinstance(self.state.strength, dict):
            for n in BAND_NAMES:
                v = _finite(self.state.strength.get(n))
                if v is not None and v > 0.0:
                    return v
        return 1.0

    # ── 超参读取（脏值 -> 默认值 + note，绝不抛）────────────────────────
    def _spec_float(self, name: str) -> float:
        v = _finite(getattr(self.spec, name, None))
        if v is None:
            v = _finite(getattr(_DEFAULTS, name, None))
            _add_note(self.notes, f"spec.{name} 不是有限数值，用默认值 {v}", self._seen)
        return float(v)

    def _spec_int(self, name: str) -> int:
        v = _count(getattr(self.spec, name, None))
        if v is None:
            v = _count(getattr(_DEFAULTS, name, None)) or 0
            _add_note(self.notes, f"spec.{name} 不是非负整数，用默认值 {v}", self._seen)
        return int(v)

    def _target_win(self, band: str) -> float:
        raw = self.spec.target_win if isinstance(self.spec.target_win, dict) else {}
        v = _finite(raw.get(band))
        if v is None:
            v = _finite(_DEFAULTS.target_win.get(band))
            _add_note(self.notes, f"spec.target_win[{band!r}] 缺失或非数值，用默认 {v}", self._seen)
        return float(v)

    def _reach_target(self) -> float:
        v = _finite(self.state.reach_target)
        if v is None:
            v = _finite(_DEFAULTS.reach_target)
            _add_note(self.notes, "state.reach_target 非有限值，用默认值", self._seen)
        return _clamp01(float(v))

    def _sanitize_reach_target(self, raw: Any, *, source: str) -> float:
        v = _finite(raw)
        if v is None or not (0.0 < v <= 1.0):
            fallback = float(_DEFAULTS.reach_target)
            _add_note(self.notes, f"{source}={raw!r} 非法，用默认 {fallback}")
            return fallback
        return float(v)

    def _sanitize_strength(self, raw: Any, *, source: str) -> Dict[str, float]:
        """补齐/校正初始强度：三个 band 都要有限正数，缺的按最近的已给 band 兜底。"""
        values = raw if isinstance(raw, dict) else {}
        if not isinstance(raw, dict):
            _add_note(self.notes, f"{source} 不是 dict（{type(raw).__name__}）")
        given: Dict[str, float] = {}
        missing: List[str] = []
        for name in BAND_NAMES:
            v = _finite(values.get(name))
            if v is not None and v > 0.0:
                given[name] = float(v)
            else:
                missing.append(name)
        out: Dict[str, float] = {}
        for name in BAND_NAMES:
            if name in given:
                out[name] = given[name]
                continue
            nearest = _nearest_band(name, given)
            out[name] = given[nearest] if nearest is not None else 1.0
        if missing:
            level = "全部缺失，用 1.0 起算" if not given else "按最近 band 暂代"
            _add_note(self.notes, f"{source} 缺少合法强度 {missing}，{level}", self._seen)
        return out

    # ── 一轮闭环 ──────────────────────────────────────────────────────
    def _update_inner(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        st: Dict[str, Any] = stats if isinstance(stats, dict) else {}
        if not isinstance(stats, dict):
            _add_note(self.notes, f"stats 不是 dict（{type(stats).__name__}），按空统计处理",
                      self._seen)

        deadband = abs(self._spec_float("deadband"))
        eta = self._spec_float("eta")
        max_step = abs(self._spec_float("max_step"))
        min_combats = max(1, self._spec_int("min_combats"))
        reach_turn = self._spec_int("reach_turn")

        episodes = _count(st.get("episodes")) or 0
        reach = st.get("reach") if isinstance(st.get("reach"), dict) else None
        if st.get("reach") is not None and reach is None:
            _add_note(self.notes, "stats['reach'] 不是 dict，reach 相关判定跳过", self._seen)
        bands_raw = st.get("bands") if isinstance(st.get("bands"), dict) else {}
        if st.get("bands") is not None and not bands_raw:
            _add_note(self.notes, "stats['bands'] 不是 dict，全部 band 跳过", self._seen)

        reach_by_turn: Dict[int, Optional[float]] = {
            t: _reach_rate(reach, t, episodes) for t in REACH_TURNS
        }

        win: Dict[str, float] = {}
        step: Dict[str, float] = {}

        # 1) 战斗胜率 band（mid / late）
        for name in ("mid", "late"):
            win[name] = -1.0  # 摘要约定：样本不足填 -1.0
            step[name] = 0.0
            parsed = _combat_stats(bands_raw.get(name))
            if parsed is None:
                _add_note(self.notes, f"bands[{name!r}] 缺失或含非数值，跳过该 band", self._seen)
                continue
            w, combats = parsed
            if combats < min_combats:
                _add_note(self.notes,
                          f"bands[{name!r}].combats={combats} < min_combats={min_combats}，"
                          "样本不足跳过", self._seen)
                continue
            win[name] = w
            err = w - self._target_win(name)
            if abs(err) <= deadband:
                continue  # deadband 内不动：小样本胜率抖动不该改难度
            step[name] = _step_of(err, eta, max_step)

        # 2) 前期 band：用 reach 事件，不用胜率
        reach_rate = reach_by_turn.get(reach_turn)
        early = _combat_stats(bands_raw.get("early"))
        win["early"] = early[0] if early is not None and early[1] >= min_combats else -1.0
        step["early"] = 0.0
        rt = self._reach_target()
        if episodes <= 0:
            _add_note(self.notes, "episodes<=0，early band 跳过（无样本）", self._seen)
        elif reach_by_turn.get(reach_turn) is None:
            _add_note(self.notes,
                      f"reach[{str(reach_turn)!r}] 缺失或非数值，early band 跳过", self._seen)
        else:
            err = reach_rate - rt
            if abs(err) > abs(self._spec_float("reach_deadband")):
                step["early"] = _step_of(err, eta, max_step)

        # 3) 应用步长：乘性移动 + 夹紧（strength 永远是有限正数）
        for name in BAND_NAMES:
            s = step.get(name, 0.0)
            if not s:
                continue
            base = self._strength_of(name)
            self.state.strength[name] = self._clamp_band(name, base * (1.0 + s))

        # 4) 棘轮：学习者连续全面达标就把前期目标收紧（前期不能一直停在很松的档位）
        mid_ok = win.get("mid", -1.0) >= self._target_win("mid") + deadband
        late_ok = win.get("late", -1.0) >= self._target_win("late") + deadband
        reach_ok = reach_rate is not None and reach_rate >= rt
        self.state.streak = self.state.streak + 1 if (mid_ok and late_ok and reach_ok) else 0
        k = max(1, self._spec_int("ratchet_k"))
        floor = self._spec_float("reach_floor")
        if self.state.streak >= k and rt > floor:
            rt = max(floor, rt - 0.05)
            self.state.reach_target = rt
            self.state.ratchet_level += 1
            self.state.streak = 0
            _add_note(self.notes,
                      f"棘轮：连续 {k} 轮达标，reach_target 收紧到 {rt:.3f}（floor {floor:.3f}）",
                      self._seen)

        self.state.iters += 1
        self.state.last = {
            "iters": int(self.state.iters),
            "win": {n: float(win.get(n, -1.0)) for n in BAND_NAMES},
            "step": {n: float(step.get(n, 0.0)) for n in BAND_NAMES},
            "strength": {n: float(self._strength_of(n)) for n in BAND_NAMES},
            "reach_rate": None if reach_rate is None else float(reach_rate),
            "reach_target": float(rt),
            "streak": int(self.state.streak),
        }
        return self._history(win, reach_by_turn)

    def _history(self, win: Dict[str, float],
                 reach_by_turn: Dict[int, Optional[float]]) -> Dict[str, Any]:
        """扁平摘要：键名就是训练 history 的列名，**不能改名**（下游直接写曲线）。"""

        def _reach(t: int) -> float:
            v = reach_by_turn.get(t)
            return -1.0 if v is None else float(v)

        return {
            "arena_early_str": float(self._strength_of("early")),
            "arena_mid_str": float(self._strength_of("mid")),
            "arena_late_str": float(self._strength_of("late")),
            "arena_early_win": float(win.get("early", -1.0)),
            "arena_mid_win": float(win.get("mid", -1.0)),
            "arena_late_win": float(win.get("late", -1.0)),
            "arena_reach9": _reach(REACH_TURNS[0]),
            "arena_reach12": _reach(REACH_TURNS[1]),
            "arena_it": int(self.state.iters),
            "arena_reach_target": float(self._reach_target()),
            "arena_ratchet": int(self.state.ratchet_level),
        }

    # ── 持久化 ────────────────────────────────────────────────────────
    def _load(self) -> None:
        """续训恢复：合法 ``strength`` 就恢复（难度接着上次）；损坏/非法则保留 init。"""
        path = self.path
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return  # 首次训练：正常情况，不记 note
        except Exception as exc:
            _add_note(self.notes,
                      f"读取 {path} 失败（{exc.__class__.__name__}: {exc}），保留 init_strength")
            return
        if not isinstance(raw, dict):
            _add_note(self.notes, f"{path} 内容不是 dict，保留 init_strength")
            return
        strength = _valid_strength(raw.get("strength"))
        if strength is None:
            _add_note(self.notes, f"{path} 的 strength 非法（需每个 band 有限正数），保留 init_strength")
            return
        self.state.strength = strength
        rt = _number(raw.get("reach_target"))
        if rt is None or not (0.0 < rt <= 1.0):
            _add_note(self.notes, f"{path} 的 reach_target={raw.get('reach_target')!r} 非法，用 spec 默认值")
        else:
            self.state.reach_target = float(rt)
        for name in ("iters", "streak", "ratchet_level"):
            v = _count(_number(raw.get(name)))
            if v is None:
                _add_note(self.notes, f"{path} 的 {name}={raw.get(name)!r} 非法，重置为 0")
                continue
            setattr(self.state, name, int(v))

    def _save(self) -> None:
        """原子写状态：先写 ``path + '.tmp'`` 再 ``os.replace``，崩在中途也不会留半截文件。"""
        if not self.path:
            return
        payload = {
            "version": 1,
            "strength": {n: float(self._strength_of(n)) for n in BAND_NAMES},
            "reach_target": float(self._reach_target()),
            "iters": int(self.state.iters),
            "streak": int(self.state.streak),
            "ratchet_level": int(self.state.ratchet_level),
        }
        tmp = f"{self.path}.tmp"
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception as exc:  # 落盘失败不该让训练停：难度只活在内存里也还能跑
            _add_note(self.notes,
                      f"写入 {self.path} 失败（{exc.__class__.__name__}: {exc}），本轮不落盘")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


def _nearest_band(name: str, given: Dict[str, float]) -> Optional[str]:
    """``given`` 里离 ``name`` 代表回合最近的那个 band（同难度斜面，量级最接近）。"""
    best: Optional[str] = None
    best_d = float("inf")
    for other in given:
        d = abs(BAND_CENTER[other] - BAND_CENTER[name])
        if d < best_d:
            best, best_d = other, d
    return best


def _valid_strength(raw: Any) -> Optional[Dict[str, float]]:
    """文件里的 ``strength`` 必须三个 band 齐全且都是有限正数，否则视为损坏。"""
    if not isinstance(raw, dict):
        return None
    out: Dict[str, float] = {}
    for name in BAND_NAMES:
        v = _number(raw.get(name))  # 严格：字符串/布尔算写坏，不做容错转换
        if v is None or v <= 0.0:
            return None
        out[name] = float(v)
    return out


# ── kv 解析 ───────────────────────────────────────────────────────────
def _parse_bool(text: str) -> Optional[bool]:
    """``1/0/true/false/yes/no``（大小写不敏感）-> bool；其它 -> None。"""
    t = text.strip().lower()
    if t in ("1", "true", "yes"):
        return True
    if t in ("0", "false", "no"):
        return False
    return None


def _parse_int(text: str) -> Optional[int]:
    """整数（允许 ``12`` / ``12.0``）-> int；带小数/非数字/负数 -> None。"""
    return _count(text.strip())


def _parse_float(text: str) -> Optional[float]:
    """有限 float；NaN/inf/空串 -> None。"""
    return _finite(text.strip())


def parse_spec(kv: str, note: Optional[list] = None) -> ArenaSpec:
    """解析 ``"enabled=1,eta=0.8,reach_target=0.88"`` 形式的覆盖项。

    - 逗号分隔 ``k=v``；只接受 ``ArenaSpec`` 里存在的字段名，未知键忽略并记 note；
    - ``target_win`` / ``bounds`` 是 dict 字段，**不从 kv 解析**（出现时记 note 忽略）；
    - 解析失败不抛异常：记 note 并保留默认值。
    """
    notes: List[str] = note if isinstance(note, list) else []
    seen: set = set()
    spec = ArenaSpec()
    text = "" if kv is None else str(kv)
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            _add_note(notes, f"parse_spec 忽略无 '=' 的片段 {item!r}", seen)
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        val = val.strip()
        if key in _DICT_FIELDS:
            _add_note(notes, f"parse_spec 不从 kv 解析 dict 字段 {key!r}，已忽略", seen)
            continue
        if key not in _SPEC_FIELDS:
            _add_note(notes, f"parse_spec 未知字段 {key!r}，已忽略", seen)
            continue
        default = getattr(spec, key)
        if key in _BOOL_FIELDS:
            parsed = _parse_bool(val)
            if parsed is None:
                _add_note(notes, f"parse_spec {key}={val!r} 不是 bool，用默认 {default!r}", seen)
                continue
            setattr(spec, key, parsed)
        elif key in _INT_FIELDS:
            parsed_int = _parse_int(val)
            if parsed_int is None:
                _add_note(notes, f"parse_spec {key}={val!r} 不是非负整数，用默认 {default!r}", seen)
                continue
            setattr(spec, key, parsed_int)
        elif key in _FLOAT_FIELDS:
            parsed_float = _parse_float(val)
            if parsed_float is None:
                _add_note(notes, f"parse_spec {key}={val!r} 不是有限数，用默认 {default!r}", seen)
                continue
            setattr(spec, key, parsed_float)
        else:  # 理论上不会到（_SPEC_FIELDS 已拦），留作显式兜底
            _add_note(notes, f"parse_spec 字段 {key!r} 暂不支持从 kv 解析，已忽略", seen)
    return spec
