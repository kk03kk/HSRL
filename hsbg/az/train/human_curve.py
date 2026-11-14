"""真人战团强度曲线：分位 <-> 绝对强度 的双向换算。

用途
----
动态难度课程的控制器工作在**绝对强度**空间（Σ(atk+hp) 是可解释、可跨运行比较的
刻度：样本 2000 就是 2000），而采集侧要的是**分位**（``human_board_sample`` /
``strength_quantile``：在第几名的真人战团）。这两个刻度由本模块互换：

- 曲线来自 :mod:`tools.pool_strength_curve` 产出的 ``human_curve.json``
  （每回合 p10/p25/p33/p50/p67/p75/p90 的真实战团强度）；
- ``strength(turn, q)``：分位 -> 强度（单调分段线性插值，超出端点夹住）；
- ``quantile(turn, strength)``：强度 -> 分位（反查，控制器用它把强度目标交给采集侧）；
- ``band_bounds(turn)``：控制器每个分段的强度上下限（取该回合 p10..p90）。

为什么不用"倍率"：把真人战团的身材乘 1.1 会造出牌库里根本不存在的分布（一个 4 星随从
变成 2 倍身材的怪物），模型会学成"打数据怪"。按强度在真人库里就近选板子，形状与
组合都是真实的，只是"遇到的对手强弱"被匹配到了能力边界。见 D-054。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 曲线里必须有的分位键（与 tools/pool_strength_curve.py 一致）。
CURVE_QUANTILES: Tuple[Tuple[str, float], ...] = (
    ("p10", 0.10), ("p25", 0.25), ("p33", 0.33), ("p50", 0.50),
    ("p67", 0.67), ("p75", 0.75), ("p90", 0.90),
)

#: 分段（与 opponent_arena.BANDS 一致）：(名字, 起始回合, 结束回合, 代表回合)。
#: 代表回合 = 该段的强度锚点（前段取中前期、后段取 13，因为"后期"真正开始于 T12+）。
BAND_CENTERS: Tuple[Tuple[str, int, int, float], ...] = (
    ("early", 1, 7, 5.0), ("mid", 8, 11, 9.5), ("late", 12, 99, 13.0),
)


def band_center_turn(band: str) -> float:
    """分段 -> 代表回合（曲线查询用的锚点）。未知分段回退到 13。"""
    for name, _lo, _hi, center in BAND_CENTERS:
        if str(name) == str(band):
            return float(center)
    return 13.0


def band_of_turn(turn: int) -> str:
    """回合 -> 分段名（1..7 early / 8..11 mid / 12+ late）。"""
    t = int(turn)
    if t <= 7:
        return "early"
    if t <= 11:
        return "mid"
    return "late"


class HumanCurve:
    """``tools/pool_strength_curve.py`` 输出文件的只读视图（纯 stdlib）。"""

    def __init__(self, payload: Optional[Dict[str, Any]] = None):
        curve = (payload or {}).get("curve") if isinstance(payload, dict) else None
        self._raw: Dict[int, Dict[str, int]] = {}
        if isinstance(curve, dict):
            for k, row in curve.items():
                try:
                    t = int(k)
                except (TypeError, ValueError):
                    continue
                if not isinstance(row, dict):
                    continue
                pts: Dict[str, int] = {}
                for name, _q in CURVE_QUANTILES:
                    v = row.get(name)
                    try:
                        if v is not None:
                            pts[name] = int(v)
                    except (TypeError, ValueError):
                        continue
                if pts:
                    self._raw[t] = pts
        self.path = str((payload or {}).get("pool", "")) if isinstance(payload, dict) else ""
        self.ok = bool(self._raw)

    # ---- 构造 ----
    @classmethod
    def load(cls, path: str) -> "HumanCurve":
        """从 JSON 文件读；文件不存在/损坏时返回 ``ok=False`` 的空曲线（不抛）。"""
        if not path or not os.path.isfile(str(path)):
            return cls({})
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            return cls({})
        out = cls(payload if isinstance(payload, dict) else {})
        out.path = str(path)
        return out

    # ---- 查询 ----
    def turns(self) -> List[int]:
        return sorted(self._raw)

    def _nearest_turn(self, turn: float) -> Optional[int]:
        if not self._raw:
            return None
        t = float(turn)
        return min(self._raw, key=lambda x: (abs(float(x) - t), x))

    def _points(self, turn: float) -> List[Tuple[float, float]]:
        """``[(q, strength)]``：该回合（缺则最近回合）的单调化分位点。"""
        hit = self._nearest_turn(turn)
        if hit is None:
            return []
        row = self._raw[int(hit)]
        pts: List[Tuple[float, float]] = []
        last = -1.0
        for name, q in CURVE_QUANTILES:
            v = row.get(name)
            if v is None:
                continue
            s = max(float(v), last)     # 单调化：脏数据造成的非单调不该产生负斜率
            pts.append((float(q), s))
            last = s
        return pts

    def strength(self, turn: float, q: float) -> Optional[float]:
        """分位 -> 强度（分段线性；超出端点夹到端点）。"""
        pts = self._points(turn)
        if not pts:
            return None
        qq = float(q)
        if qq <= pts[0][0]:
            return float(pts[0][1])
        if qq >= pts[-1][0]:
            return float(pts[-1][1])
        for (q0, s0), (q1, s1) in zip(pts, pts[1:]):
            if q0 <= qq <= q1:
                if q1 <= q0:
                    return float(s1)
                w = (qq - q0) / (q1 - q0)
                return float(s0 + (s1 - s0) * w)
        return float(pts[-1][1])

    def quantile(self, turn: float, strength: float) -> Optional[float]:
        """强度 -> 分位（反查；分段线性，超出端点夹到 [0.10, 0.90] 的端点）。"""
        pts = self._points(turn)
        if not pts:
            return None
        s = float(strength)
        if s <= pts[0][1]:
            return float(pts[0][0])
        if s >= pts[-1][1]:
            return float(pts[-1][0])
        for (q0, s0), (q1, s1) in zip(pts, pts[1:]):
            if s0 <= s <= s1:
                if s1 <= s0:
                    return float(q1)
                w = (s - s0) / (s1 - s0)
                return float(q0 + (q1 - q0) * w)
        return float(pts[-1][0])

    def band_bounds(self, band: str) -> Optional[Tuple[float, float]]:
        """分段强度上下限 ``(p10, p90)``（控制器在该区间内移动）。"""
        t = band_center_turn(band)
        lo = self.strength(t, 0.10)
        hi = self.strength(t, 0.90)
        if lo is None or hi is None:
            return None
        return (float(lo), float(max(hi, lo + 1.0)))

    def p50(self, turn: float) -> Optional[float]:
        """该回合真人中位战团强度（"我们对真人曲线比值"的分母）。"""
        return self.strength(turn, 0.50)

    def describe(self) -> Dict[str, Any]:
        """摘要（写进 run 的 meta / 文档核对用）。"""
        return {"ok": bool(self.ok), "path": str(self.path),
                "turns": self.turns(),
                "p50_by_turn": {str(t): int(round(v)) for t in self.turns()
                                for v in [self.p50(t) or 0.0] if v},
                "band_bounds": {name: self.band_bounds(name)
                                for name, _lo, _hi, _c in BAND_CENTERS}}


def load_curve(path: str) -> HumanCurve:
    """便捷入口（与 :meth:`HumanCurve.load` 同）。"""
    return HumanCurve.load(path)


__all__ = ["CURVE_QUANTILES", "BAND_CENTERS", "HumanCurve", "load_curve",
           "band_center_turn", "band_of_turn"]
