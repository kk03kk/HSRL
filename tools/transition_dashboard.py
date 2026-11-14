#!/usr/bin/env python3
"""过渡期诊断面板：一次训练 run 的"中后期有没有真的变强"，打表看。

为什么需要
----------
``artifacts/runs/<side>-<purpose>-<YYYYMMDD>-<seq>/history.jsonl`` 每行一次训练
迭代，长训几千行，靠肉眼看不出中后期是否在变强。本面板离线读这个文件，按迭代
打两张表（对局结构 / 难度与健康），再在 ``[判读]`` 里把"末行 vs 区间首行"的变化
与难度棘轮直接讲出来，省得每次手工翻 jsonl。

只用标准库、不 import 仓库内模块：诊断工具要能在任何一台机器上对任意 run 目录跑，
也必须能跑在"仓库代码没装好/正在排障"的场景里。

脏数据口径
----------
history 由多代训练栈写出，行内结构不稳定：同一个键可能在本层，也可能躲在
``stat`` / ``stats`` / ``meta`` 子字典里；``-1`` / ``null`` / 字符串表示"这一项
没样本"；还可能混进半行 JSON。面板一律当缺失处理（显示 ``-``）、坏行跳过，
绝不中断整表——诊断工具比数据更该耐脏。

用法::

    python tools/transition_dashboard.py artifacts/runs/self-play-20261114-001
    python tools/transition_dashboard.py <run>/history.jsonl --last 80 --every 4
    python tools/transition_dashboard.py <run> --human-curve curve.json --json

表格宽 114 / 100 列（中文按 2 列算）：终端窄就重定向到文件再看。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

# ── 取值：同一个指标可能落在本层或子字典里 ──────────────────────────────────
# 查找顺序就是优先级：本层是"最新写盘格式"，子字典是历史格式兜底。
# ``collect`` 必须在列：`train_entity_sync.py` 把采集 meta（tel_*/arena_*/注入计数）
# 整块写在 ``row["collect"]`` 里，不看这一层的话面板会显示成"全部无数据"。
_NESTED_KEYS: Tuple[str, ...] = ("stat", "stats", "meta", "collect", "ppo", "profile")

# 每个字段的候选键（有的 run 写 it、有的写 iteration/iter；rank 也可能叫 rank_mean）。
_FIELDS: Dict[str, Tuple[str, ...]] = {
    "it": ("it", "iteration", "iter"),
    "elapsed": ("elapsed",),
    "rank": ("rank", "rank_mean"),
    "tel_tier_6": ("tel_tier_6",),
    "tel_tier_8": ("tel_tier_8",),
    "tel_tier_10": ("tel_tier_10",),
    "tel_own_str_8": ("tel_own_str_8",),
    "tel_own_str_10": ("tel_own_str_10",),
    "tel_own_str_12": ("tel_own_str_12",),
    "tel_dmg_9": ("tel_dmg_9",),
    "tel_dmg_12": ("tel_dmg_12",),
    "tel_reach9": ("tel_reach9",),
    "tel_reach12": ("tel_reach12",),
    "rg_gain_10": ("rg_gain_10",),
    "rg_gain_11": ("rg_gain_11",),
    "comp_s_hits": ("comp_s_hits", "std_s_hit"),
    "arena_early_str": ("arena_early_str",),
    "arena_mid_str": ("arena_mid_str",),
    "arena_late_str": ("arena_late_str",),
    "arena_early_win": ("arena_early_win",),
    "arena_mid_win": ("arena_mid_win",),
    "arena_late_win": ("arena_late_win",),
    "arena_reach9": ("arena_reach9",),
    "arena_reach12": ("arena_reach12",),
    "arena_reach_target": ("arena_reach_target",),
    "approx_kl": ("approx_kl",),
    "entropy": ("entropy",),
    "anchor_kl": ("anchor_kl",),
    "loss": ("loss",),
    "adv_std": ("adv_std",),
}

#: 拿人曲线做比值的那几个回合（板面强度只在 T8/T10/T12 有）。
_CURVE_TURNS: Tuple[int, ...] = (8, 10, 12)


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """依次在 ``row``、``row["stat"]``、``row["stats"]``、``row["meta"]`` 里找键。

    本层优先；本层是 ``None``（占位/写盘格式换代的残渣）时继续往子字典里找，
    以免一个 null 把子字典里的真实数值挡住。
    """
    if not isinstance(row, Mapping):
        return default
    if key in row and row[key] is not None:
        return row[key]
    for name in _NESTED_KEYS:
        nested = row.get(name)
        if isinstance(nested, Mapping) and nested.get(key) is not None:
            return nested[key]
    return default


def _num(value: Any) -> Optional[float]:
    """把脏值收敛成有限非负 float，其余一律 ``None``（= 无样本，显示 ``-``）。

    history 用 ``-1`` 表示"这一项没有样本"，字符串（``""``/``"null"``/``"-1"``）、
    NaN / ±inf 也当缺失：这些值一旦进入面板，就会被读成"模型很弱"或"KL 爆炸"。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _first_num(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    """按候选键顺序取第一个有样本的值（``it`` / ``iteration`` / ``iter`` 并存时用）。"""
    for key in keys:
        value = _num(_get(row, key))
        if value is not None:
            return value
    return None


# ── 读盘：坏行只计数，不抛 ──────────────────────────────────────────────────
def resolve_history(target: Path) -> Optional[Path]:
    """接受 run 目录或 history.jsonl 本身；目录里没 history.jsonl 就返回 ``None``。"""
    path = Path(target)
    if path.is_dir():
        candidate = path / "history.jsonl"
        return candidate if candidate.is_file() else None
    return path if path.is_file() else None


def read_history(path: Path) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, int]]:
    """逐行读 jsonl，返回 ``[(物理行号, 行字典)]`` 与计数。

    ``errors="replace"`` + 逐行 try：半截 JSON、非对象行（``[1,2]``、裸数字）、
    编码坏掉的行都只会让计数 ``bad`` 加一，面板继续往下读。
    """
    rows: List[Tuple[int, Dict[str, Any]]] = []
    stats = {"total": 0, "valid": 0, "bad": 0}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            stats["total"] += 1
            try:
                row = json.loads(text)
            except ValueError:
                stats["bad"] += 1
                continue
            if not isinstance(row, dict):
                stats["bad"] += 1
                continue
            stats["valid"] += 1
            rows.append((line_no, row))
    return rows, stats


def sample_rows(window: Sequence[Tuple[int, Dict[str, Any]]],
                every: int) -> List[Tuple[int, Dict[str, Any]]]:
    """每 ``every`` 条抽一条，并补上窗口末行。

    末行才是"现在的模型"，抽样不该把它吃掉；否则 ``[判读]`` 会停在一条中途的
    旧数据上，看起来像训练已经停摆。
    """
    picked = list(window)
    if every <= 1 or not picked:
        return picked
    thinned = picked[::every]
    if thinned[-1] is not picked[-1]:
        thinned = thinned + [picked[-1]]
    return thinned


# ── 记录：一行 history → 面板要的一条 ───────────────────────────────────────
def _ratio(own: Optional[float], point: Optional[Mapping[str, float]]) -> Optional[float]:
    """自建板面 ÷ 真人 p50；缺任一侧或 p50 <= 0 就是 ``None``（比值列显示 ``-``）。"""
    base = None if point is None else point.get("p50")
    if own is None or base is None or base <= 0:
        return None
    return own / base


#: 反查分位用的分位点（键 -> 分位值），与 tools/pool_strength_curve.py 输出一致。
_PCT_POINTS: Tuple[Tuple[str, float], ...] = (
    ("p10", 0.10), ("p25", 0.25), ("p33", 0.33), ("p50", 0.50),
    ("p67", 0.67), ("p75", 0.75), ("p90", 0.90))


def _percentile(own: Optional[float], point: Optional[Mapping[str, float]]) -> Optional[float]:
    """自建板面在**真人分布**里的分位（0..1）；点不够就 ``None``。

    这是比"÷p50"更硬的刻度：0.14 就是"只赢过 14% 的真人战团"，它不需要排名次、
    也不受训练场难度影响，所以可以跨运行直接比较（D-054）。
    """
    if own is None or not point:
        return None
    pts = [(q, point[name]) for name, q in _PCT_POINTS
           if isinstance(point.get(name), (int, float))]
    if len(pts) < 2:
        return None
    s = float(own)
    if s <= pts[0][1]:
        return float(pts[0][0])
    if s >= pts[-1][1]:
        return float(pts[-1][0])
    for (q0, s0), (q1, s1) in zip(pts, pts[1:]):
        if s0 <= s <= s1:
            if s1 <= s0:
                return float(q1)
            return float(q0 + (q1 - q0) * (s - s0) / (s1 - s0))
    return None


def build_record(row: Mapping[str, Any], line_no: int,
                 curve: Mapping[int, Mapping[str, float]]) -> Dict[str, Any]:
    """把一行 history 压成一条记录：缺失即 ``None``，比值/分位按人曲线现算。"""
    record: Dict[str, Any] = {"line": line_no}
    for field, keys in _FIELDS.items():
        record[field] = _first_num(row, keys)
    for turn in _CURVE_TURNS:
        own = record.get(f"tel_own_str_{turn}")
        point = curve.get(turn)
        record[f"ratio_{turn}"] = _ratio(own, point)
        record[f"pct_{turn}"] = _percentile(own, point)
    return record


# ── 真人曲线：读不到就降级，不让面板失败 ────────────────────────────────────
def load_human_curve(path: Path) -> Tuple[Dict[int, Dict[str, float]], Optional[str]]:
    """读 ``{"8": {"p50": 179, "p75": 266}, ...}`` → ``({回合: {分位: 值}}, 失败原因)``。

    曲线只是辅助对比，读不到不该让面板失败：异常在这里被收敛成一句原因，
    由调用方打一行警告、把比值列降级成 ``-``。
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {}, f"读不了 {path}（{exc}）"
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return {}, f"{path} 不是合法 JSON（{exc}）"
    if not isinstance(data, dict):
        return {}, f"{path} 顶层不是对象"
    # 允许直接喂 tools/pool_strength_curve.py 的原始产物（回合表在 "curve" 下面）。
    if isinstance(data.get("curve"), dict):
        data = data["curve"]
    curve: Dict[int, Dict[str, float]] = {}
    for key, point in data.items():
        try:
            turn = int(str(key).strip())
        except (TypeError, ValueError):
            continue
        if not isinstance(point, Mapping):
            continue
        clean: Dict[str, float] = {}
        for name in ("p10", "p25", "p33", "p50", "p67", "p75", "p90", "mean"):
            value = _num(point.get(name))
            if value is not None:
                clean[name] = value
        if clean:
            curve[turn] = clean
    if not curve:
        return {}, f"{path} 里没有可用的 p50/分位数据"
    return curve, None


# ── 棘轮：arena 强度连续上升说明"变强"可能只是难度在涨 ─────────────────────
def _trend(values: Sequence[Optional[float]], need: int = 5,
           min_gain: float = 0.02) -> Dict[str, Any]:
    """末 ``need`` 个样本单调不减且首末涨幅 > ``min_gain`` → ``up``。

    只看末行差值会把"难度跟着模型一起涨"误读成棋力上涨，所以要求连续单调；
    ``need`` 个样本凑不齐就是 ``insufficient``，不拿 2 个点硬猜趋势。
    """
    kept = [float(v) for v in values if v is not None][-need:]
    out: Dict[str, Any] = {"state": "insufficient", "samples": len(kept),
                           "first": None, "last": None, "gain": None}
    if len(kept) < need:
        return out
    out["first"], out["last"] = kept[0], kept[-1]
    base = abs(kept[0])
    gain = (kept[-1] - kept[0]) / base if base > 0 else 0.0
    out["gain"] = round(gain, 6)
    monotone = all(b >= a for a, b in zip(kept, kept[1:]))
    out["state"] = "up" if monotone and gain > min_gain else "flat"
    return out


def ratchet_state(window_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """合并中/后期两条 arena 强度曲线，给 ``up`` / ``flat`` / ``insufficient``。

    两条任一条样本不足就判"数据不足"：只凭中期上升就说难度棘轮，证据不够。
    """
    mid = _trend([r.get("arena_mid_str") for r in window_records])
    late = _trend([r.get("arena_late_str") for r in window_records])
    if mid["state"] == "insufficient" or late["state"] == "insufficient":
        state = "insufficient"
    elif mid["state"] == "up" and late["state"] == "up":
        state = "up"
    else:
        state = "flat"
    return {"state": state, "mid": mid, "late": late}


# ── 等宽排版：中文按 2 列算，否则中英混排必错位 ─────────────────────────────
def _text_width(text: str) -> int:
    """终端显示宽度：东亚宽字符（W/F）占 2 列，其余占 1 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str = "right") -> str:
    """按显示宽度把 ``text`` 补到 ``width`` 列；数字右对齐、表头居中。"""
    fill = max(0, width - _text_width(text))
    if align == "left":
        return text + " " * fill
    if align == "center":
        head = fill // 2
        return " " * head + text + " " * (fill - head)
    return " " * fill + text


def _fmt_value(value: Optional[float], digits: int) -> str:
    """固定小数位格式化；无样本输出 ``-``（不拿 0 冒充，避免把缺样本读成 0）。"""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


class _Column(NamedTuple):
    """一列：中文表头、显示宽度、小数位、取值器。"""

    title: str
    width: int
    digits: int
    pick: Callable[[Mapping[str, Any]], Optional[float]]


def _col(title: str, width: int, digits: int, key: str) -> _Column:
    """按记录键生成一列（取值器只读记录字典，渲染层不必知道 history 结构）。"""
    return _Column(title, width, digits, lambda record: record.get(key))


#: ``[对局结构]``：前期回归节奏 → 中后期强度/掉血/存活 → 留存与核心卡覆盖 → 真人比值。
_STRUCT_COLUMNS: Tuple[_Column, ...] = (
    _col("迭代", 5, 0, "it"),
    _col("用时s", 6, 0, "elapsed"),
    _col("级T6", 4, 1, "tel_tier_6"),
    _col("级T8", 4, 1, "tel_tier_8"),
    _col("级T10", 5, 1, "tel_tier_10"),
    _col("板T8", 5, 0, "tel_own_str_8"),
    _col("板T10", 5, 0, "tel_own_str_10"),
    _col("板T12", 5, 0, "tel_own_str_12"),
    _col("血T9", 5, 1, "tel_dmg_9"),
    _col("血T12", 5, 1, "tel_dmg_12"),
    _col("达T9", 5, 3, "tel_reach9"),
    _col("达T12", 5, 3, "tel_reach12"),
    _col("留T10", 5, 1, "rg_gain_10"),
    _col("留T11", 5, 1, "rg_gain_11"),
    _col("S命中", 6, 3, "comp_s_hits"),
    _col("名次", 6, 2, "rank"),
    _col("×T8", 5, 2, "ratio_8"),
    _col("×T10", 5, 2, "ratio_10"),
    _col("×T12", 5, 2, "ratio_12"),
)

#: ``[难度与健康]``：arena 控制器（对手强度/胜率/到达率）+ 训练健康度。
_DIFF_COLUMNS: Tuple[_Column, ...] = (
    _col("前强", 5, 0, "arena_early_str"),
    _col("中强", 5, 0, "arena_mid_str"),
    _col("后强", 5, 0, "arena_late_str"),
    _col("前胜", 6, 3, "arena_early_win"),
    _col("中胜", 6, 3, "arena_mid_win"),
    _col("后胜", 6, 3, "arena_late_win"),
    _col("到T9", 6, 3, "arena_reach9"),
    _col("到T12", 6, 3, "arena_reach12"),
    _col("目标", 6, 3, "arena_reach_target"),
    _col("KL", 7, 4, "approx_kl"),
    _col("熵", 7, 4, "entropy"),
    _col("锚KL", 7, 4, "anchor_kl"),
    _col("loss", 8, 4, "loss"),
    _col("advσ", 7, 4, "adv_std"),
)

#: 判读里逐个报方向的"关键指标"（label, 记录键, 小数位）。掉血/难度上升不等于变强，
#: 所以只报方向与变化量，不在这里替用户下"好/坏"结论。
_KEY_METRICS: Tuple[Tuple[str, str, int], ...] = (
    ("酒馆等级 T6", "tel_tier_6", 1),
    ("酒馆等级 T8", "tel_tier_8", 1),
    ("酒馆等级 T10", "tel_tier_10", 1),
    ("自建板面 T8", "tel_own_str_8", 0),
    ("自建板面 T10", "tel_own_str_10", 0),
    ("自建板面 T12", "tel_own_str_12", 0),
    ("掉血 T9", "tel_dmg_9", 1),
    ("掉血 T12", "tel_dmg_12", 1),
    ("到达 9 回合占比", "tel_reach9", 3),
    ("到达 12 回合占比", "tel_reach12", 3),
    ("留存增重 T10", "rg_gain_10", 1),
    ("留存增重 T11", "rg_gain_11", 1),
    ("S 级卡命中率", "comp_s_hits", 3),
    ("平均名次", "rank", 2),
    ("arena 前期强度", "arena_early_str", 0),
    ("arena 中期强度", "arena_mid_str", 0),
    ("arena 后期强度", "arena_late_str", 0),
    ("arena 前期胜率", "arena_early_win", 3),
    ("arena 中期胜率", "arena_mid_win", 3),
    ("arena 后期胜率", "arena_late_win", 3),
    ("arena 到达 9 占比", "arena_reach9", 3),
    ("arena 到达 12 占比", "arena_reach12", 3),
)


def _change_text(label: str, first_row: Mapping[str, Any], last_row: Mapping[str, Any],
                 key: str, digits: int) -> Optional[str]:
    """``指标: 旧 → 新 (+差) ↑``；任一端没样本就返回 ``None``（整行省略）。"""
    first = first_row.get(key)
    last = last_row.get(key)
    if first is None or last is None:
        return None
    low, high = round(first, digits), round(last, digits)
    delta = round(high - low, digits)
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    amount = f"{delta:+.{digits}f}" if delta else "0"
    return f"{label}: {_fmt_value(low, digits)} → {_fmt_value(high, digits)} ({amount}) {arrow}"


def _dir_of(value: Optional[float]) -> str:
    """涨 ↑ / 跌 ↓ / 平 → 的方向标签（棘轮明细用）。"""
    if value is None:
        return "?"
    return "↑" if value > 0 else ("↓" if value < 0 else "→")


def _column_widths(columns: Sequence[_Column],
                   records: Sequence[Mapping[str, Any]]) -> List[int]:
    """列宽 = max(设计宽度, 表头/数据实际宽度)。

    组内是固定宽度（正常数据下排版不抖动）；只有上千小时耗时、10 倍人曲线这类
    超宽数字才会把这一列撑开——宁可这列变宽，也不能让整行错位。
    """
    widths: List[int] = []
    for column in columns:
        cells = [_fmt_value(column.pick(record), column.digits) for record in records]
        widths.append(max([column.width, _text_width(column.title)]
                          + [_text_width(cell) for cell in cells]))
    return widths


def render_table(title: str, columns: Sequence[_Column],
                 records: Sequence[Mapping[str, Any]], note: str = "") -> List[str]:
    """渲染一块固定宽度表：标题、表头、分隔线、数据行；无数据时给一行提示。"""
    widths = _column_widths(columns, records)
    head = " ".join(_pad(column.title, width, "center")
                    for column, width in zip(columns, widths))
    rule = " ".join("─" * width for width in widths)
    lines = [f"[{title}]" + (f" {note}" if note else ""), head, rule]
    for record in records:
        lines.append(" ".join(
            _pad(_fmt_value(column.pick(record), column.digits), width)
            for column, width in zip(columns, widths)))
    if not records:
        lines.append("(无可用行)")
    return lines


def render_verdict(shown: Sequence[Mapping[str, Any]],
                   window_records: Sequence[Mapping[str, Any]]) -> List[str]:
    """``[判读]``：关键指标方向、两个当前值、难度棘轮。

    结构类指标放左栏、arena 类放右栏，两栏并排是为了不让判读比表还长。
    """
    if not shown:
        return ["[判读]", "  无可用行：history 里没有可解析的记录"]
    first_row, last_row = shown[0], shown[-1]
    lines = [f"[判读] 末行 vs 区间首行（第 {first_row['line']} → {last_row['line']} 行，"
             f"共 {len(shown)} 条）"]
    struct_cells: List[str] = []
    diff_cells: List[str] = []
    for label, key, digits in _KEY_METRICS:
        text = _change_text(label, first_row, last_row, key, digits)
        if text is None:
            continue
        (diff_cells if key.startswith("arena_") else struct_cells).append(text)
    left_width = max((_text_width(cell) for cell in struct_cells), default=0) + 3
    for left, right in itertools.zip_longest(struct_cells, diff_cells, fillvalue=""):
        lines.append(("  " + _pad(left, left_width, "left") + right).rstrip())
    reach12 = last_row.get("tel_reach12")
    target = last_row.get("arena_reach_target")
    lines.append(f"  当前值：到达 12 回合占比 {_fmt_value(reach12, 3)}"
                 f" ｜ arena 目标胜率 {_fmt_value(target, 3)}")
    # 绝对棋力刻度：自建板面在真人分布里的分位（与对手强度无关，D-054 的核心读数）。
    pcts = []
    for turn in _CURVE_TURNS:
        p0, p1 = first_row.get(f"pct_{turn}"), last_row.get(f"pct_{turn}")
        if p1 is None:
            continue
        head = f"  真人分位 T{turn}: "
        if p0 is None:
            pcts.append(f"{head}{p1 * 100:.0f}")
        else:
            pcts.append(f"{head}{p0 * 100:.0f} → {p1 * 100:.0f} ({_dir_of(p1 - p0)})")
    if pcts:
        lines.append("  ".join(pcts))
    trend = ratchet_state(window_records)
    state_text = {"up": "上升中（↑）", "flat": "未上升", "insufficient": "数据不足"}[trend["state"]]
    lines.append(f"  难度棘轮：{state_text}  "
                 f"中强 {_trend_detail(trend['mid'])} ／后强 {_trend_detail(trend['late'])}")
    return lines


def _trend_detail(trend: Mapping[str, Any]) -> str:
    """棘轮明细：样本不足只报"n/5 行"，够样本报方向 + 涨幅（不给缺样本的方向）。"""
    if trend["state"] == "insufficient":
        return f"{trend['samples']}/5 行"
    return f"{_dir_of(trend['gain'])} {trend['gain'] * 100:+.1f}%"


# ── 结构化输出（``--json``） ───────────────────────────────────────────────
def build_payload(run_label: str, history: Path, stats: Mapping[str, int],
                  shown: Sequence[Mapping[str, Any]],
                  window_records: Sequence[Mapping[str, Any]],
                  curve: Mapping[int, Mapping[str, float]], curve_error: Optional[str],
                  warnings: Sequence[str]) -> Dict[str, Any]:
    """把面板结果打包成一份可 ``json.loads`` 的结构（供其它脚本消费）。"""
    first_row = shown[0] if shown else {}
    last_row = shown[-1] if shown else {}
    metrics: Dict[str, Dict[str, Any]] = {}
    for label, key, digits in _KEY_METRICS:
        first = first_row.get(key)
        last = last_row.get(key)
        delta = None
        if first is not None and last is not None:
            delta = round(round(last, digits) - round(first, digits), digits)
        metrics[key] = {
            "label": label,
            "first": first,
            "last": last,
            "delta": delta,
            "dir": None if delta is None else ("up" if delta > 0 else
                                               ("down" if delta < 0 else "flat")),
        }
    return {
        "run": run_label,
        "history": str(history),
        "stats": dict(stats),
        "records": [dict(record) for record in shown],
        "verdict": {
            "first_line": first_row.get("line"),
            "last_line": last_row.get("line"),
            "metrics": metrics,
            "reach12": last_row.get("tel_reach12"),
            "arena_reach_target": last_row.get("arena_reach_target"),
        },
        "ratchet": ratchet_state(window_records),
        "human_curve": {
            "turns": {str(turn): dict(point) for turn, point in sorted(curve.items())},
            "error": curve_error,
        },
        "warnings": list(warnings),
    }


# ── 入口 ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """CLI：run 目录或 history.jsonl 必填，其余都是取样/降级开关。"""
    parser = argparse.ArgumentParser(
        prog="transition_dashboard.py",
        description="打印一次训练 run 的过渡期诊断面板（离线、只读、只用标准库）")
    parser.add_argument("run", help="run 目录，或直接给该目录里的 history.jsonl")
    parser.add_argument("--last", type=int, default=40,
                        help="只看最后 N 条有效行（0=全部，默认 40）")
    parser.add_argument("--every", type=int, default=1,
                        help="每 K 条抽 1 条打印（末行一定显示，默认 1）")
    parser.add_argument("--human-curve", default=None,
                        help='真人曲线 JSON：{"8": {"p50": 179}, ...}')
    parser.add_argument("--json", action="store_true",
                        help="不打印表格，直接输出结构化 JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """读 history → 两个块表 + 判读；``--json`` 时只输出结构化结果。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.last < 0 or args.every < 1:
        parser.error("--last 需 >= 0，--every 需 >= 1")

    history = resolve_history(Path(args.run))
    if history is None:
        print(f"错误：{args.run} 不是 history.jsonl，目录下也没有 history.jsonl",
              file=sys.stderr)
        return 2

    rows, stats = read_history(history)
    window = rows[-args.last:] if args.last > 0 else list(rows)
    shown_rows = sample_rows(window, args.every)
    stats = dict(stats, window=len(window), shown=len(shown_rows))

    warnings: List[str] = []
    if stats["bad"]:
        warnings.append(f"跳过 {stats['bad']} 行坏记录（非法 JSON 或非对象行）")

    curve: Dict[int, Dict[str, float]] = {}
    curve_error: Optional[str] = None
    if args.human_curve:
        curve, curve_error = load_human_curve(Path(args.human_curve))
        if curve_error:
            warnings.append(f"human-curve 不可用：{curve_error}；比值列显示 -")

    window_records = [build_record(row, line_no, curve) for line_no, row in window]
    shown = [build_record(row, line_no, curve) for line_no, row in shown_rows]
    run_label = Path(args.run).resolve().name

    if args.json:
        # JSON 走 stdout，警告走 stderr：保证 stdout 永远是可 json.loads 的整块。
        for warning in warnings:
            print(f"[警告] {warning}", file=sys.stderr)
        payload = build_payload(run_label, history, stats, shown, window_records,
                                curve, curve_error, warnings)
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    for warning in warnings:
        print(f"[警告] {warning}")
    print(f"run: {run_label}")
    print(f"history: {history}")
    print(f"有效行 {stats['valid']}/{stats['total']}（坏 {stats['bad']}）"
          f"  区间 {stats['window']} 行  显示 {stats['shown']} 行")
    print()
    for line in render_table("对局结构", _STRUCT_COLUMNS, shown,
                             "板面=Σ攻+血均值 血=累计掉血 达=到达占比 ×=我们/真人p50"):
        print(line)
    print()
    for line in render_table("难度与健康", _DIFF_COLUMNS, shown,
                             "强=强度 胜=胜率 KL/熵/锚KL/loss/advσ=训练健康"):
        print(line)
    print()
    for line in render_verdict(shown, window_records):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
