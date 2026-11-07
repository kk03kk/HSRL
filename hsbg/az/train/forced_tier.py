"""强制升本（forced tier-up）课程：按回合强制 learner 执行 Upgrade，并如实记账。

动机（见 ``docs/research/LONG1085131_REVIEW.md`` §5）：模型在 tier 3 停滞，晚局
（T12+）胜率 0–6%，因此"升本"的期望收益为负，PPO 正确地学会了不升本。要打破这个
自洽的局部最优，必须先让它**真的到达高本**、在 tier 4/5/6 的商店与战团里采样，才有
机会学到高本经营；随后再把控制权交还给它。

本模块只做三件事，全部是纯逻辑、可单测：

1. **计划解析**：``"2:2,5:3,6:4,8:5"`` → ``{2: 2, 5: 3, 6: 4, 8: 5}``（回合 → 目标等级）；
2. **判定**：在每个 learner 决策上判断"现在是否应强制升本"，以及"是否该给教师标签"；
3. **记账**：每个目标按 ``self / on_time / deferred / missed`` 分类，计数器进 collect meta。

设计约束（与 ``docs/rewards``/``PPO`` 口径一致）：

- **只强制真正合法的动作**：判据是引擎给的合法掩码里 Upgrade 那一位（``mask[3]``），
  它已经同时编码"金币够 + 未达上限 + 不在待选阶段"。本模块不读金币、不补金币、
  不改引擎状态；不合法就**不强制**，并记一次 ``blocked``。
- **被强制的决策不进策略梯度**：调用方把该条记录标成 ``forced``（与动作上限守卫、
  换位守卫同口径：``logprob=0``，PPO 的策略/辅助损失排除，只保留价值链）。这样既
  不伪造 on-policy ratio，也不会把"脚本替我按的键"当成策略的选择。
- **不许静默跳过**：目标回合没达成时按 ``carry`` 顺延（默认 1 回合），顺延仍不达成
  记 ``missed``；每条都带回合号与原因，写进 meta 与 history。
- **概率门控**：``force_prob < 1`` 时按概率强制，用于"逐步解除强制"；未强制但目标
  未达成时，调用方可以写 ``teacher_action``（DAgger 标签），让策略自己学会按。
- 关闭（空计划）时本模块不构造对象、不读引擎、不新增 record 字段，旧路径逐位一致。

统计口径（每个目标等级一条）：

    self      在目标回合之前就自己升到了（说明策略本身会升，无需强制）
    on_time   恰好在目标回合升到
    deferred  在目标回合之后的 carry 回合内补升（例如 T6 小饰品吃掉金币）
    missed    超过 dl+carry 仍未达到
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: 默认强制升本计划（回合 → 目标酒馆等级）。T9 以后自由经营，六本不强制。
DEFAULT_FORCED_TIER_PLAN: Dict[int, int] = {2: 2, 5: 3, 6: 4, 8: 5}

#: 默认计划字符串（CLI / history 用）
DEFAULT_FORCED_TIER_PLAN_SPEC = "2:2,5:3,6:4,8:5"

#: 默认顺延回合数：目标回合没做到，允许在随后这么多回合内补做（记为 deferred）
DEFAULT_FORCED_TIER_CARRY = 1

#: 引擎动作空间里 Upgrade 的下标（见 ``hsbg/az/train/common.py`` ACTION_SEGMENTS）
UPGRADE_ACTION = 3

#: tier 合法范围（1..7：7 是特殊等级）
_TIER_MIN, _TIER_MAX = 1, 7
#: 回合上限（引擎最大 20000 步，回合远小于此；仅防非法输入）
_TURN_MAX = 200

_SPEC_ITEM = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


def parse_forced_tier_plan(spec: object) -> Dict[int, int]:
    """解析 ``"2:2,5:3,6:4,8:5"`` → ``{2: 2, 5: 3, 6: 4, 8: 5}``。

    - ``None`` / ``""`` / ``"none"`` / ``"off"`` / ``"0"`` → ``{}``（关闭）；
    - 也接受 ``dict``（键值直接校验）与 ``[(turn, tier), ...]``；
    - 分隔符：逗号或分号，条目内允许空格；
    - 非法 turn / tier / 重复 turn 一律 ``ValueError``（启动期 fail-fast，避免训练中
      静默把计划吃掉）。
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        items: List[Tuple[object, object]] = list(spec.items())
    elif isinstance(spec, (list, tuple)) and spec and not isinstance(spec[0], str):
        items = []
        for item in spec:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"非法强制升本计划条目: {item!r}（应为 (turn, tier)）")
            items.append((item[0], item[1]))
    else:
        text = str(spec).strip().lower()
        if text in ("", "none", "off", "0", "false"):
            return {}
        items = []
        for raw in str(spec).replace(";", ",").split(","):
            piece = raw.strip()
            if not piece:
                continue
            m = _SPEC_ITEM.match(piece)
            if m is None:
                raise ValueError(f"非法强制升本计划片段: {piece!r}（应为 turn:tier）")
            items.append((m.group(1), m.group(2)))
    out: Dict[int, int] = {}
    for raw_turn, raw_tier in items:
        try:
            turn = int(raw_turn)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"强制升本的 turn 必须是整数: {raw_turn!r}") from exc
        try:
            tier = int(raw_tier)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"强制升本的 tier 必须是整数: {raw_tier!r}") from exc
        if not (1 <= turn <= _TURN_MAX):
            raise ValueError(f"强制升本 turn 必须在 1..{_TURN_MAX}: {turn}")
        if not (_TIER_MIN <= tier <= _TIER_MAX):
            raise ValueError(f"强制升本 tier 必须在 {_TIER_MIN}..{_TIER_MAX}: {tier}")
        if turn in out:
            raise ValueError(f"强制升本的 turn 重复: {turn}")
        out[turn] = tier
    # 目标等级必须随回合单调不减（否则"降级"计划没有意义，一定是写错了）
    last_tier = 0
    for turn in sorted(out):
        if out[turn] < last_tier:
            raise ValueError(f"强制升本计划必须按回合单调不减: turn {turn} 目标 {out[turn]} "
                             f"< 前一目标 {last_tier}")
        last_tier = out[turn]
    return out


def plan_spec_of(plan: Dict[int, int]) -> str:
    """``{2:2,5:3}`` → ``"2:2,5:3"``（history / 报告用）。"""
    return ",".join(f"{t}:{v}" for t, v in sorted(plan.items()))


#: decisions 字面量后缀（``10M`` / ``500K`` / ``1B``）
_UNITS = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9}


def parse_decisions(value: object) -> float:
    """``"20M"`` / ``"500K"`` / ``1e6`` → float；非法输入 ``ValueError``。"""
    if isinstance(value, bool):
        raise ValueError(f"decisions 不能是 bool: {value!r}")
    if isinstance(value, (int, float)):
        d = float(value)
    else:
        text = str(value).strip().replace("_", "").replace(",", "")
        if not text:
            raise ValueError("decisions 不能为空")
        suffix = text[-1].lower()
        try:
            if suffix in _UNITS and suffix:
                d = float(text[:-1]) * _UNITS[suffix]
            else:
                d = float(text)
        except ValueError as exc:
            raise ValueError(f"非法 decisions 字面量: {value!r}") from exc
    if d != d or d in (float("inf"), float("-inf")):
        raise ValueError(f"decisions 必须是有限值: {value!r}")
    if d < 0:
        raise ValueError(f"decisions 必须 >= 0: {value!r}")
    return float(d)


def parse_prob_schedule(spec: object, *, default: float = 1.0
                        ) -> List[Tuple[float, float]]:
    """解析强制概率计划 ``"0:1.0,40M:0.75,120M:0.0"`` → ``[(0,1.0),(4e7,0.75),…]``。

    空 / ``None`` / ``"none"`` → ``[(0.0, default)]``。分段常数：每段从该 decisions
    起生效，直到下一段。非法概率（不在 [0,1]）或 decisions 非单调 → ``ValueError``。
    """
    if spec is None:
        return [(0.0, float(default))]
    text = str(spec).strip().lower()
    if text in ("", "none", "off"):
        return [(0.0, float(default))]
    out: List[Tuple[float, float]] = []
    for raw in text.replace(";", ",").split(","):
        piece = raw.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"强制概率计划片段缺少 ':': {piece!r}")
        left, right = piece.split(":", 1)
        d = parse_decisions(left.strip())
        try:
            p = float(right.strip())
        except ValueError as exc:
            raise ValueError(f"强制概率必须是浮点数: {right!r}") from exc
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"强制概率必须在 [0,1]: {p}")
        out.append((d, p))
    if not out:
        return [(0.0, float(default))]
    out.sort(key=lambda x: x[0])
    if len({d for d, _ in out}) != len(out):
        raise ValueError("强制概率计划的 decisions 不能重复")
    if out[0][0] > 0.0:
        out.insert(0, (0.0, out[0][1]))
    return out


def prob_at(schedule: List[Tuple[float, float]], decisions: float) -> float:
    """分段常数查表：返回 ``decisions`` 处生效的强制概率。"""
    p = float(schedule[0][1]) if schedule else 0.0
    for d, v in schedule:
        if float(decisions) >= float(d):
            p = float(v)
        else:
            break
    return float(p)


@dataclass
class ForcedTierDecision:
    """一次 learner 决策上的强制判定结果。"""

    #: 是否强制把动作改成 Upgrade
    force: bool = False
    #: 是否写教师标签（目标未达成、Upgrade 合法、但本次没有强制）
    teacher: bool = False
    #: 本次命中的目标等级（None = 本回合没有适用目标）
    target: Optional[int] = None
    #: 该目标的截止回合
    deadline: Optional[int] = None
    #: 本回合是"顺延补做"（turn > deadline）
    deferred: bool = False
    #: 目标适用但 Upgrade 不合法（金币不足/待选阶段/已到上限）——只用于记账
    blocked: bool = False

    def as_record(self) -> Dict[str, object]:
        return {"force": bool(self.force), "target": self.target,
                "deadline": self.deadline, "deferred": bool(self.deferred)}


@dataclass
class _TargetState:
    deadline: int
    want: int
    achieved_turn: Optional[int] = None
    achieved_by: str = ""          # self | forced
    missed: bool = False
    forced_actions: int = 0
    teacher_labels: int = 0
    teacher_turn: int = -1
    blocked: int = 0
    blocked_turn: int = -1


class ForcedTierTracker:
    """按局维护强制升本状态与统计（一个实例服务多局，状态用 ``game_id/seat`` 隔离）。"""

    def __init__(self, plan: object = DEFAULT_FORCED_TIER_PLAN_SPEC, *,
                 carry: int = DEFAULT_FORCED_TIER_CARRY) -> None:
        self.plan: Dict[int, int] = parse_forced_tier_plan(plan)
        self.carry = max(0, int(carry))
        self._state: Dict[Tuple[int, int], List[_TargetState]] = {}
        #: 已结束对局的按目标累计（``reset()`` 时折叠进来，避免逐局状态无限增长）
        self._cum: Dict[str, Dict[str, int]] = {}
        # 整次采集的累计计数（不随 reset 清零）
        self.forced_actions = 0
        self.teacher_labels = 0
        self.blocked = 0
        self.prob_skips = 0
        self._last_force_turn: Dict[Tuple[int, int], Optional[int]] = {}

    # ---- 生命周期 ----
    @property
    def enabled(self) -> bool:
        return bool(self.plan)

    def reset(self) -> None:
        """一局开始：把上一局的目标结果折叠进累计，再清空按局状态。

        累计计数（``forced_actions`` 等）与折叠结果都保留，因此整次采集的
        ``stats()`` 始终是全程累计，内存不随对局数增长。
        """
        for key in self._state:
            for t in self._state[key]:
                row = self._cum.setdefault(str(t.want), {
                    "self": 0, "on_time": 0, "deferred": 0, "missed": 0,
                    "forced_actions": 0, "teacher_labels": 0, "blocked": 0})
                row["forced_actions"] += int(t.forced_actions)
                row["teacher_labels"] += int(t.teacher_labels)
                row["blocked"] += int(t.blocked)
                if t.achieved_turn is None:
                    if t.missed:
                        row["missed"] += 1
                    continue
                if t.achieved_turn < t.deadline:
                    row["self"] += 1
                elif t.achieved_turn == t.deadline:
                    row["on_time"] += 1
                elif t.achieved_turn <= t.deadline + self.carry:
                    row["deferred"] += 1
                else:
                    row["missed"] += 1
        self._state = {}
        self._last_force_turn = {}

    def _targets(self, game_id: int, seat: int) -> List[_TargetState]:
        key = (int(game_id), int(seat))
        st = self._state.get(key)
        if st is None:
            st = [_TargetState(deadline=int(dl), want=int(w))
                  for dl, w in sorted(self.plan.items())]
            self._state[key] = st
            self._last_force_turn[key] = None
        return st

    # ---- 主判定 ----
    def decide(self, *, game_id: int, seat: int, turn: int, tier: int,
               upgrade_legal: bool, force_prob: float = 1.0,
               rng: Optional[object] = None) -> ForcedTierDecision:
        """判断本次决策是否强制升本 / 是否写教师标签。

        ``upgrade_legal`` 必须来自引擎合法掩码的 Upgrade 位（唯一权威口径）。
        ``rng``：``force_prob < 1`` 时用于伯努利抽样（传 ``np.random.Generator`` 或
        ``random.Random``，只要求有 ``random()``）。
        """
        out = ForcedTierDecision()
        if not self.plan:
            return out
        turn = int(turn)
        tier = int(tier)
        targets = self._targets(game_id, seat)
        # 1) 先按当前 tier 结算"已达成"（模型自己升的也算）
        last = self._last_force_turn.get((int(game_id), int(seat)))
        for t in targets:
            if t.achieved_turn is None and tier >= t.want:
                by_force = (last is not None and int(last) in (turn, turn - 1))
                t.achieved_turn = int(last) if by_force else int(turn)
                t.achieved_by = "forced" if by_force else "self"
        # 2) 标记过期未达成的目标
        for t in targets:
            if t.achieved_turn is None and not t.missed and turn > t.deadline + self.carry:
                t.missed = True
        # 3) 找当前适用目标：截止已到、未达成、未过期、tier 仍低于目标
        active: Optional[_TargetState] = None
        for t in targets:
            if t.achieved_turn is not None or t.missed:
                continue
            if turn < t.deadline or turn > t.deadline + self.carry:
                continue
            if tier >= t.want:
                continue
            active = t
            break
        if active is None:
            return out
        out.target = int(active.want)
        out.deadline = int(active.deadline)
        out.deferred = bool(turn > active.deadline)
        if not upgrade_legal:
            # 目标适用但此刻不能升（金币不足 / 待选阶段 / 已到等级上限）：
            # 不改动作、不写教师标签，每回合最多记一次 blocked。
            if active.blocked_turn != turn:
                active.blocked_turn = turn
                active.blocked += 1
                self.blocked += 1
            out.blocked = True
            return out
        want_force = True
        if force_prob < 1.0:
            draw = self._draw(rng)
            want_force = bool(draw < float(force_prob))
            if not want_force:
                self.prob_skips += 1
        if want_force:
            out.force = True
            active.forced_actions += 1
            self.forced_actions += 1
            self._last_force_turn[(int(game_id), int(seat))] = int(turn)
        elif active.teacher_turn != turn:
            # 教师标签**每回合最多一条**：同回合后续决策不再重复标注，
            # 否则 CE 会在一整个回合的十几条决策上重复同一标签，权重失衡。
            out.teacher = True
            active.teacher_turn = turn
            active.teacher_labels += 1
            self.teacher_labels += 1
        return out

    def note_forced_action(self, game_id: int, seat: int, turn: int) -> None:
        """调用方确认已经执行了强制 Upgrade（用于"下一步结算"的归属）。"""
        self._last_force_turn[(int(game_id), int(seat))] = int(turn)

    def note_teacher_label(self) -> None:
        """调用方确认已写入一条 teacher 标签（外部写标签时同步计数）。"""
        self.teacher_labels += 1

    @staticmethod
    def _draw(rng: Optional[object]) -> float:
        if rng is None:
            return 0.0
        try:
            return float(rng.random())  # numpy Generator / random.Random
        except AttributeError:  # pragma: no cover - 防御
            return 0.0

    # ---- 统计 ----
    def game_stats(self) -> Dict[str, object]:
        """当前（尚未折叠的）对局统计：每个目标一行。

        采集器在每局结束、下一次 ``reset()`` 之前调用它，把结果写进该局的 meta；
        顶层再用 :func:`aggregate_forced_tier` 汇总。未启用时返回 ``{}``。
        """
        if not self.plan:
            return {}
        agg: Dict[str, Dict[str, int]] = {}
        for key in sorted(self._state):
            for t in self._state[key]:
                row = agg.setdefault(str(t.want), {"self": 0, "on_time": 0,
                                                   "deferred": 0, "missed": 0,
                                                   "forced_actions": 0, "teacher_labels": 0, "blocked": 0})
                row["forced_actions"] += int(t.forced_actions)
                row["teacher_labels"] += int(t.teacher_labels)
                row["blocked"] += int(t.blocked)
                if t.achieved_turn is None:
                    if t.missed:
                        row["missed"] += 1
                    continue
                if t.achieved_turn < t.deadline:
                    row["self"] += 1
                elif t.achieved_turn == t.deadline:
                    row["on_time"] += 1
                elif t.achieved_turn <= t.deadline + self.carry:
                    row["deferred"] += 1
                else:
                    row["missed"] += 1
        return {"plan": plan_spec_of(self.plan), "carry": int(self.carry),
                "targets": agg}

    def stats(self) -> Dict[str, object]:
        """返回本实例的累计强制升本统计（已折叠的对局 + 进行中的对局）。

        未启用时返回 ``{}``（不新增 meta 键）。
        """
        if not self.plan:
            return {}
        agg: Dict[str, Dict[str, int]] = {
            k: dict(v) for k, v in self._cum.items()}
        for key in sorted(self._state):
            for t in self._state[key]:
                row = agg.setdefault(str(t.want), {"self": 0, "on_time": 0,
                                                   "deferred": 0, "missed": 0,
                                                   "forced_actions": 0, "teacher_labels": 0, "blocked": 0})
                row["forced_actions"] += int(t.forced_actions)
                row["teacher_labels"] += int(t.teacher_labels)
                row["blocked"] += int(t.blocked)
                if t.achieved_turn is None:
                    if t.missed:
                        row["missed"] += 1
                    continue
                if t.achieved_turn < t.deadline:
                    row["self"] += 1
                elif t.achieved_turn == t.deadline:
                    row["on_time"] += 1
                elif t.achieved_turn <= t.deadline + self.carry:
                    row["deferred"] += 1
                else:
                    row["missed"] += 1
        out: Dict[str, object] = {
            "plan": plan_spec_of(self.plan),
            "carry": int(self.carry),
            "forced_actions": int(self.forced_actions),
            "teacher_labels": int(self.teacher_labels),
            "blocked": int(self.blocked),
            "prob_skips": int(self.prob_skips),
            "targets": {k: agg[k] for k in sorted(agg, key=lambda x: int(x))},
        }
        return out


def summarize_forced_tier(metas: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """把多局/多 worker 的 ``targets`` 计数合并（collect meta 用）。"""
    total: Dict[str, Dict[str, int]] = {}
    counts = {"forced_actions": 0, "teacher_labels": 0, "blocked": 0, "prob_skips": 0}
    plan = ""
    carry = 0
    for meta in metas:
        if not isinstance(meta, dict) or "targets" not in meta:
            continue
        plan = str(meta.get("plan") or plan)
        carry = int(meta.get("carry") or carry)
        for k in counts:
            counts[k] += int(meta.get(k) or 0)
        for tier, row in (meta.get("targets") or {}).items():
            acc = total.setdefault(str(tier), {"self": 0, "on_time": 0,
                                               "deferred": 0, "missed": 0,
                                               "forced_actions": 0, "teacher_labels": 0, "blocked": 0})
            for k in acc:
                acc[k] += int(row.get(k) or 0)
    if not total and not any(counts.values()):
        return {}
    out: Dict[str, object] = {"plan": plan, "carry": int(carry)}
    out.update(counts)
    out["targets"] = {k: total[k] for k in sorted(total, key=lambda x: int(x))}
    # 顶层计数以「按目标合计」为准：game_stats 只带 targets，逐局 meta 里没有
    # 全局计数器；这样 history / 报告读顶层就是真实执行次数。
    out["forced_actions"] = int(sum(int(r.get("forced_actions") or 0)
                                    for r in total.values()))
    out["teacher_labels"] = int(sum(int(r.get("teacher_labels") or 0)
                                    for r in total.values()))
    out["blocked"] = int(sum(int(r.get("blocked") or 0) for r in total.values()))
    return out


def forced_tier_rates(stats: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """把累计计数换成比例（报告用；分母为 0 时返回空 dict）。"""
    out: Dict[str, Dict[str, float]] = {}
    for tier, row in (stats.get("targets") or {}).items():
        tot = sum(int(row.get(k) or 0) for k in ("self", "on_time", "deferred", "missed"))
        if tot <= 0:
            continue
        out[str(tier)] = {k: float(row.get(k) or 0) / tot
                          for k in ("self", "on_time", "deferred", "missed")}
        out[str(tier)]["n"] = float(tot)
    return out


def aggregate_forced_tier(games_meta: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """从逐局 meta 里的 ``forced_tier_game`` 汇总强制升本统计（默认关闭 -> ``{}``）"""
    rows = [m["forced_tier_game"] for m in games_meta
            if isinstance(m, dict) and isinstance(m.get("forced_tier_game"), dict)]
    return summarize_forced_tier(rows)


__all__ = [
    "DEFAULT_FORCED_TIER_PLAN",
    "DEFAULT_FORCED_TIER_PLAN_SPEC",
    "DEFAULT_FORCED_TIER_CARRY",
    "UPGRADE_ACTION",
    "ForcedTierDecision",
    "ForcedTierTracker",
    "parse_forced_tier_plan",
    "plan_spec_of",
    "summarize_forced_tier",
    "forced_tier_rates",
    "aggregate_forced_tier",
]