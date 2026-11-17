"""训练健康门：把 ``docs/TRAINING_RECIPE.md`` §5 的"红线"变成代码。

为什么需要（2026-11-17 审计 P2-11）
----------------------------------
``TRAINING_RECIPE §5`` 写着「``illegal_actions``/``invalid_games``/``truncated_games``
必须 0」「非有限损失立即停」「注入失败率骤降立即停」「``pipeline guard abort`` 立即停」
「策略熵 < 0.8」「吞吐 < 1500 决策/秒」。但训练入口里**一条都没有实现**：只有
``kl_early_stop``（单轮内的早停）与 ``pipeline_guard abort`` 是代码。

分类（很重要，不把"策略选择"伪装成"工程错误"）：

* **硬门（fail-closed，默认开）**：非法动作 / 无效局 / 截断局 / 引擎异常 / 非有限损失 /
  非法动作概率非 0 / 镜像注入率崩塌。这些是 ``AGENTS.md §6`` 明令"严格停止并保留现场"
  的工程错误，继续训练只会往模型里灌错标签。
* **软警告（只记录）**：策略熵、近似 KL、吞吐、标签占比。这些是**训练动力学**指标，
  自动停机属于策略决定（本轮授权明确要求不因超参"看起来不对"就改配方），因此只打印
  并在 history 里留痕，由人或后续实验判断。

纯函数、无 I/O、可离线单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "HealthLimits",
    "hard_collect_failures",
    "hard_update_failures",
    "soft_warnings",
    "format_failures",
]


@dataclass(frozen=True)
class HealthLimits:
    """硬门阈值。``None``/0 表示**不检查该项**（便于老作业/新字段逐步接入）。"""

    #: 允许的非法动作数（必须 0）
    illegal_actions: int = 0
    #: 允许的无效局数（必须 0）
    invalid_games: int = 0
    #: 允许的截断局数（必须 0）
    truncated_games: int = 0
    #: 允许的战斗/招募上下文截断（必须 0）
    truncated_combats: int = 0
    #: 引擎主动停止的局数（`engine_stopped`）
    engine_stopped: int = 0
    #: 镜像注入成功率下限；``None`` = 不检查（纯 bot / 未开注入时）
    mirror_apply_min: Optional[float] = 0.99
    #: 非法动作概率上限（masked softmax 后应为 0）
    illegal_prob_max: float = 1e-6
    #: 软警告：策略熵下限
    entropy_warn_below: float = 0.8
    #: 软警告：近似 KL 上限
    kl_warn_above: float = 0.03
    #: 软警告：吞吐下限（决策/秒）
    throughput_warn_below: float = 1500.0

    @classmethod
    def from_args(cls, args: Any) -> "HealthLimits":
        """从 argparse 命名空间取（缺字段就用默认值）。"""
        get = lambda n, d: getattr(args, n, d) if getattr(args, n, d) is not None else d
        mir = get("health_mirror_apply_min", 0.99)
        return cls(
            illegal_actions=int(get("health_illegal_actions", 0)),
            invalid_games=int(get("health_invalid_games", 0)),
            truncated_games=int(get("health_truncated_games", 0)),
            truncated_combats=int(get("health_truncated_combats", 0)),
            engine_stopped=int(get("health_engine_stopped", 0)),
            mirror_apply_min=(None if float(mir) <= 0 else float(mir)),
            entropy_warn_below=float(get("health_entropy_warn_below", 0.8)),
            kl_warn_above=float(get("health_kl_warn_above", 0.03)),
            throughput_warn_below=float(get("health_throughput_warn_below", 1500.0)),
        )


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_failures(meta: Mapping[str, Any], key: str, limit: int) -> List[str]:
    """``meta[key]`` 超过 ``limit`` 时返回一条失败说明。缺键不报错（老作业兼容）。"""
    if key not in meta:
        return []
    value = _num(meta.get(key))
    if value is None:
        return []
    if value > float(limit):
        return [f"{key}={value:g} > 允许 {limit}"]
    return []


def _mirror_map(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """注入计数可能出现在顶层，也可能嵌在 ``collect`` 里。"""
    if any(str(k).startswith("mirror_") for k in meta):
        return meta
    inner = meta.get("collect")
    return inner if isinstance(inner, Mapping) else meta


def hard_collect_failures(meta: Any, limits: HealthLimits = HealthLimits()
                          ) -> List[str]:
    """采集侧的硬门；返回失败原因列表（空 = 通过）。"""
    if not isinstance(meta, Mapping):
        return []
    out: List[str] = []
    for key, limit in (("illegal_actions", limits.illegal_actions),
                       ("invalid_games", limits.invalid_games),
                       ("truncated_games", limits.truncated_games),
                       ("truncated_combats", limits.truncated_combats),
                       ("engine_stopped", limits.engine_stopped)):
        out += _count_failures(meta, key, limit)
    if limits.mirror_apply_min is not None:
        m = _mirror_map(meta)
        attempt = _num(m.get("mirror_attempt"))
        applied = _num(m.get("mirror_applied"))
        if attempt and attempt > 0 and applied is not None:
            rate = applied / attempt
            if rate < float(limits.mirror_apply_min):
                out.append(f"mirror_applied/attempt={rate:.4f} < "
                           f"{limits.mirror_apply_min:.4f}"
                           "（对手已不是校准战团，训练分布变了）")
    return out


def _epoch_rows(stats: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    epochs = stats.get("epochs")
    if isinstance(epochs, Sequence) and not isinstance(epochs, (str, bytes)):
        return [e for e in epochs if isinstance(e, Mapping)]
    return [stats]


def hard_update_failures(stats: Any, limits: HealthLimits = HealthLimits()
                         ) -> List[str]:
    """PPO 更新侧的硬门（非有限损失 / 非法动作概率）。"""
    if not isinstance(stats, Mapping):
        return []
    out: List[str] = []
    for i, row in enumerate(_epoch_rows(stats)):
        for key in ("total", "policy_loss", "value_loss"):
            v = _num(row.get(key))
            if v is not None and not math.isfinite(v):
                out.append(f"epoch[{i}].{key}={v!r} 非有限")
        p = _num(row.get("illegal_prob_max"))
        if p is not None and p > float(limits.illegal_prob_max):
            out.append(f"epoch[{i}].illegal_prob_max={p:g} > "
                       f"{limits.illegal_prob_max:g}（掩码泄漏）")
    return out


def soft_warnings(stats: Any, meta: Any = None,
                  limits: HealthLimits = HealthLimits()) -> List[str]:
    """软警告（只记录，不停机）：熵过低 / KL 过高 / 吞吐过低。"""
    out: List[str] = []
    if isinstance(stats, Mapping):
        for i, row in enumerate(_epoch_rows(stats)):
            ent = _num(row.get("entropy"))
            if ent is not None and ent < float(limits.entropy_warn_below):
                out.append(f"epoch[{i}].entropy={ent:.3f} < "
                           f"{limits.entropy_warn_below}（策略可能过度确定）")
            kl = _num(row.get("approx_kl"))
            if kl is not None and kl > float(limits.kl_warn_above):
                out.append(f"epoch[{i}].approx_kl={kl:.4f} > "
                           f"{limits.kl_warn_above}（单轮位移偏大）")
    if isinstance(meta, Mapping) and limits.throughput_warn_below > 0:
        dps = _num(_mirror_map(meta).get("decisions_per_s"))
        if dps is None:
            dps = _num(meta.get("decisions_per_s"))
        if dps is not None and dps < float(limits.throughput_warn_below):
            out.append(f"decisions_per_s={dps:.0f} < "
                       f"{limits.throughput_warn_below:.0f}（吞吐异常）")
    return out


def format_failures(reasons: Iterable[str], *, where: str = "collect") -> str:
    """把失败列表格式化成一行可搜索的日志（现场保留由调用方负责）。"""
    items = [str(r) for r in reasons]
    if not items:
        return ""
    return f"[health-gate] {where} 触发硬门（工程错误，已停止）：" + "; ".join(items)
