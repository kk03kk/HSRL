#!/usr/bin/env python
"""对比评测面板：冠军 vs 候选，出逐回合 M1/M2 差异 + 配对显著性 + 判胜结论。

口径唯一来源：``docs/EVAL_PROTOCOL.md``。
- 结果刻度：``models[i].outcome``（mean_rank / top4 / win / illegal / truncated）
- 过程刻度：``models[i].curves[turn]`` → ``strength_p50`` = M1、``tier_mean`` = M2
- 配对检验：``models[i].rank_sequence``（同一 ``--seed`` 下各模型打同一批种子，
  因此可以逐局配对；用 bootstrap 求 mean_rank 差的 95% 区间）

用法::

    python tools/compare_panels.py artifacts/runs/gap-xxx/panel.json \
        --champion champion14858 --candidate arena_v3

    # 只看 JSON
    python tools/compare_panels.py panel.json --champion A --candidate B --json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 判"M1 明显退化"的相对阈值（默认 5%）；只在 T>=这个回合开始看
DEFAULT_M1_TOL = 0.05
DEFAULT_M1_FROM_TURN = 8
#: mean_rank 差小于这个值视为"没变好"（噪声级）
DEFAULT_RANK_EPS = 0.05
BOOTSTRAP_ROUNDS = 10000
BOOTSTRAP_SEED = 20261114


def load_models(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """从一到多个 panel.json 里收集 models，按 name 去重（后者同名覆盖前者）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for m in d.get("models") or []:
            name = str(m.get("name") or "").strip()
            if name:
                out[name] = m
    if not out:
        raise SystemExit("面板里没有 models（检查 panel.json 结构）")
    return out


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def paired_rank_diff(a: Sequence[float], b: Sequence[float],
                     rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """mean(a) - mean(b) 的配对 bootstrap 95% 区间（负值 = a 名次更小 = a 更好）。"""
    n = min(len(a), len(b))
    if n <= 0:
        return {"n": 0}
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    point = statistics.fmean(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    # 配对 win/lose：a 名次更靠前的局数
    wins = sum(1 for d in diffs if d < 0)
    losses = sum(1 for d in diffs if d > 0)
    ties = n - wins - losses
    return {"n": n, "diff": point, "ci95": [lo, hi],
            "wins": wins, "losses": losses, "ties": ties}


def per_turn(models: Dict[str, Dict[str, Any]], turns: Sequence[int]
             ) -> List[Dict[str, Any]]:
    rows = []
    for t in turns:
        row: Dict[str, Any] = {"turn": t}
        for name, m in models.items():
            c = (m.get("curves") or {}).get(str(t)) or {}
            row[name] = {"n": c.get("n"),
                         "m1": _f(c.get("strength_p50")),
                         "m2": _f(c.get("tier_mean"))}
        rows.append(row)
    return rows


def verdict(models: Dict[str, Dict[str, Any]], champ: str, cand: str,
            m1_tol: float, m1_from: int, rank_eps: float) -> Dict[str, Any]:
    ac = models[champ].get("outcome") or {}
    bc = models[cand].get("outcome") or {}
    reasons: List[str] = []

    bad = []
    for tag, o in ((champ, ac), (cand, bc)):
        if int(o.get("illegal_actions") or 0) != 0:
            bad.append(f"{tag}: illegal_actions={o.get('illegal_actions')}")
        if int(o.get("truncated_games") or 0) != 0:
            bad.append(f"{tag}: truncated_games={o.get('truncated_games')}")
    if bad:
        return {"verdict": "INVALID", "reasons": bad}

    ra, rb = _f(ac.get("mean_rank")), _f(bc.get("mean_rank"))
    sa = models[champ].get("rank_sequence") or []
    sb = models[cand].get("rank_sequence") or []
    stat = paired_rank_diff(sa, sb) if sa and sb else {"n": 0}
    if ra is not None and rb is not None:
        reasons.append(f"mean_rank {champ}={ra:.3f} → {cand}={rb:.3f} "
                       f"(Δ={rb - ra:+.3f}，负=候选更好)")
    if stat.get("n"):
        lo, hi = stat["ci95"]
        reasons.append(f"配对 bootstrap（{stat['n']} 局）Δ 的 95% 区间 "
                       f"[{lo:+.3f}, {hi:+.3f}]；逐局 候选胜 {stat['wins']} / "
                       f"负 {stat['losses']} / 平 {stat['ties']}")
        significant = (hi < 0) or (lo > 0)
    else:
        significant = False
        reasons.append("面板里没有 rank_sequence，无法做配对检验（只能看点估计）")

    better = (ra is not None and rb is not None and (ra - rb) >= rank_eps)
    worse = (ra is not None and rb is not None and (rb - ra) >= rank_eps)

    # M1 退化检查
    regress = []
    cm = models[champ].get("curves") or {}
    dm = models[cand].get("curves") or {}
    for k in sorted(cm, key=lambda x: int(x)):
        t = int(k)
        if t < m1_from:
            continue
        a = _f((cm.get(k) or {}).get("strength_p50"))
        b = _f((dm.get(k) or {}).get("strength_p50"))
        if a and b and a > 0:
            rel = (b - a) / a
            if rel <= -m1_tol:
                regress.append((t, a, b, rel))

    if ber := regress:
        reasons.append("M1 明显退化（阈值 -%.0f%%）：" % (m1_tol * 100) +
                       "、".join(f"T{t} {a:.0f}→{b:.0f}（{r * 100:+.1f}%）"
                                 for t, a, b, r in ber))

    if better and not ber:
        v = "IMPROVED" if significant else "IMPROVED_WEAK"
    elif better and ber:
        v = "SUSPICIOUS"
    else:
        v = "NO_IMPROVEMENT" if not worse else "WORSE"
    return {"verdict": v, "reasons": reasons, "paired": stat,
            "significant": significant, "m1_regressions": ber}


VERDICT_TEXT = {
    "INVALID": "整轮作废（工程问题），先修再谈",
    "IMPROVED": "进步（配对显著，且 M1 未退化）→ 换冠军 + 重建战团库",
    "IMPROVED_WEAK": "名次变好但配对不显著 → 只记录，暂不换冠军（加样本再测）",
    "SUSPICIOUS": "可疑进步（名次变好但 M1 退化）→ 先归因，不换冠军",
    "NO_IMPROVEMENT": "没有进步 → 进入归因",
    "WORSE": "变差 → 进入归因",
}


def render(models: Dict[str, Dict[str, Any]], champ: str, cand: str,
           rows: List[Dict[str, Any]], res: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append(f"## 判胜：{res['verdict']} —— {VERDICT_TEXT.get(res['verdict'], '')}")
    out.append("")
    for r in res["reasons"]:
        out.append(f"- {r}")
    out.append("")
    out.append("## 结果刻度")
    out.append("")
    out.append("| 指标 | %s（冠军） | %s（候选） | Δ |" % (champ, cand))
    out.append("|---|---:|---:|---:|")
    ac = models[champ].get("outcome") or {}
    bc = models[cand].get("outcome") or {}
    for key, label, fmt in (("mean_rank", "平均名次 ↓", "{:.3f}"),
                            ("top4_rate", "前四率 ↑", "{:.1%}"),
                            ("win_rate", "吃鸡率 ↑", "{:.1%}"),
                            ("decisions_per_game", "每局决策", "{:.1f}")):
        a, b = _f(ac.get(key)), _f(bc.get(key))
        if a is None or b is None:
            continue
        out.append(f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {b - a:+.3f} |")
    out.append("")
    out.append("## 过程刻度（M1 中位战力 / M2 平均 tier）")
    out.append("")
    out.append("| 回合 | n | M1 冠军 | M1 候选 | M1 Δ | M2 冠军 | M2 候选 | M2 Δ |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        a, b = row[champ], row[cand]
        if a["m1"] is None or b["m1"] is None:
            continue
        d1 = (b["m1"] - a["m1"]) / a["m1"] if a["m1"] else 0.0
        d2 = (b["m2"] - a["m2"]) if (a["m2"] is not None and b["m2"] is not None) else None
        out.append("| %d | %s | %.0f | %.0f | %+.1f%% | %s | %s | %s |" % (
            row["turn"], a["n"], a["m1"], b["m1"], d1 * 100,
            f"{a['m2']:.2f}" if a["m2"] is not None else "-",
            f"{b['m2']:.2f}" if b["m2"] is not None else "-",
            f"{d2:+.2f}" if d2 is not None else "-"))
    out.append("")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="compare eval panels (champion vs candidate)")
    ap.add_argument("panels", nargs="+", help="一个或多个 panel.json")
    ap.add_argument("--champion", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--m1-tol", type=float, default=DEFAULT_M1_TOL,
                    help="M1 相对退化阈值（默认 0.05 = 5%%）")
    ap.add_argument("--m1-from-turn", type=int, default=DEFAULT_M1_FROM_TURN)
    ap.add_argument("--rank-eps", type=float, default=DEFAULT_RANK_EPS,
                    help="mean_rank 差小于它算没变（默认 0.05）")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    models = load_models(a.panels)
    for n in (a.champion, a.candidate):
        if n not in models:
            raise SystemExit(f"面板里没有模型 {n!r}；可选：{sorted(models)}")
    if a.champion == a.candidate:
        raise SystemExit("champion 与 candidate 不能是同一个模型")

    turns = sorted({int(k) for m in models.values()
                    for k in (m.get("curves") or {})})
    rows = per_turn({a.champion: models[a.champion], a.candidate: models[a.candidate]},
                    turns)
    res = verdict(models, a.champion, a.candidate,
                  float(a.m1_tol), int(a.m1_from_turn), float(a.rank_eps))
    if a.json:
        print(json.dumps({"verdict": res["verdict"],
                          "reasons": res["reasons"],
                          "paired": res.get("paired"),
                          "per_turn": rows}, ensure_ascii=False, indent=1))
    else:
        print(render(models, a.champion, a.candidate, rows, res))
    return 0 if res["verdict"] != "INVALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())
