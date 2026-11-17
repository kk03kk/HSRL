#!/usr/bin/env python
"""对比评测面板：逐回合 M1/M2 差异 + 配对显著性 + 判胜参考。

口径唯一来源：``docs/EVAL_PROTOCOL.md``。
- 结果刻度：``models[i].outcome``（mean_rank / top4 / win / illegal / truncated）
- 过程刻度：``models[i].curves[turn]`` → ``strength_p50`` = M1、``tier_mean`` = M2
- 配对检验：``models[i].rank_sequence`` + ``models[i].seed_sequence``

2026-11-17 审计修的三件事
-------------------------
1. **按 seed 配对**（P1-6）：旧实现按数组下标配对（``diff[i] = a[i] - b[i]``）。
   只要某一侧缺一个**中间**分片，两侧从缺的地方起整体错位，配对差分退化成两个独立
   样本之差 ⇒ CI 假性跨 0 ⇒ 会把真实进步判成"不显著"。现在以 ``seed_sequence``
   为键求交集后再配对；缺 seed 时才退回按下标，并在结果里显式标注 ``pairing``。
2. **协议校验**（P0-2）：不同 ``protocol_sha256`` 的面板**拒绝比较**
   （``docs/EVAL_PROTOCOL.md`` §4 规则 5）。旧面板没有指纹时用
   ``--legacy-no-protocol`` 显式降级，并在报告里标 ``UNVERIFIED``。
3. **共同存活子集 M1**（P2-5）：``curves[t].strength_p50`` 是"活到第 t 回合的局"
   的条件中位数；一个模型活得更久 ⇒ 它的 T14 样本里混进更多本该早死的弱局 ⇒
   中位战力被**拉低**，于是"运营变好"会被读成"M1 退化"。新增按
   ``strength_by_seed`` 求**两模型都活到 t 的同一批 seed** 上的中位战力
   （``paired``），并与条件中位数并列显示。

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.eval_policy import assert_same_protocol  # noqa: E402

#: 判"M1 明显退化"的相对阈值（默认 5%）；只在 T>=这个回合开始看
DEFAULT_M1_TOL = 0.05
DEFAULT_M1_FROM_TURN = 8
#: M1 退化判定要求的最小样本量：n 小于它的回合不参与判定（20 局的回合波动就是噪声）
DEFAULT_M1_MIN_N = 100
#: mean_rank 差小于这个值视为"没变好"（噪声级）
DEFAULT_RANK_EPS = 0.05
BOOTSTRAP_ROUNDS = 10000
BOOTSTRAP_SEED = 20261114


def load_models(paths: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]],
                                               List[Dict[str, Any]]]:
    """从一到多个 panel.json 里收集 models（按 name 去重）+ 原始 panel 文档。"""
    out: Dict[str, Dict[str, Any]] = {}
    docs: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        docs.append(d)
        for m in d.get("models") or []:
            name = str(m.get("name") or "").strip()
            if name:
                out[name] = m
    if not out:
        raise SystemExit("面板里没有 models（检查 panel.json 结构）")
    return out, docs


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _pair_keys(a_ranks: Sequence[Any], b_ranks: Sequence[Any],
               a_seeds: Optional[Sequence[Any]] = None,
               b_seeds: Optional[Sequence[Any]] = None) -> Tuple[List[int],
                                                                 List[int], str]:
    """返回 ``(a 的下标, b 的下标, pairing)``。

    * 两侧都有 ``seed_sequence`` ⇒ 以 seed 为键求交集（**顺序无关**，缺分片也安全）；
    * 否则退回按下标配对，返回 ``pairing="index"``（调用方必须把这当成告警）。
    """
    if a_seeds and b_seeds and len(a_seeds) == len(a_ranks) \
            and len(b_seeds) == len(b_ranks):
        b_index = {int(s): i for i, s in enumerate(b_seeds)}
        idx_a: List[int] = []
        idx_b: List[int] = []
        for i, s in enumerate(a_seeds):
            j = b_index.get(int(s))
            if j is not None:
                idx_a.append(i)
                idx_b.append(j)
        return idx_a, idx_b, "seed"
    n = min(len(a_ranks), len(b_ranks))
    return list(range(n)), list(range(n)), "index"


def paired_rank_diff(a: Sequence[float], b: Sequence[float],
                     rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = BOOTSTRAP_SEED,
                     *, pairing: str = "index",
                     a_seeds: Optional[Sequence[Any]] = None,
                     b_seeds: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """a、b 的逐局名次序列；返回 ``mean(a) - mean(b)`` 的配对 bootstrap 区间。

    **符号约定**：``diff < 0`` 表示 a 的名次数字更小 = **a 更好**。
    本脚本里 a = 冠军、b = 候选，所以 ``diff < 0`` 是"冠军更好"。
    """
    ia, ib, how = _pair_keys(a, b, a_seeds, b_seeds)
    diffs = [float(a[i]) - float(b[j]) for i, j in zip(ia, ib)]
    n = len(diffs)
    if n <= 0:
        return {"n": 0, "pairing": how}
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
    champ_better = sum(1 for d in diffs if d < 0)
    cand_better = sum(1 for d in diffs if d > 0)
    return {"n": n, "diff": point, "ci95": [lo, hi], "pairing": how,
            "champ_better": champ_better, "cand_better": cand_better,
            "ties": n - champ_better - cand_better,
            "a_only": len(a) - n, "b_only": len(b) - n}


def _turn_strength(m: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """``{turn: {seed: strength}}``（来自 ``strength_by_seed`` 的反转索引）。"""
    by_seed = m.get("strength_by_seed") or {}
    out: Dict[int, Dict[str, float]] = {}
    if not isinstance(by_seed, dict):
        return out
    for seed, turns in by_seed.items():
        if not isinstance(turns, dict):
            continue
        for t, v in turns.items():
            try:
                ti = int(t)
                fv = float(v)
            except (TypeError, ValueError):
                continue
            out.setdefault(ti, {})[str(seed)] = fv
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def per_turn(models: Dict[str, Dict[str, Any]], champ: str, cand: str,
             turns: Sequence[int]) -> List[Dict[str, Any]]:
    """逐回合 M1/M2：条件中位数（``curves``）+ 共同存活子集中位数（``strength_by_seed``）。"""
    sc = _turn_strength(models[champ])
    sd = _turn_strength(models[cand])
    rows = []
    for t in turns:
        row: Dict[str, Any] = {"turn": t}
        for tag, m, sm in ((champ, models[champ], sc), (cand, models[cand], sd)):
            c = (m.get("curves") or {}).get(str(t)) or {}
            row[tag] = {"n": c.get("n"), "m1": _f(c.get("strength_p50")),
                        "m2": _f(c.get("tier_mean"))}
            del sm
        # 共同存活子集：两模型都有该回合数值的 seed 交集
        sa, sb = sc.get(t) or {}, sd.get(t) or {}
        common = sorted(set(sa) & set(sb))
        if common:
            row["paired"] = {
                "n": len(common),
                champ: _median([sa[s] for s in common]),
                cand: _median([sb[s] for s in common]),
            }
        rows.append(row)
    return rows


def verdict(models: Dict[str, Dict[str, Any]], champ: str, cand: str,
            m1_tol: float, m1_from: int, rank_eps: float,
            m1_min_n: int = DEFAULT_M1_MIN_N,
            rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
    stat = (paired_rank_diff(sa, sb,
                             a_seeds=models[champ].get("seed_sequence"),
                             b_seeds=models[cand].get("seed_sequence"))
            if sa and sb else {"n": 0, "pairing": None})
    if ra is not None and rb is not None:
        reasons.append(f"mean_rank {champ}={ra:.3f} → {cand}={rb:.3f} "
                       f"(Δ={rb - ra:+.3f}，负=候选更好)")
    if stat.get("n"):
        lo, hi = stat["ci95"]
        how = stat.get("pairing")
        warn = "" if how == "seed" else "  ← 按下标配对（缺 seed_sequence，可能是错位）"
        reasons.append(
            f"配对 bootstrap（{stat['n']} 局，按 {how} 配对{warn}）："
            f"Δ=mean_rank(冠军)−mean_rank(候选) 的 95% 区间 "
            f"[{lo:+.3f}, {hi:+.3f}]（负=冠军更好）；逐局 "
            f"冠军更好 {stat['champ_better']} / 候选更好 {stat['cand_better']} / "
            f"平 {stat['ties']}；仅单侧有 seed：冠军 {stat.get('a_only')} / "
            f"候选 {stat.get('b_only')}")
        significant = (hi < 0) or (lo > 0)
    else:
        significant = False
        reasons.append("面板里没有 rank_sequence，无法做配对检验（只能看点估计）")

    better = (ra is not None and rb is not None and (ra - rb) >= rank_eps)
    worse = (ra is not None and rb is not None and (rb - ra) >= rank_eps)

    # M1 退化检查：**优先用共同存活子集**（P2-5），否则退回条件中位数并标注
    regress = []
    skipped_thin = []
    paired_used = 0
    src_rows = rows or []
    for row in src_rows:
        t = int(row["turn"])
        if t < m1_from:
            continue
        a = row[champ]["m1"]
        b = row[cand]["m1"]
        pa = (row.get("paired") or {}).get(champ)
        pb = (row.get("paired") or {}).get(cand)
        na = int(row[champ]["n"] or 0)
        nb = int(row[cand]["n"] or 0)
        if pa is not None and pb is not None:
            a, b = pa, pb
            na = nb = int((row.get("paired") or {}).get("n") or 0)
            paired_used += 1
        if min(na, nb) < m1_min_n:
            if a and b and a > 0 and (b - a) / a <= -m1_tol:
                skipped_thin.append((t, a, b, (b - a) / a, min(na, nb)))
            continue
        if a and b and a > 0:
            rel = (b - a) / a
            if rel <= -m1_tol:
                regress.append((t, a, b, rel))

    if ber := regress:
        reasons.append(
            "M1 明显退化（阈值 -%.0f%%，n≥%d；优先用共同存活子集）：" % (m1_tol * 100, m1_min_n)
            + "、".join(f"T{t} {a:.0f}→{b:.0f}（{r * 100:+.1f}%）"
                        for t, a, b, r in ber))
    if skipped_thin:
        reasons.append("（样本太薄、未计入判定的退化回合：%s）" %
                       "、".join(f"T{t} n={n}（{(b - a) / a * 100:+.1f}%）"
                                 for t, a, b, _r, n in skipped_thin))
    if src_rows:
        reasons.append(f"M1 判定口径：{paired_used}/{len([r for r in src_rows if int(r['turn']) >= m1_from])} "
                       "个回合用上了共同存活子集（其余回合缺 strength_by_seed）")

    if better and not ber:
        v = "IMPROVED" if significant else "IMPROVED_WEAK"
    elif better and ber:
        v = "SUSPICIOUS"
    else:
        v = "NO_IMPROVEMENT" if not worse else "WORSE"
    return {"verdict": v, "reasons": reasons, "paired": stat,
            "significant": significant, "m1_regressions": ber,
            "m1_paired_subset_turns": paired_used}


VERDICT_TEXT = {
    "INVALID": "整轮作废（工程问题），先修再谈",
    "IMPROVED": "名次进步（配对显著，且 M1 未退化）",
    "IMPROVED_WEAK": "名次变好但配对不显著 → 只记录（加样本再测）",
    "SUSPICIOUS": "名次变好但 M1 退化 → 先归因",
    "NO_IMPROVEMENT": "名次没有变好 → 进入归因",
    "WORSE": "名次变差 → 进入归因",
}


def render(models: Dict[str, Dict[str, Any]], champ: str, cand: str,
           rows: List[Dict[str, Any]], res: Dict[str, Any],
           protocol: Optional[str] = None) -> str:
    out: List[str] = []
    out.append(f"## 参考结论：{res['verdict']} —— {VERDICT_TEXT.get(res['verdict'], '')}")
    out.append("")
    out.append(f"> 用户口径（2026-11-17）：**判胜不靠单一指标**，主看逐回合 M1 与增长曲线；"
               f"名次只作参考。本表同时给条件中位数与共同存活子集中位数，两者冲突时"
               f"以共同存活子集为准（条件中位数含存活选择效应）。")
    out.append("")
    if protocol:
        out.append(f"- protocol_sha256：`{protocol}`")
    else:
        out.append("- **UNVERIFIED**：输入面板没有 `protocol_sha256`（旧代码产物或"
                   "`--legacy-no-protocol`）——**不得写进正式结论**，只能作参考。")
    for r in res["reasons"]:
        out.append(f"- {r}")
    out.append("")
    out.append("## 结果刻度（参考）")
    out.append("")
    out.append("| 指标 | %s | %s | Δ |" % (champ, cand))
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
    out.append("| 回合 | n冠 | n候 | M1 冠军 | M1 候选 | M1 Δ | M1(共同存活) 冠/候/Δ | n共同 | M2 冠 | M2 候 |")
    out.append("|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for row in rows:
        a, b = row[champ], row[cand]
        if a["m1"] is None or b["m1"] is None:
            continue
        d1 = (b["m1"] - a["m1"]) / a["m1"] if a["m1"] else 0.0
        pa = row.get("paired") or {}
        pcell = "-"
        if pa.get(champ) is not None and pa.get(cand) is not None:
            pr = (pa[cand] - pa[champ]) / pa[champ] if pa[champ] else 0.0
            pcell = f"{pa[champ]:.0f} / {pa[cand]:.0f} / {pr * 100:+.1f}%"
        out.append("| %d | %s | %s | %.0f | %.0f | %+.1f%% | %s | %s | %s | %s |" % (
            row["turn"], a["n"], b["n"], a["m1"], b["m1"], d1 * 100, pcell,
            pa.get("n", "-"),
            f"{a['m2']:.2f}" if a["m2"] is not None else "-",
            f"{b['m2']:.2f}" if b["m2"] is not None else "-"))
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
    ap.add_argument("--rank-eps", type=float, default=DEFAULT_RANK_EPS)
    ap.add_argument("--m1-min-n", type=int, default=DEFAULT_M1_MIN_N)
    ap.add_argument("--legacy-no-protocol", action="store_true",
                    help="允许比较**没有** protocol_sha256 的旧面板（结果标 UNVERIFIED，"
                         "不得写进正式结论）")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    models, docs = load_models(a.panels)
    for n in (a.champion, a.candidate):
        if n not in models:
            raise SystemExit(f"面板里没有模型 {n!r}；可选：{sorted(models)}")
    if a.champion == a.candidate:
        raise SystemExit("champion 与 candidate 不能是同一个模型")

    protocol: Optional[str] = None
    try:
        protocol = assert_same_protocol(*docs, where="输入面板")
    except ValueError as exc:
        if not a.legacy_no_protocol:
            raise SystemExit(
                f"{exc}\n（历史面板可用 --legacy-no-protocol 强制比较，结果会标 UNVERIFIED；"
                "新评测请重跑以带上协议指纹）")
        print(f"WARN: 协议未验证（--legacy-no-protocol）：{exc}", file=sys.stderr)
        protocol = None

    turns = sorted({int(k) for m in (models[a.champion], models[a.candidate])
                    for k in (m.get("curves") or {})})
    rows = per_turn(models, a.champion, a.candidate, turns)
    res = verdict(models, a.champion, a.candidate,
                  float(a.m1_tol), int(a.m1_from_turn), float(a.rank_eps),
                  int(a.m1_min_n), rows=rows)
    res["protocol_sha256"] = protocol
    if protocol is None:
        res["protocol_unverified"] = True
    if a.json:
        print(json.dumps({"verdict": res["verdict"],
                          "reasons": res["reasons"],
                          "paired": res.get("paired"),
                          "protocol_sha256": protocol,
                          "protocol_unverified": protocol is None,
                          "per_turn": rows}, ensure_ascii=False, indent=1))
    else:
        print(render(models, a.champion, a.candidate, rows, res, protocol))
    return 0 if res["verdict"] != "INVALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())
