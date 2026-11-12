#!/usr/bin/env python3
"""四件套验收：把一个 ``hsai.human_gap_panel/v1`` 面板变成可判定的通过/否决清单。

为什么需要它
------------
"模型学会了阵容"不能只看一个数：板面数值变大可能只是拖到更晚才死，卖牌变多也可能是
单纯变菜。所以每次验收固定四项，逐项给通过线，并明确写出判定依据：

1. **排名**：``mean_rank`` / ``top4`` / ``win`` / ``survive>t12`` / ``survive>t14``；
2. **板面强度对真人曲线**：逐回合 ``strength_p50 / 真人 p50``，固定回合比（T10–T16），
   不用被"终局回合分布"污染的加权总比值；
3. **核心卡覆盖**：``config/strategy_cards.json`` 的 S/A/B 核心卡命中率（复用
   ``tools/analyze_composition_coverage.py`` 的口径）；
4. **前期回归**：T2–T7 的板面与 tier 相对 ``--base`` 不能退。

另外单独盯着 **升本节奏**（T8/T10/T12 的平均 tier）：2026-11-12 的证据显示入场课程
真正的病灶就在这里（见 docs/analysis/TIER_TEMPO_20261112.md），它不是"另一个指标"，
而是解释 1/2 两项为什么会同时变差的机制变量。

用法
----
    python tools/accept_gap_panel.py \
        --panel artifacts/runs/gap-g23-20261112/panel.json \
        --base base --candidate g2 --candidate g3 \
        --out artifacts/runs/gap-g23-20261112/verdict.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

DEFAULT_HUMAN = os.path.join(ROOT, "artifacts/runs/human-baseline-20261112/baseline.json")
DEFAULT_HUMAN_LIB = os.path.join(ROOT, "data/real_library/firestone-20261112-1000/library.jsonl")
DEFAULT_STRATEGY = os.path.join(ROOT, "config/strategy_cards.json")

EARLY_TURNS = (2, 3, 4, 5, 6, 7)
LATE_TURNS = (10, 11, 12, 13, 14, 15, 16)
TEMPO_TURNS = (8, 10, 12)


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _curves(model: Dict[str, Any]) -> Dict[str, Any]:
    return model.get("curves") or {}


def _strength(model: Dict[str, Any], turn: int) -> Optional[float]:
    return _num((_curves(model).get(str(turn)) or {}).get("strength_p50"))


def _tier(model: Dict[str, Any], turn: int) -> Optional[float]:
    return _num((_curves(model).get(str(turn)) or {}).get("tier_mean"))


def _rel(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """(a-b)/b，b 为空或 0 时返回 None。"""
    if a is None or b is None or b == 0:
        return None
    return a / b - 1.0


def _pick(mapping: Any, key: Any) -> Optional[float]:
    """从 dict 里取值，兼容 int/str 键（JSON 往返会把 int 键变成 str）。"""
    if not isinstance(mapping, dict):
        return None
    for k in (key, str(key)):
        if k in mapping:
            return _num(mapping[k])
    return None


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100.0 * x:+.1f}%"


def rank_block(model: Dict[str, Any]) -> Dict[str, Any]:
    o = model.get("outcome") or {}
    sv = model.get("survival") or {}
    reached = sv.get("reached") or {}
    return {
        "games": sv.get("games") or o.get("games"),
        "mean_rank": _num(o.get("mean_rank")),
        "top4_rate": _num(o.get("top4_rate")),
        "win_rate": _num(o.get("win_rate")),
        "survive_t10": _num(reached.get("10")),
        "survive_t12": _num(reached.get("12")),
        "survive_t14": _num(reached.get("14")),
        "max_turn_p50": _num(sv.get("max_turn_p50")),
        "illegal_actions": _num(o.get("illegal_actions")),
    }


def board_block(model: Dict[str, Any], human: Dict[str, Any]) -> Dict[str, Any]:
    strengths = {t: _strength(model, t) for t in
                 sorted(set(EARLY_TURNS) | set(LATE_TURNS))}
    return {"strength_p50": {str(t): v for t, v in strengths.items() if v is not None}}


def tempo_block(model: Dict[str, Any]) -> Dict[str, Any]:
    return {"tier_mean": {str(t): _tier(model, t) for t in TEMPO_TURNS}}


def coverage_block(panel_models: Dict[str, Dict[str, Any]], human_lib: str,
                   strategy: str, min_turn: int) -> Dict[str, Any]:
    """核心卡覆盖：直接复用 analyze_composition_coverage 的 summarise 口径。"""
    import analyze_composition_coverage as cov  # noqa: E402  (tools/ 已在 sys.path)

    from pathlib import Path
    comps = cov.load_strategy(Path(strategy))
    out: Dict[str, Any] = {}
    hp = Path(human_lib)
    if hp.is_file():
        out["human-winners"] = cov.summarise(
            cov.human_final_boards(hp, min_turn), comps, "human-winners")
    for name, model in panel_models.items():
        boards = [b for b in model.get("final_boards") or []
                  if int(b.get("turn", 0)) >= min_turn]
        if boards:
            out[name] = cov.summarise(boards, comps, name)
    return out


def judge(cand: Dict[str, Any], base: Dict[str, Any],
          human_p50: Dict[int, float]) -> Dict[str, Any]:
    """逐项判定；每一项都给出数值依据，不做"综合分"。"""
    cr, br = cand["rank"], base["rank"]
    cs, bs = cand["board"]["strength_p50"], base["board"]["strength_p50"]
    ct, bt = cand["tempo"]["tier_mean"], base["tempo"]["tier_mean"]
    checks: Dict[str, Any] = {}

    # 1) 排名：不能更差；并且活到 T14 的比例不能掉 >2 个点。
    d_rank = (_num(cr["mean_rank"]) or 9) - (_num(br["mean_rank"]) or 9)
    d_surv = (_num(cr["survive_t14"]) or 0) - (_num(br["survive_t14"]) or 0)
    checks["rank"] = {
        "pass": bool(d_rank <= 1e-9 and d_surv >= -0.02),
        "mean_rank": cr["mean_rank"], "base_mean_rank": br["mean_rank"],
        "delta_mean_rank": round(d_rank, 4),
        "survive_t14": cr["survive_t14"], "base_survive_t14": br["survive_t14"],
        "delta_survive_t14": round(d_surv, 4),
    }

    # 2) 板面：固定回合 T12/T14 不得低于 base，且至少一项相对 base 提升 >= 5%。
    per_turn = {}
    regressed = []
    improved = 0.0
    for t in LATE_TURNS:
        rel = _rel(_num(cs.get(str(t))), _num(bs.get(str(t))))
        if rel is not None:
            per_turn[str(t)] = round(rel, 4)
            if rel < -0.02:
                regressed.append(t)
            improved = max(improved, rel)
    checks["board"] = {
        "pass": bool(not regressed and improved >= 0.05),
        "rel_vs_base": per_turn, "regressed_turns": regressed,
        "best_improvement": round(improved, 4),
    }

    # 3) 核心卡覆盖：≥1 S 与 ≥2 核心卡都不能退。
    cc = cand.get("coverage") or {}
    bc = base.get("coverage") or {}
    if cc and bc:
        cs_s = _pick(cc.get("any_core_by_tier"), "S")
        bs_s = _pick(bc.get("any_core_by_tier"), "S")
        cs_2 = _pick(cc.get("assembled_at_least"), 2)
        bs_2 = _pick(bc.get("assembled_at_least"), 2)
        checks["coverage"] = {
            "pass": bool(None not in (cs_s, bs_s, cs_2, bs_2)
                         and cs_s >= bs_s - 1e-9 and cs_2 >= bs_2 - 1e-9),
            "any_S": cs_s, "base_any_S": bs_s,
            "two_core": cs_2, "base_two_core": bs_2,
        }
    else:
        checks["coverage"] = {"pass": None, "reason": "面板缺 final_boards"}

    # 4) 前期回归：T2–T7 逐回合不得低于 base 超 5%。
    early = {}
    early_bad = []
    for t in EARLY_TURNS:
        rel = _rel(_num(cs.get(str(t))), _num(bs.get(str(t))))
        if rel is not None:
            early[str(t)] = round(rel, 4)
            if rel < -0.05:
                early_bad.append(t)
    checks["early_game"] = {
        "pass": bool(not early_bad), "rel_vs_base": early,
        "regressed_turns": early_bad,
    }

    # 机制观察项（不计入通过线，但要一起看）：升本节奏。
    tempo = {}
    for t in TEMPO_TURNS:
        a, b = _num(ct.get(str(t))), _num(bt.get(str(t)))
        if a is not None and b is not None:
            tempo[str(t)] = round(a - b, 3)
    checks["tempo_delta_vs_base"] = tempo

    passed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("pass") is True]
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("pass") is False]
    checks["overall"] = {
        "pass": bool(not failed),
        "passed": passed, "failed": failed,
    }
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="四件套验收")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--human-baseline", default=DEFAULT_HUMAN)
    ap.add_argument("--human-lib", default=DEFAULT_HUMAN_LIB)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--coverage-min-turn", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    panel = _load(args.panel)
    models = {m.get("name"): m for m in panel.get("models") or []}
    if args.base not in models:
        raise SystemExit(f"面板里没有 base={args.base!r}；有 {sorted(models)}")
    cands = args.candidate or [n for n in models if n != args.base]

    human: Dict[str, Any] = {}
    human_p50: Dict[int, float] = {}
    if os.path.isfile(args.human_baseline):
        human = _load(args.human_baseline)
        human_p50 = {int(t): float(v.get("p50") or 0)
                     for t, v in (human.get("strength_by_turn") or {}).items()}
    else:
        print(f"[warn] 没有真人基线 {args.human_baseline}，比值列将为空", file=sys.stderr)

    coverage = coverage_block(models, args.human_lib, args.strategy,
                              args.coverage_min_turn)

    def pack(name: str) -> Dict[str, Any]:
        m = models[name]
        return {"name": name, "rank": rank_block(m), "board": board_block(m, human),
                "tempo": tempo_block(m), "coverage": coverage.get(name)}

    base = pack(args.base)
    verdicts: List[Dict[str, Any]] = []
    for name in cands:
        cand = pack(name)
        v = judge(cand, base, human_p50)
        verdicts.append({"candidate": name, "checks": v})

    # ---- 打印 ----
    print(f"PANEL {args.panel}")
    print(f"base = {args.base}  candidates = {cands}")
    hdr = f"{'':<12}" + "".join(f"{n:>14}" for n in [args.base] + cands)
    print()
    print(hdr)
    rows = [("games", lambda p: p["rank"]["games"], 0),
            ("mean_rank", lambda p: p["rank"]["mean_rank"], 3),
            ("top4", lambda p: p["rank"]["top4_rate"], 3),
            ("win", lambda p: p["rank"]["win_rate"], 3),
            ("survive>t12", lambda p: p["rank"]["survive_t12"], 3),
            ("survive>t14", lambda p: p["rank"]["survive_t14"], 3),
            ("max_turn_p50", lambda p: p["rank"]["max_turn_p50"], 1)]
    packs = {args.base: base}
    for v in verdicts:
        packs[v["candidate"]] = pack(v["candidate"])
    for label, fn, nd in rows:
        line = f"{label:<12}"
        for n in [args.base] + cands:
            v = fn(packs[n])
            line += f"{('—' if v is None else f'{v:.{nd}f}'):>14}"
        print(line)

    print()
    print("板面 p50（比值 = 模型 / 真人 p50）")
    print(f"{'turn':>4} {'human':>8}" + "".join(f"{n:>18}" for n in [args.base] + cands))
    for t in sorted(set(EARLY_TURNS) | set(LATE_TURNS)):
        hp = human_p50.get(t)
        line = f"{t:>4} {('—' if hp is None else f'{hp:.0f}'):>8}"
        for n in [args.base] + cands:
            s = _num(packs[n]["board"]["strength_p50"].get(str(t)))
            r = (s / hp) if (s and hp) else None
            line += f"{('—' if s is None else f'{s:.0f}'):>10}{('—' if r is None else f'{r:.3f}'):>8}"
        print(line)

    print()
    print("升本节奏（平均 tier）")
    print(f"{'turn':>4} {'human':>8}" + "".join(f"{n:>14}" for n in [args.base] + cands))
    ht = {int(t): _num((human.get("tier_by_turn") or {}).get(str(t), {}).get("mean"))
          for t in TEMPO_TURNS}
    for t in TEMPO_TURNS:
        line = f"{t:>4} {('—' if ht.get(t) is None else f'{ht[t]:.2f}'):>8}"
        for n in [args.base] + cands:
            v = _num(packs[n]["tempo"]["tier_mean"].get(str(t)))
            line += f"{('—' if v is None else f'{v:.2f}'):>14}"
        print(line)

    if coverage:
        print()
        print("核心卡覆盖（终局板面，>= %d 回合）" % args.coverage_min_turn)
        print(f"{'side':<16} {'games':>6} {'>=1 S':>7} {'>=1 A':>7} {'>=2 core':>9} {'>=3 core':>9}")
        for name, s in coverage.items():
            a = s.get("any_core_by_tier") or {}
            asm = s.get("assembled_at_least") or {}
            print(f"{name:<16} {s.get('games', 0):>6} "
                  f"{a.get('S', 0):>7} {a.get('A', 0):>7} "
                  f"{asm.get(2, 0):>9} {asm.get(3, 0):>9}")

    print()
    for v in verdicts:
        c = v["checks"]
        print(f"== {v['candidate']} vs {args.base} ==")
        for key in ("rank", "board", "coverage", "early_game"):
            d = c[key]
            flag = {True: "PASS", False: "FAIL", None: "SKIP"}[d.get("pass")]
            detail = {k: val for k, val in d.items() if k != "pass"}
            print(f"  {key:<11} {flag:<5} {json.dumps(detail, ensure_ascii=False)}")
        print(f"  tempoΔ(base) {json.dumps(c['tempo_delta_vs_base'], ensure_ascii=False)}")
        o = c["overall"]
        print(f"  => {'PASS' if o['pass'] else 'FAIL'}"
              f"{('  failed=' + ','.join(o['failed'])) if o['failed'] else ''}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"schema": "hsai.acceptance/v1", "panel": args.panel,
                       "base": base, "verdicts": verdicts,
                       "coverage": coverage}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
