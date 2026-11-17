#!/usr/bin/env python3
"""从「含多模型的子面板」重建一份合并面板（不重跑评测）。

背景（2026-11-17 的 bug）：``eval_human_gap.py --dump-rows`` 把 rows 落成
``rows.jsonl.gz``，一个进程跑 4 个模型时**后者覆盖前者**，于是 ``merge_panels.py``
只能重建出最后一个模型。但**每个子面板的 ``models[]`` 本身是完整的**（每个模型各有
自己的 ``trace``），所以直接从子面板的 models 聚合即可，无需重跑。

2026-11-17 审计修的三件事
-------------------------
1. **有效性计数跨分片求和**（P0-3）：旧代码 ``"outcome": parts[0].get("outcome")``
   只取第 0 个分片，于是 ``illegal_actions``/``truncated_games`` 这道"整轮作废"的门
   只覆盖 1/N 的局数。
2. **``survival`` schema**（P0-3）：旧代码写成 ``{"1":n,"2":n,...}``，与
   ``eval_human_gap.survival()`` 的 ``{"games","max_turn_p50","reached"}`` 不同，
   任何读 ``survival["reached"]`` 的消费者都会炸或误读。
3. **配对键与协议**（P1-6 / P0-2）：带上 ``seed_sequence``；分片间
   ``protocol_sha256`` 必须一致（旧分片没有指纹 → 明确标 ``legacy``，下游拒绝混比）。

凡是**能从分片里精确还原**的统计都精确合并（``action_mix`` 的 mean/used_rate/
mean_used、``combat_by_turn`` 的胜/平/伤害、``mechanics``/``retained_gain`` 从
逐局 trace 重算），不靠加权近似；还原不了的（``opponents_by_turn`` 的分位数）显式置
``None`` 而不是编一个数。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import assert_same_protocol
from tools.panel_integrity import validate_identity, validate_rows, validate_samples

from hsbg.evaluation.metrics import (  # noqa: E402
    _counts_all, _counts_by_turn, _gold_by_turn, _strength_by_seed, _trace,
    economy_curves, mechanics_by_turn, retained_gain_by_turn, survival,
)

OTHER_OUTCOME_KEYS = ("illegal_actions", "truncated_games", "decisions")


def _rows_from_trace(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 ``trace`` 还原成聚合函数吃的 rows（补 ``eot_attack``/``eot_board_hp``）。

    ``trace`` 只存了总和 ``strength``，这里按一半拆回 atk/hp；**只有它们的和有意义**
    （``economy_curves`` 用的就是 atk+hp），所以不会污染 M1。``board_stats`` 原样带过，
    供 ``retained_gain_by_turn`` 精确重算。
    """
    ec = []
    for r in rows:
        s = r.get("strength")
        ro = r.get("roster") or []
        ec.append({
            "seed": r.get("seed"), "turn": r.get("turn"),
            "tier": r.get("tier"), "gold": r.get("gold"),
            "health": r.get("health"), "hand_n": r.get("hand_n"),
            "actions": r.get("actions"),
            "eot_attack": int(s) // 2 if isinstance(s, (int, float)) else None,
            "eot_board_hp": int(s) - (int(s) // 2) if isinstance(s, (int, float)) else None,
            "eot_board": ro, "board": ro,
            "board_stats": r.get("board_stats"),
            "trinkets": r.get("trinkets"), "hero": r.get("hero"),
            "consecutive_repeats": r.get("repeats"),
            "gold_events": [],
        })
    return ec


def _merge_outcome(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """名次类指标按拼接后的逐局名次重算；其余计数**跨分片求和**。"""
    ranks: List[int] = []
    for m in parts:
        for x in (m.get("rank_sequence") or []):
            try:
                ranks.append(int(x))
            except (TypeError, ValueError):
                pass
    base = dict(parts[0].get("outcome") or {})
    games = sum(int((m.get("outcome") or {}).get("games") or 0) for m in parts)
    out = dict(base)
    out["games"] = games
    for key in OTHER_OUTCOME_KEYS:
        out[key] = sum(int((m.get("outcome") or {}).get(key) or 0) for m in parts)
    out["decisions_per_game"] = (round(out["decisions"] / games, 1) if games else None)
    out["truncated_rank_zero"] = sum(1 for r in ranks if r == 0)
    ok = [r for r in ranks if 1 <= r <= 8]
    out["rank_known"] = len(ok)
    if ok:
        out["mean_rank"] = round(sum(ok) / len(ok), 4)
        out["top4_rate"] = round(sum(1 for x in ok if x <= 4) / len(ok), 4)
        out["win_rate"] = round(sum(1 for x in ok if x == 1) / len(ok), 4)
    return out


def _merge_action_mix(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``action_mix_by_turn`` 的精确合并（mean / used_rate / mean_used 都可还原）。"""
    acc: Dict[int, Dict[str, Dict[str, float]]] = collections.defaultdict(
        lambda: {"games": 0.0, "sum": collections.defaultdict(float),
                 "used": collections.defaultdict(float),
                 "sum_used": collections.defaultdict(float)})
    for m in parts:
        for t, d in (m.get("action_mix_by_turn") or {}).items():
            try:
                ti = int(t)
            except (TypeError, ValueError):
                continue
            g = float((d or {}).get("games") or 0)
            if g <= 0:
                continue
            a = acc[ti]
            a["games"] += g
            for group, val in ((d or {}).get("mean") or {}).items():
                a["sum"][str(group)] += float(val) * g
            for group, rate in ((d or {}).get("used_rate") or {}).items():
                used = float(rate) * g
                a["used"][str(group)] += used
                mu = ((d or {}).get("mean_used") or {}).get(group)
                if mu is not None:
                    a["sum_used"][str(group)] += float(mu) * used
    out: Dict[str, Any] = {}
    for t, a in sorted(acc.items()):
        g = a["games"]
        groups = set(a["sum"]) | set(a["used"])
        out[str(t)] = {
            "games": int(g),
            "mean": {k: round(a["sum"].get(k, 0.0) / g, 2) for k in sorted(groups)},
            "used_rate": {k: round(a["used"].get(k, 0.0) / g, 3) for k in sorted(groups)},
            "mean_used": {k: (round(a["sum_used"][k] / a["used"][k], 2)
                              if a["used"].get(k) else 0.0) for k in sorted(groups)},
        }
    return out


def _merge_combat(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``combat_by_turn`` 的精确合并（胜/平/伤害/幽灵/注入率都按次数还原后重算）。"""
    acc: Dict[int, Dict[str, float]] = collections.defaultdict(
        lambda: collections.Counter())
    for m in parts:
        for t, d in (m.get("combat_by_turn") or {}).items():
            try:
                ti = int(t)
            except (TypeError, ValueError):
                continue
            n = float((d or {}).get("combats") or 0)
            if n <= 0:
                continue
            a = acc[ti]
            a["combats"] += n
            a["win"] += float((d or {}).get("win") or 0) * n
            a["tie"] += float((d or {}).get("tie") or 0) * n
            a["dmg"] += float((d or {}).get("damage_taken_mean") or 0) * n
            a["ghost"] += float((d or {}).get("ghost") or 0) * n
            a["mirror"] += float((d or {}).get("mirror_applied") or 0) * n
    out: Dict[str, Any] = {}
    for t, a in sorted(acc.items()):
        n = a["combats"]
        out[str(t)] = {
            "combats": int(n),
            "win": round(a["win"] / n, 3),
            "tie": round(a["tie"] / n, 3),
            "damage_taken_mean": round(a["dmg"] / n, 2),
            "ghost": round(a["ghost"] / n, 3),
            "mirror_applied": round(a["mirror"] / n, 3),
        }
    return out


def _merge_opponents(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``opponents_by_turn``：``n`` 与 ``mean`` 精确；分位数无法从分片还原 → ``None``。"""
    acc: Dict[int, Dict[str, Any]] = collections.defaultdict(
        lambda: {"n": 0, "sum": 0.0, "p50s": set()})
    for m in parts:
        for t, d in (m.get("opponents_by_turn") or {}).items():
            try:
                ti = int(t)
            except (TypeError, ValueError):
                continue
            n = int((d or {}).get("n") or 0)
            if n <= 0:
                continue
            a = acc[ti]
            a["n"] += n
            a["sum"] += n * float((d or {}).get("mean") or 0)
            if (d or {}).get("p50") is not None:
                a["p50s"].add((d or {}).get("p50"))
    out: Dict[str, Any] = {}
    for t, a in sorted(acc.items()):
        p50s = a["p50s"]
        out[str(t)] = {
            "n": a["n"],
            "mean": round(a["sum"] / a["n"], 1) if a["n"] else None,
            # 分片内分位数合并后不再成立，除非所有分片一致。
            "p50": (next(iter(p50s)) if len(p50s) == 1 else None),
            "p10": None, "p90": None,
        }
    return out


def merge(srcs, out_path: Path, *, expect_models: Optional[int] = None):
    if out_path.exists():
        raise SystemExit(f'refusing to overwrite existing panel: {out_path}')
    srcs = list(srcs)
    if not srcs or len({Path(p).resolve() for p in srcs}) != len(srcs):
        raise SystemExit('empty inputs or duplicate shard panel')
    by_model: "collections.OrderedDict[str, List[Dict[str, Any]]]" = \
        collections.OrderedDict()
    protocols: set = set()
    legacy = False
    protocol_docs = []
    model_names = None
    for p in srcs:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        sha = str((doc.get("protocol") or {}).get("protocol_sha256") or "")
        if sha:
            protocols.add(sha)
            protocol_docs.append(doc)
        else:
            legacy = True
        models = doc.get('models') or []
        names = [str(m.get('name')) for m in models]
        if len(set(names)) != len(names):
            raise SystemExit(f'{p}: duplicate model names')
        if model_names is not None and set(names) != model_names:
            raise SystemExit(f'{p}: models missing from shard')
        model_names = set(names)
        for m in models:
            by_model.setdefault(str(m.get("name")), []).append(m)
    if len(protocols) > 1:
        raise SystemExit(
            f"子面板的 protocol_sha256 不一致 {sorted(s[:12] for s in protocols)}"
            "——不同协议不能混榜（docs/EVAL_PROTOCOL.md §4 规则 5）。")
    if legacy and protocols:
        raise SystemExit('cannot mix legacy and verified protocol shards')
    if protocol_docs:
        try:
            assert_same_protocol(*protocol_docs)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    merged_models = []
    for name, parts in by_model.items():
        validate_samples(parts, name=name)
        validate_identity(parts, name=name)
        rows = [r for m in parts for r in (m.get("trace") or [])]
        validate_rows(rows, name=name)
        ranks: List[int] = []
        fb: List[Dict[str, Any]] = []
        for m in parts:
            rs = m.get("rank_sequence")
            if isinstance(rs, list):
                ranks.extend(int(x) for x in rs)
            fb.extend(m.get("final_boards") or [])
        ec = _rows_from_trace(rows)
        seeds: List[int] = []
        for m in parts:
            seq = m.get("seed_sequence")
            if not isinstance(seq, list) or len(seq) != len(m.get("rank_sequence") or []):
                seeds = []
                break
            seeds.extend(int(x) for x in seq)

        entry = {
            "name": name,
            "games": len(_strength_by_seed(ec)),
            "shards": len(parts),
            "outcome": _merge_outcome(parts),
            "rank_sequence": ranks or None,
            "seed_sequence": seeds or None,
            "curves": economy_curves(ec),
            "survival": survival(ec),
            "final_boards": fb,
            "trace": rows,
            "strength_by_seed": _strength_by_seed(ec),
            "trinkets_by_turn": _counts_by_turn(ec, "trinkets"),
            "gold_by_turn": _gold_by_turn(ec),
            "heroes": _counts_all(ec, "hero"),
            "mechanics_by_turn": mechanics_by_turn(ec),
            "retained_gain_by_turn": retained_gain_by_turn(ec),
            "action_mix_by_turn": _merge_action_mix(parts),
            "combat_by_turn": _merge_combat(parts),
            "opponents_by_turn": _merge_opponents(parts),
            "action_hist": _merge_action_hist(parts),
            "field_stats": _merge_field_stats(parts),
        }
        for key in ('checkpoint', 'checkpoint_sha256', 'arch'):
            value = next((p[key] for p in parts if p.get(key) is not None), None)
            if value is not None:
                entry[key] = value
        merged_models.append(entry)

    if expect_models is not None and len(merged_models) != int(expect_models):
        raise SystemExit(f"模型数 {len(merged_models)} != 期望 {expect_models}")

    doc = {
        "models": merged_models,
        "merged_from": [str(p) for p in srcs],
        "protocol": (dict(max((d['protocol'] for d in protocol_docs), key=len)) if protocols else
                     {"protocol_sha256": None, "legacy": True,
                      "note": "子面板来自 2026-11-17 之前的旧代码，没有协议指纹；"
                              "不得与新榜混比"}),
        "merge": {"schema": "hsai.panel.multimodel/v1", "shards": len(srcs),
                  "models": len(merged_models), "legacy_sources": legacy},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(doc, ensure_ascii=False))
    for m in merged_models:
        cv = m["curves"]
        o = m["outcome"]
        print("merged %-12s games=%-5s rows=%-6d T12=%-7s T16=%-7s rank=%-6s "
              "illegal=%-3s trunc=%-3s"
              % (m["name"], m["games"], len(m["trace"]),
                 (cv.get("12") or {}).get("strength_p50"),
                 (cv.get("16") or {}).get("strength_p50"),
                 o.get("mean_rank"), o.get("illegal_actions"),
                 o.get("truncated_games")))
    print("protocol_sha256 =", doc["protocol"].get("protocol_sha256"))
    if legacy:
        print("WARN: 有子面板没有协议指纹（旧代码产物）——该合并结果不得与新榜混比")
    print("wrote", out_path)
    return doc


def _merge_action_hist(parts: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    acc: collections.Counter = collections.Counter()
    for m in parts:
        for k, v in (m.get("action_hist") or {}).items():
            try:
                acc[str(k)] += int(v)
            except (TypeError, ValueError):
                continue
    return dict(acc.most_common()) or None


def _merge_field_stats(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc: collections.Counter = collections.Counter()
    out: Dict[str, Any] = {}
    for m in parts:
        fs = m.get("field_stats") or {}
        for k, v in fs.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                acc[k] += float(v)
            elif k == "library" and "library" not in out:
                out[k] = v
    out.update({k: (int(v) if float(v).is_integer() else round(v, 4))
                for k, v in acc.items()})
    attempt = float(acc.get("mirror_attempt") or 0.0)
    applied = float(acc.get("mirror_applied") or 0.0)
    out["mirror_apply_rate"] = round(applied / attempt, 4) if attempt else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("srcs", nargs="+", help="子面板 panel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-models", type=int, default=None)
    a = ap.parse_args(argv)
    merge(a.srcs, Path(a.out), expect_models=a.expect_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
