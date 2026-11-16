#!/usr/bin/env python3
"""把 N 份「分片评测」的 panel.json 合并成一份完整 panel。

为什么需要：``workflows/eval_human_gap.py`` 是**串行**的（一个进程、一局一局跑），
512 局要 ~11 分钟。要提速最省事的办法是**按种子分片并行**：每个分片跑
``--games K --seed (base + i*K)``，各自写一份 panel，再合并。

要求：``eval_human_gap`` 已经把**逐局逐回合**的明细都写进面板
（``trace`` / ``strength_by_seed`` / ``boards_by_turn`` / ``final_boards`` /
``rank_sequence``），所以合并可以做到**无损**：拼接逐局部分，从 ``trace``
重算所有聚合曲线（``curves``），其余聚合块按样本量加权或直接拼接。

用法：
    python tools/merge_panels.py --out merged.json shard0/panel.json shard1/panel.json ...

注意：分片必须**用同一个 base seed、互不重叠的连续种子块**，否则不同模型之间
就不是同一批种子了（配对 bootstrap 会失效）。
"""
import argparse
import collections
import json
from pathlib import Path


def _pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def merge_curves(traces):
    """从合并后的 trace 重算逐回合曲线（口径与 eval_human_gap.economy_curves 一致）。"""
    by_turn = collections.defaultdict(lambda: collections.defaultdict(list))
    tier_hist = collections.defaultdict(collections.Counter)
    for r in traces:
        try:
            t = int(r["turn"])
        except (KeyError, TypeError, ValueError):
            continue
        b = by_turn[t]
        if isinstance(r.get("tier"), int):
            b["tier"].append(float(r["tier"]))
            tier_hist[t][int(r["tier"])] += 1
        if isinstance(r.get("gold"), int):
            b["gold"].append(float(r["gold"]))
        if isinstance(r.get("hand_n"), int):
            b["hand"].append(float(r["hand_n"]))
        if isinstance(r.get("actions"), int):
            b["actions"].append(float(r["actions"]))
        s = r.get("strength")
        if isinstance(s, (int, float)):
            b["strength"].append(float(s))
        ro = r.get("roster") or r.get("board_stats") or []
        if isinstance(ro, list):
            b["board_n"].append(float(len(ro)))
        elif isinstance(ro, dict):
            b["board_n"].append(float(len(ro)))

    out = {}
    for t, b in by_turn.items():
        if len(b["tier"]) < 10:
            continue
        hist = tier_hist[t]
        n = len(b["tier"])
        st = sorted(b["strength"], reverse=True)
        out[str(t)] = {
            "n": n,
            "tier_mean": _r(sum(b["tier"]) / n, 2),
            "tier_mode": hist.most_common(1)[0][0],
            "tier6_share": _r(sum(1 for x in b["tier"] if x >= 6) / n, 3),
            "tier5plus_share": _r(sum(1 for x in b["tier"] if x >= 5) / n, 3),
            "tier_hist": dict(sorted(hist.items())),
            "gold_p50": _r(_pct(b["gold"], 0.5)),
            "hand_p50": _r(_pct(b["hand"], 0.5)),
            "strength_p50": _r(_pct(st, 0.5)),
            "strength_p90": _r(_pct(st, 0.90)),
            "strength_p99": _r(_pct(st, 0.99)),
            "strength_max": _r(st[0]) if st else None,
            "strength_min": _r(st[-1]) if st else None,
            "strength_mean": _r(sum(st) / len(st)) if st else None,
            "strength_all": [_r(x) for x in st],
            "strength_top10": [_r(x) for x in st[:10]],
            "board_n_p50": _r(_pct(b["board_n"], 0.5)),
            "actions_mean": _r(sum(b["actions"]) / len(b["actions"])) if b["actions"] else None,
        }
    return out


def merge(list_of_models, out_path):
    models = []
    for p in list_of_models:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for m in d.get("models") or []:
            m["_src"] = str(p)
            models.append(m)
    by_name = collections.OrderedDict()
    for m in models:
        by_name.setdefault(m.get("name") or "model", []).append(m)

    merged_models = []
    for name, ms in by_name.items():
        trace = []
        sbs = {}
        bbt = {}
        fb = []
        ranks = []
        for m in ms:
            trace.extend(m.get("trace") or [])
            for k, v in (m.get("strength_by_seed") or {}).items():
                sbs[str(k)] = v
            for k, v in (m.get("boards_by_turn") or {}).items():
                bbt[str(k)] = v
            fb.extend(m.get("final_boards") or [])
            r = m.get("rank_sequence")
            if isinstance(r, list):
                ranks.extend(r)
        out = dict(ms[0])
        out.pop("_src", None)
        out["trace"] = trace
        out["strength_by_seed"] = sbs
        out["boards_by_turn"] = bbt
        out["final_boards"] = fb
        if ranks:
            out["rank_sequence"] = ranks
        out["curves"] = merge_curves(trace)
        out["shards"] = [m.get("_src") for m in ms]
        out["games"] = len(sbs) or out.get("games")
        merged_models.append(out)

    doc = {"models": merged_models, "merged_from": [str(p) for p in list_of_models]}
    Path(out_path).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    for m in merged_models:
        cv = m["curves"]
        print("merged %-14s games=%-5s turns=%s  T12=%s  T16=%s"
              % (m.get("name"), m.get("games"), sorted(cv, key=lambda x: int(x))[-1:],
                 (cv.get("12") or {}).get("strength_p50"), (cv.get("16") or {}).get("strength_p50")))
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a.shards, a.out)


if __name__ == "__main__":
    main()
