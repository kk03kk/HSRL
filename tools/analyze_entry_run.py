#!/usr/bin/env python3
"""读一次「中后期入场」实验的 history.jsonl，输出学习曲线与机制健康度。

它只读**已经落盘的**记录，不重跑任何东西；所以段与段之间随时可以看。

关键指标（都是实测口径，不是代理）：

1. ``entry_attempt/applied/rejected``：入场注入是否真的生效；``rejected_by_reason``
   给出为什么没成（``not_alive`` 正常、``pending_choice`` 偶发、``sample_error`` 才是 bug）。
2. ``learner_final_str_by_turn``：learner 终局板面 Σ(atk+hp) 按终局回合分组求均值。
3. **``ratio``**：同一终局回合上，learner 平均板面 ÷ 真人 ≥8k MMR 吃鸡局 p50。
   这是本次实验最主要的「有没有变强」信号 —— 不是自比的，是对着真人曲线比的。
4. ``human_board_applied`` / ``mirror_applied``：中后期的对手到底是谁。
5. ``action_hist`` 的 sell/refresh 占比：换挡是否发生（辅证，不作唯一判据）。

用法::

    python tools/analyze_entry_run.py --run artifacts/runs/entry-s1-20261112 \
        --out artifacts/runs/entry-s1-20261112/curve.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BANK = os.path.join(ROOT, "data/real_library/firestone-bank-1000/summary-m8k.json")


def _load_history(run_dir):
    for cand in (os.path.join(run_dir, "history.jsonl"),
                 os.path.join(run_dir, "train", "history.jsonl"),
                 run_dir):
        if os.path.isfile(cand):
            rows = []
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return cand, rows
    raise SystemExit(f"找不到 history.jsonl：{run_dir}")


def _sec(rec):
    """注入/遥测键可能在记录顶层，也可能在 ``collect`` 段；两处都看。"""
    out = dict(rec)
    inner = rec.get("collect")
    if isinstance(inner, dict):
        for k, v in inner.items():
            out.setdefault(k, v)
    return out


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="运行目录（含 history.jsonl 或 train/history.jsonl）")
    ap.add_argument("--bank-summary", default=DEFAULT_BANK,
                    help="真人对局库 summary.json（取 per_turn[*].p50）")
    ap.add_argument("--out", default="")
    ap.add_argument("--tail", type=int, default=0, help="只看最后 N 轮（0=全部）")
    args = ap.parse_args(argv)

    human = {}
    if os.path.isfile(args.bank_summary):
        with open(args.bank_summary, "r", encoding="utf-8") as f:
            summ = json.load(f)
        human = {int(t): int(v.get("p50") or 0)
                 for t, v in (summ.get("per_turn") or {}).items()}
    else:
        print(f"[warn] 没有真人库摘要：{args.bank_summary}（ratio 将为空）", file=sys.stderr)

    path, rows = _load_history(args.run)
    if args.tail and len(rows) > args.tail:
        rows = rows[-args.tail:]
    print(f"[history] {path}  rows={len(rows)}", flush=True)

    curve = []
    for i, rec in enumerate(rows):
        s = _sec(rec)
        sum_t = s.get("learner_final_str_sum_by_turn") or {}
        n_t = s.get("learner_final_str_n_by_turn") or {}
        by_turn, ratios, wsum, wtot = {}, {}, 0.0, 0.0
        for tstr, tot in sum_t.items():
            n = int(n_t.get(tstr, 0) or 0)
            if n <= 0:
                continue
            mean = float(tot) / n
            by_turn[int(tstr)] = round(mean, 1)
            hp = human.get(int(tstr))
            if hp:
                r = mean / float(hp)
                ratios[int(tstr)] = round(r, 3)
                wsum += r * n
                wtot += n
        # 加权（按局数）总比值：跨回合也按局数加权，避免「后期样本少」把曲线拉高。
        ratio_all = round(wsum / wtot, 3) if wtot else None
        ah = s.get("action_hist") or {}
        tot_acts = sum(int(v) for v in ah.values()) or 0
        sell = int(ah.get("sell", 0))
        refresh = int(ah.get("refresh", 0))
        curve.append({
            "iter": rec.get("iter", i),
            "cum_decisions": rec.get("cum_decisions"),
            "decisions": s.get("decisions"),
            "games": s.get("games"),
            "records": s.get("records"),
            "entry_attempt": s.get("entry_attempt"),
            "entry_applied": s.get("entry_applied"),
            "entry_rejected": s.get("entry_rejected"),
            "entry_rejected_by_reason": s.get("entry_rejected_by_reason"),
            "entry_hand_applied": s.get("entry_hand_applied"),
            "entry_turns_hist": s.get("entry_turns_hist"),
            "human_board_applied": s.get("human_board_applied"),
            "human_board_rejected": s.get("human_board_rejected"),
            "mirror_applied": s.get("mirror_applied"),
            "learner_final_str_mean": s.get("learner_final_str_mean"),
            "learner_final_str_p50": s.get("learner_final_str_p50"),
            "learner_final_str_p90": s.get("learner_final_str_p90"),
            "learner_final_str_by_turn": by_turn,
            "human_p50_by_turn": {t: human[t] for t in by_turn if t in human},
            "ratio_by_turn": ratios,
            "ratio_vs_human_all": ratio_all,
            "sell_share": round(sell / tot_acts, 4) if tot_acts else None,
            "refresh_share": round(refresh / tot_acts, 4) if tot_acts else None,
        })

    # 控制台表：只打最关键的几列。
    print(f"{'iter':>6} {'dec':>9} {'ent_ok/att':>11} {'hb_ok':>7} "
          f"{'str_mean':>9} {'ratio':>6} {'sell%':>6} {'refr%':>6}")
    for c in curve:
        ea = c.get("entry_applied")
        att = c.get("entry_attempt")
        print(f"{str(c['iter']):>6} {str(c.get('decisions')):>9} "
              f"{str(ea) + '/' + str(att):>11} {str(c.get('human_board_applied')):>7} "
              f"{str(c.get('learner_final_str_mean')):>9} "
              f"{str(c.get('ratio_vs_human_all')):>6} "
              f"{('%.2f' % (100 * c['sell_share']) if c.get('sell_share') is not None else '-'):>6} "
              f"{('%.2f' % (100 * c['refresh_share']) if c.get('refresh_share') is not None else '-'):>6}",
              flush=True)
    if curve:
        f, l = curve[0], curve[-1]
        print("\n[first]", json.dumps({k: f[k] for k in
              ("ratio_by_turn", "learner_final_str_by_turn", "entry_rejected_by_reason")},
              ensure_ascii=False), flush=True)
        print("[last ]", json.dumps({k: l[k] for k in
              ("ratio_by_turn", "learner_final_str_by_turn", "entry_rejected_by_reason")},
              ensure_ascii=False), flush=True)
        r0, r1 = f.get("ratio_vs_human_all"), l.get("ratio_vs_human_all")
        if r0 and r1:
            print(f"\n[ratio] {r0} -> {r1}  ({100 * (r1 / r0 - 1):+.1f}%)", flush=True)
    out = {"schema": "hsai.entry_curve/v1", "history": path,
           "bank_summary": os.path.abspath(args.bank_summary),
           "human_p50": {str(k): v for k, v in sorted(human.items())},
           "curve": curve}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
