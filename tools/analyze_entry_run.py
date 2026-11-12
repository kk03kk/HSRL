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
            # 奖励机制健康度：配了但没生效是历史踩过的坑，所以逐轮记账。
            "tier_checks": s.get("tier_checks"),
            "tier_hits": s.get("tier_hits"),
            "tier_misses": s.get("tier_misses"),
            "tier_hit_rate": s.get("tier_hit_rate"),
            "tier_target_reward_sum": s.get("tier_target_reward_sum"),
            "tier_gate_k": s.get("tier_gate_k"),
            "tier_gate_blocked": s.get("tier_gate_blocked"),
            "board_target_reward_sum": s.get("board_target_reward_sum"),
            "board_target_str_delta_mean": s.get("board_target_str_delta_mean"),
            # 阵容覆盖奖励（comp_target）：四件套里"核心卡覆盖"的训练侧对应物。
            "comp_target_turns": s.get("comp_target_turns"),
            "comp_target_best_mean": s.get("comp_target_best_mean"),
            "comp_target_s_turns": s.get("comp_target_s_turns"),
            "comp_target_reward_sum": s.get("comp_target_reward_sum"),
        })

    # 控制台表：只打最关键的几列。
    print(f"{'iter':>6} {'dec':>9} {'ent_ok/att':>11} {'hb_ok':>7} "
          f"{'str_mean':>9} {'ratio':>6} {'sell%':>6} {'refr%':>6} "
          f"{'t_hit%':>7} {'t_rw':>7} {'bt_rw':>7} {'c_best':>7} {'cS%':>5} {'c_rw':>6} "
          f"{'g_k':>5} {'gate%':>6}")
    for c in curve:
        ea = c.get("entry_applied")
        att = c.get("entry_attempt")
        thr = c.get("tier_hit_rate")
        print(f"{str(c['iter']):>6} {str(c.get('decisions')):>9} "
              f"{str(ea) + '/' + str(att):>11} {str(c.get('human_board_applied')):>7} "
              f"{str(c.get('learner_final_str_mean')):>9} "
              f"{str(c.get('ratio_vs_human_all')):>6} "
              f"{('%.2f' % (100 * c['sell_share']) if c.get('sell_share') is not None else '-'):>6} "
              f"{('%.2f' % (100 * c['refresh_share']) if c.get('refresh_share') is not None else '-'):>6} "
              f"{('%.1f' % (100 * float(thr)) if thr is not None else '-'):>7} "
              f"{(('%.1f' % float(c['tier_target_reward_sum']))
                 if c.get('tier_target_reward_sum') is not None else '-'):>7} "
              f"{(('%.1f' % float(c['board_target_reward_sum']))
                 if c.get('board_target_reward_sum') is not None else '-'):>7} "
              f"{(('%.3f' % float(c['comp_target_best_mean']))
                 if c.get('comp_target_best_mean') is not None else '-'):>7} "
              f"{(('%.1f' % (100.0 * float(c['comp_target_s_turns'])
                             / max(1.0, float(c['comp_target_turns'] or 1))))
                 if c.get('comp_target_turns') else '-'):>5} "
              f"{(('%.0f' % float(c['comp_target_reward_sum']))
                 if c.get('comp_target_reward_sum') is not None else '-'):>6} "
              f"{(('%.2f' % float(c['tier_gate_k']))
                 if c.get('tier_gate_k') else '-'):>5} "
              f"{(('%.1f' % (100.0 * float(c['tier_gate_blocked'])
                             / max(1.0, float(c['tier_gate_blocked'] or 0)
                                   + float(c.get('tier_checks') or 0))))
                 if c.get('tier_gate_blocked') else '-'):>6}",
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
        # 逐回合**绝对**板面序列：``ratio_vs_human_all`` 会被"终局回合分布"污染
        # （游戏结束得早，比值天然更高），所以真正要看的趋势是固定回合上的绝对值。
        print("\n[per-turn absolute learner board strength]", flush=True)
        for t in range(5, 17):
            vals = [c["learner_final_str_by_turn"].get(t) for c in curve]
            vals = [v for v in vals if v]
            if not vals:
                continue
            half = max(1, len(vals) // 2)
            print(f"  T{t}: n={len(vals)} first_half={sum(vals[:half]) / half:.0f} "
                  f"last_half={sum(vals[-half:]) / half:.0f} "
                  f"({100 * (sum(vals[-half:]) / sum(vals[:half]) - 1):+.1f}%) "
                  f"human_p50={human.get(t)}", flush=True)
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
