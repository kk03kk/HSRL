#!/usr/bin/env python3
"""从 V1.1 200 局面板产物重算动作行并生成 Markdown 报告。"""
from __future__ import annotations
import argparse, gzip, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

def atype(name: str) -> str:
    return str(name).split("(", 1)[0].split(" {", 1)[0].strip()

def pct(n: int, d: int) -> float:
    return 0.0 if not d else 100.0 * n / d

def f(x, digits=2):
    if x is None: return "—"
    if isinstance(x, float): return f"{x:.{digits}f}"
    return str(x)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run', required=True, type=Path); a=ap.parse_args()
    rp=a.run/'report'/'report.json'; report=json.loads(rp.read_text())
    games=json.loads((a.run/'report'/'game_summaries.json').read_text())
    # Recompute action rows from all action logs, independent of event action_type.
    rows={int(r['turn']): r for r in report['actions_by_turn']}
    by_action=defaultdict(Counter); by_game=defaultdict(lambda: defaultdict(Counter))
    buy_min=defaultdict(int)
    for seed in [int(g['seed']) for g in games]:
        p=a.run/'actions'/f'actions_{seed:08d}.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as fh:
            for line in fh:
                e=json.loads(line); t=int(e['turn']); typ=atype(str(e.get('action_name','')))
                by_action[t][typ]+=1
                by_game[seed][t][typ]+=1
                if typ=='Buy' and e.get('bought_minion'): buy_min[t]+=1
    for t,row in rows.items():
        reached=int(row['reached'])
        counts=by_game
        # per-game action count for reached games, including EndTurn
        game_turns=[g['seed'] for g in games if t in set(g.get('reached_turns') or [])]
        totals=[sum(by_game[s][t].values()) for s in game_turns]
        nonend=[sum(v for k,v in by_game[s][t].items() if k!='EndTurn') for s in game_turns]
        row['avg_actions']=sum(totals)/reached if reached else 0
        row['avg_non_end_actions']=sum(nonend)/reached if reached else 0
        row['max_actions']=max(totals) if totals else 0
        row['hit_80']=sum(1 for n in totals if n>=80)
        row['hit_80_rate']=pct(row['hit_80'],reached)
        row['avg_upgrade']=by_action[t]['Upgrade']/reached if reached else 0
        row['upgrade_rate']=pct(sum(1 for s in game_turns if by_game[s][t]['Upgrade']>0),reached)
        row['avg_refresh']=by_action[t]['Refresh']/reached if reached else 0
        row['avg_buy_minion']=buy_min[t]/reached if reached else 0
        row['avg_sell']=by_action[t]['Sell']/reached if reached else 0
        row['avg_move']=by_action[t]['Move']/reached if reached else 0
        row['by_type']=dict(by_action[t])
    report['actions_by_turn']=[rows[t] for t in sorted(rows)]
    (a.run/'report'/'actions_by_turn.json').write_text(json.dumps(report['actions_by_turn'],ensure_ascii=False,indent=2)+'\n')
    rp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # Markdown report.
    L=[]; add=L.append
    o=report['overall']
    add('# HSAI V1.1 vs 7 engine bots — 200 局报告\n')
    add(f"模型：`{o['model']['path']}`  \nSHA：`{o['model']['sha256']}`  \n引擎：`{o['engine']['game_version']}` / `{o['engine']['source_hash']}`  \n局数：{o['games']}；动作口径：greedy，最多 80 动作/回合，模型座位轮转。\n")
    add('## 1. 整体表现\n')
    add('| 指标 | 值 |\n|---|---:|\n')
    add(f"| 平均排名 | {f(o['mean_rank'])} |\n| 前四率 | {f(o['top4_rate']*100)}% |\n| 吃鸡率 | {f(o['win_rate']*100)}% |\n| 平均结束回合 | {f(o['mean_end_turn'])} |\n")
    add('\n| 最终排名 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    d=o['rank_distribution']; add('| 局数 | '+' | '.join(str(d.get(str(k),0)) for k in range(1,9))+' |\n')
    add('\n## 2. 逐回合动作统计\n')
    add('动作数包含 EndTurn；`80达上限` 指该回合总动作数≥80。升本/刷新/买怪/卖怪/交换位置均为模型实际执行次数；分母为该回合到达局数。\n\n')
    add('| 回合 | 到达 | 平均动作 | 非End | 最大 | 80达上限 | 平均升本 | 升本率 | 平均刷新 | 平均买随从 | 平均卖 | 平均交换 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in report['actions_by_turn']:
        add(f"| {r['turn']} | {r['reached']} | {f(r['avg_actions'])} | {f(r['avg_non_end_actions'])} | {r['max_actions']} | {r['hit_80']} ({f(r['hit_80_rate'])}%) | {f(r['avg_upgrade'])} | {f(r['upgrade_rate'])}% | {f(r['avg_refresh'])} | {f(r['avg_buy_minion'])} | {f(r['avg_sell'])} | {f(r['avg_move'])} |\n")
    add('\n## 3. 随从购买偏好\n')
    for tier in range(1,7):
        add(f'### {tier} 本\n\n| 排名 | 随从 | 购买次数 |\n|---:|---|---:|\n')
        for i,x in enumerate(report['purchase_top_by_tier'].get(str(tier),[]),1): add(f"| {i} | {x['name']} | {x['count']} |\n")
        add('\n')
    add('### 全等级合并 Top 10\n\n| 排名 | 随从 | 酒馆等级 | 购买次数 |\n|---:|---|---:|---:|\n')
    for i,x in enumerate(report['purchase_top_all'],1): add(f"| {i} | {x['name']} | {x['tier']} | {x['count']} |\n")
    add('\n## 4. 随机 10 局最终阵容\n')
    for x in report['final_sample']:
        add(f"### seed {x['seed']} / 排名 {x['rank']} / 结束回合 {x['end_turn']} / 最后EOT T{x['final_eot_turn']}\n\n")
        add('| # | 随从 | 攻击 | 生命 | 等级 | 金色 |\n|---:|---|---:|---:|---:|---|\n')
        for m in x.get('final_board') or []:
            add(f"| {m['position']+1} | {m['name']} | {m['atk']} | {m['hp']} | {m['card_tier']} | {'是' if m['golden'] else '否'} |\n")
        add('\n')
    add('## 5. 战团库\n')
    lm=report['library']; man=lm['manifest']
    add(f"路径：`{lm['path']}/train`  \n帧数：{lm['frames']}；跳过空场：{lm['skipped']}  \nmanifest：`driver={man['driver']}`，games={man['games']}，last_turn={man['last_turn']}  \n验证：{man['validation']}；严格镜像：{man['strict_audit']}\n")
    add('\n## 6. 逐回合酒馆等级分布\n')
    add('`当前` 为 EOT 时精确等级占比；`累计≥2..6` 为 EOT 样本中已到达该等级的比例。\n\n')
    add('| 回合 | EOT | 1本 | 2本 | 3本 | 4本 | 5本 | 6本 | 累计≥2 | 累计≥3 | 累计≥4 | 累计≥5 | 累计≥6 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in report['tier_by_turn']:
        td=r['tier_distribution']; add(f"| {r['turn']} | {r['eot']} | "+' | '.join(f(td[str(k)]) for k in range(1,7))+' | '+' | '.join(f(r[f'reach_tier{k}']) for k in range(2,7))+' |\n')
    add('\n## 7. 逐回合剩余金币\n\n| 回合 | 到达 | EOT | 平均剩余金币 | 中位 | 金币>0比例 |\n|---:|---:|---:|---:|---:|---:|\n')
    for r in report['turn_stats']: add(f"| {r['turn']} | {r['reached']} | {r['eot']} | {f(r['avg_gold'])} | {f(r['median_gold'])} | {f(r['gold_positive_rate'])}% |\n")
    add('\n## 8. 逐回合战团强度\n\n| 回合 | EOT | 平均随从数 | 平均总攻血 | P50总攻血 | P90总攻血 |\n|---:|---:|---:|---:|---:|---:|\n')
    for r in report['turn_stats']: add(f"| {r['turn']} | {r['eot']} | {f(r['avg_board_size'])} | {f(r['avg_attack_health_total'])} | {f(r['p50_attack_health_total'])} | {f(r['p90_attack_health_total'])} |\n")
    add('\n## 9. 逐回合存活数\n\n| 回合 | 到达局数 |\n|---:|---:|\n')
    for r in report['turn_stats']: add(f"| {r['turn']} | {r['reached']} |\n")
    (a.run/'report'/'report.md').write_text(''.join(L))
    print((a.run/'report'/'report.md'))
if __name__=='__main__': main()
