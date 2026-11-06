#!/usr/bin/env python3
"""汇总新库覆盖与真实对位暴露；给出建议，不把棋力成绩变成止训条件。"""
from collections import defaultdict, Counter
from pathlib import Path
from workflows.run_learning_experiment import read


def summarize(root):
    root = Path(root)
    calibration = read(root/'calibrated/train/calibration.json')
    exposure = {}
    for path in sorted((root/'evaluations').glob('*.json')):
        if not path.stem.endswith('-b10m'):
            continue  # 主报告只保留主模型暴露；其余完整明细仍在独立评测文件
        buckets = defaultdict(list)
        for seat in read(path)['per_seat']:
            for row in seat.get('combat_exposure', []):
                buckets[row['turn']].append(row)
        exposure[path.stem] = {str(turn):dict(combats=len(rows),
            injected=sum(r['mirror_applied'] for r in rows), ghosts=sum(r['ghost'] for r in rows),
            win_rate=sum(r['win'] for r in rows)/len(rows),
            mean_damage_taken=sum(r['damage_taken'] for r in rows)/len(rows),
            injected_sources=dict(Counter(r.get('source') or 'unlabelled' for r in rows if r['mirror_applied'])),
            injected_mixture=dict(Counter(r.get('mixture_source') or 'unlabelled' for r in rows if r['mirror_applied'])),
            independent_seeds=len({r['seed'] for r in rows})) for turn, rows in sorted(buckets.items())}
    missing = [int(t) for t in range(8, 16)
               if not calibration['turns'].get(str(t), {}).get('target_reachable')]
    source_comparison = {}
    for name, turns in calibration['source_statistics'].items():
        source_comparison[name] = {}
        for turn, stats in turns.items():
            scores = [r['combat']['score'] for r in calibration['scored_frames']
                      if r['source'] == name and str(r['turn']) == turn and r.get('combat')]
            source_comparison[name][turn] = dict(source_games=stats['source_games'], frames=stats['n'],
                attack=stats['attack'], health=stats['health'], combined=stats['combined'],
                unique_compositions=stats['unique_compositions'], unique_cards=stats['unique_cards'],
                candidate_combat_score=sum(scores)/len(scores) if scores else None)
    return dict(schema='hsrl.opponent_research_result/v1',
        checkpoint_benchmark=read(root/'checkpoint-benchmark.json'),
        source_comparison=source_comparison,
        calibrated_turns={turn:{k:v for k,v in value.items() if k not in ('mechanics','reference_games')}
                          for turn,value in calibration['turns'].items()},
        unreachable_target_turns=missing, combat_exposure=exposure,
        numerical_recipe='保持FP32、16局、epochs1、LR6e-6；本轮只优化快照发布，不改变PPO数学。',
        next_training_proposal=dict(parent='B10M version2328完整trainer',
            adaptation=dict(decisions=2000000, library='validated-mixtures/adapt/train', control_fraction=.5),
            main=dict(decisions=8000000, library='validated-mixtures/main/train', control_fraction=.25),
            wall_hours=9, monitor_every_decisions=2000000, performance_stops=False,
            engineering_stops=True, ready=False,
            note='待核对实际压力与缺口后冻结新计划；未启动长训；不是已证明最优配方。'),
        limitations=['64局/面板仅8个独立seed，用于压力定位，不能据此可靠排名。',
                     '新库逐帧完整快照来自引擎；真人仅作近似条件战斗参考，未直接混入。',
                     '不可达回合明确列出，不能把候选库宣称为完整真人校准成果。'])
