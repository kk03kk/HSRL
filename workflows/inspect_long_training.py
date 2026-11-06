"""登录节点只读长训状态/最新详细评测，支持按回合查看具体阵容。"""
import argparse
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--turn',type=int)
    p.add_argument('--seed',type=int)
    p.add_argument('--seat',type=int)
    a=p.parse_args()
    status=a.run/'status.json'
    if status.exists():print(json.dumps(read(status),ensure_ascii=False,indent=2))
    paths=list(a.run.glob('baselines/diagnostics.json'))+list(a.run.glob('milestones/*/diagnostics.json'))+list(a.run.glob('final/*/diagnostics.json'))
    if not paths:
        print('尚无已完成的评测报告');return
    latest=max(paths,key=lambda f:f.stat().st_mtime_ns)
    report=read(latest)
    print(f'\n最新详细评测：{latest}')
    candidate=report.get('candidate',report)
    for key in ('mean_rank','top4_rate','win_rate'):
        print(key,candidate.get(key))
    if 'paired' in report:print('paired',json.dumps(report['paired'],ensure_ascii=False))
    if a.turn is None:
        print(json.dumps(candidate['turns'],ensure_ascii=False,indent=2));return
    print('回合汇总',json.dumps(candidate['turns'].get(str(a.turn)),ensure_ascii=False,indent=2))
    raw=Path(candidate['raw_evaluation'])
    if raw.stat().st_size>32*1024*1024 and os.name!='nt' and not os.environ.get('SLURM_JOB_ID'):
        print(f'原始评测大于32MiB，登录节点不整份解析；请在Slurm计算节点或本地查看：{raw}')
        return
    evaluation=read(raw)
    matches=[]
    for seat in evaluation['per_seat']:
        if a.seat is not None and seat['policy_seat']!=a.seat:continue
        for row in seat['end_turn_economy']:
            if row['turn']!=a.turn or (a.seed is not None and row['seed']!=a.seed):continue
            battles=[r for r in seat['combat_exposure'] if r['seed']==row['seed'] and r['turn']==a.turn]
            matches.append(dict(seat=seat['policy_seat'],economy=row,combat=battles))
            if a.seed is None:break
        if matches and a.seed is None:break
    print(json.dumps(matches,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
