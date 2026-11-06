"""本地流式复盘完整history和阶段评测；牌面强度统一突出P50。"""
import argparse,collections,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from workflows.report_personal_run import summarize
from workflows.report_firestone_learning import by_turn

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def analyze(capture,out):
    from hsbg.az.train.personal import write_json
    capture,out=Path(capture),Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (capture/'train/history.jsonl').open(encoding='utf8') if line.strip()]
    computed=summarize(rows);saved=read(capture/'training-report.json')
    if computed!=saved:raise ValueError('完整history重算与服务器training-report不同')
    plan=read(capture/'plan.json');status=read(capture/'status.json');latest=read(capture/'train/latest.json')
    if status['latest']!=latest or plan['inherited']['actual_decisions']+computed['totals']['decisions']!=latest['cum_decisions']:raise ValueError('累计决策账本不同')
    evaluations=[];budgets=[];pools=[]
    for path in sorted((capture/'milestones').glob('*/hard.json')):
        ev=read(path);cp=read(path.parent/'checkpoint.json');record=read(path.parent/'report.json')
        if not ev['valid'] or ev['checkpoint']['sha256']!=cp['sha256']['model.pt']:raise ValueError('阶段评测来源不符')
        if ev['protocol_sha256']!=record['paired']['hard']['protocol_sha256']:raise ValueError('阶段协议不一致')
        for seat in ev['per_seat']:
            for e in seat['end_turn_economy']:
                if e['action_counts'].get('move',0)>5:raise ValueError('换位超限')
                if e['actions']>80:
                    if not e['forced']:raise ValueError('超预算没有合法收尾标记')
                    budgets.append(dict(version=cp['version'],seed=e['seed'],turn=e['turn'],actions=e['actions'],counts=e['action_counts']))
        evaluations.append(dict(version=cp['version'],new_decisions=cp['actual_decisions']-plan['inherited']['actual_decisions'],mean_rank=ev['mean_rank'],top4=ev['top4_rate'],win=ev['win_rate'],protocol_sha256=ev['protocol_sha256'],turns=by_turn(ev)))
    if len({e['protocol_sha256'] for e in evaluations})!=1:raise ValueError('阶段协议不同，不能合并趋势')
    sums=collections.Counter();metrics=[]
    for row in rows:
        if row.get('iter') is None:continue
        p,q,c=row['profile'],row['ppo'],row['collect'];t=p['timeline']
        for k in ('ppo_s','parent_read_s','build_batch_s','checkpoint_save_publish_s','pipeline_fallback_wait_s'):sums[k]+=p.get(k,0)
        sums['ppo_inner_s']+=q['seconds'];sums['batch_wait_s']+=t['t_collect_done']-t['t_iter_start'];sums['iteration_s']+=t['t_history_done']-t['t_iter_start']
        if c.get('mirror_pool_observations'):pools.append(dict(iter=row['iter'],observations=c['mirror_pool_observations']))
        metrics.append(dict(iter=row['iter'],**{k:(q['epochs'][-1] if q.get('epochs') else {}).get(k) for k in ('entropy','approx_kl','anchor_kl','value_loss')}))
    def window(v):return {k:statistics.mean(x[k] for x in v if isinstance(x.get(k),(int,float))) for k in ('entropy','approx_kl','anchor_kl','value_loss')}
    result=dict(schema='hsai.training_review/v1',passed=True,history_recomputed_exactly=True,source_base=plan['inherited']['actual_decisions'],latest=latest,training=computed,timing=dict(sums),early200=window(metrics[:200]),late200=window(metrics[-200:]),evaluations=evaluations,late_observations_conditional_on_survival=True,dev_selection_bias=True,final512_available=False,performance_note='fixed96-game repeated Dev; selecting best checkpoint does not establish unseen-game superiority')
    write_json(out/'review.json',result);write_json(out/'budget-events.json',budgets);write_json(out/'pool-observations.json',pools)
    print(json.dumps(dict(passed=True,iterations=len(computed['iterations']),new_decisions=computed['totals']['decisions'],stages=len(evaluations),budget_events=len(budgets),pool_observation_batches=len(pools))))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();analyze(a.capture,a.out)
