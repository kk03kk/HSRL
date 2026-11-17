"""Imitate complete economic turns and assess actual closed-loop resource use."""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hsbg.az.train.ppo import PPOEvaluator,PPOGreedyPolicy
from hsbg.runtime.artifacts import file_identity
from workflows.learn_cycle_process import run
from workflows.probe_whole_turn_economy import metrics
from tools.probe_economic_starts import run_turn,canonical_items


def summarize(rows):
    def group(rr):
        return {'tasks':len(rr),
            'eot_body_mean':float(np.mean([r['student']['eot_body'] for r in rr])),
            'eot_body_p50':float(np.median([r['student']['eot_body'] for r in rr])),
            'gold_spent_mean':float(np.mean([r['student']['gold_spent'] for r in rr])),
            'core_count_mean':float(np.mean([r['student']['core_count'] for r in rr])),
            'recruit_deaths':sum(not r['student'].get('alive',True) for r in rr),
            'teacher_eot_body_mean':float(np.mean([r['teacher']['eot_body'] for r in rr]))}
    return {'all':group(rows),'kinds':{k:group([r for r in rows if r['kind']==k]) for k in sorted({r['kind'] for r in rows})},'rows':rows}


def reuse_baseline(root,tasks,checkpoint):
    root=Path(root);complete=json.loads((root/'complete.json').read_text())
    run_info=json.loads((root/'run.json').read_text())
    if not complete['valid'] or file_identity(root/'run.json')!=complete['run']:
        raise ValueError('invalid baseline evidence')
    if run_info['checkpoint']!=file_identity(checkpoint):raise ValueError('baseline uses another checkpoint')
    indexed={(r['source_seed'],r['variant']):r for r in complete['rows']}
    rows=[];changed_teacher=0
    for task in tasks:
        key=(task['seed'],task['kind']);row=indexed[key]
        trace_path=root/(str(key[0])+'-'+key[1]+'.jsonl.gz')
        if file_identity(trace_path)!=row['trace']:raise ValueError('baseline trace checksum differs')
        with gzip.open(trace_path,'rt',encoding='utf-8') as stream:
            original=json.loads(next(stream))
        if original['policy']!='parent' or original['initial']!=task['initial']:
            raise ValueError('baseline initial state is not identical')
        changed_teacher+=task['teacher']!=row['operator']
        rows.append({'seed':key[0],'kind':key[1],'student':row['parent'],'teacher':task['teacher']})
    return {**summarize(rows),'reused_baseline':file_identity(root/'complete.json'),
            'changed_teacher_cases':changed_teacher,
            'validation':'all initial states, source checkpoint and trace checksums identical; teacher metrics come from the new validated dataset; no repeated parent games'}


def make_closure(args):
    @torch.no_grad()
    def assess(model,tasks,device,path):
        if path.name=='before.jsonl.gz':
            return reuse_baseline(args.baseline,tasks,args.checkpoint)
        policy=PPOGreedyPolicy(PPOEvaluator(model,device=device));ids=canonical_items();rows=[]
        with gzip.open(path,'xt',encoding='utf-8',compresslevel=3) as stream:
            for task in tasks:
                _,trace=run_turn(policy,task['initial'],ids,allow_recruit_death=True)
                row={'seed':task['seed'],'kind':task['kind'],'student':metrics(trace),'teacher':task['teacher']}
                rows.append(row);stream.write(json.dumps({'result':row,**trace},separators=(',',':'))+'\n')
        return summarize(rows)
    return assess


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--mode',choices=('all',),default='all')
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--anchor-kl',type=float,default=10.);p.add_argument('--per-kind-batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=1193801)
    args=p.parse_args()
    run(args,expected_schema='hsai.whole-turn-process-dataset/v1',
        closed_loop_fn=make_closure(args),learning_tag='whole_turn_learning')
