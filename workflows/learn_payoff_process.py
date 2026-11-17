"""Learn multi-turn payoff operation inside a declared golden-engine context."""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.conditional_economy import fork_conditional
from workflows.learn_cycle_process import run
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.evaluation.quality import suffix_quality

def validate_dataset(root,checkpoint):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('probe') or (manifest['source_games'],manifest['training_sources'],manifest['development_sources'])!=(256,207,49):
        raise ValueError('requires the complete predeclared source split, not a probe/subset')
    if manifest['run']!=file_identity(root/'run.json'):raise ValueError('dataset run changed')
    identity=json.loads((root/'run.json').read_text())
    if identity['parent']!=file_identity(checkpoint):raise ValueError('demonstrator fallback parent differs')
    from hsbg.az.train.conditional_economy import current_gate_spec
    from hsbg.az.train.common import engine_info
    from hsbg.az.train.eval_policy import TRAINING_CONTRACT
    if identity['gate']!=current_gate_spec('gold_engine') or identity['engine']!=engine_info():
        raise ValueError('dataset gate or engine differs')
    if identity['action_contract']!=dict(TRAINING_CONTRACT):raise ValueError('dataset action contract differs')
    if file_identity(root/'tasks.jsonl.gz')!=manifest['files']['tasks.jsonl.gz']:raise ValueError('dataset tasks changed')
    evaluations=[];source_seeds=set();total_actions=0
    with gzip.open(root/'tasks.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['seed'] in source_seeds:raise ValueError('duplicate demonstration source')
            source_seeds.add(row['seed'])
            if row['split']=='train':
                path=root/'sources'/str(row['seed'])/'evaluation.json';result=json.loads(path.read_text())
            elif row['split']=='dev':
                path=Path(row['evidence']['reused_trace']).parent/'result.json';result=json.loads(path.read_text())['evaluation']
            else:raise ValueError('unknown demonstration split')
            checked=suffix_quality(result)
            if result['rank_sequence']!=[row['evidence']['rank']]:raise ValueError('demonstration rank identity differs')
            if checked['actions']!=row['evidence']['trace']['actions']:raise ValueError('demonstration action count differs')
            total_actions+=checked['actions'];evaluations.append({'path':str(path),'identity':file_identity(path)})
    if len(source_seeds)!=256:raise ValueError('incomplete source quality coverage')
    return {'passed':True,'sources':len(source_seeds),'actions':total_actions,'evaluations':evaluations,
        'scope':'All training and reused development outcomes checked for missing/fallback/rejected injection and truncated combat'}

def separate_assessment(model,tasks,device,path):
    return {'kinds':{},'tasks':len(tasks),'status':'not run here',
        'scope':'Closed-loop ability must be assessed by a separate sealed full-suffix job; label agreement is insufficient'}

EVENTS={
    'progression':('unlock_growth_payoff','complete_engine','restore_core','space_for_core','triple_core','invest_core_pair'),
    'resources':('wealth','friendly_pirates','friendly_fallback','buy_economy','buy_bounty','pirate_majority_space'),
    'play':('cycle_hand','spell','pending'),
    'liquidation':('liquidate_cycle','keep_stronger_cycle','space_for_cycle','remove_health_risk'),
    'search':('search',),
    'finish':('fill_board','end'),
}

def mixed_event_sampling(raw):
    import numpy as np
    train=~raw['dev'];reasons=raw['reasons'];groups={};sizes={}
    if set(reasons)-{r for rs in EVENTS.values() for r in rs}:raise ValueError('unknown demonstrator event')
    for phase in ('T8_9','T10_12','T13_plus'):
        key='phase/'+phase;groups[key]=np.flatnonzero(train&(raw['kinds']==phase));sizes[key]=16
    for event,reasons_in_group in EVENTS.items():
        key='event/'+event;groups[key]=np.flatnonzero(train&np.isin(reasons,reasons_in_group));sizes[key]=8
    if any(not len(v) for v in groups.values()):raise ValueError('empty predeclared sampling group')
    return groups,sizes

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--mode',default='all',choices=('all',))
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--anchor-kl',type=float,default=10.);p.add_argument('--per-kind-batch',type=int,default=32)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--sampling',choices=('phase','mixed_events'),default='phase')
    p.add_argument('--seed',type=int,default=1203801);args=p.parse_args()
    if args.sampling=='mixed_events' and args.per_kind_batch!=32:
        p.error('mixed_events fixes the total demonstration batch at 96; retain --per-kind-batch 32')
    args.dataset_validation=validate_dataset(args.dataset,args.checkpoint)
    if args.validate_only:
        atomic_json(Path(args.out)/'complete.json',args.dataset_validation)
        print(json.dumps({k:v for k,v in args.dataset_validation.items() if k!='evaluations'}),flush=True)
        sys.exit(0)
    run(args,expected_schema='hsai.payoff-process-dataset/v1',closed_loop_fn=separate_assessment,
        learning_tag='gold_payoff_learning',
        model_transform=lambda model,meta:fork_conditional(model,meta,mechanisms=True,gate_mode='gold_engine'),
        sampling_plan_fn=mixed_event_sampling if args.sampling=='mixed_events' else None)
