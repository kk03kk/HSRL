"""Compare economic teaching and a student over complete conditional suffixes.

All starts are explicitly endowed. This is not natural-game performance.
"""
import argparse,gzip,hashlib,json,multiprocessing as mp,sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from hsbg.evaluation.trace import DecisionRecorder,validate_trace
from tools.probe_economic_starts import select_sources
from workflows.probe_whole_turn_economy import fixture

_CACHE={}

class EconomyRecorder(DecisionRecorder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.economy={};self.option_actions=0;self.deployment_actions=0;self.gate_retention_actions=0
    def after(self,game,before,decision,executed_action,forced_tier=False):
        self.option_actions+=bool(decision.info.get('research_option_active'))
        self.deployment_actions+=bool(decision.info.get('deployment_override_executed'))
        self.gate_retention_actions+=bool(decision.info.get('gate_retention_override_executed'))
        if executed_action==0:
            p=before['state_before']['players'][before['seat']]
            self.economy[int(before['turn'])]={'gold_spent':p['stats']['gold_spent_turn'],
                'gold_end':p['gold'],'forced_reason':decision.info.get('forced_reason')}
        super().after(game,before,decision,executed_action,forced_tier)

def trial(task):
    import torch,hsbg_engine as E
    from hsbg.az.train.eval_policy import EvalProtocol,make_eval_policy
    from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
    torch.set_num_threads(1)
    case,variant,replica,name,checkpoint,operator,root,device,library=task
    key=(name,checkpoint,operator,device)
    if key not in _CACHE:
        built=make_eval_policy(checkpoint,protocol=EvalProtocol(mirror_snaps=library),device=device)
        policy=built['policy']
        if operator=='retain_gate':
            from hsbg.az.train.gate_retention_probe import GateRetentionPolicy
            policy=GateRetentionPolicy(policy.evaluator)
        elif operator=='deploy_payoff':
            from hsbg.az.train.shadow_economy import PayoffDeploymentPolicy
            policy=PayoffDeploymentPolicy(policy.evaluator)
        elif operator:policy=EconomicOptionPolicy(policy.evaluator,
            operator_mode=operator if isinstance(operator,str) else 'legacy')
        _CACHE[key]=(policy,file_identity(checkpoint))
    if ('pool',library) not in _CACHE:_CACHE[('pool',library)]=MirrorSnapshotPool(library,sample='p50band')
    policy,checkpoint_identity=_CACHE[key]
    initial,edits=fixture(case['state'],variant)
    seed=11960000+case['case']*100+replica
    fresh=json.loads(E.Game(seed=seed).state_json());initial['seed']=fresh['seed'];initial['rng']=fresh['rng']
    path=Path(root)/'trials'/name/f"case-{case['case']:03d}-{variant}-r{replica:02d}"
    path.mkdir(parents=True,exist_ok=False)
    identity={'name':name,'checkpoint_sha256':checkpoint_identity['sha256'],
        'source_seed':case['seed'],'source_turn':case['turn'],'variant':variant,'future_seed':seed,
        'operator_intervention':operator,'scope':'explicitly endowed conditional suffix, not natural performance',
        'interventions':edits,'initial_opponents_preserved':True,'reset_controller':True}
    from hsbg.az.train import mechanism_obs as M
    schema=M.schema_of(getattr(policy.evaluator,'model',None))
    if schema:identity.update(mechanism_schema=schema,mechanism_schema_sha256=M.SCHEMA_SHA256)
    recorder=EconomyRecorder(path/'decisions.jsonl.gz',identity=identity)
    try:
        result=evaluate_policy(policy,EvalConfig(games=1,seed=seed,research_initial_state=initial,
            collect_economy=True,decision_recorder=recorder,mirror_pool=_CACHE[('pool',library)],
            mirror_strict=False,mirror_seats=7,mirror_p=1.))
        from hsbg.evaluation.quality import suffix_quality
        suffix_quality(result)
        trace=recorder.finish(result);validation=validate_trace(path/'decisions.jsonl.gz',replay=16)
    except BaseException:
        recorder.abort();raise
    turns=[]
    for row in result['end_turn_economy']:
        if row['eot_board'] is None:raise ValueError('missing surviving EOT frame')
        turns.append({'turn':row['turn'],'strength':row['eot_attack']+row['eot_board_hp'],
            'tier':row['tier'],'health':row['health'],'actions':row['actions'],
            **recorder.economy[row['turn']]})
    row={'case':case['case'],'source_seed':case['seed'],'initial_turn':case['turn'],
        'variant':variant,'replica':replica,'future_seed':seed,'name':name,
        'initial_sha256':hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'rank':result['rank_sequence'][0],'turns':turns,'operator_actions':recorder.option_actions,
        'trace':trace,'validation':validation}
    if operator=='deploy_payoff':row['deployment_actions']=recorder.deployment_actions
    if operator=='retain_gate':row['gate_retention_actions']=recorder.gate_retention_actions
    atomic_json(path/'result.json',{'summary':row,'evaluation':result});return row

def compare(rows,parent='parent'):
    indexed=defaultdict(dict)
    for r in rows:
        key=(r['case'],r['variant'],r['replica'])
        if r['name'] in indexed[key]:raise ValueError('duplicate suffix outcome')
        indexed[key][r['name']]=r
    names=sorted({r['name'] for r in rows});rng=np.random.default_rng(11960000);out={}
    for variant in sorted({r['variant'] for r in rows}):
        for name in names:
            if name==parent:continue
            cases=defaultdict(list);growth=defaultdict(list)
            for (case,v,replica),arms in indexed.items():
                if v!=variant:continue
                if set(arms)!=set(names):raise ValueError('unpaired suffix outcome')
                a,b=arms[name],arms[parent]
                if a['source_seed']!=b['source_seed']:raise ValueError('paired source seed differs')
                if a['future_seed']!=b['future_seed']:raise ValueError('paired future seed differs')
                if a['initial_sha256']!=b['initial_sha256']:raise ValueError('paired initial state differs')
                cases[case].append(a['rank']-b['rank'])
                ta={t['turn']:t for t in a['turns']};tb={t['turn']:t for t in b['turns']}
                if all(t in ta and t in tb for t in (9,12)):
                    growth[case].append(np.log1p(ta[12]['strength'])-np.log1p(ta[9]['strength'])
                        -np.log1p(tb[12]['strength'])+np.log1p(tb[9]['strength']))
            def estimate(groups):
                x=np.asarray([np.mean(v) for _,v in sorted(groups.items())])
                if not len(x):return {'sources':0,'mean':None,'ci95':None}
                boot=np.mean(rng.choice(x,size=(5000,len(x)),replace=True),axis=1)
                return {'sources':len(x),'paired_replicates':sum(map(len,groups.values())),
                    'mean':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
            out[name+'/'+variant]={'rank_minus_'+parent:estimate(cases),
                'common_survivor_T9_T12_log_growth_minus_'+parent:estimate(growth)}
    curves={}
    for name in names:
        for variant in sorted({r['variant'] for r in rows}):
            rr=[r for r in rows if r['name']==name and r['variant']==variant];turns=defaultdict(list)
            for r in rr:
                for t in r['turns']:turns[t['turn']].append(t)
            curves[name+'/'+variant]={'games':len(rr),'rank_mean':float(np.mean([r['rank'] for r in rr])),
                'turns':{str(t):{'n':len(v),'sources':len({r['case'] for r in rr if any(x['turn']==t for x in r['turns'])}),
                    'strength_median':float(np.median([x['strength'] for x in v])),
                    'gold_spent_median':float(np.median([x['gold_spent'] for x in v])),
                    'cap_count':sum(x['forced_reason']=='action_cap' for x in v)} for t,v in sorted(turns.items())}}
    return {'reference':parent,'paired':out,'curves':curves,
        'limitations':['Explicit component endowments; not natural acquisition or promotion',
            'Bootstrap units are source games, after averaging future RNG replicas',
            'Growth uses common survivors only and may select a different cohort',
            'Median curves pool surviving replicas; n is not the number of independent sources']}

def main(args):
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    sources=select_sources(plan['bank'])
    if args.probe:sources=sources[:1]
    for i,s in enumerate(sources):s['case']=i
    variants=plan['variants'][:1] if args.probe else plan['variants']
    replicas=1 if args.probe else plan['replicas']
    models={n:m for n,m in plan['models'].items()
            if not args.probe or n in plan.get('probe_models',('parent','operator'))}
    if not models:raise ValueError('no study models selected')
    identities={n:file_identity(m['checkpoint']) for n,m in models.items()}
    for n,m in models.items():
        if m.get('sha256') and identities[n]['sha256']!=m['sha256']:raise ValueError('model identity differs: '+n)
    library=Path(plan['library']);library_identity={p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}
    if not library_identity:raise ValueError('no training snapshot field')
    with gzip.open(out/'cases.jsonl.gz','xt',encoding='utf-8') as f:
        for s in sources:f.write(json.dumps(s,separators=(',',':'))+'\n')
    from hsbg.az.train.common import engine_info
    from hsbg.az.train.eval_policy import TRAINING_CONTRACT
    run={'plan':file_identity(plan_path),'args':vars(args),'models':identities,
        'engine':engine_info(),'action_contract':dict(TRAINING_CONTRACT),
        'source_manifest':file_identity(Path(plan['bank'])/'manifest.json'),
        'cases':file_identity(out/'cases.jsonl.gz'),'source':source_identity(Path(__file__).resolve().parents[1]),
        'environment':environment_identity(),'snapshot_files':library_identity}
    if run['source']['tracked_dirty']:raise ValueError('committed source required')
    atomic_json(out/'run.json',run)
    tasks=[(s,v,r,n,m['checkpoint'],m['operator'],str(out),args.device,str(library))
           for s in sources for v in variants for r in range(replicas) for n,m in models.items()]
    rows=[];print(json.dumps({'sources':len(sources),'rollouts':len(tasks),'probe':args.probe}),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(trial,tasks):
            rows.append(row)
            if len(rows)%16==0:print(json.dumps({'finished':len(rows),'total':len(tasks)}),flush=True)
    if {n:file_identity(m['checkpoint']) for n,m in models.items()}!=identities:raise ValueError('models changed')
    if {p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}!=library_identity:raise ValueError('library changed')
    result={'valid':True,'probe':args.probe,'run':file_identity(out/'run.json'),
        'rollouts':len(rows),'models':identities,'comparison':compare(rows,plan.get('reference','parent')),'rows':rows}
    atomic_json(out/'complete.json',result);print(json.dumps(result['comparison']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--probe',action='store_true')
    main(p.parse_args())
