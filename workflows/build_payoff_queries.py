"""One fixed round of student-visited public teacher queries on training sources.

The student executes every action. Teachers may abstain, deaths stay in evidence,
and all held-out rows are copied unchanged from the original demonstration set.
"""
import argparse,base64,gzip,json,multiprocessing,shutil,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.az.train import mechanism_obs as M
from hsbg.evaluation.trace import DecisionRecorder,validate_trace
from hsbg.evaluation.quality import suffix_quality
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from workflows.build_payoff_dataset import label,save_rows
from workflows.learn_payoff_process import validate_dataset
from workflows.probe_whole_turn_economy import fixture

_CACHE={}


def selected_sources(bank,original,count=64):
    tasks=[]
    with gzip.open(Path(original)/'tasks.jsonl.gz','rt',encoding='utf-8') as stream:
        tasks=[json.loads(line) for line in stream]
    train={t['seed'] for t in tasks if t['split']=='train'}
    if len(train)!=207 or any(seed%5==0 for seed in train):raise ValueError('original source split differs')
    groups={}
    with gzip.open(Path(bank)/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            source=json.loads(line);p=source['state']['players'][0]
            if source['seat']!=0 or not source['state']['lobby_races']&64:continue
            if not 8<=source['turn']<=9 or p['health']<15 or p['pending'] or p['ended_turn']:continue
            if source['seed'] not in groups or source['turn']<groups[source['seed']]['turn']:groups[source['seed']]=source
    ordered=[groups[k] for k in sorted(groups)]
    indexed=[(i,s) for i,s in enumerate(ordered) if s['seed'] in train]
    if {s['seed'] for _,s in indexed}!=train:raise ValueError('original training sources missing')
    chosen=np.linspace(0,len(indexed)-1,count,dtype=int)
    if len(set(chosen.tolist()))!=count:raise ValueError('nonunique training selection')
    return [indexed[i] for i in chosen]


class QueryRecorder(DecisionRecorder):
    def __init__(self,*args,policy,**kwargs):
        super().__init__(*args,**kwargs);self.policy=policy;self.labels=[];self.excluded=Counter();self.recruit_deaths=[]
    def after(self,game,before,decision,executed_action,forced_tier=False):
        if executed_action!=decision.action or forced_tier:raise ValueError('student action was intervened on')
        info=decision.info
        if info.get('forced'):self.excluded['forced']+=1
        elif not info['shadow_gold_context']:self.excluded['outside_gold_gate']+=1
        elif info['shadow_teacher_abstention']:self.excluded['abstention/'+info['shadow_teacher_abstention']]+=1
        else:
            target=info['shadow_teacher_action']
            if target is None:raise ValueError('missing target without explicit abstention')
            f=np.frombuffer(base64.b64decode(before['observation']['floats_f32le']),dtype='<f4')
            c=np.frombuffer(base64.b64decode(before['observation']['cards_i32le']),dtype='<i4')
            teacher=SimpleNamespace(action=target,info={'forced':False,'research_option_active':True,
                'research_operator_reason':info['shadow_teacher_reason']})
            row=label(f,c,self.policy.evaluator.last_mask,teacher,before['turn'])
            if row is None:raise ValueError('accepted query outside gate')
            self.labels.append(row)
        after=json.loads(game.player_json(before['seat']))
        if executed_action!=0 and not after['alive']:
            self.recruit_deaths.append({'turn':before['turn'],'action_index':before['action_index'],
                'action':executed_action,'health':after['health']})
        super().after(game,before,decision,executed_action,forced_tier)


def worker(spec):
    import torch,hsbg_engine as E
    from hsbg.az.train.eval_policy import EvalProtocol,make_eval_policy
    from hsbg.az.train.shadow_economy import ShadowEconomyPolicy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
    torch.set_num_threads(1)
    index,source,args=spec;seed=source['seed'];out=Path(args['out'])/'sources'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    if not _CACHE:
        built=make_eval_policy(args['checkpoint'],protocol=EvalProtocol(mirror_snaps=args['library']),device=args['device'])
        _CACHE['policy']=ShadowEconomyPolicy(built['policy'].evaluator)
        _CACHE['pool']=MirrorSnapshotPool(args['library'],sample='p50band')
    policy=_CACHE['policy'];policy.reset()
    state,edits=fixture(source['state'],'complete_gold');future_seed=12300000+index*100
    fresh=json.loads(E.Game(seed=future_seed).state_json());state['seed']=fresh['seed'];state['rng']=fresh['rng']
    identity={'source_seed':seed,'source_index':index,'future_seed':future_seed,'split':'train',
        'checkpoint_sha256':args['checkpoint_sha256'],'interventions':edits,
        'mechanism_schema':M.SCHEMA,'mechanism_schema_sha256':M.SCHEMA_SHA256,
        'scope':'student executes all actions; public teacher queries only; artificial golden initial components'}
    recorder=QueryRecorder(out/'decisions.jsonl.gz',identity=identity,policy=policy)
    try:
        result=evaluate_policy(policy,EvalConfig(games=1,seed=future_seed,research_initial_state=state,
            collect_economy=True,decision_recorder=recorder,mirror_pool=_CACHE['pool'],
            mirror_strict=False,mirror_seats=7,mirror_p=1.))
        quality=suffix_quality(result);trace=recorder.finish(result)
        validation=validate_trace(out/'decisions.jsonl.gz',replay=16)
    except BaseException:
        recorder.abort();raise
    save_rows(out/'queries.npz',recorder.labels,seed)
    atomic_json(out/'evaluation.json',result)
    row={'seed':seed,'source_index':index,'future_seed':future_seed,'split':'train','labels':len(recorder.labels),
        'queries_file':file_identity(out/'queries.npz'),'evaluation':file_identity(out/'evaluation.json'),
        'quality':quality,'trace':trace,'validation':validation,'excluded':dict(recorder.excluded),
        'recruit_deaths':recorder.recruit_deaths,'rank':result['rank_sequence'][0]}
    atomic_json(out/'complete.json',row);return row


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);plan_path=Path(args.plan)
    plan=json.loads(plan_path.read_text());original=Path(plan['original_dataset']);bank=Path(plan['bank'])
    manifest=json.loads((original/'manifest.json').read_text());bank_meta=json.loads((bank/'manifest.json').read_text())
    if file_identity(original/'manifest.json')['sha256']!=plan['original_manifest_sha256']:raise ValueError('original data changed')
    for name,identity in manifest['files'].items():
        if file_identity(original/name)!=identity:raise ValueError('original file changed: '+name)
    if file_identity(bank/'states.jsonl.gz')!=bank_meta['data']:raise ValueError('source bank changed')
    admission=validate_dataset(original,plan['parent'])
    student=file_identity(plan['student'])
    if student['sha256']!=plan['student_sha256']:raise ValueError('querying student changed')
    from hsbg.az.train.conditional_economy import current_gate_spec
    from hsbg.az.train.common import engine_info
    from hsbg.az.train.eval_policy import TRAINING_CONTRACT
    library=Path(plan['library']);snapshots={p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}
    original_run=json.loads((original/'run.json').read_text())
    if original_run['snapshot_files']!=snapshots:raise ValueError('query field differs from demonstration field')
    if original_run['source_manifest']!=file_identity(bank/'manifest.json'):raise ValueError('query source bank differs')
    sources=selected_sources(bank,original,plan['training_sources'])
    if args.probe:sources=[sources[0],sources[-1]]
    source=source_identity(Path(__file__).resolve().parents[1])
    if source['tracked_dirty']:raise ValueError('committed source required')
    run={'plan':file_identity(plan_path),'args':vars(args),'student':student,'source':source,
        'environment':environment_identity(),'original_root':str(original),'original_manifest':file_identity(original/'manifest.json'),
        'original_admission':admission,'source_manifest':file_identity(bank/'manifest.json'),
        'engine':engine_info(),'action_contract':dict(TRAINING_CONTRACT),'snapshot_files':snapshots,
        'gate':current_gate_spec('gold_engine'),'selected_sources':[s['seed'] for _,s in sources],
        'source_selection':'64 deterministic evenly spaced indices of all 207 original training sources; no outcome filtering'}
    atomic_json(out/'run.json',run)
    worker_args={'out':str(out),'checkpoint':plan['student'],'checkpoint_sha256':student['sha256'],
        'device':args.device,'library':plan['library']}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=[]
        for row in pool.map(worker,[(i,s,worker_args) for i,s in sources]):
            results.append(row);print(json.dumps({'completed':len(results),'total':len(sources),'labels':row['labels']}),flush=True)
    query_arrays={}
    for row in results:
        path=out/'sources'/str(row['seed'])/'queries.npz'
        if file_identity(path)!=row['queries_file']:raise ValueError('source queries changed')
        with np.load(path,allow_pickle=False) as data:
            if data['dev'].any():raise ValueError('development source leaked into queries')
            for k in data.files:query_arrays.setdefault(k,[]).append(data[k])
    query={k:np.concatenate(v) for k,v in query_arrays.items()}
    np.savez_compressed(out/'queries.npz',**query)
    with np.load(original/'labels.npz',allow_pickle=False) as old:
        if set(old.files)!=set(query):raise ValueError('query schema differs')
        merged={k:np.concatenate((old[k],query[k])) for k in old.files}
        merged['origin']=np.concatenate((np.zeros(len(old['dev']),dtype=np.int8),np.ones(len(query['dev']),dtype=np.int8)))
    np.savez_compressed(out/'labels.npz',**merged)
    for name in ('anchors.npz','tasks.jsonl.gz'):shutil.copyfile(original/name,out/name)
    with gzip.open(out/'query_tasks.jsonl.gz','xt',encoding='utf-8') as stream:
        for row in results:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    if file_identity(plan['student'])!=student:raise ValueError('student changed during queries')
    if snapshots!={p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}:raise ValueError('query field changed')
    atomic_json(out/'manifest.json',{'schema':'hsai.payoff-aggregate-dataset/v1','probe':args.probe,
        'run':file_identity(out/'run.json'),'mechanism_schema_sha256':M.SCHEMA_SHA256,
        'query_sources':len(results),'query_labels':len(query['dev']),'original_labels':manifest['labels'],
        'source_games':256,'training_sources':207,'development_sources':49,
        'files':{n:file_identity(out/n) for n in ('labels.npz','queries.npz','anchors.npz','tasks.jsonl.gz','query_tasks.jsonl.gz')},
        'scope':'One student-state aggregation round; unchanged original development labels; no new held-out games',
        'limitations':['Heuristic targets are not optimal actions','Unsupported recovery states are explicit abstentions',
            'Student deaths retained as legal bad outcomes','Teacher-private memory is not added to student input']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--probe',action='store_true');main(p.parse_args())
