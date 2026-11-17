"""Multi-turn growth-controller demonstrations, isolated by source game.

Only the explicitly validated golden engine context supplies labels. Existing
development suffixes are replayed, not sampled again. Training futures are new.
"""
import argparse,base64,gzip,json,multiprocessing,shutil,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.conditional_economy import current_gate_spec
from hsbg.evaluation.trace import DecisionRecorder,validate_trace
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from workflows.probe_whole_turn_economy import fixture

_CACHE={}

def gate(features,cards):
    s=current_gate_spec('gold_engine')
    owned=np.concatenate([cards[slice(*s[k])] for k in ('board_slice','hand_slice')])
    return bool(features[s['lobby_feature']]>.5 and
        sum(any(int(c) in group for c in owned) for group in s['embedding_ids'])>=s['minimum_distinct_groups'])

def label(features,cards,mask,decision,turn):
    if features.shape!=(M.OBS_FLOATS,) or cards.shape!=(124,) or mask.shape!=(245,):
        raise ValueError('demonstration observation layout differs')
    if decision.info.get('forced') or not decision.info.get('research_option_active') or not gate(features,cards):return None
    action=int(decision.action)
    if not mask[action]:raise ValueError('teacher label violates actual guarded mask')
    target=np.zeros(245,dtype=bool);target[action]=True
    kind='T8_9' if turn<=9 else 'T10_12' if turn<=12 else 'T13_plus'
    return (features.copy(),cards.astype(np.int16),mask.copy(),target,kind,int(turn),
            decision.info['research_operator_reason'])

class Recorder(DecisionRecorder):
    def __init__(self,*args,policy,**kwargs):
        super().__init__(*args,**kwargs);self.policy=policy;self.labels=[];self.excluded=Counter()
    def after(self,game,before,decision,executed_action,forced_tier=False):
        if executed_action!=decision.action or forced_tier:raise ValueError('unexpected demonstration intervention')
        f=np.frombuffer(base64.b64decode(before['observation']['floats_f32le']),dtype='<f4')
        c=np.frombuffer(base64.b64decode(before['observation']['cards_i32le']),dtype='<i4')
        row=label(f,c,self.policy.evaluator.last_mask,decision,before['turn'])
        if row is not None:self.labels.append(row)
        else:self.excluded['forced' if decision.info.get('forced') else 'outside_gate']+=1
        if executed_action!=0 and (not json.loads(game.player_json(before['seat']))['alive']):
            raise ValueError('teacher died during recruitment; preserve failure evidence')
        super().after(game,before,decision,executed_action,forced_tier)

def resources(checkpoint,library):
    import torch
    from hsbg.az.train.eval_policy import EvalProtocol,make_eval_policy
    from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    torch.set_num_threads(1)
    if not _CACHE:
        built=make_eval_policy(checkpoint,protocol=EvalProtocol(mirror_snaps=library),device='cpu')
        _CACHE['policy']=EconomicOptionPolicy(built['policy'].evaluator,operator_mode='growth')
        _CACHE['pool']=MirrorSnapshotPool(library,sample='p50band')
    _CACHE['policy'].reset()
    return _CACHE['policy'],_CACHE['pool']

def save_rows(path,rows,seed):
    if not rows:raise ValueError('no valid golden-context demonstration')
    f,c,m,t,k,turns,reasons=zip(*rows)
    np.savez_compressed(path,features=f,cards=c,masks=m,targets=t,
        seeds=np.full(len(rows),seed,dtype=np.int64),dev=np.full(len(rows),seed%5==0,dtype=bool),
        kinds=k,turns=turns,reasons=reasons)

def task(spec):
    import hsbg_engine as E
    from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
    source,index,args,old=spec;seed=source['seed'];out=Path(args['out'])
    dest=out/'sources'/str(seed);dest.mkdir(parents=True,exist_ok=False)
    policy,pool=resources(args['checkpoint'],args['library'])
    split='dev' if seed%5==0 else 'train';excluded=Counter()
    if split=='dev':
        case=old['case'];root=Path(args['development'])
        path=root/'trials/growth'/f'case-{case:03d}-complete_gold-r00'/'decisions.jsonl.gz'
        if file_identity(path)!={k:old['trace'][k] for k in ('bytes','sha256')}:raise ValueError('sealed teacher trace differs')
        labels=[];actions=0
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line)
                if r['event']!='action':continue
                game=E.Game.from_json(json.dumps(r['state_before']))
                decision=policy.decide(game,r['seat'])
                if decision.action!=r['action']:raise ValueError('replayed teacher action differs')
                f,c=M.observe(game,r['seat'],M.SCHEMA)
                row=label(np.frombuffer(f,dtype='<f4'),np.frombuffer(c,dtype='<i4'),
                    policy.evaluator.last_mask,decision,r['turn'])
                if row is not None:labels.append(row)
                else:excluded['forced' if decision.info.get('forced') else 'outside_gate']+=1
                game.step(r['seat'],r['action'])
                if json.loads(game.state_json())!=r['state_after'] or any(game.diagnostic_counts()):
                    raise ValueError('development native replay differs')
                actions+=1
        evidence={'reused_trace':str(path),'trace':old['trace'],'native_replayed_actions':actions,'rank':old['rank']}
    else:
        state,edits=fixture(source['state'],'complete_gold')
        future_seed=12070000+index*100
        fresh=json.loads(E.Game(seed=future_seed).state_json());state['seed']=fresh['seed'];state['rng']=fresh['rng']
        identity={'source_seed':seed,'future_seed':future_seed,'split':split,'teacher_mode':'growth',
            'mechanism_schema':M.SCHEMA,'mechanism_schema_sha256':M.SCHEMA_SHA256,
            'checkpoint_sha256':args['parent_sha256'],'interventions':edits,
            'scope':'explicit golden-engine source; heuristic demonstration, not optimal actions'}
        recorder=Recorder(dest/'decisions.jsonl.gz',identity=identity,policy=policy)
        try:
            result=evaluate_policy(policy,EvalConfig(games=1,seed=future_seed,research_initial_state=state,
                collect_economy=True,decision_recorder=recorder,mirror_pool=pool,
                mirror_strict=False,mirror_seats=7,mirror_p=1.))
            if result['illegal_actions'] or result['truncated_games']:raise ValueError('demonstration quality failure')
            trace=recorder.finish(result);validation=validate_trace(dest/'decisions.jsonl.gz',replay=16)
        except BaseException:
            recorder.abort();raise
        labels=recorder.labels;excluded=recorder.excluded
        evidence={'trace':trace,'validation':validation,'rank':result['rank_sequence'][0]}
        atomic_json(dest/'evaluation.json',result)
    save_rows(dest/'labels.npz',labels,seed)
    complete={'seed':seed,'split':split,'kind':'gold_engine','labels':len(labels),
        'labels_file':file_identity(dest/'labels.npz'),'excluded':dict(excluded),'evidence':evidence}
    atomic_json(dest/'complete.json',complete);return complete

def main(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    bank=Path(args.bank);manifest=json.loads((bank/'manifest.json').read_text())
    if file_identity(bank/'states.jsonl.gz')!=manifest['data']:raise ValueError('source bank identity differs')
    groups={}
    with gzip.open(bank/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            source=json.loads(line);p=source['state']['players'][0]
            if source['seat']!=0 or not source['state']['lobby_races']&64:continue
            if not 8<=source['turn']<=9 or p['health']<15 or p['pending'] or p['ended_turn']:continue
            if source['seed'] not in groups or source['turn']<groups[source['seed']]['turn']:groups[source['seed']]=source
    sources=[groups[k] for k in sorted(groups)]
    if args.probe:sources=[next(s for s in sources if s['seed']%5!=0),next(s for s in sources if s['seed']%5==0)]
    development=Path(args.development);old=json.loads((development/'complete.json').read_text())
    if not old.get('valid') or old.get('probe') or old['run']!=file_identity(development/'run.json'):
        raise ValueError('development study not sealed')
    parent=file_identity(args.checkpoint)
    if old['models']['growth']!=parent:raise ValueError('teacher parent differs')
    by_seed={r['source_seed']:r for r in old['rows'] if r['name']=='growth' and r['variant']=='complete_gold' and r['replica']==0}
    source_meta=source_identity(Path(__file__).resolve().parents[1])
    if source_meta['tracked_dirty']:raise ValueError('committed source required')
    anchor_root=Path(args.anchors);anchor=json.loads((anchor_root/'manifest.json').read_text())
    if file_identity(anchor_root/'anchors.npz')!=anchor['files']['anchors.npz']:raise ValueError('anchor identity differs')
    from hsbg.az.train.common import engine_info
    from hsbg.az.train.eval_policy import TRAINING_CONTRACT
    library=Path(args.library);snapshots={p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}
    previous=json.loads((development/'run.json').read_text())
    if not snapshots or previous['snapshot_files']!=snapshots or previous['engine']!=engine_info():
        raise ValueError('development engine or snapshot field differs')
    if previous['source_manifest']!=file_identity(bank/'manifest.json'):raise ValueError('source bank differs from development')
    atomic_json(root/'run.json',{'args':vars(args),'source':source_meta,'environment':environment_identity(),
        'engine':engine_info(),'action_contract':dict(TRAINING_CONTRACT),'snapshot_files':snapshots,
        'parent':parent,'source_manifest':file_identity(bank/'manifest.json'),
        'development':file_identity(development/'complete.json'),'gate':current_gate_spec('gold_engine')})
    worker_args={**vars(args),'parent_sha256':parent['sha256']}
    tasks=[(s,i,worker_args,by_seed.get(s['seed'])) for i,s in enumerate(sources)]
    completed=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(task,tasks):
            completed.append(row);print(json.dumps({'completed':len(completed),'total':len(tasks),'labels':row['labels']}),flush=True)
    arrays={}
    for row in completed:
        path=root/'sources'/str(row['seed'])/'labels.npz'
        if file_identity(path)!=row['labels_file']:raise ValueError('source labels changed')
        with np.load(path,allow_pickle=False) as data:
            for k in data.files:arrays.setdefault(k,[]).append(data[k])
    np.savez_compressed(root/'labels.npz',**{k:np.concatenate(v) for k,v in arrays.items()})
    shutil.copyfile(anchor_root/'anchors.npz',root/'anchors.npz')
    with gzip.open(root/'tasks.jsonl.gz','xt',encoding='utf-8') as stream:
        for row in completed:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    if file_identity(args.checkpoint)!=parent:raise ValueError('parent changed')
    if {p.name:file_identity(p) for p in sorted(library.glob('snap_*.json.gz'))}!=snapshots:raise ValueError('snapshot field changed')
    atomic_json(root/'manifest.json',{'schema':'hsai.payoff-process-dataset/v1',
        'mechanism_schema_sha256':M.SCHEMA_SHA256,'source':source_meta,'probe':args.probe,
        'run':file_identity(root/'run.json'),'source_games':len(sources),'labels':sum(r['labels'] for r in completed),
        'training_sources':sum(s['seed']%5!=0 for s in sources),'development_sources':sum(s['seed']%5==0 for s in sources),
        'scope':'multi-turn heuristic growth only inside gold_engine gate; no rank/value training targets',
        'limitations':['Explicit endowments, not natural acquisition','Development sources reused; exploratory only',
            'Labels exclude forced actions and any state outside the declared gold_engine gate'],
        'files':{n:file_identity(root/n) for n in ('labels.npz','anchors.npz','tasks.jsonl.gz')}})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True)
    p.add_argument('--anchors',required=True);p.add_argument('--development',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--library',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--probe',action='store_true');main(p.parse_args())
