"""Read-only public-history sidecar, exactly aligned with a sealed teacher dataset.

No new model observations are enabled and no demonstration is changed. The
sidecar keeps source splits and original row indices for a future controlled
history-input experiment.
"""
import argparse,gzip,json,multiprocessing,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.az.train.public_turn_history import PublicTurnHistory,project_own_record,SCHEMA,SCHEMA_SHA256


def native_gate(state,seat,groups):
    p=state['players'][seat]
    owned={int(m['card'])+1 for m in p['board']}
    owned.update(int(h['item']['Minion']['card'])+1 for h in p['hand'] if 'Minion' in h['item'])
    return bool(state['lobby_races']&64 and all(owned.intersection(group) for group in groups))


def annotate(spec):
    root,task,out,groups=spec;root=Path(root);dest=Path(out)/'sources'/str(task['seed']);dest.mkdir(parents=True,exist_ok=False)
    labels=root/'sources'/str(task['seed'])/'labels.npz'
    if file_identity(labels)!=task['labels_file']:raise ValueError('source labels changed')
    trace=Path(task['evidence'].get('reused_trace',str(labels.parent/'decisions.jsonl.gz')))
    expected=task['evidence']['trace']
    if file_identity(trace)!={k:expected[k] for k in ('bytes','sha256')}:raise ValueError('source trace changed')
    with np.load(labels,allow_pickle=False) as raw:
        data={k:raw[k] for k in ('targets','turns','reasons','seeds','dev')}
    history=PublicTurnHistory();marked=[];counts=[];indices=[];total=0
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for row in map(json.loads,stream):
            if row['event']!='action':continue
            before=project_own_record(row['state_before'],row['seat'])
            after=project_own_record(row['state_after'],row['seat'])
            flags,count=history.before(before);info=row['policy_info']
            keep=(not info.get('forced') and info.get('research_option_active')
                and native_gate(row['state_before'],row['seat'],groups))
            if keep:
                i=len(marked)
                if (i>=len(data['targets']) or int(data['targets'][i].sum())!=1
                    or int(data['targets'][i].argmax())!=row['action'] or data['turns'][i]!=row['turn']
                    or data['reasons'][i]!=info['research_operator_reason']):
                    raise ValueError('source label chronology differs')
                marked.append(flags);counts.append(count);indices.append(row['action_index'])
            history.after(before,after,row['action']);total+=1
    if total!=expected['actions'] or len(marked)!=task['labels'] or len(marked)!=len(data['targets']):
        raise ValueError('source action or label coverage differs')
    path=dest/'history.npz'
    np.savez_compressed(path,played_by_slot=np.stack(marked),actions_so_far=np.asarray(counts,dtype=np.int32),
        trace_action_index=np.asarray(indices,dtype=np.int32),seeds=data['seeds'],dev=data['dev'],turns=data['turns'])
    result={'seed':task['seed'],'split':task['split'],'rows':len(marked),'actions':total,
        'trace':{'path':str(trace),'identity':file_identity(trace)},'labels':file_identity(labels),'history':file_identity(path)}
    atomic_json(dest/'complete.json',result);return result


def main(args):
    root=Path(args.dataset);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    if file_identity(root/'manifest.json')['sha256']!=args.expected_manifest_sha256:
        raise ValueError('expected original manifest identity differs')
    if manifest['schema']!='hsai.payoff-process-dataset/v1' or manifest['probe']:
        raise ValueError('complete original teacher dataset required')
    for name,item in manifest['files'].items():
        if file_identity(root/name)!=item:raise ValueError('original file changed: '+name)
    with gzip.open(root/'tasks.jsonl.gz','rt',encoding='utf-8') as stream:tasks=[json.loads(s) for s in stream]
    offsets={};cursor=0
    for task in tasks:offsets[task['seed']]=cursor;cursor+=task['labels']
    if cursor!=manifest['labels']:raise ValueError('task row counts differ from original manifest')
    if args.probe:tasks=[next(t for t in tasks if t['split']=='train'),next(t for t in tasks if t['split']=='dev')]
    from hsbg.az.train.conditional_economy import current_gate_spec
    groups=current_gate_spec('gold_engine')['embedding_ids']
    source=source_identity(Path(__file__).resolve().parents[1])
    if source['tracked_dirty']:raise ValueError('committed source required')
    run={'args':vars(args),'source':source,'original_manifest':file_identity(root/'manifest.json'),
        'history_schema':SCHEMA,'history_schema_sha256':SCHEMA_SHA256,'gate_groups':groups}
    atomic_json(out/'run.json',run)
    completed=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(annotate,[(str(root),t,str(out),groups) for t in tasks]):
            completed.append(result)
            if len(completed)%16==0:print(json.dumps({'sources':len(completed),'total':len(tasks)}),flush=True)
    arrays={}
    for row in completed:
        path=out/'sources'/str(row['seed'])/'history.npz'
        if file_identity(path)!=row['history']:raise ValueError('annotation changed')
        with np.load(path,allow_pickle=False) as raw:
            for k in raw.files:arrays.setdefault(k,[]).append(raw[k])
    combined={k:np.concatenate(v) for k,v in arrays.items()}
    if not args.probe:
        with np.load(root/'labels.npz',allow_pickle=False) as old:
            if any(not np.array_equal(old[k],combined[k]) for k in ('seeds','dev','turns')):
                raise ValueError('combined original row alignment differs')
        if len(combined['seeds'])!=manifest['labels']:raise ValueError('combined label count differs')
    combined['original_row']=np.concatenate([np.arange(offsets[r['seed']],offsets[r['seed']]+r['rows'],dtype=np.int64) for r in completed])
    np.savez_compressed(out/'history.npz',**combined)
    atomic_json(out/'complete.json',{'passed':True,'probe':args.probe,'run':file_identity(out/'run.json'),
        'rows':len(combined['seeds']),'training_rows':int((~combined['dev']).sum()),'development_rows':int(combined['dev'].sum()),
        'sources':completed,'history':file_identity(out/'history.npz'),
        'scope':'Read-only aligned public-history annotations; not new labels, learned capability or changed live/PPO observations',
        'limitations':['History starts at the declared controller reset; no pre-reset history invented',
            'A newly introduced entity can be a Play-created token, not only the played hand entity',
            'Original teacher labels and development split unchanged; teacher optimality not assumed']})
    print(json.dumps({'sources':len(completed),'rows':len(combined['seeds']),'probe':args.probe}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--probe',action='store_true');main(p.parse_args())
