"""Matched extra-training control versus one student-state aggregation round."""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import file_identity
from hsbg.evaluation.quality import suffix_quality
from workflows.learn_cycle_process import run
from workflows.learn_payoff_process import validate_dataset,mixed_event_sampling,separate_assessment


def mixed_origin_sampling(raw):
    groups,sizes=mixed_event_sampling(raw);origin=raw['origin'];out={};batches={}
    if set(origin.tolist())!={0,1} or np.any(raw['dev']&(origin!=0)):
        raise ValueError('invalid query origin or held-out query leakage')
    for name,indices in groups.items():
        if sizes[name]%2:raise ValueError('source mixture must split batches equally')
        for source in (0,1):
            key=name+'/'+str(source);out[key]=indices[origin[indices]==source];batches[key]=sizes[name]//2
            if not len(out[key]):raise ValueError('empty predeclared query/original sampling group: '+key)
    return out,batches


def validate_queries(root,plan_path):
    root=Path(root);plan=json.loads(Path(plan_path).read_text());original=Path(plan['original_dataset'])
    meta=json.loads((root/'manifest.json').read_text());identity=json.loads((root/'run.json').read_text())
    if meta['schema']!='hsai.payoff-aggregate-dataset/v1' or meta['probe'] or meta['query_sources']!=64:
        raise ValueError('complete predeclared 64-source query round required')
    if meta['run']!=file_identity(root/'run.json') or identity['plan']!=file_identity(plan_path):
        raise ValueError('query run or plan changed')
    if identity['student']!=file_identity(plan['student']) or identity['student']['sha256']!=plan['student_sha256']:
        raise ValueError('fixed starting student changed')
    if identity['original_manifest']!=file_identity(original/'manifest.json') or identity['original_manifest']['sha256']!=plan['original_manifest_sha256']:
        raise ValueError('original data identity differs')
    original_admission=validate_dataset(original,plan['parent'])
    for name,item in meta['files'].items():
        if file_identity(root/name)!=item:raise ValueError('query file changed: '+name)
    original_meta=json.loads((original/'manifest.json').read_text())
    for name,item in original_meta['files'].items():
        if file_identity(original/name)!=item:raise ValueError('original file changed: '+name)
    old_run=json.loads((original/'run.json').read_text())
    for key in ('engine','action_contract','source_manifest','snapshot_files','gate'):
        if identity[key]!=old_run[key]:raise ValueError('query field differs: '+key)
    with gzip.open(original/'tasks.jsonl.gz','rt',encoding='utf-8') as stream:old_tasks=[json.loads(s) for s in stream]
    all_seeds=sorted(t['seed'] for t in old_tasks);train=sorted(t['seed'] for t in old_tasks if t['split']=='train')
    expected=[train[i] for i in np.linspace(0,len(train)-1,64,dtype=int)]
    if identity['selected_sources']!=expected:raise ValueError('query source selection changed')
    with gzip.open(root/'query_tasks.jsonl.gz','rt',encoding='utf-8') as stream:tasks=[json.loads(s) for s in stream]
    if [t['seed'] for t in tasks]!=expected:raise ValueError('query source coverage differs')
    actions=0;labels=0;recruit_deaths=0;excluded={};evaluations=[]
    from collections import Counter
    excluded=Counter()
    for task in tasks:
        if task['split']!='train' or task['seed']%5==0:raise ValueError('development source used in queries')
        i=all_seeds.index(task['seed'])
        if task['source_index']!=i or task['future_seed']!=12300000+i*100:raise ValueError('query future RNG differs')
        dest=root/'sources'/str(task['seed']);path=dest/'evaluation.json'
        if file_identity(path)!=task['evaluation'] or file_identity(dest/'queries.npz')!=task['queries_file']:
            raise ValueError('source query evidence changed')
        result=json.loads(path.read_text());quality=suffix_quality(result)
        if result['rank_sequence']!=[task['rank']] or quality['actions']!=task['trace']['actions']:
            raise ValueError('source query outcome differs')
        actions+=quality['actions'];labels+=task['labels'];recruit_deaths+=len(task['recruit_deaths']);excluded.update(task['excluded'])
        evaluations.append({'path':str(path),'identity':task['evaluation']})
    if labels!=meta['query_labels'] or actions!=labels+sum(excluded.values()):raise ValueError('query accounting is incomplete')
    with np.load(original/'labels.npz',allow_pickle=False) as old,np.load(root/'labels.npz',allow_pickle=False) as data,np.load(root/'queries.npz',allow_pickle=False) as queries:
        count=len(old['dev'])
        if set(data.files)!=set(old.files)|{'origin'} or set(queries.files)!=set(old.files):raise ValueError('aggregate schema differs')
        for key in old.files:
            if not np.array_equal(data[key][:count],old[key]) or not np.array_equal(data[key][count:],queries[key]):
                raise ValueError('aggregation changed original or query rows: '+key)
        if queries['dev'].any() or set(queries['seeds'].tolist())!=set(expected):raise ValueError('query labels source split differs')
        if not (queries['targets'].sum(-1)==1).all() or np.any(queries['targets']&~queries['masks']):
            raise ValueError('query targets violate their actual guards')
        if np.any(data['origin'][:count]!=0) or np.any(data['origin'][count:]!=1):raise ValueError('wrong row origins')
        mixed_origin_sampling(data)
    return {'passed':True,'query_sources':64,'query_actions':actions,'query_labels':labels,'recruit_deaths':recruit_deaths,
        'excluded':dict(excluded),'evaluations':evaluations,'original_admission':original_admission,
        'original_development_rows_exact':True,'query_manifest':file_identity(root/'manifest.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--queries',required=True)
    p.add_argument('--arm',choices=('replay','aggregate'),required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    args.dataset_validation=validate_queries(args.queries,args.plan)
    if args.validate_only:
        from hsbg.runtime.artifacts import atomic_json
        atomic_json(Path(args.out)/'complete.json',args.dataset_validation);sys.exit(0)
    plan=json.loads(Path(args.plan).read_text());cfg=plan['comparison']
    args.checkpoint=plan['student'];args.dataset=args.queries if args.arm=='aggregate' else plan['original_dataset']
    args.mode='all';args.updates=cfg['updates'];args.lr=cfg['learning_rate'];args.seed=cfg['seed']
    args.anchor_kl=cfg['anchor_kl_coefficient'];args.per_kind_batch=32
    if (cfg['demonstrations_per_update'],cfg['anchors_per_update'],cfg['anchor_kl_stop'],cfg['gradient_clip'])!=(96,192,.03,1):
        raise ValueError('refinement budget differs from the validated learner')
    run(args,expected_schema='hsai.payoff-aggregate-dataset/v1' if args.arm=='aggregate' else 'hsai.payoff-process-dataset/v1',
        closed_loop_fn=separate_assessment,learning_tag='payoff_query_refinement',
        sampling_plan_fn=mixed_origin_sampling if args.arm=='aggregate' else mixed_event_sampling)
