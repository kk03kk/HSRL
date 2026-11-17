"""Read every sealed suffix result and inventory quality failures, without filtering outcomes."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.evaluation.quality import suffix_quality,MIRROR_ERRORS

def inspect(roots,datasets):
    reports=[]
    for root in map(Path,roots):
        complete=json.loads((root/'complete.json').read_text());paths=[]
        for r in complete['rows']:
            paths.append(root/'trials'/r['name']/f"case-{r['case']:03d}-{r['variant']}-r{r['replica']:02d}"/'result.json')
        reports.append(scan(root,paths,False))
    for root in map(Path,datasets):
        reports.append(scan(root,sorted(root.glob('sources/*/evaluation.json')),True))
    return {'reports':reports,'scope':'All observed results retained; partial dataset scans are not complete dataset validation'}

def scan(root,paths,dataset):
    failures=[];totals=Counter();by=Counter();games=[]
    for path in paths:
        raw=json.loads(path.read_text());r=raw if dataset else raw['evaluation']
        name='dataset_training' if dataset else raw['summary']['name']+'/'+raw['summary']['variant']
        by[name]+=1
        for k in MIRROR_ERRORS:totals[k]+=r['mirror'][k]
        try:suffix_quality(r)
        except ValueError as e:
            failures.append({'path':str(path),'identity':file_identity(path),'arm':name,'error':str(e),
                'seed':r['seed_sequence'],'rank':r['rank_sequence'],
                'initial_turn':r.get('initial_turn'),'last_learner_eot':max([t['turn'] for t in r['end_turn_economy']],default=None),
                'mirror':r['mirror'],'truncated_combats':r['truncated_combats']})
    return {'directory':str(root),'dataset':dataset,'dataset_complete':(root/'manifest.json').exists() if dataset else None,
        'results':len(paths),'by_arm':dict(by),'failure_count':len(failures),'failure_totals':dict(totals),'failures':failures}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='*',default=[])
    p.add_argument('--datasets',nargs='*',default=[]);p.add_argument('--out',required=True);args=p.parse_args()
    result=inspect(args.runs,args.datasets);atomic_json(args.out,result)
    for r in result['reports']:
        print(json.dumps({k:v for k,v in r.items() if k!='failures'}),flush=True)
        for failure in r['failures'][:3]:print(json.dumps(failure),flush=True)
