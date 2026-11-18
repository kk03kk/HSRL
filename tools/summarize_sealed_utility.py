"""Recompute an explicit utility from already admitted complete policy arms."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import file_identity,atomic_json
from tools.suffix_outcomes import summarize


def main(args):
    analysis=json.loads(Path(args.analysis).read_text())
    if not analysis['quality']['passed']:raise ValueError('strict admitted analysis required')
    supplied={file_identity(p)['sha256']:Path(p) for p in args.completes}
    rows=[];sources=[];policies=analysis['selected_whole_policy_arms'];variants=analysis['selected_whole_variants']
    for entry in analysis['inputs']:
        path=supplied.get(entry['complete']['sha256'])
        if path is None or file_identity(path)!=entry['complete']:raise ValueError('admitted complete asset missing or changed')
        data=json.loads(path.read_text())
        if not data['valid'] or data['probe']:raise ValueError('complete ability study required')
        rows.extend(r for r in data['rows'] if (policies is None or r['name'] in policies)
            and (variants is None or r['variant'] in variants))
        sources.append({'path':str(path),'identity':file_identity(path)})
    result=summarize(rows,args.references,[float(s) for s in args.utility.split(',')])
    result.update(admitted_analysis=file_identity(args.analysis),inputs=sources,
        declaration='This is an explicit supplementary scoring transform of sealed final ranks, not a new formal promotion criterion')
    atomic_json(args.out,result)
    print(json.dumps({k:v for k,v in result['paired'].items() if k.endswith('/complete_gold')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True);p.add_argument('--completes',nargs='+',required=True)
    p.add_argument('--references',nargs='+',default=['parent']);p.add_argument('--utility',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
