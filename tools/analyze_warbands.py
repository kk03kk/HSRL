"""Compile a read-only manual collection into reference native combat fixtures."""
import argparse
import json
from local_offline import prepare

if __name__=='__main__':
    handles=prepare()
    from taverngym.research.manual_bank import analyze
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='runs/warband-collector/collection.json')
    p.add_argument('--out',required=True)
    p.add_argument('--after-start-turns',type=int,nargs='*',default=[])
    p.add_argument('--trials',type=int,default=32)
    a=p.parse_args()
    result=analyze(a.input,a.out,after_start_turns=a.after_start_turns,trials=a.trials)
    print(json.dumps({'usable':result['usable_records'],'total':len(result['records']),'report':a.out+'/report.json'}))
