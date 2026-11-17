"""Extract public entry/active/exit decisions from one sealed development game."""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.compare_research_panels import load_sealed_panel
from tools.audit_conditional_coverage import public_gate

def main():
    import hsbg_engine as E
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',required=True)
    p.add_argument('--model',required=True);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--out',required=True);args=p.parse_args()
    panel=load_sealed_panel(args.panel);model=next(m for m in panel['models'] if m['name']==args.model)
    refs=[r for r in model['decision_traces'] if args.seed in r['seeds']]
    if len(refs)!=1:raise ValueError('expected one sealed trace containing this seed')
    ref=refs[0];path=Path(ref['directory'])/ref['file']
    if file_identity(path)['sha256']!=ref['sha256']:raise ValueError('trace identity differs')
    selected=[]
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['event']!='action' or r['seed']!=args.seed:continue
            before=public_gate(r['state_before'],r['seat']);after=public_gate(r['state_after'],r['seat'])
            if not before['active'] and not after['active']:continue
            def view(state):
                v=json.loads(E.Game.from_json(json.dumps(state)).view(r['seat']))['me']
                return {k:v[k] for k in ('board','hand','shop','gold','health','tier','pending','casting')}
            selected.append({k:r[k] for k in ('seed','turn','action_index','action','policy_info')} |
                {'before_gate':before,'after_gate':after,'public_before':view(r['state_before']),
                 'public_after':view(r['state_after'])})
    result={'panel':file_identity(args.panel),'trace':ref,'model_sha256':model['checkpoint_sha256'],
        'seed':args.seed,'scope':'Descriptive public decisions only, not demonstration labels','decisions':selected}
    atomic_json(args.out,result)
    for r in selected:
        print(json.dumps({k:r[k] for k in ('turn','action_index','action','before_gate','after_gate')},ensure_ascii=False))

if __name__=='__main__':main()
