"""Describe recorded end-turn behaviour and export bounded decision examples.

These are diagnostic observations, not counterfactual optimality judgments.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.evaluation.metrics import _trace
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.inspect_decisions import extract
from workflows.eval_human_gap import _safe_name


def audit(panel_path,out):
    panel_path,out=Path(panel_path),Path(out)
    panel=json.loads(panel_path.read_text(encoding='utf-8'))
    if out.exists(): raise FileExistsError(out)
    result={'schema':'hsai.economy-audit/v1','panel':file_identity(panel_path),'models':{}}
    examples=[]
    for model in panel['models']:
        rows=[]
        for ref in model['decision_traces']:
            path=Path(ref['directory'])/f'rows-{_safe_name(model["name"])}.jsonl.gz'
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                rows.extend(json.loads(line) for line in stream)
        actual=sorted(_trace(rows),key=lambda r:(r['seed'],r['turn']))
        if actual != sorted(model['trace'],key=lambda r:(r['seed'],r['turn'])):
            raise ValueError('raw turn rows differ from the validated panel')
        late=[r for r in rows if r['turn']>=6]
        idle=[r for r in late if r['gold']>=3]
        counts=Counter()
        for row in rows: counts.update(row.get('action_counts',{}))
        result['models'][model['name']]={
            'turn_rows':len(rows),'turn6plus_rows':len(late),'turn6plus_end_gold_ge3':len(idle),
            'turn6plus_end_gold_ge3_share':len(idle)/len(late) if late else None,
            'policy_forced_end_turn_rows':sum(bool(r.get('forced')) for r in rows),
            'action_counts':dict(counts),
            'turn6plus_hand_left_ge3':sum((r.get('hand_n') or 0)>=3 for r in late),
            'example_selection':'first two seed/turn pairs at turn >=6 with >=3 end-turn gold; examples are not automatically mistakes',
        }
        for row in sorted(idle,key=lambda r:(r['seed'],r['turn']))[:2]:
            detail=extract(panel_path,model['name'],row['seed'],turn=row['turn'])
            decisions=detail['decisions']
            end=next((d for d in reversed(decisions) if d['action']==0),None)
            name=f'{_safe_name(model["name"])}-{row["seed"]}-t{row["turn"]}.json'
            examples.append((name,detail))
            result['models'][model['name']].setdefault('examples',[]).append({
                'seed':row['seed'],'turn':row['turn'],'gold':row['gold'],'tier':row['tier'],
                'health':row['health'],'actions':[d['action'] for d in decisions],
                'end_policy_info':end['policy_info'] if end else None,
                'native_legal_at_end':[i for i,v in enumerate(end['legal_mask']) if v] if end else [],
                'file':name})
    out.mkdir(parents=True)
    for name,detail in examples: atomic_json(out/name,detail)
    atomic_json(out/'economy-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.panel,args.out)))
