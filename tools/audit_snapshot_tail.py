"""Read sealed panels and training histories; identify unobserved snapshot tails.

No policy execution, new games, filtering, or promotion. A late terminal state
does not by itself prove a missing injection occurred in that particular game.
"""
import argparse,gzip,json,re,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity

FIELDS=('mirror_attempt','mirror_applied','mirror_missing','mirror_fallback',
        'mirror_rejected','mirror_errors','mirror_pool_unbalanced','mirror_pool_species_unbalanced')

def scan_trace(task):
    panel,name,trace,last_turn=task
    path=Path(trace['directory'])/trace['file']
    if file_identity(path)!={k:trace[k] for k in ('bytes','sha256')}:
        raise ValueError('trace identity differs')
    late=[];actions=0;games=0;last_action=None;start=None
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('{"event":"action",'):
                actions+=1;last_action=line
            elif line.startswith('{"event":"game_start",'):
                start=json.loads(line);last_action=None
            elif line.startswith('{"event":"game_end",'):
                row=json.loads(line);games+=1
                if row['state']['turn']>last_turn:
                    action=json.loads(last_action) if last_action else None
                    p=action['state_after']['players'][action['seat']] if action else None
                    late.append({'panel':panel,'model':name,'seed':row['seed'],'rank':row['rank'],
                        'world_final_turn':row['state']['turn'],'library_last_turn':last_turn,
                        'last_learner_action_turn':action['turn'] if action else None,
                        'learner_alive_after_last_action':p['alive'] if p else None,
                        'learner_health_after_last_action':p['health'] if p else None,
                        'learner_actions_beyond_library':bool(action and action['turn']>last_turn),
                        'trace':str(path),'trace_identity':{k:trace[k] for k in ('bytes','sha256')},
                        'checkpoint_sha256':trace['identity']['checkpoint_sha256']})
    if actions!=trace['actions'] or games!=trace['games']:raise ValueError('trace coverage mismatch')
    return {'model':name,'games':games,'actions':actions,'late':late}

def main(args):
    panels=[];tasks=[]
    for path in map(Path,args.panels):
        data=json.loads(path.read_text())
        rows=[]
        for model in data['models']:
            field=model['field_stats'];last=field['library']['turn_max']
            rows.append({'name':model['name'],'games':model['games'],
                'checkpoint':model['checkpoint'],'checkpoint_sha256':model['checkpoint_sha256'],
                **{k:field[k] for k in FIELDS}})
            if field['mirror_missing'] or field['mirror_fallback']:
                tasks.extend((str(path),model['name'],t,last) for t in model['decision_traces'])
        panels.append({'path':str(path),'identity':file_identity(path),
            'protocol':data['protocol'],'models':rows})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:scans=list(pool.map(scan_trace,tasks))
    runs={'legacy_2m':'nus-input-base-20261118-1','mechanism_2m':'nus-input-mech-20261118-1'}
    for i,seed in enumerate((1181811,2181811)):
        for arm in ('control','process'):runs[f'transfer_{arm}_s{i+1}']=f'nus-transfer-{arm}-20261118-{seed}'
    for arm in ('natural','endowed'):runs[f'curriculum_{arm}_2m']=f'nus-curriculum-{arm}-20261118-5182001-2'
    histories={}
    for name,directory in runs.items():
        path=Path('artifacts/runs')/directory/'train/history.jsonl';counts=Counter();affected=[]
        with path.open() as stream:
            for line in stream:
                r=json.loads(line)
                if 'ppo' not in r:continue
                c=r['collect'];counts['iterations']+=1;counts['decisions']+=c['decisions']
                for k in FIELDS:counts[k]+=int(c.get(k,0))
                if c.get('mirror_missing') or c.get('mirror_fallback'):
                    affected.append({'iteration':r.get('iteration',r.get('iter')),
                        **{k:c.get(k) for k in FIELDS}})
        histories[name]={'path':str(path),'identity':file_identity(path),'totals':dict(counts),'affected':affected}
    result={'panels':panels,'trace_scan':scans,'training_histories':histories,
        'late_games':[x for s in scans for x in s['late']],
        'scope':'Read-only missing/fallback inventory; late games are diagnostic candidates, not proof of causal effect or grounds for removing outcomes'}
    atomic_json(Path(args.out)/'complete.json',result)
    print(json.dumps({'late_games':result['late_games'],'training':{k:v['totals'] for k,v in histories.items()}}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8)
    main(p.parse_args())
