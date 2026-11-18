"""Read-only accounting of a deployed payoff card through the same recruit turn.

Tracks the exact entity separately from same-card replacements/triples. A sale
or disappearance is descriptive, not automatically a tactical error.
"""
import argparse,gzip,json,multiprocessing,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.evaluation.quality import suffix_quality


def account(actions,payoff_ids):
    pending=[];events=[];action_count=0
    for row in actions:
        action_count+=1
        seat=row['seat'];before=row['state_before']['players'][seat];after=row['state_after']['players'][seat]
        before_ids={m['eid'] for m in before['board']};after_ids={m['eid'] for m in after['board']}
        for event in pending:
            if event['turn']!=row['turn']:raise ValueError('deployed entity has no recruit-turn terminal')
            eid=event['eid']
            if event['first_departure'] is None and eid in before_ids and eid not in after_ids and row['action']!=0:
                sold=(16<=row['action']<23 and before['board'][row['action']-16]['eid']==eid)
                event['first_departure']={'action_index':row['action_index'],'action':row['action'],
                    'kind':'sold' if sold else 'other_native_transition',
                    'payoff_still_on_board':any(m['card'] in payoff_ids for m in after['board'])}
        if row['policy_info'].get('deployment_override_executed'):
            created=[m for m in after['board'] if m['eid'] not in before_ids and m['card'] in payoff_ids]
            if len(created)!=1:raise ValueError('intervention did not create exactly one payoff entity')
            pending.append({'turn':row['turn'],'action_index':row['action_index'],'eid':created[0]['eid'],
                'original_greedy_action':row['policy_info']['original_greedy_action'],'first_departure':None})
        if row['action']==0 or not after['alive']:
            # EndTurn's AFTER state may contain combat or the next turn.
            board=before['board'] if row['action']==0 else after['board']
            for event in pending:
                event.update(end_action_index=row['action_index'],end='end_turn' if row['action']==0 else 'recruit_death',
                    entity_at_end=any(m['eid']==event['eid'] for m in board),
                    any_payoff_at_end=any(m['card'] in payoff_ids for m in board))
                events.append(event)
            pending=[]
    if pending:raise ValueError('unfinished deployment accounting')
    return events,action_count


def scan(spec):
    import hsbg_engine as E
    root,row=spec;trial=Path(root)/'trials'/row['name']/f"case-{row['case']:03d}-{row['variant']}-r{row['replica']:02d}"
    result=json.loads((trial/'result.json').read_text())
    if result['summary']!=row:raise ValueError('trial summary changed')
    suffix_quality(result['evaluation'])
    path=trial/'decisions.jsonl.gz'
    if file_identity(path)!={k:row['trace'][k] for k in ('bytes','sha256')}:raise ValueError('trace changed')
    ncols,raw=E.card_static();n=len(raw)//(4*ncols)
    ids={i-1 for i in range(1,n) if E.card_info(i)[0] in ('BG36_344','BG36_344_G')}
    if len(ids)!=2:raise ValueError('payoff vocabulary differs')
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        events,count=account((r for r in map(json.loads,stream) if r['event']=='action'),ids)
    if count!=row['trace']['actions'] or len(events)!=row['deployment_actions']:
        raise ValueError('intervention or action coverage differs')
    return {'case':row['case'],'source_seed':row['source_seed'],'replica':row['replica'],
        'trace':file_identity(path),'actions':count,'events':events}


def main(args):
    root=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    complete=json.loads((root/'complete.json').read_text())
    if not complete['valid'] or complete['probe'] or complete['run']!=file_identity(root/'run.json'):
        raise ValueError('complete sealed ability diagnostic required')
    rows=complete['rows']
    if len(rows)!=98 or {r['name'] for r in rows}!={'deploy_payoff'} or {r['variant'] for r in rows}!={'complete_gold'}:
        raise ValueError('entire predeclared deployment arm required')
    if len({(r['case'],r['replica']) for r in rows})!=98 or len({r['source_seed'] for r in rows})!=49:
        raise ValueError('source coverage differs')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        details=list(pool.map(scan,[(str(root),r) for r in rows]))
    counts=Counter()
    for d in details:
        for e in d['events']:
            counts['deployments']+=1;counts[e['end']]+=1
            counts['entity_at_end']+=int(e['entity_at_end']);counts['any_payoff_at_end']+=int(e['any_payoff_at_end'])
            counts['first_departure_'+(e['first_departure']['kind'] if e['first_departure'] else 'none')]+=1
    result={'input':{'root':str(root),'complete':file_identity(root/'complete.json'),'run':complete['run']},
        'counts':dict(counts),'trials':details,
        'scope':'Read-only within-turn retention after sealed interventions; no new games, no training labels',
        'limitations':['Events can overlap and recur within the same game','Sales need not be mistakes',
            'EndTurn metrics use pre-combat state','An entity can disappear through native triples or transforms']}
    atomic_json(out/'complete.json',result);print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=4);main(p.parse_args())
