"""Read sealed T12 action-cap traces; never change the action contract or policy."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.env.action_space_std import action_name


def compact_action(row):
    seat=row['seat'];before=row['state_before']['players'][seat]
    after=row['state_after']['players'][seat]
    return {'action':row['action'],'kind':action_name(row['action']).split('(')[0].split(' {')[0],
            'gold_before':before['gold'],'gold_after':after['gold'],
            'body_before':sum(m['atk']+m['hp'] for m in before['board']),
            'body_after':sum(m['atk']+m['hp'] for m in after['board']),
            'forced_reason':row['policy_info'].get('forced_reason'),
            'hand_before':len(before['hand']),'board_before':len(before['board']),
            'pending_before':bool(before['pending'])}


def summarize_turn(actions):
    if not actions or actions[-1]['action']!=0 or actions[-1]['forced_reason']!='action_cap':
        raise ValueError('expected a recorded forced EOT action')
    end=actions[-1];tail=actions[:-1][-20:]
    if not tail:raise ValueError('empty capped action tail')
    return {'actual_non_eot_actions':len(actions)-1,'tail_n':len(tail),
            'tail_kinds':dict(Counter(r['kind'] for r in tail)),
            'tail_gold_outflow':sum(max(0,r['gold_before']-r['gold_after']) for r in tail),
            'tail_body_delta':end['body_before']-tail[0]['body_before'],
            'gold_at_forced_eot':end['gold_before'],'hand_at_forced_eot':end['hand_before'],
            'board_at_forced_eot':end['board_before'],'pending_at_forced_eot':end['pending_before']}


def inspect(task):
    root,row=task
    path=Path(root)/'trials'/row['name']/f"case-{row['case']:03d}-{row['variant']}-r{row['replica']:02d}"/'decisions.jsonl.gz'
    identity=file_identity(path)
    if identity!={k:row['trace'][k] for k in ('bytes','sha256')}:raise ValueError('trace identity differs')
    actions=[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        header=json.loads(next(f))
        if header['identity']!=row['trace']['identity']:raise ValueError('trace header differs')
        for line in f:
            event=json.loads(line)
            if event.get('event')!='action':continue
            if event['turn']>12:break
            if event['turn']==12:actions.append(compact_action(event))
    expected=next(t for t in row['turns'] if t['turn']==12)
    if len(actions)!=expected['actions']:raise ValueError('turn action count differs')
    result=summarize_turn(actions)
    if result['gold_at_forced_eot']!=expected['gold_end']:raise ValueError('gold record differs')
    return {'name':row['name'],'source_seed':row['source_seed'],'replica':row['replica'],
            'trace':identity,**result}


def analyze(analysis_path,workers=8):
    sealed=json.loads(Path(analysis_path).read_text())
    if sealed['quality']['passed'] is not True:raise ValueError('strict analysis required')
    wanted=set(sealed['selected_whole_policy_arms']);tasks=[];coverage=defaultdict(list)
    for item in sealed['inputs']:
        root=Path(item['directory']);complete=root/'complete.json'
        if file_identity(complete)!=item['complete']:raise ValueError('sealed input differs')
        for row in json.loads(complete.read_text())['rows']:
            if row['name'] not in wanted or row['variant']!='complete_gold':continue
            turns=[t for t in row['turns'] if t['turn']==12]
            if not turns:continue
            capped=turns[0]['forced_reason']=='action_cap'
            coverage[row['name']].append({'source_seed':row['source_seed'],'capped':capped})
            if capped:tasks.append((str(root),row))
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(inspect,tasks))
    groups={}
    for name,denom in sorted(coverage.items()):
        selected=[r for r in rows if r['name']==name];types=Counter()
        for row in selected:types.update(row['tail_kinds'])
        groups[name]={'T12_survivors':len(denom),'T12_source_games':len({r['source_seed'] for r in denom}),
            'capped_turns':len(selected),'capped_source_games':len({r['source_seed'] for r in selected}),
            'last20_kind_counts':dict(types),'last20_actual_actions':sum(r['tail_n'] for r in selected),
            'medians':{key:statistics.median(r[key] for r in selected) if selected else None for key in
                ('tail_gold_outflow','tail_body_delta','gold_at_forced_eot','hand_at_forced_eot','board_at_forced_eot')},
            'pending_at_forced_eot':sum(r['pending_at_forced_eot'] for r in selected)}
    return {'input':file_identity(analysis_path),'groups':groups,'rows':rows,
        'scope':'Descriptive T12 capped survivor subset; no cap intervention, causal productivity or natural-game claim',
        'gold_outflow':'Sum of positive observed gold decrements, not gross spending when effects simultaneously return gold',
        'body_delta':'Recruit board total attack plus health; not combat value; forced EOT combat transition excluded'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if Path(args.out).exists():raise FileExistsError(args.out)
    result=analyze(args.analysis,args.workers);atomic_json(args.out,result);print(json.dumps(result['groups']),flush=True)
