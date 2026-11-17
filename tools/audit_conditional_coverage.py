"""Describe public conditional-gate reachability in already sealed natural traces."""
import argparse,gzip,json,sys
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.compare_research_panels import load_sealed_panel

GROUPS=(('BG_LOE_077','TB_BaconUps_045'),('BG33_823','BG33_823_G'),
        ('BG33_825','BG33_825_G'))

@lru_cache(maxsize=1)
def card_groups():
    import hsbg_engine as E
    cols,raw=E.card_static();n=len(raw)//(4*cols)
    groups={cid:g for g,ids in enumerate(GROUPS) for cid in ids}
    return {i-1:groups[E.card_info(i)[0]] for i in range(1,n) if E.card_info(i)[0] in groups}

def item_group(item,groups):
    for key in ('item','Minion','Spell'):
        if key in item:return item_group(item[key],groups)
    return groups.get(item['card'])

def public_gate(state,seat,legal=None):
    p=state['players'][seat];groups=card_groups()
    owned={g for item in p['board']+p['hand'] if (g:=item_group(item,groups)) is not None}
    pirates=bool(state['lobby_races']&(1<<6))
    offers=[]
    if legal is not None:
        offers=[i for i,item in enumerate(p['shop'][:8])
                if legal[8+i] and (g:=item_group(item,groups)) is not None and g not in owned]
    return {'pirates':pirates,'owned_groups':sorted(owned),
            'active':pirates and len(owned)>=2,'missing_core_buy_slots':offers}

def audit_trace(ref):
    path=Path(ref['directory'])/ref['file']
    if file_identity(path)['sha256']!=ref['sha256']:raise ValueError('trace identity differs')
    events=Counter();games=defaultdict(set);turns=defaultdict(set);active_by_turn=Counter();examples=[]
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['event']!='action':continue
            seed=r['seed'];turn=r['turn'];a=r['action']
            info=public_gate(r['state_before'],r['seat'],r['legal_mask'])
            flags=['all']
            if info['pirates']:
                flags+=['pirate_lobby','owned_'+str(len(info['owned_groups']))]
                if info['active']:
                    flags.append('gate_active');active_by_turn[turn]+=1
                    if a==0:flags.append('active_end_turn')
                if len(info['owned_groups'])==1 and info['missing_core_buy_slots']:
                    flags.append('one_core_missing_second_buy_available')
                    if 8<=a<16 and a-8 in info['missing_core_buy_slots']:
                        flags.append('second_core_bought')
                    if len(examples)<12:examples.append({'seed':seed,'turn':turn,
                        'action_index':r['action_index'],'action':a,**info})
            for flag in flags:
                events[flag]+=1;games[flag].add(seed);turns[flag].add((seed,turn))
    return {'model':ref['identity']['name'],'events':dict(events),
        'games':{k:sorted(v) for k,v in games.items()},
        'turns':{k:sorted(v) for k,v in turns.items()},
        'active_by_turn':dict(active_by_turn),'examples':examples}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('panels',nargs='+')
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--model',action='append',default=[]);args=p.parse_args();refs=[]
    for path in args.panels:
        panel=load_sealed_panel(path)
        refs.extend(r for m in panel['models'] for r in m['decision_traces']
                    if not args.model or r['identity']['name'] in args.model)
    if not refs:raise ValueError('no sealed traces selected')
    merged={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(audit_trace,refs):
            m=merged.setdefault(result['model'],{'events':Counter(),'games':defaultdict(set),
                'turns':defaultdict(set),'active_by_turn':Counter(),'examples':[]})
            m['events'].update(result['events']);m['active_by_turn'].update(result['active_by_turn'])
            for field in ('games','turns'):
                for k,values in result[field].items():
                    m[field][k].update(tuple(v) if isinstance(v,list) else v for v in values)
            m['examples']+=result['examples']
    for m in merged.values():
        m['counts']={k:{'actions':m['events'][k],'games':len(m['games'][k]),'turns':len(m['turns'][k])}
                     for k in m['events']}
        m['active_by_turn']=dict(m['active_by_turn']);m['examples']=m['examples'][:24]
        m['game_seeds']={k:sorted(v) for k,v in m['games'].items()}
        for field in ('events','games','turns'):del m[field]
    result={'models':merged,'panels':{p:file_identity(p) for p in args.panels},
        'scope':'All recorded turns, descriptive natural reachability only; no demonstration labels or training states exported',
        'gate':'Pirates in lobby and >=2 distinct owned Brann/Rogers/Privateer groups; shop excluded'}
    atomic_json(args.out,result)
    for name,m in merged.items():print(json.dumps({'model':name,'counts':m['counts']}),flush=True)

if __name__=='__main__':main()
