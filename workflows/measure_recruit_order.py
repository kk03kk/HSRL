"""Test alternative orderings of the same two affordable operations in sealed natural games."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.compare_research_panels import load_sealed_panel

ROOT=Path('artifacts/runs/nus-recruit-order-20261118-2')
PANEL=Path('artifacts/runs/nus-natural12m-eval-20261118-1/panel.json')


def describe(p,a):
    if 8<=a<16:
        return ('buy',p['shop'][a-8]['sid'])
    if 16<=a<30:
        return ('sell' if a<23 else 'activate',p['board'][(a-16)%7]['eid'])
    if 46<=a<126:
        hand,pos=divmod(a-46,8)
        h=p['hand'][hand]
        anchor=p['board'][pos]['eid'] if pos<len(p['board']) and 'Minion' in h['item'] else None
        return ('play',h['hid'],anchor)
    if 126<=a<196:
        hand,pos=divmod(a-126,7)
        return ('magnet',p['hand'][hand]['hid'],p['board'][pos]['eid'])
    return None


def resolve(p,key):
    board=[m['eid'] for m in p['board']]
    kind=key[0]
    if kind=='buy':
        ids=[s['sid'] for s in p['shop']]
        return 8+ids.index(key[1]) if key[1] in ids else None
    if kind in ('sell','activate'):
        return (16 if kind=='sell' else 23)+board.index(key[1]) if key[1] in board else None
    items=[h['hid'] for h in p['hand']]
    if items.count(key[1])!=1:
        return None
    hand=items.index(key[1])
    if key[2] is None:
        return 46+8*hand+(len(board) if 'Minion' in p['hand'][hand]['item'] else 0)
    if key[2] not in board:
        return None
    pos=board.index(key[2])
    return 46+8*hand+pos if kind=='play' else 126+7*hand+pos


def sequence(state,seat,keys,seed,rng):
    import hsbg_engine as engine
    game=engine.Game.from_json(json.dumps(dict(state,seed=seed,rng=rng),separators=(',',':')))
    for key in keys:
        p=json.loads(game.player_json(seat))
        a=resolve(p,key)
        if a is None or p['pending'] or not game.legal_mask(seat)[a]:
            return None
        game.step(seat,a)
        require_clean_engine(game)
        if game.is_over() or game.turn!=state['turn'] or not game.needs_action(seat):
            return None
    p=json.loads(game.player_json(seat))
    return None if p['pending'] else p


def compare(old,new):
    common=('tier','health','armor','gold_next_turn','free_refreshes','max_gold_bonus','upgrade_cost',
            'shop','effects','scheduled','owed','quests','rewards','counters','levels','powers')
    if any(old[k]!=new[k] for k in common):
        return None
    if sorted(map(str,old['hand']))!=sorted(map(str,new['hand'])):
        return None
    statkeys=('atk','hp','max_hp','temp_atk','temp_hp')
    def assets(p):
        return [{k:v for k,v in m.items() if k not in statkeys} for m in p['board']]
    if assets(old)!=assets(new):
        return None
    changes=[new['gold']-old['gold']]
    for before,after in zip(old['board'],new['board']):
        values=[after[k]-before[k] for k in statkeys]
        changes.extend([*values[:3],values[0]-values[3],values[1]-values[4]])
    power=lambda p:sum(m['atk']+m['hp'] for m in p['board'])
    return min(changes),max(changes),changes[0],power(new)-power(old)


def scan(ref):
    import hsbg_engine as engine
    path=Path(ref['directory'])/ref['file']
    if file_identity(path)['sha256']!=ref['sha256']:
        raise ValueError('sealed archive changed')
    counts=Counter();positives=[];seeds=[];actions=0
    futures=[json.loads(engine.Game(seed=90500001+i).state_json()) for i in (0,1)]
    with gzip.open(path,'rt') as stream:
        if json.loads(next(stream))['identity']!=ref['identity']:
            raise ValueError('trajectory identity mismatch')
        for line in stream:
            event=json.loads(line)
            if event['event']=='game_start':
                seeds.append(event['seed']);previous=None;budget=Counter()
            if event['event']!='action':
                continue
            actions+=1
            t=event['turn']
            if not 7<=t<=12:
                previous=None
                continue
            seat=event['seat'];p=event['state_before']['players'][seat]
            key=describe(p,event['action'])
            pair=previous
            previous=(t,key,event) if key and not p['pending'] else None
            if not previous or not pair or pair[0]!=t or key==pair[1] or budget[t]>=5:
                continue
            a,b=pair[1],key
            left=pair[2];state=left['state_before'];start=state['players'][seat]
            reverse=resolve(start,b)
            if reverse is None or not left['legal_mask'][reverse] or resolve(start,a)!=left['action']:
                continue
            budget[t]+=1;counts['attempts']+=1
            deltas=[]
            for future in futures:
                args=future['seed'],future['rng']
                old=sequence(state,seat,(a,b),*args);new=sequence(state,seat,(b,a),*args)
                if old is None or new is None:
                    break
                delta=compare(old,new)
                if delta is None:
                    break
                deltas.append(delta)
            if len(deltas)!=2:
                continue
            counts['same_assets']+=1
            if all(d[0]>=0 and (d[2]>0 or d[3]>=2) for d in deltas):
                positives.append(dict(seed=seeds[-1],turn=t,ops=[a[0],b[0]],coin=min(d[2] for d in deltas),
                                      stats=min(d[3] for d in deltas)))
            elif all(d[1]<=0 and (d[2]<0 or d[3]<=-2) for d in deltas):
                counts['original_better']+=1
    if seeds!=ref['seeds'] or actions!=ref['actions']:
        raise ValueError('archive coverage mismatch')
    return counts,positives


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=False)
    sealed=load_sealed_panel(PANEL)
    refs=next(m for m in sealed['models'] if m['name']=='parent')['decision_traces']
    with ProcessPoolExecutor(8) as pool:
        parts=list(pool.map(scan,refs))
    all_counts=sum((p[0] for p in parts),Counter())
    cases=[case for _,rows in parts for case in rows]
    atomic_json(ROOT/'assessment.json',dict(panel=file_identity(PANEL),games=sum(len(r['seeds']) for r in refs),
                counts=all_counts,improved=len(cases),changed_games=len({r['seed'] for r in cases}),cases=cases))
