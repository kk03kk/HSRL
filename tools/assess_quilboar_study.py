"""Paired natural and quill-eligible reports, with survivor and legacy diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity

UTILITY=np.array([1,.35,-1,-1.15,-1.30,-1.45,-1.60,-1.75])
RID='BG31_323'
SOURCE={'BG31_326','BG36_331'}
CORE={RID,*SOURCE}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def summary(g):
    rank=g['rank']
    turns={int(t):(n,tuple(c.removesuffix('_G') for c in board)) for t,(n,board) in g['eot'].items()}
    together={t for t,(_,b) in turns.items() if RID in b and SOURCE.intersection(b)}
    procs={e[0] for e in g['events']}
    return dict(rank=rank,utility=UTILITY[rank-1],first=int(rank==1),top2=int(rank<=2),top4=int(rank<=4),
                duo=int(bool(together)),sustained=int(any(t+1 in together for t in together)),
                rider=int(any(RID in b for _,b in turns.values())),proc=int(bool(procs)),repeat=int(len(procs)>=2),
                triggered_turns=len(procs),events=len(g['events']),native_gems=sum(e[2] for e in g['events']),
                reach12=int(12 in turns),reach15=int(15 in turns),
                strength={t:n for t,(n,_) in turns.items()},boards={t:b for t,(_,b) in turns.items()})


def interval(x):
    a=np.asarray(x,dtype=float)
    if not len(a):
        return dict(n=0)
    rng=np.random.default_rng(49171)
    samples=[a[rng.integers(len(a),size=(500,len(a)))].mean(1) for _ in range(20)]
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(np.concatenate(samples),[.025,.975]).tolist())


def cohort(rows):
    delta={}
    for k in ('rank','utility','first','top2','top4','rider','duo','sustained','proc','repeat',
              'triggered_turns','events','native_gems','reach12','reach15'):
        a=np.asarray([r[0][k] for r in rows])
        b=np.asarray([r[1][k] for r in rows])
        delta[k]=dict(parent=float(a.mean()),student=float(b.mean()),paired=interval(b-a))
    delta['turns']={}
    for t in range(1,16):
        data={}
        for i,name in enumerate(('parent','student')):
            v=sorted(r[i]['strength'][t] for r in rows if t in r[i]['strength'])
            data[name]=dict(n=len(v),p50=v[round(.5*(len(v)-1))] if v else None)
        pairs=[(a['strength'][t],b['strength'][t]) for a,b in rows if t in a['strength'] and t in b['strength']]
        data['common_log_delta']=interval([np.log1p(y)-np.log1p(x) for x,y in pairs])
        delta['turns'][t]=data
    for start,end in ((9,12),(10,12),(10,14),(10,15)):
        growths=[]
        for a,b in rows:
            x,y=a['strength'],b['strength']
            if all(t in x and t in y for t in (start,end)):
                growths.append(np.log1p(y[end])-np.log1p(y[start])-np.log1p(x[end])+np.log1p(x[start]))
        delta[f'growth_{start}_{end}']=interval(growths)
    return delta


def families(g,cards):
    flags=set()
    for t,board in g['boards'].items():
        if t<9:continue
        before=g['boards'].get(t-1,())
        for label in ('MECHANICAL','DRAGON'):
            if all(sum(label in cards[k] for k in b)>=4 for b in (before,board)):
                flags.add(label)
    return flags


def assess(root):
    plan=read(root/'plan.json')['plan']
    sel=read(root/'selection.json')
    end=read(root/'complete.json')
    if not end['valid'] or end['selected']!=sel['model']:
        raise ValueError('finalist identity mismatch')
    cards={c['id']:c['races'] for c in read(Path('engine/crates/hsbg-engine/data/cards.json'))['cards']}
    report={}
    for stage in ('natural','special'):
        d=root/stage
        if file_identity(d/'complete.json')!=end['panels'][stage]:
            raise ValueError('wrong final panel')
        receipt=read(d/'complete.json')
        arms=read(d/'arms.json')
        if arms!={'parent':plan['base'],'student':sel['path']} or arms!=receipt['arms']:
            raise ValueError('unexpected final panel arms')
        games=[]
        for i,source in enumerate(receipt['source_shards']):
            file=d/f'{i}.json'
            if file_identity(file)!=source:raise ValueError('changed evidence shard')
            games.extend(read(file))
        seeds=sorted(g['parent']['seed'] for g in games)
        if seeds!=receipt['seeds'] or len(set(seeds))!=plan['sizes'][stage]:
            raise ValueError('wrong final seeds')
        if stage=='natural' and seeds!=list(range(plan['seeds'][stage],plan['seeds'][stage]+len(seeds))):
            raise ValueError('natural cohort mismatch')
        report[stage]={}
        cohorts={'all':[]}
        legacy={k:Counter() for k in ('parent','student')}
        examples=[]
        unchanged=0
        for g in games:
            a,b=g['parent'],g['student']
            if (a['seed'],a['lobby'])!=(b['seed'],b['lobby']):
                raise ValueError('mispaired game')
            pig=bool(a['lobby']&128)
            if not pig and a!=b:raise ValueError('off-tribe behavior drift')
            unchanged+=a==b
            if stage=='special' and not pig:raise ValueError('special cohort mismatch')
            pair=(summary(a),summary(b))
            cohorts['all'].append(pair)
            if stage=='natural':
                cohorts.setdefault('quill_in' if pig else 'quill_out',[]).append(pair)
                for k,bit in [('dragon_in',2),('mech_in',4)]:
                    if pig and a['lobby'] & (1<<bit):cohorts.setdefault(k,[]).append(pair)
            for j,k in enumerate(('parent','student')):
                fam=families(pair[j],cards)
                legacy[k].update(fam)
                if j==0 and stage=='natural':
                    for name in fam:cohorts.setdefault('parent_path_'+name,[]).append(pair)
            if pair[1]['repeat']:
                examples.append(dict(seed=b['seed'],rank=b['rank'],parent_rank=a['rank'],events=b['events'],
                            strength=pair[1]['strength'],parent_strength=pair[0]['strength'],
                            final=b['eot'][str(max(pair[1]['strength']))]))
        report[stage]=dict(cohorts={k:cohort(r) for k,r in cohorts.items()},legacy=legacy,
                           unchanged=unchanged,repeat_examples=examples)
    report['selection']=sel
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();atomic_json(a.out,assess(a.root))
