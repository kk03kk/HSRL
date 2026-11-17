"""Read sealed full traces to distinguish learned skill from natural opportunity coverage."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.compare_research_panels import load_sealed_panel

CORE=('BG33_823','BG33_825','BG_LOE_077','BG33_821','BG36_364')


def audit_trace(ref):
    import hsbg_engine as engine
    cards=json.loads(Path('engine/crates/hsbg-engine/data/cards.json').read_text(encoding='utf-8'))['cards']
    normal={c['id']:c.get('normal') or c['id'] for c in cards}
    ids={i:normal[engine.card_info(i+1)[0]] for i in range(len(cards))}
    def item(x):
        if 'item' in x:return item(x['item'])
        if 'Minion' in x:return item(x['Minion'])
        if 'Spell' in x:return item(x['Spell'])
        return ids[x['card']]
    path=Path(ref['directory'])/ref['file']
    if file_identity(path)['sha256']!=ref['sha256']:raise ValueError('trace checksum differs')
    opportunities=defaultdict(set);buys=defaultdict(set);counts=Counter()
    cases=defaultdict(set);turns={};examples=[]
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['event']!='action':continue
            counts['actions']+=1
            if r['turn']<9:continue
            seed=r['seed'];turn=r['turn'];a=r['action'];seat=r['seat']
            p=r['state_before']['players'][seat];q=r['state_after']['players'][seat]
            board=[item(x) for x in p['board']];hand=[item(x) for x in p['hand']]
            legal=r['legal_mask'];key=(seed,turn)
            tr=turns.setdefault(key,{'seed':seed,'turn':turn,'income':0,'actions':0,'max_gold':p['gold']})
            tr['actions']+=1;tr['max_gold']=max(tr['max_gold'],p['gold'])
            # EndTurn can advance to the next turn; never count its gold reset as income.
            if a!=0:tr['income']+=max(0,q['gold']-p['gold'])
            if a==0:
                tr.update(gold_left=p['gold'],spent=p['stats']['gold_spent_turn'],
                          strength=sum(m['atk']+m['hp'] for m in p['board']),
                          full_board=len(board)==7,core=[c for c in CORE if c in board])
                if all(c in board for c in CORE[:3]):cases['three_core_end_turn'].add(seed)
            for i,slot in enumerate(p['shop'][:8]):
                c=item(slot)
                if c not in CORE or not legal[8+i]:continue
                opportunities[c].add(seed)
                if a==8+i:buys[c].add(seed)
                if c=='BG33_825' and c not in board and hand.count('BG33_815')>=2:
                    cases['privateer_wealth2_purchase_available'].add(seed)
                    if len(board)==7:cases['privateer_wealth2_purchase_full_board'].add(seed)
                    if a==8+i:cases['privateer_wealth2_purchased'].add(seed)
                if c=='BG33_821' and all(x in board for x in ('BG33_825','BG_LOE_077')):
                    cases['rascal_amplified_purchase_available'].add(seed)
                    if a==8+i:cases['rascal_amplified_purchased'].add(seed)
            if 'BG33_825' in hand and 'BG33_825' not in board and 'BG33_815' in hand:
                cases['privateer_wealth_order_available'].add(seed)
                if len(board)==7:cases['privateer_wealth_order_full_board'].add(seed)
                selected=hand[(a-46)//8] if 46<=a<126 else None
                if selected=='BG33_825':cases['privateer_played_before_wealth'].add(seed)
                if selected=='BG33_815':cases['wealth_played_before_privateer'].add(seed)
                if len(examples)<8:examples.append({'seed':seed,'turn':turn,'action_index':r['action_index'],
                    'action':a,'board':board,'hand':hand,'gold':p['gold']})
            if 8<=a<16:kind='buy';chosen=item(p['shop'][a-8])
            elif 16<=a<23:kind='sell';chosen=board[a-16]
            elif 46<=a<126:kind='play';chosen=hand[(a-46)//8]
            else:continue
            if chosen in CORE or chosen.startswith('BG33_81'):counts[kind+':'+chosen]+=1
    return {'model':ref['identity']['name'],'opportunities':{k:sorted(v) for k,v in opportunities.items()},
            'buys':{k:sorted(v) for k,v in buys.items()},'cases':{k:sorted(v) for k,v in cases.items()},
            'counts':dict(counts),'turns':list(turns.values()),'examples':examples}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('panels',nargs='+')
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();refs=[]
    for path in args.panels:
        panel=load_sealed_panel(path)
        refs.extend(r for m in panel['models'] for r in m['decision_traces'])
    merged={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(audit_trace,refs):
            m=merged.setdefault(r['model'],{'opportunities':defaultdict(set),'buys':defaultdict(set),
                'cases':defaultdict(set),'counts':Counter(),'turns':[],'examples':[]})
            for field in ('opportunities','buys','cases'):
                for k,v in r[field].items():m[field][k].update(v)
            m['counts'].update(r['counts']);m['turns']+=r['turns'];m['examples']+=r['examples']
    for m in merged.values():
        for field in ('opportunities','buys','cases'):
            m[field]={k:{'games':len(v),'seeds':sorted(v)} for k,v in m[field].items()}
        m['counts']=dict(m['counts'])
    atomic_json(args.out,{'models':merged,'panels':{x:file_identity(x) for x in args.panels},
        'scope':'T9+ only. Event counts and unique-game counts are distinct. Pre-EndTurn economy only.'})
    for name,m in merged.items():print(json.dumps({'model':name,'cases':{k:v['games'] for k,v in m['cases'].items()}}))


if __name__=='__main__':main()
