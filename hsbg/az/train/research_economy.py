"""Public-view whole-turn economic heuristic for explicitly endowed research tasks.

This is not an optimal teacher or a replacement for the deployed policy. It must
earn a measured advantage on paired native rollouts before producing labels.
Only own public view and legal actions enter Operator; no game handle or RNG.
"""
import json
from pathlib import Path

CARDS=json.loads((Path(__file__).resolve().parents[3]/'engine/crates/hsbg-engine/data/cards.json').read_text(encoding='utf-8'))['cards']
NORMAL={c['id']:c.get('normal') or c['id'] for c in CARDS}
CORES=('BG_LOE_077','BG33_823','BG33_825','BG36_523')
ECON=('BG33_821','BGS_123','BGS_116','BGS_115')
# This specialized economic demonstrator is not a Demon health-management
# expert. Do not cycle health-loss Battlecries, or introduce persistent health
# spenders. In particular, golden Percussionist + golden Brann can be lethal
# across a chain of compulsory discoveries despite a healthy starting hero.
HEALTH_RISK=('BG26_522','BG26_525','BG26_528','CORE_LOOT_013','LOOT_013',
             'BGS_004','BG32_892','BG26_524','BG27_011','BGDUO31_210',
             'BG28_905','BG30_106','BG32_893')
def cid(m):return NORMAL.get(m.get('id',''),m.get('id',''))
def strength(m):return int(m.get('atk',0))+int(m.get('hp',0))

class Operator:
    def __init__(self):self.cycled=set()
    def protect(self,board):
        out=set()
        keep=CORES+('BG33_821',) if all(any(cid(x)==c for x in board) for c in CORES) else CORES
        for c in keep:
            same=[m for m in board if cid(m)==c]
            if same:out.add(max(same,key=lambda m:(m.get('golden',False),strength(m)))['eid'])
        return out
    def act(self,v):
        if set(v)-{'me','mask','turn','phase','lobby_races'}:
            raise ValueError('operator accepts only a restricted public own view')
        p=v['me'];m=list(v['mask']);board=p['board'];hand=p['hand'];shop=p['shop']
        for i,x in enumerate(shop):
            if x.get('costs_health',False):m[8+i]=False
        protected=self.protect(board);gold=p['gold']
        lobby=v.get('lobby_races') or sorted({r for x in board for r in x.get('races',[]) if r!='ALL'})
        race_counts={r:sum(r in x.get('races',[]) or 'ALL' in x.get('races',[]) for x in board) for r in lobby}
        pirate_majority=race_counts.get('PIRATE',0)>max([n for r,n in race_counts.items() if r!='PIRATE'] or [0])
        if p['pending']:
            opts=(p['pending'].get('targets') if p['pending'].get('kind')=='target' else p['pending'].get('options')) or []
            legal=[i for i in range(16) if m[30+i]]
            if p['pending'].get('kind')=='target':
                by={x['eid']:x for x in board}
                best=max(legal,key=lambda i: strength(by.get(opts[i],{})) if i<len(opts) else -1)
            else:
                def score(i):
                    x=opts[i] if i<len(opts) and isinstance(opts[i],dict) else {}
                    c=cid(x)
                    return (-10000 if c in HEALTH_RISK else
                            10000 if c in CORES and not any(cid(b)==c for b in board)
                            else 5000 if c in ECON else strength(x))
                allowed=[i for i in legal if i>=len(opts) or not isinstance(opts[i],dict) or opts[i].get('id')!='BG36_MagicItem_414']
                best=max(allowed or legal,key=score)
            return 30+best,'pending'
        for i,x in enumerate(board):
            if cid(x) in HEALTH_RISK and m[16+i]:return 16+i,'remove_health_risk'
        def play(i):
            if cid(hand[i]) in HEALTH_RISK:return None
            a=[46+8*i+j for j in range(8) if m[46+8*i+j]]
            return a[-1] if a else None
        def play_card(c):
            for i,x in enumerate(hand):
                if cid(x)==c and play(i) is not None:return play(i)
        def free_slot():
            choices=[i for i,x in enumerate(board) if x['eid'] not in protected and m[16+i]]
            return 16+min(choices,key=lambda i:strength(board[i])) if choices else None
        for c in CORES:
            if not any(cid(x)==c for x in board):
                a=play_card(c)
                if a is not None:return a,'restore_core'
                if any(cid(x)==c for x in hand):
                    a=free_slot()
                    if a is not None:return a,'space_for_core'
        # Component completion only with an established partial engine. A lone
        # core name is not sufficient evidence that acquisition pays for itself.
        if len(hand)<9 and sum(any(cid(x)==c for x in board) for c in CORES)>=2:
            for c in CORES:
                if not any(cid(x)==c for x in board+hand):
                    for i,x in enumerate(shop):
                        if cid(x)==c and m[8+i]:return 8+i,'complete_engine'
        if gold<90:
            a=play_card('BG33_815')
            if a is not None:return a,'wealth'
        for i,x in enumerate(board):
            if x['eid'] in self.cycled and x['eid'] not in protected:
                future=gold>=3 or p['free_refreshes']>0 or any(y.get('type')=='minion' or cid(y) in ('BG33_814','BG33_815') for y in hand)
                if future:
                    others=[j for j,y in enumerate(board) if y['eid'] not in protected and j!=i and m[16+j]]
                    j=min(others,key=lambda j:strength(board[j])) if others else i
                    if j!=i and strength(x)>strength(board[j])+20:return 16+j,'keep_stronger_cycle'
                    if m[16+i]:return 16+i,'liquidate_cycle'
                self.cycled.discard(x['eid'])
        if len(hand)<=7:
            a=play_card('BG33_814')
            if a is not None and pirate_majority:return a,'friendly_pirates'
            if a is not None and any(cid(x)=='BG33_821' for x in board):
                slot=free_slot()
                if slot is not None:return slot,'pirate_majority_space'
        # Play incoming minions to realize their Battlecry / sell return. The
        # policy reads no future offers, opponent hand or simulator RNG.
        order=sorted(range(len(hand)),key=lambda i:(cid(hand[i]) not in ECON,-strength(hand[i]),i))
        for i in order:
            if hand[i].get('type')!='minion':continue
            if cid(hand[i]) in HEALTH_RISK:continue
            # Keep the second normal core for a native triple. Cycling it would
            # destroy the route from the normal engine to larger multipliers.
            if cid(hand[i]) in CORES and not hand[i].get('golden',False) and any(cid(x)==cid(hand[i]) and not x.get('golden',False) for x in board):continue
            a=play(i)
            if a is not None:return a,'cycle_hand'
            a=free_slot()
            if a is not None:return a,'space_for_cycle'
        for i,x in enumerate(hand):
            if x.get('type')!='minion':
                if cid(x)=='BG33_814' and not pirate_majority:continue
                a=play(i)
                if a is not None:return a,'spell'
        if len(hand)<9:
            priority=('BG33_815','BG33_814','BG33_821','BGS_123','BGS_116','BGS_115')
            for c in priority:
                for i,x in enumerate(shop):
                    if cid(x)==c and m[8+i]:return 8+i,'buy_economy'
            for i,x in enumerate(shop):
                if cid(x).startswith('BG33_81') and m[8+i]:return 8+i,'buy_bounty'
                if cid(x) in CORES and sum(cid(y)==cid(x) for y in board+hand)>=2 and m[8+i]:return 8+i,'triple_core'
            if gold>=6:
                for i,x in enumerate(shop):
                    if cid(x) in CORES and sum(cid(y)==cid(x) for y in board+hand)==1 and not any(cid(y)==cid(x) and y.get('golden',False) for y in board) and m[8+i]:return 8+i,'invest_core_pair'
        if m[1] and (p['free_refreshes']>0 or gold>=4):return 1,'search'
        if len(board)<7 and len(hand)<10:
            choices=[i for i,x in enumerate(shop) if x.get('type')=='minion' and m[8+i]]
            if choices:return 8+max(choices,key=lambda i:strength(shop[i])),'fill_board'
        a=play_card('BG33_814')
        if a is not None:return a,'friendly_fallback'
        return 0,'end'
    def note(self,before,after,action):
        if 46<=action<126:
            x=before['me']['hand'][(action-46)//8]
            if x.get('type')=='minion':
                old={m['eid'] for m in before['me']['board']}
                self.cycled.update(m['eid'] for m in after['me']['board'] if m['eid'] not in old)
        present={m['eid'] for m in after['me']['board']}
        self.cycled.intersection_update(present)

