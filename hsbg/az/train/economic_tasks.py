"""Native, short-horizon economic process tasks, not a full-game reward.

The first family concerns Privateer / Wealthy Bounty investment and ordering.
Fixtures are explicitly synthetic edits of fresh natural recruit states. The
oracle considers only known cards and never Refresh, EndTurn or hidden offers.
It maximizes realized Gold plus the liquidation value of a newly acquired
Privateer. Original board units cannot be sold by the oracle. Every plan is
executed by the unchanged simulator; this is a local skill target, not proof
that investing is optimal in a live full game.
"""
from __future__ import annotations
import json
from functools import lru_cache

import numpy as np

PRIVATEER='BG33_825'
WEALTHY='BG33_815'


@lru_cache(maxsize=1)
def card_ids():
    import hsbg_engine as e
    ncols,raw=e.card_static()
    return {e.card_info(i)[0]:i-1 for i in range(1,len(raw)//(4*ncols))}


def own(game,seat=0):return json.loads(game.player_json(seat))


def item_card(value):
    if 'item' in value:return item_card(value['item'])
    if 'Minion' in value:return int(value['Minion']['card'])
    if 'Spell' in value:return int(value['Spell']['card'])
    return int(value['card'])


def debug(game,cmd,seat=0,**args):
    game.debug(json.dumps(dict(cmd=cmd,p=seat,**args)))


def fixture(source,*,kind,bounties=2,gold=5,seed=0,seat=0):
    """Return a new simulator and a complete list of fixture interventions."""
    import hsbg_engine as e
    if not int(source['lobby_races']) & (1 << 6):
        raise ValueError('Privateer fixtures require a Pirate lobby')
    game=e.Game.from_json(json.dumps(source))
    p=own(game,seat)
    if p['pending'] or p['ended_turn']:
        raise ValueError('fixture needs an actionable recruit state')
    ids=card_ids()
    interventions=[]
    def edit(cmd,**kwargs):
        debug(game,cmd,seat,**kwargs);interventions.append(dict(cmd=cmd,**kwargs))
    edit('clear_zone',zone='hand')
    # Avoid accidental triples and allow room for an amplifier. Never mutate source.
    p=own(game,seat)
    for i in reversed(range(len(p['board']))):
        if p['board'][i]['card'] in (ids[PRIVATEER],ids['BG33_825_G']):
            edit('remove',zone='board',index=i)
    while len(own(game,seat)['board'])>5:
        board=own(game,seat)['board']
        i=min(range(len(board)),key=lambda j:board[j]['atk']+board[j]['hp'])
        edit('remove',zone='board',index=i)
    # Keep the natural shop, except one declared replacement in purchase tasks.
    if kind in ('buy_privateer','redundant_privateer'):
        shop=own(game,seat)['shop']
        if len(shop)>=8:edit('remove',zone='shop',index=len(shop)-1)
        edit('add_minion',zone='shop',card=PRIVATEER)
    elif kind!='order_privateer':raise ValueError(kind)
    if kind=='redundant_privateer':edit('add_minion',zone='board',card=PRIVATEER)
    edit('set_player',gold=int(gold))
    rng=np.random.default_rng(seed)
    hand=[WEALTHY]*int(bounties)
    if kind=='order_privateer':hand.append(PRIVATEER)
    rng.shuffle(hand)
    for card in hand:
        edit('add_minion' if card==PRIVATEER else 'add_spell',zone='hand',card=card)
    # Capture AFTER fixture setup. The demonstration itself contains only native steps.
    return game,interventions


def candidates(game,seat=0):
    p=own(game,seat);mask=bytes(game.legal_mask(seat));ids=card_ids()
    if p['pending']:return []
    out=[]
    for i,slot in enumerate(p['shop'][:8]):
        if item_card(slot)==ids[PRIVATEER] and mask[8+i]:out.append(8+i)
    for i,card in enumerate(p['hand'][:10]):
        if item_card(card) not in (ids[PRIVATEER],ids[WEALTHY]):continue
        legal=[46+8*i+j for j in range(8) if mask[46+8*i+j]]
        # For this gold-only family, positions do not change the intended payoff;
        # use a canonical legal position for plans, and certify alternatives below.
        if legal:out.append(legal[-1])
    return out


def wealth_remaining(game,seat=0):
    return sum(item_card(x)==card_ids()[WEALTHY] for x in own(game,seat)['hand'])


def score(game,original_eids,seat=0):
    p=own(game,seat);ids=card_ids()
    acquired=sum(1 for m in p['board'] if m['card']==ids[PRIVATEER] and m['eid'] not in original_eids)
    acquired+=sum(1 for x in p['hand'] if item_card(x)==ids[PRIVATEER])
    return float(p['gold']+acquired)


def solve(game,*,seat=0,max_depth=5,original_eids=None):
    """Exhaust all bounded relevant native action orders; return value and plans."""
    original_eids=set(m['eid'] for m in own(game,seat)['board']) if original_eids is None else original_eids
    def visit(g,depth):
        if wealth_remaining(g,seat)==0:
            return score(g,original_eids,seat),[]
        if depth==0:return float('-inf'),[]
        best=float('-inf');plan=[]
        for action in candidates(g,seat):
            child=g.copy();child.step(seat,action)
            value,tail=visit(child,depth-1)
            if value>best:best,plan=value,[action]+tail
        return best,plan
    values={};plans={}
    for action in candidates(game,seat):
        child=game.copy();child.step(seat,action)
        value,tail=visit(child,max_depth-1)
        values[action]=value;plans[action]=[action]+tail
    finite={a:v for a,v in values.items() if np.isfinite(v)}
    if not finite:raise ValueError('no complete bounded economic plan')
    best=max(finite.values())
    good=[a for a,v in finite.items() if abs(v-best)<1e-6]
    # Certify every equivalent placement with the native simulator, rather than
    # assuming that other own-board effects leave the same economic outcome.
    for action in list(good):
        if 46<=action<=125:
            slot=(action-46)//8
            for pos in range(8):
                other=46+8*slot+pos
                if other in values or not bytes(game.legal_mask(seat))[other]:continue
                child=game.copy();child.step(seat,other)
                value,tail=visit(child,max_depth-1)
                values[other]=value;plans[other]=[other]+tail
                if abs(value-best)<1e-6:good.append(other)
    return dict(value=best,optimal_actions=sorted(good),
                action_values={str(a):v for a,v in values.items() if np.isfinite(v)},
                plan=plans[min(good)],original_eids=sorted(original_eids))
