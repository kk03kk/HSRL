"""Explicit precursor fixture for testing the value of a native core triple.

This is a counterfactual research setup, not a natural replay or demonstration.
A legal purchase is never labeled good without evaluating its continuation.
"""
import json
from hsbg.az.train.economic_tasks import debug,own
from workflows.probe_whole_turn_economy import fixture

def privateer_triple(source):
    import hsbg_engine as E
    p=source['players'][0]
    if not source['lobby_races']&64 or not 5<=p['tier']<=6 or p['gold']<3:
        raise ValueError('bridge requires a Pirate lobby, native tier >=5 and purchase gold')
    state,edits=fixture(source,'complete_gold')
    game=E.Game.from_json(json.dumps(state))
    def edit(cmd,**kw):
        debug(game,cmd,**kw);edits.append({'cmd':cmd,**kw})
    board=json.loads(game.view(0))['me']['board']
    index=next(i for i,m in enumerate(board) if m['id']=='BG33_825_G')
    edit('remove',zone='board',index=index)
    edit('add_minion',zone='board',card='BG33_825')
    edit('add_minion',zone='hand',card='BG33_825')
    if len(own(game)['shop'])>=8:edit('remove',zone='shop',index=7)
    edit('add_minion',zone='shop',card='BG33_825')
    shop=json.loads(game.view(0))['me']['shop']
    slot=len(shop)-1;action=8+slot
    if shop[slot]['id']!='BG33_825' or not bytes(game.legal_mask(0))[action]:
        raise ValueError('bridge third-copy purchase is not legal')
    for key in ('gold','health','tier'):
        if own(game)[key]!=p[key]:raise ValueError('bridge changed original resources')
    if any(game.diagnostic_counts()):raise ValueError('bridge native diagnostics')
    return json.loads(game.state_json()),edits,action
