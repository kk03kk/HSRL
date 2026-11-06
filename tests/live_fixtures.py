"""Offline oracle only. Never import this privileged simulator fixture in live code."""
import json
from pathlib import Path
import numpy as np
import hsbg_engine as t
from hsbg.env import obs_std as O

ROOT=Path(__file__).resolve().parents[1]

def vocabulary():return {t.card_info(i)[0]:i for i in range(1,t.N_CARDS)}

def native_view(game,seat=0):
    """Translate public simulator JSON, with explicit engine counters as oracle.

    This validates the new encoder against independent Rust observe bytes, not
    the real log projector. It must not be presented as a live-data test.
    """
    v=json.loads(game.view(seat));state=json.loads(game.state_json());pl=state['players'][seat];me=v['me']
    def cid(n):return t.card_info(n+1)[0] if n is not None else None
    def identity(x):return x['id'] if x else None
    def minion(m):
        return dict(card_id=m['id'],attack=m['atk'],health=m['hp'],max_health=m['max_hp'],tier=m['tier'],
                    golden=m['golden'],attached=len(m['attached']),shield_hits=m['shield_hits'],
                    keywords=m['keywords'],races=m['races'])
    def item(c,zone):
        is_m=c.get('type')=='minion' or 'eid' in c and 'shield_hits' in c
        result=dict(card_id=c['id'],is_minion=is_m,cost=c.get('price',0) if zone=='shop' else 0,
            locked=c.get('locked',0),temporary=c.get('temporary',False),combined=c.get('combined',False),
            frozen=c.get('frozen',False) if zone=='shop' else c.get('doomed',False),
            kept=c.get('kept',False) if zone=='shop' else False,
            marked=c.get('ticket',False) if zone=='shop' else bool(c.get('group',0)))
        if is_m:result['minion']=minion(c)
        else:result.update(spell_data=c['data'],tier=c['tier'])
        return result
    g={k:me[k] for k in ('health','armor','gold','tier','max_gold_bonus','gold_next_turn','upgrade_price','refresh_price',
                         'minion_price','free_refreshes','frozen','next_is_ghost')}
    g.update(turn=v['turn'],hero_select=v['phase']=='hero',game_over=v['phase']=='over',alive=v['alive'],
             me_alive=me['alive'],can_dark_gift=me['dark_gift']['available'],dark_gift_uses=me['dark_gift']['uses'],
             trinket_count=len(me['trinkets']),deity_present=me['deity'] is not None,
             deity_atk=(me['deity'] or {}).get('atk',0),deity_hp=(me['deity'] or {}).get('hp',0),
             board_len=len(me['board']),hand_len=len(me['hand']),hand_queue_len=len(pl['hand_queue']),
             effects_len=len(pl['effects']),chronum=pl['chronum_saved'])
    # View's phase is spelled HeroSelect in some builds; state is authoritative.
    g['hero_select']=state['phase']=='HeroSelect';g['game_over']=state['phase']=='GameOver'
    for k in ('won_last','lost_last','spell_atk','spell_hp','gem_atk','gem_hp','tavern_atk','tavern_hp'):g[k]=me['stats'][k]
    for i in range(2):
        power=me['hero_powers'][i] if i<len(me['hero_powers']) else {}
        g.update({f'power{i}_usable':power.get('usable',False),f'power{i}_price':power.get('price') or 0,f'power{i}_uses_game':power.get('uses_game',0)})
    pending=me['pending'];kind=1 if g['hero_select'] else 0
    if pending and not kind:raise ValueError('fixture supports hero/ordinary phases only')
    assert not me['quests']
    others=[]
    for o in v['opponents']:
        others.append(dict(hero=identity(o['hero']),alive=o['alive'],health=o['health'],armor=o['armor'],tier=o['tier'],
            placement=o['placement'],last=o['was_last_opponent'],next=o['is_next_opponent'],seen_board=[minion(m) for m in o['seen_board']]))
    return dict(format='hsrl.live-public-view.v1',**{'global':g},lobby_races=[int(r in v['lobby_races']) for r in O.RACE_NAMES],
        hero=identity(me['hero']),power0=identity(me['hero_powers'][0]) if me['hero_powers'] else None,
        power1=identity(me['hero_powers'][1]) if len(me['hero_powers'])>1 else None,deity=identity(me['deity']['card']) if me['deity'] else None,
        pending_source=None,quests=[],pending=kind,secrets=[identity(x) for x in me['secrets']],
        board=[minion(m) for m in me['board']],hand=[item(c,'hand') for c in me['hand']],shop=[item(c,'shop') for c in me['shop']],
        choices=[dict(card_id=c['id'],card_only=True) for c in pending['options']] if pending else [],
        trinkets=[identity(x) for x in me['trinkets']],opponents=others,unknown=[])

def game_to_turn(turn=2,seed=91):
    g=t.Game(seed=seed)
    for _ in range(3000):
        if g.turn>=turn:return g
        p=next(p for p in range(8) if g.needs_action(p));g.step(p,g.bot_action(p))
    raise AssertionError('fixture did not progress')

def card(eid,kind='MINION',cid='BG20_100',pos=1,**tags):
    return dict(entity_id=eid,card_id=cid,card_type=kind,position=pos,controller=1,zone='PLAY',
                tier=1,cost=0,tags={'ATK':'2','HEALTH':'1','CARDRACE':'43',**tags})

def snapshot():
    view=native_view(t.Game(seed=91));view.update(pending=0,power_entities=[])
    data=dict(turn=1,phase='recruit',local_player_id=1,board=[],hand=[],shop=[],choices=[],controls=[],
        interaction=None,tavern_tier=1,log_position=100,resolutions=[],options_pending=False,client_task_list=None)
    return dict(game_id='game-1',revision=1,client=data,options=[],view=view,visual=dict(window_token='fixture'))

def option(eid,oid=1,targets=(),**fields):
    return dict(entity=eid,id=oid,error='NONE',type='POWER',targets=list(targets),has_targets=bool(targets),**fields)
