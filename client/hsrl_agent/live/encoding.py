"""Public canonical view -> exact AutoTavern 3227/124 encoding.

No Game is constructed and no hidden state or random lobby completes missing fields.
The canonical view contains raw logical quantities, never pre-normalized guesses.
"""
import numpy as np
from hsbg.env import obs_std as O
from .contracts import Blocked

RACES=O.RACE_NAMES
KEYWORDS=O.KEYWORD_NAMES

def minion(value):
    if value is None: return [0.]*31
    if value.get('unverified_keywords'):raise Blocked('minion keyword absence has no complete static/dynamic producer')
    required=('attack','health','max_health','tier','golden','attached','shield_hits','keywords','races')
    if any(k not in value or value[k] is None for k in required): raise Blocked('incomplete minion features')
    if set(value['keywords'])-set(KEYWORDS) or set(value['races'])-set(RACES):
        raise Blocked('unmapped keyword/race')
    return [1.,value['attack']/10,value['health']/10,value['max_health']/10,value['tier']/6,
        float(value['golden']),value['attached']/3,value['shield_hits']/3]+[
        float(k in value['keywords']) for k in KEYWORDS]+[float(k in value['races']) for k in RACES]

def item(value):
    if value is None: return [0.]*40
    for k in ('is_minion','cost','locked','temporary','combined','frozen','kept','marked'):
        if k not in value or value[k] is None: raise Blocked('incomplete item feature:'+k)
    head=[1.,float(value['is_minion']),value['cost']/5,value['locked'],float(value['temporary']),
          float(value['combined']),float(value['frozen']),float(value['kept']),float(value['marked'])]
    if value['is_minion']: return head+minion(value['minion'])
    if value.get('spell_data') is None or value.get('tier') is None: raise Blocked('missing spell data/tier')
    return head+[1.,value['spell_data']/6,0.,0.,value['tier']/6]+[0.]*26

def encode(view,vocabulary):
    try:return _encode(view,vocabulary)
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        raise Blocked('invalid/missing model input: '+str(exc)) from exc


def _encode(view,vocabulary):
    if view.get('format')!='hsrl.live-public-view.v1': raise Blocked('unknown public view')
    if view.get('unknown'): raise Blocked('unknown public fields: '+','.join(view['unknown']))
    g=view['global']
    if set(O.GLOBAL_FIELD_NAMES)-set(g): raise Blocked('global fields missing')
    if len(view['lobby_races'])!=11: raise Blocked('lobby race vector missing')
    f=[float(g[k])/O.GLOBAL_FIELD_SCALE.get(k,1.) for k in O.GLOBAL_FIELD_NAMES]
    f.extend(view['lobby_races'])
    quests=view['quests']
    if len(quests)>3: raise Blocked('too many quests')
    for i in range(3): f.extend([1.,quests[i]['progress']/max(quests[i]['goal'],1)] if i<len(quests) and quests[i] else [0.,0.])
    pending=view['pending']
    if type(pending) is not int or not 0<=pending<9: raise Blocked('pending kind')
    f.extend(float(i==pending) for i in range(9))
    c=[]
    def index(cid):
        if not cid: return 0
        if cid not in vocabulary: raise Blocked('unknown model card: '+str(cid))
        return vocabulary[cid]
    c.extend(index(view[k]) for k in ('hero','power0','power1','deity','pending_source'))
    for i in range(3):
        q=(quests[i] or {}) if i<len(quests) else {}
        owned=view.get('owned_rewards',[])
        c.extend((index(q.get('card_id')),index(q.get('reward')),
                  index(owned[i] if i<len(owned) else q.get('owned_reward'))))
    secrets=view['secrets']
    if len(secrets)>5: raise Blocked('too many secrets')
    c.extend(index(secrets[i]) if i<len(secrets) else 0 for i in range(5))
    for zone,n in (('board',7),('hand',10),('shop',8),('choices',16)):
        values=view[zone]
        if len(values)>n: raise Blocked('slot overflow:'+zone)
        for i in range(n):
            value=values[i] if i<len(values) else None
            if zone=='board': f.extend(minion(value))
            elif value and value.get('card_only'):
                f.extend([1.]+[0.]*39)
            else: f.extend(item(value))
            c.append(index(value['card_id']) if value else 0)
    trinkets=view['trinkets']
    if len(trinkets)>8: raise Blocked('trinket overflow')
    c.extend(index(trinkets[i]) if i<len(trinkets) else 0 for i in range(8))
    others=view['opponents']
    if len(others)!=7: raise Blocked('seven public opponent slots required')
    # Native observation hides opponent identity/stats at hero selection. This
    # branch is part of the trained encoding, independent of real hero health.
    if g['hero_select']:
        others=[dict(hero=None,alive=o['alive'],health=40,armor=0,tier=1,placement=0,last=False,next=False,seen_board=[]) if o else None for o in others]
    for o in others:
        if o is None:
            f.extend([0.]*9);c.append(0)
        else:
            if len(o['seen_board'])>7: raise Blocked('opponent board overflow')
            f.extend([1.,float(o['alive']),o['health']/40,o['armor']/20,o['tier']/6,
                      len(o['seen_board'])/7,o['placement']/8,float(o['last']),float(o['next'])])
            c.append(index(o['hero']))
    for o in others:
        board=o['seen_board'] if o else []
        for i in range(7):
            value=board[i] if i<len(board) else None
            f.extend(minion(value));c.append(index(value['card_id']) if value else 0)
    f,c=np.asarray(f,np.float32),np.asarray(c,np.int32)
    if f.shape!=(3227,) or c.shape!=(124,) or not np.isfinite(f).all():
        raise Blocked('model encoding shape/numeric error')
    return f,c
