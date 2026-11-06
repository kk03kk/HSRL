"""Visible log entities/history -> canonical model view, with explicit unknowns.

Engine-only counters require named, sourced context facts. They are never obtained
by constructing a random simulator Game or treating an old candidate model as input.
"""
import copy
import time
from hsbg.env import obs_std as O
from .contracts import Blocked

RACE_IDS={'20':'BEAST','15':'DEMON','24':'DRAGON','18':'ELEMENTAL','17':'MECHANICAL',
          '14':'MURLOC','23':'PIRATE','43':'QUILBOAR','11':'UNDEAD','126':'ABERRATION','92':'NAGA'}

def number(tags,key,default=None):
    value=tags.get(key,default)
    try: return int(value) if value is not None else None
    except (TypeError,ValueError): raise Blocked('invalid numeric tag:'+key)

def control_cost(card):
    # Client action-control quote overrides raw card COST; no universal price.
    return number(card['tags'],'INTERACTABLE_OBJECT_COST',card.get('cost'))

def card_minion(card):
    t=card['tags']
    for key in ('ATK','HEALTH'):
        if key not in t: raise Blocked('minion stat missing:'+key)
    races=str(t.get('CARDRACE','')).split(',')
    races=[RACE_IDS.get(r,r) for r in races]
    races=list(O.RACE_NAMES) if races==['ALL'] or races==['26'] else [] if races in (['0'],['INVALID'],['NONE']) else [r for r in races if r]
    if set(races)-set(O.RACE_NAMES): raise Blocked('unknown minion race')
    max_health=number(t,'HEALTH')
    aliases={'IMMUNE_ATTACKING':'IMMUNE_WHILE_ATTACKING'}
    unverified=[k for k in O.KEYWORD_NAMES if k not in t and aliases.get(k,k) not in t]
    return dict(card_id=card['card_id'],attack=number(t,'ATK'),health=max_health-number(t,'DAMAGE',0),
        max_health=max_health,tier=card['tier'],golden=bool(number(t,'PREMIUM',0)),
        attached=number(t,'MODEL_ATTACHED_COUNT'),shield_hits=number(t,'MODEL_SHIELD_HITS',number(t,'DIVINE_SHIELD')),
        keywords=[k for k in O.KEYWORD_NAMES if number(t,aliases.get(k,k),number(t,k,0))>0],
        unverified_keywords=unverified,races=races if 'CARDRACE' in t else None)

class Projector:
    def __init__(self,profile,vocabulary_info):
        self.profile=profile;self.info=vocabulary_info;self.memory={};self.game_id=None
    def _item(self,card,zone,price=None):
        t=card['tags'];is_minion=card['card_type']=='MINION'
        c=copy.deepcopy(card)
        if card['card_id'] not in self.info: raise Blocked('card absent from trained vocabulary:'+card['card_id'])
        c['tier']=self.info[card['card_id']]['tier'] if c.get('tier') is None else c['tier']
        # Engine-specific counters/flags require a producer with reviewed
        # semantics. Absent tags are unknown, including flags that often are 0.
        result=dict(card_id=c['card_id'],is_minion=is_minion,cost=0 if zone=='hand' else price,
            locked=number(t,'MODEL_LOCKED_TURNS') if zone=='hand' else 0,
            temporary=number(t,'MODEL_TEMPORARY') if zone=='hand' else False,
            combined=number(t,'MODEL_COMBINED') if zone=='hand' else False,
            frozen=bool(number(t,'FROZEN',0)) if zone=='shop' else number(t,'MODEL_DOOMED_THIS_TURN') if zone=='hand' else False,
            kept=number(t,'MODEL_KEPT_FROM_LAST_TURN') if zone=='shop' else False,
            marked=number(t,'MODEL_DARKMOON_TICKET') if zone=='shop' else number(t,'MODEL_LINKED_GROUP') if zone=='hand' else False,
            tier=c['tier'])
        if is_minion: result['minion']=card_minion(c)
        else:
            # Most spells' simulator data is zero; only a fact producer can
            # certify this. Do not turn absent SCRIPT_DATA into a guessed zero.
            result['spell_data']=number(t,'MODEL_SPELL_DATA')
        return result

    def build(self,reader,data,*,now=None,facts=None,allow_owned_choice_block=False,allow_idle_live_tail=False):
        now=time.time() if now is None else now
        if reader.spectating:raise Blocked('spectator session cannot authorize player input')
        if not reader.caught_up or not reader.history_complete: raise Blocked('need complete CREATE_GAME history')
        current_tail=allow_idle_live_tail and reader.live_tail_current()
        if reader.latest_time is None or not (-.5<=now-reader.latest_time<=self.profile.get('max_source_age',15)
                or current_tail and reader.latest_time<=now):
            raise Blocked('stale source log; no input')
        if data['build'] not in self.profile['allowed_builds']: raise Blocked('unreviewed client build')
        if data['mode']!='GT_BATTLEGROUNDS': raise Blocked('only solo Battlegrounds is supported')
        if data['phase'] not in ('recruit','choice'): raise Blocked('not a decision phase')
        if 'client_combat_playback_pending' in data.get('warnings',[]):
            raise Blocked('client combat playback has not reached server recruit; input paused')
        if not data['ready'] or data['options_pending'] or data.get('client_task_list') is not None:
            raise Blocked('server snapshot or visual playback is pending')
        if data['warnings']: raise Blocked('client state warning:'+','.join(data['warnings']))
        if reader.state.block_stack and not (allow_owned_choice_block and data.get('choices')
                and (data.get('interaction') or {}).get('owner')==data.get('local_player_id')):
            raise Blocked('server block is still open')
        if self.game_id!=data['game_id']:
            self.game_id=data['game_id'];self.memory={}
        state=reader.state;pid=data['local_player_id']
        if pid is None: raise Blocked('local player unknown')
        ids=[c['entity_id'] for z in ('board','hand','shop') for c in data[z]]
        if len(ids)!=len(set(ids)): raise Blocked('duplicate entity across zones')
        for z in ('board','hand','shop'):
            if [c['position'] for c in data[z]]!=list(range(1,len(data[z])+1)):
                raise Blocked('noncontiguous visible zone:'+z)
            if z in ('board','hand') and any(c['controller']!=pid for c in data[z]):
                raise Blocked('wrong local controller:'+z)
        pt=state.entities[state.players[pid]].tags
        hero=data['hero'];hero_select=bool(data['choices']) and all(c['card_type']=='HERO' for c in data['choices'])
        if hero is None and not hero_select: raise Blocked('local hero missing')
        facts=copy.deepcopy(facts or {})
        if facts and (facts.get('game_id')!=data['game_id'] or facts.get('revision')!=data['revision']):
            raise Blocked('public context facts are not bound to this revision')
        global_values=copy.deepcopy(facts.get('global',{}))
        derived=dict(turn=data['turn'],hero_select=int(hero_select),game_over=0,me_alive=1,
            health=hero['health'] if hero else None,armor=number(hero['tags'],'ARMOR',0) if hero else None,
            gold=data['gold'],tier=data['tavern_tier'],upgrade_price=data['upgrade_cost'],
            board_len=len(data['board']),hand_len=len(data['hand']))
        global_values.update({k:v for k,v in derived.items() if v is not None})
        # Gem/spell tags may denote total value rather than extra stats; deity
        # tags may refer to lobby/owned state. No unreviewed identity mapping.
        aliases={'free_refreshes':'BACON_FREE_REFRESH_COUNT','gold_next_turn':'BACON_PLAYER_EXTRA_GOLD_NEXT_TURN',
                 'won_last':'BACON_WON_LAST_COMBAT'}
        for name,tag in aliases.items():
            if tag in pt: global_values[name]=number(pt,tag)
        for name,spec in self.profile.get('reviewed_feature_aliases',{}).items():
            if name not in O.GLOBAL_FIELD_NAMES or not spec.get('evidence'):
                raise Blocked('unreviewed feature transformation')
            if spec['tag'] in pt:global_values[name]=number(pt,spec['tag'])+spec.get('offset',0)
        controls=data['controls']
        control=lambda token:next((c for c in controls if token in c['card_id'].lower()),None)
        for name,token in [('refresh_price','reroll_button')]:
            c=control(token)
            if c and control_cost(c) is not None: global_values[name]=control_cost(c)
        control_by_id={c['entity_id']:c for c in controls}
        slot_prices={}
        for opt in state.options:
            source=control_by_id.get(opt['entity'])
            if not source or 'dragbuy' not in source['card_id'].lower() or control_cost(source) is None:continue
            for row in opt.get('target_rows',[]):
                slot_prices.setdefault(row['entity'],set()).add(control_cost(source))
        # A slot-specific discount/free card does not establish the model's
        # global minion_price. Keep it unknown without a separate producer.
        powers=sorted([c for c in controls if c['card_type']=='HERO_POWER'],key=lambda c:(c['position'],c['entity_id']))
        if len(powers)>2: raise Blocked('hero power slot overflow')
        if len(powers)>1:
            if any('MODEL_POWER_SLOT' not in c['tags'] for c in powers):
                raise Blocked('two power slots require explicit slot identity')
            powers=sorted(powers,key=lambda c:number(c['tags'],'MODEL_POWER_SLOT'))
            if [number(c['tags'],'MODEL_POWER_SLOT') for c in powers]!=[0,1]:raise Blocked('invalid power slots')
        for i in range(2):
            if i>=len(powers):
                global_values.update({f'power{i}_usable':0,f'power{i}_price':0,f'power{i}_uses_game':0})
            else:
                c=powers[i]
                global_values[f'power{i}_usable']=int(any(o['entity']==c['entity_id'] and o['error']=='NONE' for o in state.options))
                if control_cost(c) is not None: global_values[f'power{i}_price']=control_cost(c)
                if 'MODEL_USES_GAME' in c['tags']: global_values[f'power{i}_uses_game']=number(c['tags'],'MODEL_USES_GAME')
        view=copy.deepcopy(facts.get('view',{}))
        view.update(format='hsrl.live-public-view.v1',global_=None)
        view.pop('global_',None);view['global']=global_values
        view.update(hero=hero['card_id'] if hero else None,power0=powers[0]['card_id'] if powers else None,
            power1=powers[1]['card_id'] if len(powers)>1 else None,power_entities=[c['entity_id'] for c in powers])
        for zone in ('board','hand','shop'):
            values=[]
            for c in data[zone]:
                patched=copy.deepcopy(c)
                patched['tags'].update(facts.get('entity_tags',{}).get(str(c['entity_id']),{}))
                prices=slot_prices.get(c['entity_id'],set())
                price=number(patched['tags'],'MODEL_PURCHASE_PRICE')
                if price is None and len(prices)==1:price=next(iter(prices))
                value=self._item(patched,zone,price=price if zone=='shop' else 0)
                values.append(value['minion'] if zone=='board' else value)
            view[zone]=values
        interaction=data.get('interaction') or {}
        view['pending']=1 if hero_select else facts.get('pending',None if data['choices'] else 0)
        view['pending_source']=next((e.card_id for eid,e in state.entities.items() if eid==interaction.get('source_entity')),None)
        view['choices']=[]
        for c in data['choices']:
            if view['pending']==3:
                if not view['pending_source']: raise Blocked('Choose-One source missing')
                view['choices'].append(dict(card_id=view['pending_source'],card_only=True))
            elif hero_select or view['pending']==5: view['choices'].append(dict(card_id=c['card_id'],card_only=True))
            else:
                patched=copy.deepcopy(c);patched['tags'].update(facts.get('entity_tags',{}).get(str(c['entity_id']),{}))
                # Native Quest encodes the quest goal as cost, Trinket its
                # observation quote (even when a free transaction is offered),
                # Timewarp Chronum. A raw universal COST=0 is not equivalent.
                quote=number(patched['tags'],'MODEL_OFFER_PRICE') if view['pending'] in (6,7,8) else 0
                view['choices'].append(self._item(patched,'choices',price=quote))
        required=('lobby_races','quests','owned_rewards','secrets','trinkets','opponents','deity')
        unknown=list(view.get('unknown',[]))+[k for k in required if k not in view]
        unknown += ['global.'+k for k in O.GLOBAL_FIELD_NAMES if k not in global_values]
        if view['pending'] is None: unknown.append('pending.kind')
        # Collect all missing numeric/flag fields for actionable coverage reports.
        def missing(value,path):
            if isinstance(value,dict):
                for k,v in value.items():
                    if k=='unverified_keywords':
                        unknown.extend(path+'.keywords.'+name for name in v)
                    if v is None and k not in ('card_id',): unknown.append(path+'.'+k)
                    elif isinstance(v,(dict,list)): missing(v,path+'.'+k)
            elif isinstance(value,list):
                for i,v in enumerate(value): missing(v,path+'.'+str(i))
        for z in ('board','hand','shop','choices'):
            missing(view[z],z)
        view['unknown']=unknown
        return dict(game_id=data['game_id'],revision=data['revision'],build=data['build'],
            captured_at=data['captured_at'],source_time=reader.latest_time,client=data,
            options=copy.deepcopy(state.options),view=view)
