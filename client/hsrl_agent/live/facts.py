"""Evidence-only context enrichment. Missing engine counters are explicit blockers."""
import copy
from .contracts import Blocked
from .projection import number

def context_facts(reader,data,card_info,*,lobby_races=None,supplied=None):
    """Supplied facts must name the exact game/revision; never reused across frames."""
    if supplied is not None:
        if supplied.get('game_id')!=data['game_id'] or supplied.get('revision')!=data['revision']:
            raise Blocked('context sidecar game/revision mismatch')
        return copy.deepcopy(supplied)
    state=reader.state;pid=data['local_player_id']
    if pid not in state.players: raise Blocked('local player tags missing')
    own=state.entities[state.players[pid]].tags
    game=state.entities[state.game_entity].tags if state.game_entity in state.entities else {}
    view={};g={};provenance={}
    # Explicit bonus counters only. Absence never means zero; these names are
    # not used to infer internal effect-list length, ownership or total buffs.
    for key,tag in {'spell_atk':'TAVERN_SPELL_ATTACK_INCREASE','spell_hp':'TAVERN_SPELL_HEALTH_INCREASE',
                    'gem_atk':'BACON_BLOODGEMBUFFATKVALUE','gem_hp':'BACON_BLOODGEMBUFFHEALTHVALUE'}.items():
        if tag in own:
            g[key]=number(own,tag);provenance['global.'+key]='explicit_player_bonus_tag:'+tag
    if reader.combat_seen:
        # Combat clones and recruit boards must not be confused. Until a
        # validated pre-combat memory producer exists, empty memory is false.
        view['unknown']=['opponents.seen_board_history']
    if lobby_races is not None:
        from .encoding import RACES
        if len(lobby_races)!=5 or len(set(lobby_races))!=5 or set(lobby_races)-set(RACES):
            raise Blocked('lobby races must be five distinct observed tribes')
        view['lobby_races']=[int(r in lobby_races) for r in RACES]
        provenance['lobby_races']='observed_lobby_configuration'
    # dbf -> string id is static catalogue identity, not a generated game state.
    dbf=number(game,'BACON_GLOBAL_OLD_GOD_DBID')
    deity=next((cid for cid,v in card_info.items() if v.get('dbf')==dbf),None)
    if deity:
        # Lobby deity identity is not ownership. Native pl.deity is conditional
        # on this player's state; never turn a global DBF into deity_present=1.
        provenance['lobby_deity_identity']=deity
    sigils=[e for e in state.entities.values() if e.card_id=='BG_OldGod' and
        number(e.tags,'CONTROLLER')==pid and e.tags.get('ZONE')=='SECRET' and
        e.tags.get('BACON_DEITY_SIGIL')=='1']
    # Ownership is witnessed by this player's actual secret-zone sigil, not
    # merely by the lobby deity DBF. Both current stats must also be explicit.
    if str(data.get('build',state.build))=='255790' and len(sigils)==1 and deity and all(
            tag in own for tag in ('BACON_OLD_GOD_ATTACK','BACON_OLD_GOD_HEALTH')):
        view['deity']=deity;g.update(deity_present=1,deity_atk=number(own,'BACON_OLD_GOD_ATTACK'),
            deity_hp=number(own,'BACON_OLD_GOD_HEALTH'))
        provenance['deity']='owned_sigil:'+str(sigils[0].id)+'+explicit_player_stats+global_dbf'
    powers=[c for c in data['controls'] if c['card_type']=='HERO_POWER']
    for i,p in enumerate(sorted(powers,key=lambda c:(c['position'],c['entity_id']))[:2]):
        if 'NUM_TIMES_HERO_POWER_USED_THIS_GAME' in p['tags']:
            g[f'power{i}_uses_game']=number(p['tags'],'NUM_TIMES_HERO_POWER_USED_THIS_GAME')
        elif reader.history_complete:
            g[f'power{i}_uses_game']=reader.power_uses.get(p['entity_id'],0)
    dark=next((c for c in data['controls'] if c['card_id']=='BG36_Button_DarkGift'),None)
    if dark:
        view['dark_gift_entity']=dark['entity_id']
        g['can_dark_gift']=int(any(o['entity']==dark['entity_id'] and o['error']=='NONE' for o in state.options))
        g['dark_gift_uses']=reader.power_uses.get(dark['entity_id'],0)
    # Public leaderboard heroes are not the combat clone's controller. They carry
    # stable PLAYER_ID and SETASIDE/PLAY visibility; never publish opponent hands.
    heroes={}
    for entity in state.entities.values():
        t=entity.tags;card=entity.card();opid=number(t,'PLAYER_ID')
        if card.card_type=='HERO' and opid is not None and t.get('BACON_COMBAT_PHASE_HERO')!='1' and card.zone in ('PLAY','SETASIDE'):
            if 'PLAYER_LEADERBOARD_PLACE' in t or entity.id==(data.get('hero') or {}).get('entity_id'):
                heroes[opid]=card
    others=sorted(p for p in heroes if p!=pid)
    if len(others)==7:
        # Native ordering is cyclic player order, not current leaderboard rank.
        order=sorted(others,key=lambda p:((p-pid)%8))
        opponents=[]
        for p in order:
            c=heroes[p];alive=c.health is not None and c.health>0
            tier=number(c.tags,'PLAYER_TECH_LEVEL',c.tier)
            if c.health is None or tier is None: break
            opponents.append(dict(hero=c.card_id,alive=alive,health=c.health,armor=number(c.tags,'ARMOR',0),
                tier=tier,placement=0 if alive else number(c.tags,'PLAYER_LEADERBOARD_PLACE'),
                last=p==reader.last_opponent,next=p==number(own,'NEXT_OPPONENT_PLAYER_ID'),
                seen_board=copy.deepcopy(reader.seen_boards.get(p,[]))))
        if len(opponents)==7:
            view['opponents']=opponents;g['alive']=sum(int(o['alive']) for o in opponents)+1
            next_pid=number(own,'NEXT_OPPONENT_PLAYER_ID')
            if next_pid in heroes:
                g['next_is_ghost']=int((heroes[next_pid].health or 0)<=0)
    trinkets=[c['card_id'] for c in data['controls'] if c['card_type']=='BATTLEGROUND_TRINKET' and c.get('tags',{}).get('IS_CURRENT_TRINKET')=='1']
    if number(game,'BACON_TRINKETS_ACTIVE')==0: view['trinkets']=[];g['trinket_count']=0
    elif trinkets: view['trinkets']=trinkets;g['trinket_count']=len(trinkets)
    if str(data.get('build',state.build))=='255790' and reader.history_complete:
        # Two permanent placeholders become selected trinkets via CHANGE_ENTITY.
        # Both have ZONE_POSITION=0, so retain their witnessed creation slots.
        slots={};ambiguous=False;known_ids=set(reader.trinket_slots)
        for eid,slot in reader.trinket_slots.items():
            e=state.entities.get(eid)
            if not e or number(e.tags,'CONTROLLER')!=pid:continue
            # Hero replacement can retire an earlier placeholder pair. Only
            # current PLAY slots establish ownership; archived graveyard slots
            # must not make the later, explicitly created pair ambiguous.
            if e.tags.get('ZONE')!='PLAY':continue
            if e.card().card_type!='BATTLEGROUND_TRINKET' or slot in slots:
                ambiguous=True;continue
            if e.card_id in ('BG30_Trinket_1st','BG30_Trinket_2nd'):slots[slot]=None
            elif e.card_id in card_info and e.tags.get('BACON_TRINKET')=='1':slots[slot]=e.card_id
            else:ambiguous=True
        extra=[e for e in state.entities.values() if e.id not in known_ids and
            number(e.tags,'CONTROLLER')==pid and e.tags.get('ZONE')=='PLAY' and
            e.card().card_type=='BATTLEGROUND_TRINKET']
        if set(slots)=={0,1} and not ambiguous and not extra:
            view['trinkets']=[slots[i] for i in (0,1) if slots[i] is not None]
            g['trinket_count']=len(view['trinkets']);provenance['trinkets']='complete_history_placeholder_slots_and_CHANGE_ENTITY'
    if number(game,'BACON_QUESTS_ACTIVE')==0:view['quests']=[]
    secret=[e.card_id for e in state.entities.values() if number(e.tags,'CONTROLLER')==pid
            and e.tags.get('ZONE')=='SECRET' and e.card_id and e not in sigils]
    if any(number(e.tags,'CONTROLLER')==pid and e.tags.get('ZONE')=='7' for e in state.entities.values()):
        view.setdefault('unknown',[]).append('secrets.numeric_zone_mapping')
    view['secrets']=secret
    if data['shop']: g['frozen']=int(all(number(c['tags'],'FROZEN',0) for c in data['shop']))
    elif 'MODEL_FROZEN' in own: g['frozen']=number(own,'MODEL_FROZEN')
    # Generic aliases intentionally use exact semantic producer tags. For
    # example MAXRESOURCES=99 is a client ceiling, NOT model max_gold_bonus.
    for key,tag in {'max_gold_bonus':'MODEL_MAX_GOLD_BONUS','lost_last':'MODEL_LOST_LAST',
        'hand_queue_len':'MODEL_HAND_QUEUE_COUNT','tavern_atk':'MODEL_TAVERN_ATTACK',
        'tavern_hp':'MODEL_TAVERN_HEALTH','effects_len':'MODEL_EFFECT_COUNT','chronum':'MODEL_CHRONUM'}.items():
        if tag in own: g[key]=number(own,tag);provenance['global.'+key]='explicit_public_tag:'+tag
    return dict(game_id=data['game_id'],revision=data['revision'],view=view,**{'global':g},
                provenance=provenance,entity_tags={})
