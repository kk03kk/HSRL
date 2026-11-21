"""Public client evidence for the Player150 observation and its missing-data flags."""
import re

import numpy as np

from hsbg.az.train import player_observation
from hsbg.env import obs_std as O
from .contracts import Blocked
from .encoding import minion
from .projection import rogers_remaining


GROWTH_TAGS = {
    8: 'BACON_BLOODGEMBUFFATKVALUE',
    9: 'BACON_BLOODGEMBUFFHEALTHVALUE',
}
WRATHGUARDS = {'BG36_362', 'BG36_362_G'}


def known_number(tags, tag):
    value = tags.get(tag)
    if isinstance(value, (int, str)) and re.fullmatch(r'-?\d+', str(value)):
        return int(value)
    return None


def observed_context(reader, data):
    """Extract only owned counters and an explicitly announced next opponent."""
    state = reader.state
    pid = data['local_player_id']
    player = state.entities.get(state.players.get(pid))
    tags = player.tags if player else {}
    growth = [None]*10
    for i, tag in GROWTH_TAGS.items():
        growth[i] = known_number(tags, tag)
    next_pid = known_number(tags, 'NEXT_OPPONENT_PLAYER_ID')
    encounter = None
    seen_turn = None
    if (data.get('phase') in ('recruit','choice') and type(pid) is int and type(next_pid) is int
            and 1<=pid<=8 and 1<=next_pid<=8 and next_pid!=pid):
        heroes = [entity for entity in state.entities.values()
                  if known_number(entity.tags, 'PLAYER_ID')==next_pid
                  and entity.tags.get('CARDTYPE')=='HERO' and entity.tags.get('PLAYER_LEADERBOARD_PLACE') is not None
                  and entity.tags.get('BACON_COMBAT_PHASE_HERO')!='1'
                  and entity.tags.get('ZONE') in ('PLAY','SETASIDE') and entity.card_id]
        if len(heroes)==1:
            encounter=dict(slot=(next_pid-pid)%8-1, hero=heroes[0].card_id)
            seen_turn=getattr(reader,'encounter_turns',{}).get(next_pid)
            if seen_turn is None and reader.history_complete and not reader.combat_seen:
                seen_turn=0
    return dict(growth=growth, health_prices=[None]*len(data.get('shop',[])), encounter=encounter,
                last_seen_turn=seen_turn)


def rendered_text(info, cid, card):
    text=info[cid].get('text')
    if not isinstance(text,str):
        return None
    if not card:
        return text
    remaining=rogers_remaining(card)
    if remaining is not None:
        return re.sub(r'(?<=\()9(?= Gold left!\))',str(remaining),text)
    if cid in WRATHGUARDS:
        tags=card.get('tags',{})
        attack, health=(known_number(tags,'TAG_SCRIPT_DATA_NUM_'+n) for n in ('1','2'))
        if attack is not None and health is not None and attack>=0 and health>=0:
            return re.sub(r'\+\d+/\+\d+',f'+{attack}/+{health}',text,count=1)
    return text


def card_faces(snapshot, info):
    view=snapshot['view'];data=snapshot['client']
    by_id={c['entity_id']:c for c in data.get('controls',[])}
    faces=[]
    def put(slot, cid, card=None):
        if cid in info:
            text=rendered_text(info,cid,card)
            if text is not None:faces.append([slot,cid,text])
    for i,eid in enumerate(view.get('power_entities',[])[:2]):
        card=by_id.get(eid)
        if card:put(1+i,card['card_id'],card)
    for i,quest in enumerate(view.get('quests',[])[:3]):
        if quest:
            put(5+3*i,quest.get('card_id'));put(6+3*i,quest.get('reward'))
    for i,reward in enumerate(view.get('owned_rewards',[])[:3]):put(7+3*i,reward)
    rewards={r['entity_id']:r for r in data.get('choice_rewards',[])}
    for zone,base,width in (('board',19,7),('hand',26,10),('shop',36,8),('choices',44,16)):
        if zone=='choices' and view['pending'] not in (2,6,7,8):
            continue  # Native Pending::Hero, Branch and Target carry no extra faces.
        for i,card in enumerate(data.get(zone,[])[:width]):
            put(base+i,card['card_id'],card)
            if zone=='choices' and view['pending']==7:
                reward=rewards.get(known_number(card.get('tags',{}),'TAG_SCRIPT_DATA_ENT_1'))
                if reward:put(base+i,reward['card_id'],reward)
    for i,cid in enumerate(view.get('trinkets',[])[:8]):put(60+i,cid)
    return faces


def public_facts(snapshot, info):
    context=snapshot.get('player150_public')
    remaining=snapshot.get('remaining_actions')
    if not isinstance(context,dict) or type(remaining) is not int or not 0<=remaining<=150:
        raise Blocked('Player150 requires a witnessed public snapshot and controller action budget')
    return {**context,'remaining':remaining,'faces':card_faces(snapshot,info)}


def encode_observation(features, cards, snapshot, vocabulary, info):
    facts=public_facts(snapshot,info)
    features=features.copy();cards=cards.copy()
    features[O.SLICE_OPPONENTS]=0.;features[O.SLICE_OPP_BOARDS]=0.
    cards[O.SLICE_CARD_OPP_HEROES]=0;cards[O.SLICE_CARD_OPP_BOARDS]=0
    other=None if snapshot['view']['global'].get('hero_select') else facts['encounter']
    if other:
        i=other['slot']
        if type(i) is not int or not 0<=i<O.N_OPP:
            raise Blocked('Player150 encounter slot is invalid')
        at=O.SLICE_OPPONENTS.start+i*O.OPP_F
        features[at]=features[at+8]=1.
        cards[O.SLICE_CARD_OPP_HEROES.start+i]=vocabulary.get(other['hero'],0)
        # Only certified memories of this encounter can populate these fields.
        board=other.get('board',[])
        if board:
            if len(board)>O.N_BOARD:raise Blocked('Player150 encounter board overflow')
            features[at+4]=other['tier']/6
            features[at+5]=len(board)/7
            for index,item in enumerate(board,i*O.N_BOARD):
                start=O.SLICE_OPP_BOARDS.start+index*O.MINION_F
                features[start:start+O.MINION_F]=minion(item)
                cards[O.SLICE_CARD_OPP_BOARDS.start+index]=vocabulary.get(item['card_id'],0)
    return np.concatenate((features,player_observation.encode(facts))),cards
