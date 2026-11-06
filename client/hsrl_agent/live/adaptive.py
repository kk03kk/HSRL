"""Explicit lossy public-state adapter for execution-focused client integration.

Never completes a real game with a simulator. Strategic unknowns are estimated;
server authorizations, identities, turn/playback and transaction gates stay real.
"""
import copy
from hsbg.env import obs_std as O
from .contracts import Blocked
from .projection import Projector,number,RACE_IDS
from .facts import context_facts


class AdaptiveProjector(Projector):
    def __init__(self,profile,info):
        super().__init__(profile,copy.deepcopy(info));self.estimates={}

    def build_public(self,reader,data,*,now=None):
        # Published entities are immutable. Only replace top-level metadata;
        # copying nested historical receipts every inference is unnecessary.
        data=dict(data);est={}
        def estimate(path,value,reason):est[path]=dict(value=value,reason=reason);return value
        choices=data.get('choices',[])
        hero_choice=bool(choices) and all(c['card_type']=='HERO' for c in choices)
        # Resource tags need not exist during the initial hero selection.
        if hero_choice:
            if data['gold'] is None:data['gold']=estimate('global.gold',0,'hero-selection pre-recruit baseline')
            if data['turn'] is None:data['turn']=estimate('global.turn',0,'hero-selection phase')
            data['warnings']=[w for w in data['warnings'] if w!='gold_unknown']
        interaction=data.get('interaction') or {}
        if choices and data['ready'] and interaction.get('owner')==data['local_player_id']:
            # EntityChoices independently authorizes selection while the next
            # ordinary Options list is intentionally suspended until selection.
            data['options_pending']=False;data['client_task_list']=None
        facts=context_facts(reader,data,self.info)
        facts['view'].pop('unknown',None)
        g=facts['global'];v=facts['view'];entity_tags=facts['entity_tags']
        for name in O.GLOBAL_FIELD_NAMES:
            if name not in g:g[name]=estimate('global.'+name,0,'unreconstructed engine counter; explicit neutral approximation')
        # Common semantic prices are observation estimates only. They NEVER
        # authorize an action or override the live Options legal mask.
        g['minion_price']=estimate('global.minion_price',3,'base purchase-price prior; actual action legality remains server-authorized')
        g['refresh_price']=estimate('global.refresh_price',1,'base refresh prior; explicit live quote overrides')
        for key,default in [('lobby_races',[0]*11),('quests',[]),('owned_rewards',[]),
                            ('trinkets',[]),('deity',None),('opponents',[None]*7)]:
            if key not in v:v[key]=estimate(key,default,'not reconstructed from current public evidence')
        if reader.combat_seen:
            estimate('opponents.seen_board',None,'unavailable pre-combat memory; unseen slots remain empty')
        g['alive']=sum(bool(o and o['alive']) for o in v['opponents'])+1 if any(v['opponents']) else 8
        g['trinket_count']=len(v['trinkets']);g['deity_present']=int(v['deity'] is not None)
        for z in ('board','hand','shop','choices'):
            for i,c in enumerate(data.get(z,[])):
                cid=c['card_id'];t=c['tags']
                if not cid:raise Blocked('visible actionable entity has no card identity')
                if cid not in self.info:
                    self.info[cid]=dict(id=cid,tier=number(t,'TECH_LEVEL',0),atk=number(t,'ATK',0),
                        hp=number(t,'HEALTH',1),cost=number(t,'COST',0),races=[],keywords=[])
                    estimate(z+'.'+str(i)+'.card_embedding',0,'out-of-vocabulary card: padding identity, live stats and legal entity retained')
                info=self.info[cid];patch={}
                def put(tag,value,reason):
                    if tag not in t:patch[tag]=str(estimate(f'{z}.{i}.{tag}',value,reason))
                for tag in ('MODEL_LOCKED_TURNS','MODEL_TEMPORARY','MODEL_COMBINED','MODEL_DOOMED_THIS_TURN',
                            'MODEL_KEPT_FROM_LAST_TURN','MODEL_DARKMOON_TICKET','MODEL_LINKED_GROUP','MODEL_SPELL_DATA'):
                    put(tag,0,'missing private item state; neutral approximation')
                put('MODEL_PURCHASE_PRICE',number(t,'INTERACTABLE_OBJECT_COST',number(t,'COST',
                    3 if c['card_type']=='MINION' else info.get('cost',0))),'public/base price approximation')
                put('MODEL_OFFER_PRICE',number(t,'COST',info.get('cost',0)),'displayed/base choice price')
                if c['card_type']=='MINION':
                    put('ATK',info.get('atk',0),'base attack when client omits current stat')
                    put('HEALTH',max(1,info.get('hp',1)),'base health when client omits current stat')
                    races=info.get('races',[])
                    put('CARDRACE',','.join(races) if races else 'NONE','base race; unreported dynamic additional races omitted')
                    aliases={'IMMUNE_ATTACKING':'IMMUNE_WHILE_ATTACKING'}
                    for kw in O.KEYWORD_NAMES:
                        tag=aliases.get(kw,kw)
                        if kw not in t and tag not in t:put(kw,int(kw in info.get('keywords',[])),'static keyword prior; unreported modifications omitted')
                    put('MODEL_SHIELD_HITS',number(t,'DIVINE_SHIELD',int('DIVINE_SHIELD' in info.get('keywords',[]))),
                        'explicit shield count or static prior')
                    put('MODEL_ATTACHED_COUNT',0,'engine attachment count not reconstructed')
                entity_tags[str(c['entity_id'])]=patch
        powers=sorted((c for c in data['controls'] if c['card_type']=='HERO_POWER'),key=lambda c:(c['position'],c['entity_id']))
        for i,c in enumerate(powers[:2]):
            entity_tags.setdefault(str(c['entity_id']),{})['MODEL_POWER_SLOT']=str(i)
            g[f'power{i}_price']=number(c['tags'],'INTERACTABLE_OBJECT_COST',number(c['tags'],'COST',
                self.info.get(c['card_id'],{}).get('cost',0)))
        if choices:
            if hero_choice:kind=1
            elif all(c['card_type']=='HERO_POWER' for c in choices):kind=5
            elif all(c['card_type']=='BATTLEGROUND_TRINKET' for c in choices):kind=6
            elif any(c['tags'].get('QUEST')=='1' for c in choices):kind=7
            else:kind=estimate('pending.kind',2,'generic discover projection; unknown selection subtypes approximated')
            facts['pending']=kind
        result=super().build(reader,data,now=now,facts=facts,allow_owned_choice_block=True,allow_idle_live_tail=True)
        result['source_evidence']='current complete connected tail' if reader.live_tail_current() else 'recent source event'
        # Some inference-only quantities are derived late by the strict projector.
        # Every remaining imputation is recorded rather than called exact.
        view=result['view']
        for key in view.get('unknown',[]):
            if key.startswith('global.'):
                name=key[7:]
                view['global'][name]=estimate(key,g.get(name,0),'late missing scalar')
            else:raise Blocked('adapter has an unhandled structural missing field:'+key)
        view['unknown']=[]
        est={k:r for k,r in est.items() if not (k.startswith('global.') and view['global'].get(k[7:])!=r['value'])}
        result['approximation']=dict(mode='public-approx-v1',lossless=False,fields=est,
            legal_actions='real server Options / EntityChoices; never estimated')
        result['comparison']='semantic-v1'
        self.estimates=est
        result['target_heroes']=[]
        if data.get('hero'):
            result['target_heroes'].append({**data['hero'],'is_local':True})
        for entity in reader.state.entities.values():
            if entity.card_id=='TB_BaconShopBob' and entity.tags.get('ZONE')=='PLAY':
                from dataclasses import asdict
                result['target_heroes'].append({**asdict(entity.card()),'is_local':False})
        return result


def prospective_target(snapshot,decision):
    """Model target decision before server commit; explicitly prospective."""
    from .interactions import target_snapshot
    s=target_snapshot(snapshot,(decision,snapshot,0),dict(target_open=True),allow_public_hero=True)
    s.setdefault('approximation',{})['target_timing']='prospective precommit public state'
    return s
