"""Fresh post-input target views; cached pre-play features are never reused."""
import copy
from .contracts import Blocked,digest


def choice_context(data):
    interaction=data.get('interaction') or {}
    return digest(dict(game_id=data.get('game_id'),turn=data.get('turn'),
        request_id=interaction.get('request_id'),owner=interaction.get('owner'),
        source_entity=interaction.get('source_entity'),choice_type=interaction.get('choice_type'),
        candidates=[(c['entity_id'],c['card_id']) for c in data.get('choices',[])]))


def target_snapshot(snapshot,pending,visual,*,allow_public_hero=False):
    """Target options came from the original server option; UI proves the stage.

    Only target objects represented by the trained pending block are supported.
    Hero targets and unknown entities fail; they are not substituted with zeros.
    """
    if not pending or not pending[0].binding.targets or not visual.get('target_open'):return snapshot
    decision,before,_=pending;b=decision.binding;data=snapshot['client']
    if data.get('choices') or data.get('interaction'):return snapshot
    if snapshot['game_id']!=decision.game_id or data['turn']!=before['client']['turn']:
        raise Blocked('target context changed from the pinned source action')
    if data.get('log_position',0)<before['client'].get('log_position',0):raise Blocked('target source cursor regressed')
    sources={c['entity_id']:c for z in ('hand','board','controls') for c in before['client'].get(z,[])}
    source=sources.get(b.source)
    if source is None:raise Blocked('target source identity is missing')
    current={};canonical={}
    for zone in ('board','shop','hand'):
        cards=data.get(zone,[]);values=snapshot['view'].get(zone,[])
        if len(cards)!=len(values):raise Blocked('target zone projection disagrees with source entities')
        for card,value in zip(cards,values):
            current[card['entity_id']]=card
            if zone=='board':
                canonical[card['entity_id']]=dict(card_id=value['card_id'],is_minion=True,minion=copy.deepcopy(value))
            else:canonical[card['entity_id']]=copy.deepcopy(value)
    if allow_public_hero:
        for card in snapshot.get('target_heroes',[]):
            current[card['entity_id']]=card
            canonical[card['entity_id']]=dict(card_id=card['card_id'],card_only=True)
    if len(set(b.targets))!=len(b.targets) or len(b.targets)>16:
        raise Blocked('target authorization is duplicated or exceeds model slots')
    if any(e not in current for e in b.targets):raise Blocked('target is outside the trained board/shop/hand representation')
    result=copy.deepcopy(snapshot);view=result['view'];client=result['client']
    choices=[]
    for eid in b.targets:
        item=canonical[eid]
        # Pending::Target encodes an Item, not its hand/shop container flags.
        item.update(cost=0,locked=0,temporary=False,combined=False,frozen=False,kept=False,marked=False)
        choices.append(item)
    view.update(pending=4,pending_source=source['card_id'],choices=choices)
    client['choices']=[copy.deepcopy(current[e]) for e in b.targets]
    client['interaction']=dict(mode='target',request_id='target:'+decision.stamp,min_select=1,max_select=1,
        owner=data['local_player_id'],source_entity=b.source,
        authority='pinned_server_option',option_id=b.option_id)
    if allow_public_hero:
        result['approximation']=copy.deepcopy(result.get('approximation',{}))
        result['approximation']['target_timing']='public target stage; uncommitted source may still appear in hand'
        if any(e in {c['entity_id'] for c in snapshot.get('target_heroes',[])} for e in b.targets):
            result['approximation']['hero_target']='known visible hero encoded as a card-only pending item'
        if b.kind=='play' and source['card_type']=='MINION' and b.source not in {c['entity_id'] for c in data['board']}:
            result['target_preview']=dict(source=source,position=b.position)
    return result


def branch_snapshot(snapshot,pending,visual):
    if not pending or not pending[0].binding.branches or not visual.get('branch_open'):return snapshot
    d,before,_=pending;b=d.binding;data=snapshot['client']
    if b.sub_option_id is not None:return snapshot
    if data.get('choices') or data.get('interaction'):return snapshot
    if snapshot['game_id']!=d.game_id or data['turn']!=before['client']['turn']:
        raise Blocked('branch source context changed')
    source=next((c for z in ('hand','board','controls') for c in before['client'].get(z,[]) if c['entity_id']==b.source),None)
    # A generic two-card panel is insufficient to identify which source opened it.
    if not source or visual.get('branch_source_card_id')!=source['card_id']:
        raise Blocked('branch panel source identity is unproven')
    branch_ids=[r[0] for r in b.branches]
    if visual.get('branch_entities')!=branch_ids:raise Blocked('visible branches differ from pinned authorization')
    current={c['entity_id']:c for c in data.get('branches',[])}
    if any(e not in current for e in branch_ids):raise Blocked('fresh branch entities are unavailable')
    result=copy.deepcopy(snapshot)
    result['view'].update(pending=3,pending_source=source['card_id'],
        choices=[dict(card_id=source['card_id'],card_only=True) for e in branch_ids])
    result['client']['choices']=[copy.deepcopy(current[e]) for e in branch_ids]
    result['client']['interaction']=dict(mode='branch',request_id='branch:'+d.stamp,min_select=1,max_select=1,
        owner=data['local_player_id'],source_entity=b.source,branch_options=b.branches,
        authority='pinned_server_subOptions')
    return result
