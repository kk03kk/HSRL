"""Magnetic placement and causal attachment evidence, independent of card IDs."""
from .model import Kind


def magnetic(card):
    return bool(card and card.card_type=='MINION' and card.tags.get('MAGNETIC')=='1' and card.tags.get('SILENCED')!='1')


def mechanical(card):
    return bool(card and card.tags.get('CARDRACE') in ('MECHANICAL','ALL','17','26'))


def magnetic_context(obs,action):
    source=next((c for c in obs.hand if c.entity_id==action.source),None)
    target=next((c for c in obs.board if c.entity_id==action.target),None)
    return bool(action.kind==Kind.PLAY and magnetic(source) and source.controller==obs.local_player_id
        and source.tags.get('BATTLECRY')!='1' and action.sub_option_id is None
        and target and mechanical(target) and target.controller==obs.local_player_id
        and action.interaction and action.interaction.placement=='magnetize')


def magnetic_authorized(obs,action):
    if not magnetic_context(obs,action):return False
    return any(a.kind==Kind.PLAY and a.source==action.source and a.option_id==action.option_id
        and a.sub_option_id is None and (
            a.target is None and a.evidence in ('server_options','verified_insertion') or
            a==action and a.evidence=='verified_magnetic') for a in obs.actions)


def fusion_target(action,before,after,receipt):
    """新附魔需关联来源；旧附魔叠加需同一根内的目标关联和精确增量。"""
    source=next((c for c in before.hand if c.entity_id==action.source),None)
    if (not magnetic(source) or not receipt.complete or receipt.kind!=Kind.PLAY
            or receipt.source!=action.source or after.interaction is not None):return None
    if any(c.entity_id==action.source for c in after.hand+after.board):return None
    chain=[(old,new) for eid,old,new in receipt.zone_transitions if eid==action.source]
    if ('HAND','PLAY') not in chain or ('PLAY','SETASIDE') not in chain:return None
    if chain.index(('PLAY','SETASIDE'))<chain.index(('HAND','PLAY')):return None
    old_controls={c.entity_id:c for c in before.controls}
    old_board={c.entity_id:c for c in before.board};new_board={c.entity_id:c for c in after.board}
    targets=set()
    for c in after.controls:
        if c.card_type!='ENCHANTMENT' or c.zone!='PLAY':continue
        if c.controller!=before.local_player_id or c.tags.get('MAGNETIC')!='1':continue
        try:target=int(c.tags.get('ATTACHED',''))
        except ValueError:continue
        prior=old_board.get(target);current=new_board.get(target)
        if not (prior and current and mechanical(prior) and prior.controller==current.controller==before.local_player_id and prior.card_id==current.card_id):continue
        old=old_controls.get(c.entity_id)
        if old is None:
            if c.tags.get('CREATOR')==str(action.source):targets.add(target)
            continue
        if (old.card_type!='ENCHANTMENT' or old.zone!='PLAY' or old.controller!=c.controller
                or old.card_id!=c.card_id or old.tags.get('MAGNETIC')!='1'
                or any(old.tags.get(k)!=c.tags.get(k) for k in ('ATTACHED','CREATOR'))):continue
        # 不能凭属性上涨判成功：来源必须在本根明确记录此目标，且旧附魔的两项
        # 数值在同根从原值增长到新值，增量分别等于打出来源的攻血。
        changes=receipt.entity_changes
        if not any(eid==action.source and tag=='TAG_SCRIPT_DATA_NUM_1' and new==str(target)
                   for eid,tag,_,new in changes):continue
        matched=True
        for tag,amount in (('TAG_SCRIPT_DATA_NUM_1',source.attack),('TAG_SCRIPT_DATA_NUM_2',source.health)):
            try:a=int(old.tags[tag]);b=int(c.tags[tag])
            except (KeyError,TypeError,ValueError):matched=False;break
            if amount is None or b-a!=amount or not any(eid==c.entity_id and changed==tag and
                    previous==str(a) and new==str(b) for eid,changed,previous,new in changes):
                matched=False;break
        if matched:targets.add(target)
    return next(iter(targets)) if len(targets)==1 else None
