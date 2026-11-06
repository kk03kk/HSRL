"""Shared identity gate for initial and recovered single-choice confirmation.

Button appearance proves neither which option is selected nor what it costs.
Hero selection requires independent halo/slot identity evidence; an enabled
button alone does not prove the requested hero was selected.
"""


def request_identity(obs):
    ctx=obs.interaction
    return (obs.game_id,obs.local_player_id,ctx.request_id,ctx.source_entity,ctx.task_list,ctx.owner,ctx.choice_type) if ctx else None


def confirmation_reasons(before,current,action,*,selected,enabled):
    reasons=[]
    if str(current.phase)!='choice' or request_identity(before)!=request_identity(current):
        reasons.append('choice_request_context_changed')
    if not current.interaction or current.interaction.request_id!=action.choice_id:
        reasons.append('choice_request_mismatch')
    if tuple(c.entity_id for c in before.choices)!=tuple(c.entity_id for c in current.choices):
        reasons.append('choice_candidate_order_changed')
    previous=next((c for c in before.choices if c.entity_id==action.source),None)
    candidate=next((c for c in current.choices if c.entity_id==action.source),None)
    if not previous or not candidate or (previous.card_id,previous.card_type,previous.controller)!=(candidate.card_id,candidate.card_type,candidate.controller):
        reasons.append('choice_candidate_identity_changed')
    if candidate and candidate.tags.get('BACON_LOCKED_MULLIGAN_HERO') in ('1',1):
        reasons.append('hero_choice_locked')
    if candidate and candidate.card_type=='BATTLEGROUND_TRINKET':
        if candidate.cost is None or candidate.cost<0 or current.gold is None:reasons.append('trinket_price_unknown_at_confirmation')
        elif candidate.cost>current.gold:reasons.append('insufficient_choice_gold_at_confirmation')
    if tuple(selected)!=(action.source,):reasons.append('selected_choice_identity_unproven')
    if not enabled:reasons.append('choice_confirm_enabled_state_unknown')
    if current.interaction and current.interaction.display_state in ('waiting','hidden'):
        reasons.append('choice_client_not_shown')
    return tuple(reasons)


def unselected_hero_evidence(mapper,obs,image,window,action):
    """证明本地英雄选择未提交；只供有界同项重选，不用于确认或消耗型动作重发。"""
    import cv2
    import numpy as np
    from .dependency_readiness import choice_slot_structure
    if str(obs.phase)!='choice' or not obs.choices or any(c.card_type!='HERO' for c in obs.choices):return False
    if not obs.interaction or obs.interaction.selected or obs.interaction.display_state in ('hidden','waiting'):return False
    if getattr(mapper,'mode',None)!='geometry':return False
    viewport=mapper._viewport(image,window)
    projection=mapper.resolver.resolve(obs,viewport,session=mapper.session)
    spec=next((s for s in projection.layouts if s.zone=='choices' and s.status!='candidate'),None)
    confirm=next((m.rect for m in projection.mappings if m.key=='control:confirm'),None)
    own=next((m.rect for m in projection.mappings if m.entity_id==action.source),None)
    if spec is None or confirm is None or own is None:return False
    # 所有候选当前可见；原来源必须仍然是未锁定的同一实体。
    if confirmation_reasons(obs,obs,action,selected=(action.source,),enabled=True):return False
    if not all(choice_slot_structure(image,mapper._bounds(spec,viewport,i)) for i in range(len(obs.choices))):return False
    state=mapper.confirmation(obs,image,window)
    if state is None or state[0] is not None or state[1]:return False
    hsv=cv2.cvtColor(np.asarray(image.crop((confirm.x,confirm.y,confirm.x+confirm.width,confirm.y+confirm.height))),cv2.COLOR_RGB2HSV)
    neutral=(hsv[:,:,1]<65)&(hsv[:,:,2]>65)&(hsv[:,:,2]<245)
    return bool(neutral.mean()>.65)
