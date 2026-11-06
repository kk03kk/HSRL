"""State relevant to a bound gesture and its target pixel evidence."""
import hashlib,json


def input_context_token(obs,action,*,ignore_playback=False):
    def cards(group):
        return [(c.entity_id,c.card_id,c.controller,c.zone,c.position,c.card_type,c.attack,c.health,c.cost,
                 tuple(c.tags.get(k) for k in ('FROZEN','MAGNETIC','BATTLECRY','CARDRACE','EXHAUSTED','INTERACTABLE_OBJECT','LITERALLY_UNPLAYABLE'))) for c in group]
    ctx=obs.interaction
    control_ids={action.source,action.authorization_source}
    controls=[(c.entity_id,c.card_id,c.cost,c.tags.get('EXHAUSTED'),c.tags.get('INTERACTABLE_OBJECT_COST'),c.tags.get('GAME_MODE_BUTTON_SLOT'))
              for c in obs.controls if c.entity_id in control_ids]
    advertised=action in obs.actions
    if action.evidence=='verified_magnetic':
        from .magnetic import magnetic_authorized
        advertised=magnetic_authorized(obs,action)
    if action.evidence=='verified_board_move':
        advertised=any(a.key==action.key and a.evidence in ('bg_move_adapter','verified_board_move') for a in obs.actions)
    if action.evidence=='verified_insertion':
        advertised=action.position is not None and 1<=action.position<=len(obs.board)+1 and any(
            a.kind==action.kind and (a.source,a.target,a.option_id,a.sub_option_id)==(action.source,action.target,action.option_id,action.sub_option_id)
            and a.evidence in ('server_options','verified_insertion') for a in obs.actions)
    if action.evidence=='verified_hand_target':
        advertised=any(a.kind==action.kind and (a.source,a.target,a.option_id,a.sub_option_id)==(action.source,action.target,action.option_id,action.sub_option_id) and a.evidence in ('server_options','verified_hand_target') for a in obs.actions)
    if action.evidence=='verified_target_cancel':
        from .client_protocol import cancellable_target
        advertised=cancellable_target(obs,action)
    if action.evidence=='verified_phase_wait':
        advertised=obs.phase=='recruit' and obs.interaction is None
    if ctx and ctx.protocol=='verified_multi_select' and action.choice_id==ctx.request_id:
        if action.kind=='select':
            advertised=action.source in {c.entity_id for c in obs.choices} and action.source not in ctx.selected and action.selected==ctx.selected+(action.source,) and len(action.selected)<=ctx.max_select
        elif action.kind=='confirm':advertised=action.selected==ctx.selected and ctx.min_select<=len(ctx.selected)<=ctx.max_select
    state=dict(game=obs.game_id,mode=obs.mode,phase=obs.phase,player=obs.local_player_id,options_pending=obs.options_pending,
        client_playback_busy=bool(not ignore_playback and str(obs.phase)=='recruit' and obs.interaction is None and obs.client_task_list is not None),
        client_submission_position=obs.client_submission_position if ignore_playback else None,
        hero=(obs.hero.entity_id,obs.hero.tags.get('HERO_POWER_ENTITY')) if obs.hero else None,
        turn=obs.turn,gold=obs.gold,tier=obs.tavern_tier,upgrade=obs.upgrade_cost,
        zones=[cards(getattr(obs,z)) for z in ('hand','board','shop','choices')],controls=controls,
        branches=cards(obs.branches) if action.sub_option_id is not None else (),
        request=(ctx.mode,ctx.request_id,ctx.min_select,ctx.max_select,ctx.selected,
                 ctx.source_entity,ctx.task_list,ctx.owner,ctx.choice_type,ctx.display_state) if ctx else None,
        advertised=advertised)
    # 战场手势只绑定相关区域；服务器仍需提供原动作，不能借此沿用已撤销的动作。
    board={c.entity_id:c for c in obs.board}
    board_action=action.kind in ('sell','reorder','activate') and action.source in board
    board_target=action.kind=='target' and action.target in board
    if (board_action or board_target) and (action.target is None or action.target in board):
        state['zones']=[[(c.entity_id,c.card_id,c.controller,c.zone,c.position,c.card_type) for c in obs.board]]
        state['tier']=state['upgrade']=None
        if action.kind in ('sell','reorder','target'):state['gold']=None
        if action.kind=='activate':
            card=board[action.source]
            state['ability_source']=(cards((card,)),tuple(sorted(card.tags.items())))
        if action.target in board:state['board_target']=cards((board[action.target],))
    source=next((c for c in obs.hand if c.entity_id==action.source),None)
    if (action.kind=='play' and source and source.card_type in ('SPELL','BATTLEGROUND_SPELL')
            and action.target in board and action.sub_option_id is None and ctx is None):
        # 直接定向法术依赖手牌重排和战场身份/位置；宝石造成的攻血变化不移动目标。
        # 仍绑定来源全部标签、费用、金币与服务器当前授权，不能复用撤销的动作。
        state['zones']=[[(c.entity_id,c.card_id,c.controller,c.zone,c.position,c.card_type)
                         for c in group] for group in (obs.hand,obs.board)]
        state['spell_source']=(cards((source,)),tuple(sorted(source.tags.items())))
        state['tier']=state['upgrade']=None
    return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
