"""Shared protocol facts. Pure functions; never grant input admission."""


def owned_selection(data):
    return bool(data.get('choices') and data.get('local_player_id') is not None
                and (data.get('interaction') or {}).get('owner')==data['local_player_id'])


def continuations(binding,data,*,confirm_sent=False,target_sent=False,branch_sent=False):
    result=[]
    if owned_selection(data):result.append('server_selection')
    if binding.kind=='choose' and not confirm_sent:result.append('selection_confirmation')
    if binding.targets and binding.target is None and not target_sent:result.append('target')
    if binding.branches and not branch_sent:result.append('branch')
    return tuple(result)


def protocol_state(controller,data):
    pending=controller.pending;halted=controller.halted
    steps=continuations(pending[0].binding,data,confirm_sent=controller.confirm_sent,
        target_sent=controller.target_sent,branch_sent=controller.branch_sent) if pending else ()
    if halted:stage='delivery_unknown' if halted=='input interrupted; delivery unknown' else 'halted'
    elif pending:stage='continuation' if steps else 'receipt'
    elif controller.ended==(data.get('game_id'),data.get('turn')) and not owned_selection(data):stage='next_turn'
    else:stage='decision'
    return dict(protocol_stage=stage,continuations=list(steps),parent_depth=len(controller.parents))


def physical_context(data,pending=None):
    interaction=data.get('interaction') or {};stage=None
    if pending and (pending[0].binding.targets or pending[0].binding.branches):
        stage=(pending[0].stamp,pending[0].binding.sub_option_id)
    if owned_selection(data) and interaction.get('mode')=='choice':stage=None
    mode=interaction.get('mode');authority=interaction.get('authority')
    synthetic=bool(pending and authority=={'branch':'pinned_server_subOptions','target':'pinned_server_option'}.get(mode)
        and mode in ('branch','target') and data['game_id']==pending[0].game_id
        and data.get('local_player_id') is not None and interaction.get('owner')==data['local_player_id']
        and interaction.get('source_entity')==pending[0].binding.source
        and interaction.get('request_id')==mode+':'+pending[0].stamp)
    return (data['game_id'],data.get('turn'),data['phase'],None if synthetic else interaction.get('request_id'),stage,
            () if synthetic else tuple(c['entity_id'] for c in data.get('choices',[])))


def terminal_proof(state,data,*,history_complete):
    if not history_complete or data.get('mode')!='GT_BATTLEGROUNDS' or data.get('local_player_id') is None:return None
    player=state.entities.get(state.players.get(data['local_player_id']));game=state.entities.get(state.game_entity)
    result=player.tags.get('PLAYSTATE') if player else None
    if result not in ('WON','LOST','TIED','CONCEDED','4','5','6','8'):result=game.tags.get('STATE') if game else None
    if result not in ('WON','LOST','TIED','CONCEDED','4','5','6','8','COMPLETE','3'):return None
    return dict(game_id=data['game_id'],local_player_id=data['local_player_id'],proof=result,log_position=data.get('log_position'))
