"""Observable combat corpus only. This is NOT the model's recruit-time memory."""
from dataclasses import asdict


def combat_frame(state):
    pid=state.local_player_id;player=state.entities.get(state.players.get(pid));tags=player.tags if player else {}
    cards=[e.card() for e in state.entities.values()]
    heroes=[c for c in cards if c.card_type=='HERO' and c.zone=='PLAY' and
            c.tags.get('BACON_COMBAT_PHASE_HERO')=='1' and c.controller!=pid]
    reasons=[]
    if tags.get('PLAYSTATE')!='PLAYING':reasons.append('local player is not explicitly playing')
    current=tags.get('BACON_CURRENT_COMBAT_PLAYER_ID')
    if current is not None and str(current)!=str(pid):reasons.append('client is showing another player combat')
    hero=heroes[0] if len(heroes)==1 else None
    if hero is None:reasons.append('combat opponent hero identity is ambiguous')
    opponent=int(hero.tags['PLAYER_ID']) if hero and str(hero.tags.get('PLAYER_ID','')).isdigit() else None
    owner=hero.controller if hero else None
    # PLAYER_ID is stable leaderboard identity. CONTROLLER is the temporary
    # combat/Bob-side owner (14 in the actual 254477 corpus), not that player ID.
    def board(controller):
        if controller is None:return []
        return [asdict(c) for c in sorted((c for c in cards if c.controller==controller and c.zone=='PLAY'
                    and c.card_type=='MINION'),key=lambda c:(c.position,c.entity_id))]
    own,other=board(pid),board(owner)
    if pid is None or opponent is None or opponent==pid or owner==pid:reasons.append('combat player mapping is incomplete')
    for name,values in [('own',own),('opponent',other)]:
        if not values:reasons.append(name+' board absent at an attack boundary')
        if len(values)>7 or [c['position'] for c in values]!=list(range(1,len(values)+1)):
            reasons.append(name+' board positions are incomplete')
        if any(not c['card_id'] for c in values):reasons.append(name+' card identity missing')
    return dict(game=state.generation,turn=tags.get('TURN'),build=state.build,local_player=pid,
        opponent=opponent,combat_opponent_controller=owner,own=own,opponent_board=other,
        timing='before_first_observed_attack',opening_effects_completion='not_certified',public_only=True,
        structural_check_passed=not reasons,exclusion_reasons=reasons,
        model_memory_compatible=False,training_library_eligible=False)
