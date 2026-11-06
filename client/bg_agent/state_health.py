"""Channel and projection health, separate from observation capture timestamps."""
import time
from .network_state import blocks_game_input


def projection_errors(obs):
    errors=[];seen=set()
    zones=('choices',) if str(obs.phase)=='choice' else ('hand','board','shop','choices')
    for name in zones:
        cards=getattr(obs,name)
        ids=[c.entity_id for c in cards]
        if len(ids)!=len(set(ids)) or seen.intersection(ids):errors.append('duplicate_entity:'+name)
        seen.update(ids)
        # Choice order is supplied by its own request, not always ZONE_POSITION.
        if name!='choices' and cards and sorted(c.position for c in cards)!=list(range(1,len(cards)+1)):
            errors.append('invalid_slot_sequence:'+name)
        if name in ('hand','board') and obs.local_player_id is not None and any(c.controller!=obs.local_player_id for c in cards):
            errors.append('wrong_controller:'+name)
    if str(obs.phase)!='choice':
        if len(obs.hand)>10:errors.append('hand_overflow_snapshot')
        if len(obs.board)>7:errors.append('board_overflow_snapshot')
    return tuple(errors)


class ChannelHealth:
    """Silence is permitted, indefinitely stale authority is not.

    No file observer can distinguish a genuinely idle game from a silently
    frozen producer immediately. Explicit I/O/network faults fail immediately;
    lack of semantic progress has a bounded lease independent of captured_at.
    """
    def __init__(self,silence_limit=180):
        self.silence_limit=silence_limit;self.token=None;self.progress_at=None

    def inspect(self,obs,*,session,error=None,network='unknown',now=None):
        now=time.monotonic() if now is None else now
        token=(str(session),obs.game_id,obs.log_position,obs.phase,obs.turn)
        if token!=self.token:self.token=token;self.progress_at=now
        elapsed=now-self.progress_at
        reasons=[]
        if error:reasons.append('log_io_unhealthy')
        if blocks_game_input(network,obs.phase):reasons.append('network_'+network)
        if str(obs.phase) in ('recruit','choice','combat') and elapsed>self.silence_limit:
            reasons.append('semantic_progress_lease_expired')
        # Combat/death logs can legitimately retain gaps, and menu/result
        # snapshots retain the last board. They are not recruit authority.
        if str(obs.phase) in ('recruit','choice'):reasons.extend(projection_errors(obs))
        return dict(healthy=not reasons,reasons=reasons,silence_seconds=elapsed,
            authority='log_progress_and_projection',network=network)
