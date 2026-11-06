"""Small client protocols with no guessed controls or input side effects."""
from .model import Kind,Phase


def passive_end(action,obs):
    return (action.kind==Kind.END and action.evidence=='verified_phase_wait'
        and action in obs.actions and obs.mode=='GT_BATTLEGROUNDS'
        and obs.phase==Phase.RECRUIT and obs.interaction is None and obs.ready
        and obs.turn is not None and not obs.options_pending and not obs.warnings)


def interrupt_phase_wait(action,before,current):
    """END is an interruptible local wait, never an unacknowledged click."""
    if action.kind!=Kind.END or action.evidence!='verified_phase_wait':return False
    if (before.game_id,before.turn,before.local_player_id,before.mode)!=(current.game_id,current.turn,current.local_player_id,current.mode):return False
    if current.phase==Phase.CHOICE and current.interaction:return True
    if current.phase!=Phase.RECRUIT or not current.ready or current.options_pending or current.warnings:return False
    from .action_context import input_context_token
    from .model import Action
    if input_context_token(before,Action(Kind.WAIT),ignore_playback=True)!=input_context_token(current,Action(Kind.WAIT),ignore_playback=True):return True
    def available(obs):
        return {(a.kind,a.source,a.target,a.sub_option_id,a.resource_cost,a.authorization_source)
                for a in obs.actions if a.kind not in (Kind.WAIT,Kind.END,Kind.REORDER)}
    return bool(available(current)-available(before))


def cancellable_target(obs,action=None):
    ctx=obs.interaction
    return bool(ctx and ctx.mode=='target' and ctx.min_select==0
        and ctx.protocol in ('verified_target','server_targets')
        and ctx.owner in (None,obs.local_player_id)
        and (action is None or action.kind==Kind.CANCEL and action.choice_id==ctx.request_id))
