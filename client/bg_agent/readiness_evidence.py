"""Short-lived UI evidence shared by actions, not by unrelated requests."""
from dataclasses import dataclass

from .model import Phase


def readiness_binding(obs):
    return (obs.gold,obs.tavern_tier,obs.upgrade_cost,
        tuple(tuple((c.entity_id,c.card_id,c.position,c.card_type,c.attack,c.health,c.tags.get('LITERALLY_UNPLAYABLE')) for c in getattr(obs,z)) for z in ('shop','board','hand','choices')),
        tuple((c.entity_id,c.card_id,c.cost,c.tags.get('EXHAUSTED'),c.tags.get('INTERACTABLE_OBJECT_COST')) for c in obs.controls))


def observation_context(obs):
    return (obs.mode,layout_context(obs,()),readiness_binding(obs),obs.client_task_list,obs.options_pending,
            obs.client_submission_position,obs.client_playback_position)


def layout_context(obs,layouts):
    ctx=obs.interaction
    return (obs.game_id,obs.turn,str(obs.phase),obs.build,obs.local_player_id,
        obs.hero.entity_id if obs.hero else None,
        tuple((s.id,s.zone,s.count,s.variant) for s in layouts),
        tuple((c.entity_id,c.tags.get('GAME_MODE_BUTTON_SLOT')) for c in obs.controls),
        (ctx.mode,ctx.request_id,ctx.owner,ctx.source_entity,ctx.task_list,ctx.selected,ctx.display_state) if ctx else None)


@dataclass(frozen=True)
class ReadinessEvidence:
    window_token: str
    context: tuple
    captured_at: float
    frame_id: str
    submission: int|None
    playback_position: int|None
    ready: bool=True
    binding: tuple|None=None
    log_completion_verified: bool=False

    def accepts(self,obs,window,layouts,now,*,verified_log_position=None):
        if not self.ready or window.blockers or window.token!=self.window_token:return False
        if not 0<=now-self.captured_at<=.5 or not obs.ready or obs.warnings:return False
        if layout_context(obs,layouts)!=self.context:return False
        if obs.phase!=Phase.RECRUIT or obs.interaction is not None or obs.client_task_list is not None:return False
        changed=self.binding is not None and self.binding!=readiness_binding(obs)
        if changed and obs.client_submission_position==self.submission:return False
        if obs.client_submission_position!=self.submission:
            # A server receipt alone is insufficient: its visual playback must
            # also have ended after that root. Never infer this from revision.
            end=getattr(obs,'client_playback_position',None)
            if not self.log_completion_verified or verified_log_position is None or end is None or end<=verified_log_position:return False
        elif obs.client_playback_position!=self.playback_position:return False
        return True
