"""Conservative real-client readiness: semantic AND window AND target pixels.

READY is a shadow-plan prerequisite, never an input hit-test or authorization.
Unknown modals and missing evidence fail closed. No card identity recognition.
"""
import time
import numpy as np
from .model import Kind,Phase,Observation,Decision
from .planning import ShadowPlanner
from .action_context import input_context_token
from .mapping import StabilityBasis
from .readiness_window import CaptureHistory,capture_blockers


def logical_reasons(row,action,ui_state=None):
    obs=row['observation'];reasons=list(row['blockers'])+capture_blockers(row.get('capture'))
    if not row['log_snapshot_unchanged']:reasons.append('log_capture_not_synchronized')
    if not obs.get('ready',True):reasons.append('logical_state_not_ready')
    if obs['phase']==Phase.RECRUIT and not obs.get('interaction') and obs.get('options_pending'):reasons.append('server_options_pending')
    if obs['phase']==Phase.RECRUIT and not obs.get('interaction') and obs.get('client_task_list') is not None:
        reasons.append('client_task_list_processing')
    if obs['mode']!='GT_BATTLEGROUNDS':reasons.append('solo_mode_unconfirmed')
    if obs['phase'] not in (Phase.RECRUIT,Phase.CHOICE):reasons.append('non_action_phase')
    if action.kind==Kind.WAIT:reasons.append('no_confirmed_action')
    if row.get('lifecycle',{}).get('transition_pending'):reasons.append('scene_transition_pending')
    if (ui_state or {}).get('mode')=='pause_menu':reasons.append('pause_menu_visible')
    if obs['phase']==Phase.RECRUIT and (ui_state or {}).get('mode')=='targeting_suspected' and action.kind!=Kind.TARGET:reasons.append('unresolved_visual_target_mode')
    if obs['phase']==Phase.RECRUIT and (ui_state or {}).get('mode')=='recruit_transition':reasons.append('recruit_transition_overlay')
    ctx=obs.get('interaction')
    if ctx and ctx.get('mode')=='choice':
        if ctx.get('display_state') in ('waiting','hidden'):reasons.append('choice_client_not_displayed')
        if action.kind in (Kind.CHOOSE,Kind.SELECT,Kind.CONFIRM) and action.choice_id!=ctx.get('request_id'):reasons.append('choice_request_changed')
        if ctx.get('owner') not in (None,obs.get('local_player_id')):reasons.append('choice_owner_changed')
    known_target=bool(action.kind==Kind.TARGET and ctx and ctx.get('mode')=='target' and ctx.get('protocol') in ('verified_target','server_targets','fixture'))
    if ctx and ctx.get('mode')!='choice' and not known_target:reasons.append('modal_protocol_not_verified')
    from .stage_admission import target_visible
    if action.kind==Kind.TARGET and (not known_target or not target_visible(Observation.from_dict(obs),ui_state,action.authorization_source)):
        reasons.append('target_request_not_verified')
    if obs.get('warnings'):reasons.extend('observation_warning:'+w for w in obs['warnings'])
    return reasons


class ClientReadiness:
    def __init__(self,quiet=.25,max_gap=2,log_settle=.15):
        self.quiet,self.max_gap,self.log_settle=quiet,max_gap,log_settle
        self.previous=None;self.since=None;self.frames=0
        self.capture_history=CaptureHistory(max_gap);self.context_event_signature=None

    def invalidate(self):
        self.previous=None;self.since=None;self.frames=0
        self.capture_history.invalidate();self.context_event_signature=None

    def inspect(self,row,image,mappings,action,ui_state=None,*,proposal=None):
        timestamp=row['capture'].get('end',0);obs=row['observation']
        reasons=logical_reasons(row,action,ui_state)
        mapped={m.key:m for m in mappings if m.status=='matched' and m.rect}
        scene='scene:choice' if obs['phase']==Phase.CHOICE else 'scene:recruit'
        if scene not in mapped:reasons.append('visual_scene_unconfirmed')
        # Use the actual planned regions; includes drag destination and targets.
        # Geometry/context checks never consume causal receipt snapshots.
        observation=Observation.from_dict({**obs,'resolutions':[]})
        if proposal is None:
            proposal=ShadowPlanner(row,mappings,replay=True,compact=True).build(observation,
                Decision(action,obs['revision'],'readiness geometry probe'))
        keys=sorted(set(s['region'] for s in proposal['steps'] if s.get('region')))
        unknown=[r for r in proposal['rejection_reasons'] if r.startswith(('unknown_mapping:','target_selection_','special_placement_','activate_control_','branch_selection_','drop_region_'))]
        reasons.extend(unknown)
        if not keys:reasons.append('target_regions_unknown')
        # These bases are assigned only after measuring the CURRENT frame.
        # Two fresh local geometry checks can replace a blanket art-quiet
        # budget, just as on the calibrated geometry path. Modals/targets and
        # any pixel-only region keep the conservative legacy gate.
        def local_basis(key):
            m=mapped.get(key)
            return bool(m and (m.stability_basis in (StabilityBasis.GEOMETRY,StabilityBasis.CONTROL_FRAME) or
                m.stability_basis==StabilityBasis.RECEIVING_REGION and (
                    action.kind==Kind.BUY and key=='region:hand' or action.kind==Kind.SELL and key=='region:bob')))
        local_geometry=bool(keys and obs['phase']==Phase.RECRUIT and not obs.get('interaction') and
            action.kind in (Kind.BUY,Kind.SELL,Kind.PLAY,Kind.REFRESH,Kind.FREEZE,Kind.UPGRADE) and
            action.target is None and action.sub_option_id is None and not action.interaction and all(local_basis(k) for k in keys))
        crops={};rects={};crop_boxes={}
        width,height=image.size
        signature=(row.get('session'),obs['game_id'],input_context_token(observation,action),action.key,row['window']['token'],tuple(keys))
        previous=self.previous
        if self.context_event_signature!=signature:
            self.context_event_at=obs.get('log_event_at')
            self.context_event_signature=signature
        # Opponents' unrelated log traffic must not restart this action's
        # settle guard forever. Changes of the actual action context do reset
        # it, as do window/game/request/geometry stability checks below.
        event_at=getattr(self,'context_event_at',None)
        if not local_geometry and event_at is not None and timestamp-event_at<self.log_settle:reasons.append('log_visual_settle_guard')
        # The controller/planner may inspect the SAME capture twice. Reusing
        # that evidence must neither reset the quiet window nor manufacture
        # another distinct stable frame. Backward timestamps still reset it.
        relation=self.capture_history.inspect(('visual',signature),row)
        if relation.reason:reasons.append(relation.reason)
        comparable=bool(previous and previous[0]==signature and relation.comparable and not relation.reason)
        for key in keys:
            if key not in mapped:
                reasons.append('unknown_mapping:'+key);continue
            box=mapped[key].rect
            if box.x<0 or box.y<0 or box.x+box.width>width or box.y+box.height>height:
                reasons.append('target_outside_client:'+key);continue
            rects[key]=(box.x,box.y,box.width,box.height)
            # A fresh circular HUD rim can breathe by a few physical pixels.
            # Its safe inner click box permits bounded relative jitter; other
            # region types retain the original strict four-pixel bound.
            tolerance=max(4,min(box.width,box.height)*.25) if mapped[key].stability_basis==StabilityBasis.CONTROL_FRAME else 4
            close=bool(comparable and key in previous[3] and max(abs(a-b) for a,b in zip(rects[key],previous[3][key]))<=tolerance)
            comparable &= close
            if close:rects[key]=previous[3][key]
            # Inspect the hit-point neighborhood, excluding animated card rims.
            # Preserve the exact crop across small measured rim jitter.
            cx,cy=box.point();rx=min(40,max(12,box.width*.2));ry=min(40,max(12,box.height*.2))
            crop=previous[4][key] if close else (max(0,round(cx-rx)),max(0,round(cy-ry)),min(width,round(cx+rx)),min(height,round(cy+ry)))
            crop_boxes[key]=crop
            crops[key]=np.asarray(image.crop(crop).convert('RGB').resize((80,80)),dtype=np.float32)
        blocked=any(r!='log_visual_settle_guard' for r in reasons)
        regions={};stable=bool(crops) and comparable
        for key,pixels in crops.items():
            delta=np.abs(pixels-previous[2][key]) if comparable and key in previous[2] else None
            mean=float(delta.mean()) if delta is not None else None
            fraction=float(np.mean(np.max(delta,axis=2)>18)) if delta is not None else None
            pixel_stable=mean is not None and mean<=3 and fraction<=.04
            # Fresh complete hand-marker rows and independently checked empty
            # wood patches already establish the hit geometry. Buff particles
            # may change their colors without moving the card or drop area.
            measured_geometry=comparable and mapped[key].stability_basis==StabilityBasis.GEOMETRY
            measured_geometry |= comparable and mapped[key].stability_basis==StabilityBasis.CONTROL_FRAME
            # BUY/SELL release into an anchor-bound receiving region, rather
            # than clicking artwork there. Animated hand buffs/Bob portraits do
            # not change its geometry or the server's legal source entity.
            measured_geometry |= comparable and mapped[key].stability_basis==StabilityBasis.RECEIVING_REGION and (
                action.kind==Kind.BUY and key=='region:hand' or action.kind==Kind.SELL and key=='region:bob')
            ok=pixel_stable or measured_geometry
            regions[key]=dict(stable=ok,pixel_stable=pixel_stable,mean_rgb_delta=mean,changed_fraction=fraction,
                measured_rect=(mapped[key].rect.x,mapped[key].rect.y,mapped[key].rect.width,mapped[key].rect.height),
                reference_rect=rects[key],
                stability_basis='fresh_measured_geometry' if measured_geometry and not pixel_stable else 'pixels')
            stable &= ok
        if not stable or reasons:self.since=timestamp;self.frames=0
        elif previous and relation.fresh:self.frames+=1
        if self.since is None:self.since=timestamp
        quiet_for=timestamp-self.since
        if not stable:reasons.append('target_pixels_not_stable')
        elif quiet_for<(min(self.quiet,.033) if local_geometry else self.quiet) or self.frames<1:reasons.append('target_quiet_window_pending')
        # A blocked frame cannot seed the next comparison. Keep only the
        # bound settle event, so unrelated log traffic cannot reset it forever.
        if blocked:
            self.previous=None;self.capture_history.invalidate()
        else:self.previous=(signature,timestamp,crops,rects,crop_boxes)
        reasons=sorted(set(reasons))
        return dict(state='READY' if not reasons else 'BLOCKED' if any(r not in ('target_pixels_not_stable','target_quiet_window_pending','ui_animation_or_stability_unknown','log_visual_settle_guard') for r in reasons) else 'WAITING',
            ready=not reasons,reasons=reasons,regions=regions,quiet_for_seconds=quiet_for,
            comparable=comparable,consecutive_stable_frames=self.frames,checked_at=timestamp,
            readiness_protocol='fresh_local_geometry_33ms' if local_geometry else 'local_pixel_quiet',
            input_interactability='unknown',scope='shadow prerequisites only; not a hit test')


class ReadyShadowPlanner(ShadowPlanner):
    def __init__(self,row,mappings,image,gate,ui_state=None,**kwargs):
        super().__init__(row,mappings,**kwargs);self.image,self.gate,self.ui_state=image,gate,ui_state
    def build(self,obs,decision):
        started=time.perf_counter();plan=super().build(obs,decision)
        readiness=self.gate.inspect(self.row,self.image,self.mappings,decision.action,self.ui_state,proposal=plan)
        plan['ui_readiness']=readiness;plan['lifecycle']=self.row.get('lifecycle',{})
        if readiness['reasons']:
            plan['status']='refused';plan['rejection_reasons']=sorted(set(plan['rejection_reasons']+readiness['reasons']))
        plan['readiness_ms']=(time.perf_counter()-started)*1000
        return plan
