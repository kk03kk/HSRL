"""One phase readiness window; calibrated actions do not restart it."""
import time

from .client_readiness import ClientReadiness,logical_reasons
from .model import Observation,Phase
from .readiness_evidence import ReadinessEvidence,layout_context,readiness_binding
from .windows import WindowSnapshot
from .action_context import input_context_token


class GeometryReadiness(ClientReadiness):
    uses_layout_evidence=True
    def __init__(self,mapper,**kw):
        super().__init__(**kw);self.mapper=mapper;self.phase_context=None
        self.phase_since=None;self.layout_previous=None;self.evidence=None;self.phase_frames=0
        self.motion_previous=None
        self.layout_frame_id=None
        self.motion_anchor=None

    def invalidate(self):
        super().invalidate()
        self.phase_context=None;self.phase_since=None;self.layout_previous=None
        self.evidence=None;self.phase_frames=0;self.motion_previous=None;self.layout_frame_id=None
        self.motion_anchor=None

    def inspect(self,row,image,mappings,action,ui_state=None,*,proposal=None):
        raw=row['observation'];obs=Observation.from_dict({**raw,'resolutions':[]})
        if not self.mapper.authorize_geometry(obs,action):
            # Revoke geometry evidence once when leaving that path. Clearing
            # the base gate on every recovery frame prevents it from ever
            # comparing two captures and permanently blocks all fallback input.
            if self.phase_context is not None:self.invalidate()
            return super().inspect(row,image,mappings,action,ui_state,proposal=proposal)
        stamp=row['capture'].get('end',0)
        completion_verified=bool(self.mapper.projection.layouts) and all(s.probes.get('log_completion_verified',False) for s in self.mapper.projection.layouts)
        phase=(row.get('session'),row['window']['token'],layout_context(obs,self.mapper.projection.layouts),
            input_context_token(obs,action),obs.client_submission_position,obs.client_playback_position)
        reasons=logical_reasons(row,action,ui_state)
        reused=False
        if row.get('readiness_reused') and self.evidence:
            window=WindowSnapshot.from_dict(row['window'])
            reused=self.evidence.accepts(obs,window,self.mapper.projection.layouts,time.time(),
                verified_log_position=row.get('readiness_verified_log_position'))
            if reused:reasons=[r for r in reasons if r!='log_capture_not_synchronized']
            else:reasons.append('readiness_evidence_expired')
        if proposal:reasons.extend(proposal['rejection_reasons'])
        relation=self.capture_history.inspect(('geometry',phase),row)
        if relation.reason:reasons.append(relation.reason)
        if self.phase_context!=phase or not relation.comparable:
            self.phase_context=phase;self.phase_since=stamp;self.phase_frames=0;self.evidence=None;self.motion_previous=None;self.motion_anchor=None
        if reasons:
            self.invalidate()
            return dict(state='BLOCKED',ready=False,reasons=sorted(set(reasons)),regions={},checked_at=stamp,
                comparable=False,consecutive_stable_frames=0,stability_basis='invalid_evidence_revoked')
        elif relation.comparable and relation.fresh:self.phase_frames+=1
        quiet=stamp-self.phase_since
        motion=getattr(self.mapper,'motion_evidence',lambda *a:None)(obs,action,image) if not reasons else None
        if motion is False:
            # A static background at an obsolete coordinate is not a safe
            # fallback. Force independently measured recovery before input.
            self.evidence=None;self.motion_previous=None
            reject=getattr(self.mapper,'reject_motion',None)
            if reject:reject(obs)
            self.invalidate()
            return dict(state='BLOCKED',ready=False,reasons=['layout_contour_mismatch'],
                regions={},checked_at=stamp,stability_basis='recovery_required')
        # Re-inspecting a proven capture is allowed; it cannot create a fresh
        # sample or extend its capture time. Controller and planner both inspect
        # the same frame before a gesture.
        fast=bool(relation.comparable and not relation.fresh and self.evidence is not None
            and self.evidence.captured_at==stamp and self.evidence.binding==readiness_binding(obs))
        if motion is not None:
            previous=self.motion_previous
            if self.motion_anchor is None:self.motion_anchor=motion
            drift=len(self.motion_anchor)!=len(motion) or max((abs(a-b) for a,b in zip(self.motion_anchor,motion)),default=0)>3
            if drift:
                self.phase_since=stamp;quiet=0;self.phase_frames=0;self.evidence=None;self.motion_anchor=motion;fast=False
            if not drift and relation.fresh and previous and .033<=stamp-previous[0]<=self.max_gap and len(previous[2])==len(motion):
                fast=max((abs(a-b) for a,b in zip(previous[2],motion)),default=999)<=3
                if not fast:
                    self.phase_since=stamp;quiet=0;self.phase_frames=0;self.evidence=None
            if previous is None or relation.fresh and stamp-previous[0]>=.033:self.motion_previous=(stamp,row.get('id'),motion)
        else:self.motion_previous=None
        # Choice entry is infrequent and may still be enabling input handlers.
        # Retain its safe quiet budget, now measured after actual contour motion.
        if obs.phase==Phase.CHOICE and quiet<self.quiet:fast=False
        if not reasons and not reused and not fast and (motion is not None or self.phase_frames<1 or quiet<self.quiet):reasons.append('target_quiet_window_pending')
        if not reasons and not reused:
            self.evidence=ReadinessEvidence(row['window']['token'],layout_context(obs,self.mapper.projection.layouts),
                stamp,str(row['id']),obs.client_submission_position,obs.client_playback_position,
                binding=readiness_binding(obs),log_completion_verified=completion_verified)
        self.layout_previous=stamp
        self.layout_frame_id=row['id']
        # Compatibility with the provider's old gate introspection. The new
        # path never manufactures previous target crops from this value.
        self.previous=((row.get('session'),obs.game_id,'geometry',action.key),stamp,{}, {},{})
        return dict(state='READY' if not reasons else 'WAITING' if reasons==['target_quiet_window_pending'] else 'BLOCKED',
            ready=not reasons,reasons=sorted(set(reasons)),regions={},quiet_for_seconds=quiet,
            comparable=self.phase_frames>0,consecutive_stable_frames=self.phase_frames,checked_at=stamp,
            stability_basis='independent_local_contours_33ms' if fast else 'calibrated_layout_and_current_scene_probe',input_interactability='unknown_until_game_ack',
            scope='geometry prerequisite; server action and result verification remain mandatory')
