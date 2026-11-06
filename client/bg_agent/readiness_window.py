"""Capture continuity shared by normal actions and deferred UI stages.

Clocks measure capture time, never time spent polling an old image. No input,
card recognition, log revision assumptions, or input retry authority lives here.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FrameRelation:
    timestamp: float
    comparable: bool=False
    fresh: bool=False
    reason: str|None=None


class CaptureHistory:
    def __init__(self,max_gap=2):
        self.max_gap=max_gap;self.context=None;self.previous=None;self.seen={}

    def invalidate(self):
        self.context=None;self.previous=None

    def inspect(self,context,row):
        capture=row.get('capture') or {};stamp=capture.get('end')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<0:
            self.invalidate();return FrameRelation(0,reason='capture_timestamp_missing')
        window=row.get('window',{}).get('token')
        if capture.get('window_token') and window and capture['window_token']!=window:
            self.invalidate();return FrameRelation(stamp,reason='capture_window_changed')
        context=(row.get('session'),window,context)
        # Broker sequence is the physical capture identity. Row IDs can change
        # when different modules inspect the same coalesced frame.
        identity=(row.get('session'),row.get('window',{}).get('token'),
            ('sequence',capture['frame_sequence']) if capture.get('frame_sequence') is not None
            else ('row',row['id']) if row.get('id') is not None else ('time',stamp))
        old_time=self.seen.get(identity)
        if old_time is not None and old_time!=stamp:
            self.invalidate();return FrameRelation(stamp,reason='capture_identity_time_mismatch')
        self.seen[identity]=stamp
        if len(self.seen)>128:self.seen.pop(next(iter(self.seen)))
        previous=self.previous if self.context==context else None
        if previous and stamp<previous[1]:
            self.invalidate();return FrameRelation(stamp,reason='capture_time_reversed')
        comparable=bool(previous and 0<=stamp-previous[1]<=self.max_gap)
        fresh=not previous or identity!=previous[0] and stamp>previous[1]
        if fresh or not comparable:self.previous=(identity,stamp)
        self.context=context
        return FrameRelation(stamp,comparable,fresh)


def capture_blockers(capture):
    capture=capture or {};reasons=[]
    if not capture.get('geometry_unchanged',True):reasons.append('capture_window_changed')
    if capture.get('stage_context_changed'):reasons.append('stage_context_changed_during_capture')
    before=capture.get('window_token');after=capture.get('window_after',{}).get('token')
    if before and after and before!=after:reasons.append('capture_window_changed')
    reasons.extend(capture.get('window_after',{}).get('blockers') or ())
    return reasons


class StageReadiness:
    """Two valid independent captures, one bound stage and a quiet hit region."""
    def __init__(self,quiet=.15,max_gap=2,tolerance=4):
        self.history=CaptureHistory(max_gap);self.quiet=quiet;self.tolerance=tolerance
        self.anchor=None;self.since=None;self.counted_at=None;self.frames=0;self.ready=False

    def invalidate(self):
        self.history.invalidate();self.anchor=None;self.since=None
        self.counted_at=None;self.frames=0;self.ready=False

    def inspect(self,context,capture,geometry,*,valid=True,window=None,session=None):
        row=dict(capture=capture or {},window={'token':window},session=session)
        relation=self.history.inspect(context,row);stamp=relation.timestamp
        reasons=capture_blockers(capture)
        if relation.reason:reasons.append(relation.reason)
        if not valid or not geometry:reasons.append('stage_not_visible_or_eligible')
        if reasons:
            self.invalidate();return dict(ready=False,reasons=reasons,checked_at=stamp)
        moved=self.anchor is not None and (len(self.anchor)!=len(geometry) or
            max((abs(a-b) for a,b in zip(self.anchor,geometry)),default=0)>self.tolerance)
        if not relation.comparable or moved or self.since is None:
            self.anchor=tuple(geometry);self.since=stamp;self.counted_at=stamp;self.frames=1;self.ready=False
        elif relation.fresh and stamp-self.counted_at>=.033:
            self.frames+=1;self.counted_at=stamp
        self.ready=self.frames>=2 and stamp-self.since>=self.quiet
        return dict(ready=self.ready,reasons=[] if self.ready else ['target_quiet_window_pending'],
            checked_at=stamp,quiet_for_seconds=stamp-self.since,consecutive_stable_frames=self.frames,
            comparable=relation.comparable,fresh_capture=relation.fresh)


def stage_context(obs,action,name):
    """Candidate identity, display phase and parent action, not card art."""
    from .action_context import input_context_token
    return (name,action.key,input_context_token(obs,action),obs.client_task_list,
            obs.client_submission_position,obs.client_playback_position)
