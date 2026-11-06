"""Bounded re-offering of never-sent actions after new measured UI evidence.

This module never sends input or changes delivery status. Dispatched actions,
including proven no-effect deliveries, are excluded by the caller.
"""
import hashlib
import time

from .action_context import input_context_token
from .model import Decision, Kind, Phase
from .planning import ShadowPlanner


def evidence(provider,obs,action,maps):
    row=getattr(provider,'last_row',None);image=getattr(provider,'last_image',None)
    if row is None or image is None or row.get('blockers') or not obs.ready:return None
    # The raw current observation must advertise the same action. Generated
    # insertion actions use the existing context adapter on their next plan.
    if action not in obs.actions:return None
    plan=ShadowPlanner({**row,'observation':obs.to_dict()},maps,replay=True,compact=True).build(
        obs,Decision(action,obs.revision,'read-only geometry recovery'))
    if plan['rejection_reasons']:return None
    keys=sorted({s['region'] for s in plan['steps'] if s.get('region')})
    found={m.key:m.rect for m in maps if m.status=='matched' and m.rect}
    if not keys or any(k not in found for k in keys):return None
    signature=[row['window']['token'],input_context_token(obs,action)]
    for key in keys:
        box=found[key];x,y=box.point();r=8
        if x-r<0 or y-r<0 or x+r>image.width or y+r>image.height:return None
        patch=image.crop((round(x-r),round(y-r),round(x+r),round(y+r))).convert('RGB').resize((8,8))
        signature.append((key,box.x,box.y,box.width,box.height,hashlib.sha256(patch.tobytes()).hexdigest()))
    return tuple(signature)


def remember(provider,action,reason):
    obs=provider.last_obs
    if obs.phase!=Phase.RECRUIT or obs.interaction is not None:return
    if reason.startswith('live_plan_blocked:'):
        parts=reason.removeprefix('live_plan_blocked:').split(',')
        if not all(r in ('target_pixels_not_stable','target_quiet_window_pending','target_regions_unknown')
                   or r.startswith('unknown_mapping:') for r in parts):return
    elif reason not in ('live_hand_slot_binding_unconfirmed','live_hand_source_slot_mismatch',
                        'live_source_on_tier_badge','live_source_outside_semantic_region:hand'):return
    key=(obs.game_id,obs.turn,str(action.kind),action.source)
    states=getattr(provider,'geometry_deferrals',{})
    old=states.get(key,{})
    states[key]=dict(action=action,failed=evidence(provider,obs,action,getattr(provider,'mappings',[])),
                     skipped_at=time.monotonic(),recoveries=old.get('recoveries',0),candidate=None,
                     capture=None,since=None)
    provider.geometry_deferrals=states


def recover(provider,obs,maps):
    states={k:v for k,v in getattr(provider,'geometry_deferrals',{}).items() if k[:2]==(obs.game_id,obs.turn)}
    provider.geometry_deferrals=states
    if not maps or obs.phase!=Phase.RECRUIT or obs.interaction is not None or not obs.ready:return
    row=getattr(provider,'last_row',None)
    if not row:return
    capture=row.get('capture',{}).get('end')
    if capture is None:return
    unavailable=getattr(provider,'unavailable_actions',{})
    for key,state in states.items():
        if key not in unavailable or state['recoveries']>=2:continue
        now=time.monotonic()
        if now-state['skipped_at']<2:continue
        signature=evidence(provider,obs,state['action'],maps)
        if signature is None or signature==state['failed']:
            state.update(candidate=None,capture=None,since=None);continue
        if signature!=state['candidate'] or state['capture'] is None or capture<state['capture']:
            state.update(candidate=signature,capture=capture,since=capture);continue
        # Re-reading the same screenshot cannot count as a second sample.
        if capture<=state['capture'] or capture-state['since']<.25:continue
        state['recoveries']+=1
        unavailable.pop(key)
        provider.ui_waits={k:v for k,v in getattr(provider,'ui_waits',{}).items()
                           if k[:2]!=(obs.game_id,obs.turn) or k[-1]!=state['action'].key}
        provider.audit.write('live_geometry_action_reoffered',action_key=state['action'].key,
            game_id=obs.game_id,turn=obs.turn,recovery=state['recoveries'],input_sent=False,
            evidence='changed_geometry_or_local_pixels_stable_across_distinct_captures')
