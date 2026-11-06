"""Current-frame action signals. No quiet clock or multi-frame admission."""
import math,time
import cv2
import numpy as np

from .action_geometry import route
from .client_readiness import logical_reasons
from .model import Observation,Kind,Decision
from .readiness_window import capture_blockers
from .mapping import StabilityBasis
from .windows import Box


def needs_green_signal(obs,action):
    if action.kind in (Kind.BUY,Kind.SELL,Kind.REORDER,Kind.ENTER,Kind.DISMISS,Kind.END,Kind.WAIT,Kind.CANCEL):return False
    if action.kind in (Kind.CHOOSE,Kind.SELECT):
        card=next((c for c in obs.choices if c.entity_id==action.source),None)
        if card and card.card_type=='HERO':return False
    return True


def hand_top_cap(image,hit,other_hits):
    """Visible top rim of one calibrated slot in an overlapping hand fan.

    Connected contours span several playable cards. Inspect only the narrow
    interior of this slot, bounded well before either neighbouring hit point.
    This supplies eligibility, never a source coordinate or entity identity.
    """
    width,height=image.size;x,y=hit.point()
    distances=[abs(ox-x) for ox,oy in other_hits if abs(ox-x)>2]
    if not distances:return dict(ready=False,reason='hand_cap_slot_neighbours_missing')
    half=min(width*.012,min(distances)*.23)
    left,right=max(0,round(x-half)),min(width,round(x+half))
    top,bottom=round(height*.84),min(round(height*.93),round(y-height*.02))
    if right-left<max(12,width*.006) or bottom-top<5:return dict(ready=False,reason='hand_cap_search_empty')
    hsv=cv2.cvtColor(np.asarray(image.crop((left,top,right,bottom))),cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv);mask=(h>=35)&(h<=100)&(s>95)&(v>110)
    exists=mask.any(axis=0)
    if exists.mean()<.85:return dict(ready=False,reason='hand_cap_outline_absent')
    xs=np.flatnonzero(exists);ys=mask[:,exists].argmax(axis=0)
    fit=np.polyfit(xs,ys,2);residual=float(np.percentile(np.abs(np.polyval(fit,xs)-ys),90))
    # A thin exterior strip, rather than a large green area of card artwork.
    thickness=[]
    for xx,yy in zip(xs,ys):
        stops=np.flatnonzero(~mask[yy:,xx])
        thickness.append(int(stops[0]) if len(stops) else mask.shape[0]-yy)
    ready=residual<=height*.004 and 2<=float(np.median(thickness))<=height*.025
    return dict(ready=ready,reason=None if ready else 'hand_cap_shape_unconfirmed',signal='green_hand_top_rim',
        search=[left,top,right,bottom],support=float(exists.mean()),residual=residual,
        median_thickness=float(np.median(thickness)),edge_point=[(left+right)/2,float(top+np.median(ys))])


def choice_green_sides(image,hit,bounds,other_hits):
    """Search this fixed slot's exterior sides without requiring a closed rim."""
    x,y=hit.point();w,h=image.size
    if not bounds.x<x<bounds.x+bounds.width or not bounds.y<y<bounds.y+bounds.height:return None
    neighbours=sorted(ox for ox,oy in other_hits)
    left=max(((ox+x)/2 for ox in neighbours if ox<x),default=0)
    right=min(((ox+x)/2 for ox in neighbours if ox>x),default=w)
    y0=max(0,round(bounds.y+bounds.height*.18));y1=min(h,round(bounds.y+bounds.height*.88))
    margin=max(4,round(bounds.width*.28));proof=[]
    for side,edge in (('left',bounds.x),('right',bounds.x+bounds.width)):
        x0=max(round(left)+2,edge-margin,0);x1=min(round(right)-2,edge+margin,w)
        # Keep green card artwork near the click point out of both rim probes.
        if side=='left':x1=min(x1,round(x-bounds.width*.12))
        else:x0=max(x0,round(x+bounds.width*.12))
        if x1-x0<3 or y1-y0<5:return None
        hsv=cv2.cvtColor(np.asarray(image.crop((x0,y0,x1,y1))),cv2.COLOR_RGB2HSV)
        hh,ss,vv=cv2.split(hsv);mask=((hh>=35)&(hh<=100)&(ss>95)&(vv>110)).astype('uint8')
        closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((max(3,round(len(mask)*.02)),1),np.uint8))
        columns=(closed.mean(axis=0)>=.70).astype('uint8')
        runs=np.diff(np.flatnonzero(np.diff(np.r_[0,columns,0])))
        widths=runs[::2]
        if not any(2<=n<=bounds.width*.10 for n in widths):return None
        proof.append(dict(side=side,search=[x0,y0,x1,y1],support=float(closed.mean(axis=0).max())))
    return dict(ready=True,reason=None,signal='slot_local_green_sides',hit=[x,y],sides=proof)


def green_outline(image,hit,*,bounds=None,zone=None,other_hits=()):
    """A green exterior around this hit, not green artwork elsewhere."""
    width,height=image.size;x,y=hit.point()
    if zone=='choices' and bounds is not None:
        sides=choice_green_sides(image,hit,bounds,other_hits)
        if sides:return sides
    if bounds is not None and bounds.width>width*.025 and bounds.height>height*.035:
        if zone in ('choices','hand') and bounds.x<x<bounds.x+bounds.width and bounds.y<y<bounds.y+bounds.height:
            from .choice_visibility import frame_rails
            if frame_rails(image,bounds):
                return dict(ready=True,reason=None,signal='paired_green_sides',hit=[x,y],
                    bounds=dict(x=bounds.x,y=bounds.y,width=bounds.width,height=bounds.height))
        pad=max(6,round(min(bounds.width,bounds.height)*.15))
        crop=(max(0,bounds.x-pad),max(0,bounds.y-pad),
              min(width,bounds.x+bounds.width+pad),min(height,bounds.y+bounds.height+pad))
    else:
        rx=width*(.13 if zone in ('hand','choices') else .075)
        ry=height*(.23 if zone=='choices' else .16 if zone=='hand' else .13)
        crop=(max(0,round(x-rx)),max(0,round(y-ry)),min(width,round(x+rx)),min(height,round(y+ry)))
        if zone=='hand':crop=(*crop[:3],height)
    if min(crop[2]-crop[0],crop[3]-crop[1])<3:return dict(ready=False,reason='green_search_region_empty')
    hsv=cv2.cvtColor(np.asarray(image.crop(crop)),cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv);mask=((h>=35)&(h<=100)&(s>95)&(v>110)).astype('uint8')*255
    diagnostics={}
    if zone in (None,'controls'):
        diagnostics=dict(control_ring_best_support=0.,control_ring_quadrants=[],
            green_pixels=int(np.count_nonzero(mask)))
        cx,cy=x-crop[0],y-crop[1]
        angles=np.linspace(0,2*np.pi,48,endpoint=False)
        spread=max(2,round(width*.002))
        expanded=cv2.dilate(mask,np.ones((spread*2+1,spread*2+1),np.uint8))
        for radius in np.linspace(width*.022,width*.06,24):
            xs=np.rint(cx+radius*np.cos(angles)).astype(int);ys=np.rint(cy+radius*np.sin(angles)).astype(int)
            inside=(xs>=0)&(xs<expanded.shape[1])&(ys>=0)&(ys<expanded.shape[0])
            hits=np.zeros(48,dtype=bool);hits[inside]=expanded[ys[inside],xs[inside]]>0
            support=float(hits.mean());quadrants=[float(part.mean()) for part in np.split(hits,4)]
            if support>=diagnostics['control_ring_best_support']:
                diagnostics.update(control_ring_best_support=support,control_ring_quadrants=quadrants,
                    control_ring_radius=float(radius))
            if hits.mean()>=.55 and min(part.mean() for part in np.split(hits,4))>=.25:
                return dict(ready=True,reason=None,signal='green_control_ring',hit=[x,y],radius=float(radius),
                    search=list(crop),**diagnostics)
    # Close small breaks in the same outline, never bridge separate cards.
    kernel=max(3,round(width/640))
    joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
    contours,_=cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    matches=[]
    for contour in contours:
        bx,by,bw,bh=cv2.boundingRect(contour)
        # An outline cut by this search crop can belong to a larger object
        # spanning several slots. Only the natural client edge may clip it.
        if (bx==0 and crop[0]>0 or by==0 and crop[1]>0 or
            bx+bw>=mask.shape[1] and crop[2]<width or
            by+bh>=mask.shape[0] and crop[3]<height):continue
        bx+=crop[0];by+=crop[1]
        if bw<width*.018 or bh<height*.025:continue
        if not bx+2<=x<=bx+bw-2 or not by+2<=y<=by+bh-2:continue
        if any(bx<=ox<=bx+bw and by<=oy<=by+bh for ox,oy in other_hits):continue
        patch=mask[by-crop[1]:by-crop[1]+bh,bx-crop[0]:bx-crop[0]+bw]>0
        fill=float(patch.mean())
        if not .008<=fill<=.65:continue
        # A small interior green picture cannot supply a surrounding frame.
        cx,cy=round(x-crop[0]),round(y-crop[1]);radius=max(2,round(width/640))
        center=mask[max(0,cy-radius):cy+radius+1,max(0,cx-radius):cx+radius+1]
        if center.size and float((center>0).mean())>.7:continue
        matches.append(dict(bounds=dict(x=bx,y=by,width=bw,height=bh),green_fraction=fill))
    return dict(ready=bool(matches),reason=None if matches else 'action_green_outline_absent',
                matches=matches,hit=[x,y],search=list(crop),**diagnostics)


def active_minion_rim(image,bounds):
    """Ability glow follows the upper portrait arc, not a rectangular card."""
    x0=max(0,bounds.x-round(bounds.width*.18));y0=max(0,bounds.y-round(bounds.height*.15))
    x1=min(image.width,bounds.x+round(bounds.width*1.18));y1=min(image.height,bounds.y+bounds.height)
    hsv=cv2.cvtColor(np.asarray(image.crop((x0,y0,x1,y1))),cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv);mask=(h>=35)&(h<=100)&(s>95)&(v>110)
    cx,cy=bounds.point();pad=max(2,round(image.width*.002))
    for rx in (.47,.52,.57,.62):
        hits=[]
        for angle in np.linspace(np.pi,2*np.pi,32):
            x=round(cx+rx*bounds.width*np.cos(angle))-x0;y=round(cy+.45*bounds.height*np.sin(angle))-y0
            patch=mask[max(0,y-pad):y+pad+1,max(0,x-pad):x+pad+1]
            hits.append(bool(patch.size and patch.any()))
        if np.mean(hits)>=.65 and min(np.mean(hits[:10]),np.mean(hits[-10:]))>=.6:
            return dict(ready=True,reason=None,signal='green_active_minion_arc',support=float(np.mean(hits)))
    return dict(ready=False,reason='action_green_outline_absent')


def offered_trinket_signal(mapper,obs,action,image,visible):
    """未选饰品没有绿框：用当前请求、报价、整组紫框与灰确认按钮证明初次可选。"""
    ctx=obs.interaction
    if (str(obs.phase)!='choice' or not obs.ready or obs.warnings or ctx is None
        or ctx.mode!='choice' or ctx.protocol!='verified_trinket_select'
        or ctx.owner!=obs.local_player_id or ctx.request_id is None
        or ctx.display_state not in ('shown','unknown') or ctx.selected
        or (ctx.min_select,ctx.max_select)!=(1,1)
        or action.kind!=Kind.CHOOSE or action not in obs.actions
        or action.evidence!='server_choices' or action.choice_id!=ctx.request_id):return None
    cards=sorted(obs.choices,key=lambda c:c.position)
    if (len(cards)!=4 or [c.position for c in cards]!=[1,2,3,4]
        or any(c.card_type!='BATTLEGROUND_TRINKET' for c in cards)):return None
    selected=next((c for c in cards if c.entity_id==action.source),None)
    if selected is None or selected.cost is None or obs.gold is None or not 0<=selected.cost<=obs.gold:return None
    detector=getattr(mapper,'trinkets',None)
    if detector is None:return None
    boxes,title,evidence=detector.detect(image,4)
    if (title is None or len(boxes)!=4 or evidence.get('confirm_appearance')!='gray_unselected_like'
        or not evidence.get('confirm_region')):return None
    # 所有实体点击点都必须落在当前帧重新测得的对应紫框内，不能只靠旧坐标放行。
    for card,box in zip(cards,boxes):
        region=visible.get(f'entity:{card.entity_id}')
        if region is None:return None
        x,y=region.rect.point()
        if not box.x<=x<=box.x+box.width or not box.y<=y<=box.y+box.height:return None
    return dict(ready=True,reason=None,signal='offered_trinket_panel',
        request_id=ctx.request_id,price=selected.cost,gold=obs.gold,
        hit=list(visible[f'entity:{action.source}'].rect.point()))


def action_signal(mapper,obs,action,image,mappings):
    visible={m.key:m for m in mappings if m.status=='matched' and m.rect}
    geometry=route(action,obs,visible);source=visible.get(geometry.source)
    if source is None:return dict(ready=False,reason='unknown_mapping:'+str(geometry.source))
    entity=action.target if action.kind==Kind.TARGET else action.source
    zone=next((z for z in ('hand','shop','board','choices','controls') if any(c.entity_id==entity for c in getattr(obs,z))),None)
    bounds=None
    projection=getattr(mapper,'projection',None);viewport=getattr(mapper,'viewport',None)
    if projection is not None and viewport is not None:
        spec=next((s for s in projection.layouts if s.zone==zone),None)
        # 已映射的合法HUD按钮未必出现在实体行；不能把None当成属性名。
        cards=tuple(sorted(getattr(obs,zone,()),key=lambda c:c.position)) if zone and zone!='choices' else tuple(obs.choices) if zone=='choices' else ()
        index=next((i for i,c in enumerate(cards) if c.entity_id==entity),None)
        if spec is not None and index is not None:
            bounds=mapper._choice_bounds(spec,viewport,index) if zone=='choices' else mapper._bounds(spec,viewport,index)
    other_hits=[m.rect.point() for m in visible.values() if m.entity_id is not None and m.entity_id!=entity
                and any(c.entity_id==m.entity_id for c in getattr(obs,zone,()))] if zone else []
    evidence=green_outline(image,source.rect,bounds=bounds,zone=zone,other_hits=other_hits)
    if not evidence['ready'] and zone=='choices':
        offered=offered_trinket_signal(mapper,obs,action,image,visible)
        if offered:evidence=offered
    if not evidence['ready'] and action.kind==Kind.ACTIVATE and zone=='board':
        card=next(c for c in obs.board if c.entity_id==entity)
        # Only a server-advertised active ability may use a minion's green arc.
        if card.tags.get('INTERACTABLE_OBJECT') in ('1',1) and action in obs.actions:
            outer=bounds or source.rect
            if outer.width>image.width*.035 and outer.height>image.height*.08:
                evidence=active_minion_rim(image,outer)
    if not evidence['ready'] and zone=='hand' and getattr(mapper,'authorize_geometry',lambda *a:False)(obs,action,source.rect.point()):
        cap=hand_top_cap(image,source.rect,other_hits)
        if cap['ready']:evidence=cap
    if not evidence['ready'] and action.kind==Kind.CONFIRM:
        patch=np.asarray(image.crop((source.rect.x,source.rect.y,source.rect.x+source.rect.width,source.rect.y+source.rect.height)))
        hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
        enabled=float(((hsv[:,:,0]>=80)&(hsv[:,:,0]<=125)&(hsv[:,:,1]>65)&(hsv[:,:,2]>100)).mean())>.15
        if enabled:evidence=dict(ready=True,reason=None,signal='enabled_confirm_control')
    # Initial hero offers use purple portrait frames. Selection is later
    # independently proved by the green halo before the enabled Confirm click.
    card=next((c for c in obs.choices if c.entity_id==entity),None)
    if not evidence['ready'] and zone=='choices' and card and card.card_type=='HERO':
        from .choice_visibility import frame_rails
        if bounds and frame_rails(image,bounds,family='hero'):
            evidence=dict(ready=True,reason=None,signal='offered_hero_portrait',hit=list(source.rect.point()))
    return {**evidence,'source_region':geometry.source}


def recruit_signal(mapper,obs,image,mappings):
    """Current enabled UI elsewhere can gate unlit BUY/SELL/MOVE sources.

    A glow is client input readiness, not a receipt for the preceding action.
    The controller still verifies each action before planning the next one.
    """
    if str(obs.phase)!='recruit' or obs.interaction or obs.options_pending or not obs.ready:
        return dict(ready=False,reason='recruit_signal_context_unavailable')
    if not any(m.key=='scene:recruit' and m.status=='matched' for m in mappings):
        return dict(ready=False,reason='visual_recruit_scene_unconfirmed')
    priorities={Kind.FREEZE:0,Kind.REFRESH:1,Kind.UPGRADE:2,Kind.HERO_POWER:3,Kind.ACTIVATE:4,Kind.PLAY:5}
    controls={c.entity_id for c in obs.controls}
    candidates=sorted((a for a in obs.actions if a.kind in priorities),key=lambda a:priorities[a.kind])
    seen=set()
    for action in candidates:
        key=(action.kind,action.source)
        if key in seen:continue
        seen.add(key)
        signal=action_signal(mapper,obs,action,image,mappings)
        if signal['ready']:return {**signal,'witness_kind':str(action.kind),'witness_entity':action.source}
    return dict(ready=False,reason='recruit_green_signal_absent' if candidates else 'recruit_green_source_unavailable')


def source_geometry_ready(mapper,obs,action,mappings):
    visible={m.key:m for m in mappings if m.status=='matched' and m.rect}
    geometry=route(action,obs,visible)
    if geometry.error or any(k not in visible for k in geometry.required):return False
    if getattr(mapper,'authorize_geometry',lambda *a:False)(obs,action):return True
    return visible[geometry.source].stability_basis==StabilityBasis.GEOMETRY and all(
        visible[k].stability_basis in (StabilityBasis.GEOMETRY,StabilityBasis.RECEIVING_REGION)
        for k in geometry.required)


def ordinary_choice_panel(mapper,obs,action,image,mappings):
    """无绿框普通发现的专用路径，绝不替代英雄/饰品/任务/多选协议。"""
    ctx=obs.interaction
    if (str(obs.phase)!='choice' or action.kind!=Kind.CHOOSE or ctx is None
            or ctx.protocol!='server_single' or ctx.mode!='choice'
            or ctx.owner!=obs.local_player_id or ctx.request_id!=action.choice_id
            or (ctx.min_select,ctx.max_select)!=(1,1) or ctx.selected
            or ctx.display_state not in ('shown','unknown') or action not in obs.actions
            or action.evidence!='server_choices' or not 2<=len(obs.choices)<=4
            or any(c.card_type in ('HERO','BATTLEGROUND_TRINKET') or c.tags.get('QUEST')=='1' for c in obs.choices)):
        return False
    projection=getattr(mapper,'projection',None)
    spec=next((s for s in getattr(projection,'layouts',()) if s.zone=='choices'),None)
    if spec is None or spec.status=='candidate':return False
    visible={m.key:m for m in mappings if m.status=='matched' and m.rect}
    if 'scene:choice' not in visible:return False
    # 某槽暗、邻槽亮时是逐选项资格差异，不能用无绿框面板协议越权。
    if any(action_signal(mapper,obs,a,image,mappings)['ready'] for a in obs.actions
            if a.kind==Kind.CHOOSE and a.choice_id==ctx.request_id):return False
    boxes=[mapper._choice_bounds(spec,mapper.viewport,i) for i in range(len(obs.choices))]
    for card,box in zip(obs.choices,boxes):
        hit=visible.get(f'entity:{card.entity_id}')
        if box is None or hit is None:return False
        x,y=hit.rect.point()
        if not box.x<x<box.x+box.width or not box.y<y<box.y+box.height:return False
    from .choice_visibility import neutral_discover_frames
    return neutral_discover_frames(image,boxes)


class GreenReadiness:
    uses_green_signal=True
    uses_layout_evidence=False
    def __init__(self,mapper):self.mapper=mapper;self.evidence=None;self.previous=None
    def invalidate(self):self.evidence=None;self.previous=None
    def signal(self,obs,action,image,mappings,ui_state=None):
        from .stage_admission import panel_visible,target_visible
        if action.kind==Kind.TARGET:
            valid=target_visible(obs,ui_state if ui_state is not None else getattr(self.mapper,'last_ui_state',{}),action.authorization_source)
            return dict(ready=valid,reason=None if valid else 'target_stage_pending',signal='current_target_stage')
        if action.kind in (Kind.CHOOSE,Kind.SELECT):
            valid=panel_visible(obs,image)
            return dict(ready=valid,reason=None if valid else 'choice_panel_pending',signal='current_choice_panel')
        return action_signal(self.mapper,obs,action,image,mappings)
    def inspect(self,row,image,mappings,action,ui_state=None,*,proposal=None):
        obs=Observation.from_dict({**row['observation'],'resolutions':[]})
        reasons=logical_reasons(row,action,ui_state)
        if proposal is None:
            from .planning import ShadowPlanner
            proposal=ShadowPlanner(row,mappings,compact=True).build(obs,Decision(action,obs.revision,'current action signal'))
        reasons.extend(proposal['rejection_reasons'])
        reasons=[r for r in reasons if r not in ('client_task_list_processing','recruit_action_processing','log_visual_settle_guard',
            'ui_animation_or_stability_unknown','target_quiet_window_pending','target_pixels_not_stable')]
        capture=row.get('capture') or {};stamp=capture.get('end')
        if isinstance(stamp,(int,float)) and stamp<=row.get('input_released_at',float('-inf')):
            reasons.append('capture_precedes_input_release')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not 0<=time.time()-stamp<=.75:
            reasons.append('capture_expired')
        if capture.get('window_token') and capture['window_token']!=row['window']['token']:
            reasons.append('capture_window_changed')
        signal=self.signal(obs,action,image,mappings,ui_state)
        if not signal['ready']:reasons.append(signal['reason'])
        return dict(state='READY' if not reasons else 'WAITING',ready=not reasons,
            reasons=sorted(set(reasons)),signal=signal,checked_at=row['capture'].get('end'),
            protocol='current_frame_action_signal',requires_another_frame=False)


class SharedRecruitReadiness(GreenReadiness):
    def signal(self,obs,action,image,mappings,ui_state=None):
        signal=recruit_signal(self.mapper,obs,image,mappings)
        if not source_geometry_ready(self.mapper,obs,action,mappings):
            return dict(ready=False,reason='action_source_geometry_unconfirmed')
        return signal


class ActionReadiness:
    """Green-capable objects and unlit physical rows have separate protocols."""
    uses_green_signal=True
    uses_layout_evidence=True
    def __init__(self,mapper):
        self.green=GreenReadiness(mapper);self.shared=SharedRecruitReadiness(mapper)
        self.evidence=None;self.previous=None
    def invalidate(self):
        self.green.invalidate();self.shared.invalidate();self.evidence=None;self.previous=None
    def inspect(self,row,image,mappings,action,ui_state=None,*,proposal=None):
        from .stage_admission import ORDINARY
        obs=Observation.from_dict({**row['observation'],'resolutions':[]})
        if action.kind in ORDINARY and obs.interaction is None:
            result=self.shared.inspect(row,image,mappings,action,ui_state,proposal=proposal)
            signal=result.get('signal',{})
            if (signal.get('reason')=='recruit_green_source_unavailable'
                    and action.kind in (Kind.SELL,Kind.REORDER)
                    and not set(result['reasons'])-{'recruit_green_source_unavailable'}):
                return {**result,'ready':True,'state':'READY','reasons':[],
                    'protocol':'no_green_source_logged_state',
                    'signal':dict(ready=True,signal='logged_state_no_green_source')}
            return {**result,'protocol':'shared_recruit_green_signal','source_green_required':False}
        result=self.green.inspect(row,image,mappings,action,ui_state,proposal=proposal)
        return {**result,'protocol':'current_interaction_stage'}


class SignalReadiness:
    """Follow-up stages already provide a measured enabled/eligible signal."""
    def __init__(self,**_):pass
    def invalidate(self):pass
    def inspect(self,context,capture,geometry,*,valid=True,window=None,session=None):
        reasons=capture_blockers(capture)
        stamp=(capture or {}).get('end')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<0:reasons.append('capture_timestamp_missing')
        elif not 0<=time.time()-stamp<=.75:reasons.append('capture_expired')
        if capture and capture.get('reused_capture'):reasons.append('capture_reused_for_new_input')
        if capture and capture.get('window_token') and window and capture['window_token']!=window:
            reasons.append('capture_window_changed')
        if not valid or not geometry:reasons.append('action_signal_absent')
        return dict(ready=not reasons,reasons=reasons,checked_at=(capture or {}).get('end'),
                    protocol='current_frame_action_signal',requires_another_frame=False)
