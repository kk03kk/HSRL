"""Log-bound generic discover, HUD slots and deferred target submission.

No card identities or hero-specific gestures. Unknown geometry fails closed.
"""
from dataclasses import replace,asdict
import time,queue
import cv2
import numpy as np
from .mapping import Mapping,StabilityBasis
from .chain_metrics import measured_stage
from .model import Kind,Phase
from .windows import Box
from .geometry_cache import pixel_detector
from .geometry import client_to_screen
from .readiness_window import stage_context
from .action_budgets import CHOICE_SECONDS,TARGET_SECONDS
from .green_readiness import SignalReadiness


@pixel_detector
def discover_regions(image,count,*,selected_frames=False,full=False):
    if not 1<=count<=10:return []
    width,height=image.size;scale=min(1,1000/width)
    rgb=cv2.resize(np.asarray(image),None,fx=scale,fy=scale)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([12 if selected_frames else 35,95,100]),np.array([112,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c);x,y,w,h=[v/scale for v in (x,y,w,h)]
        min_width=.12 if count<=4 else .045
        min_height=.29 if count<=4 else .18
        if min_width*width<w<.30*width and min_height*height<h<.63*height and .18*height<y<.48*height and .3<w/h<1:
            boxes.append(Box(round(x),round(y),round(w),round(h)))
    boxes.sort(key=lambda b:b.x)
    if len(boxes)!=count:return []
    if max(b.y for b in boxes)-min(b.y for b in boxes)>height*.045:return []
    if count==1:
        if abs(boxes[0].point()[0]-width*.5)>=width*.1:return []
    else:
        gaps=np.diff([b.point()[0] for b in boxes])
        if min(gaps)<width*(.15 if count<=4 else .055) or max(gaps)/min(gaps)>1.2:return []
        if any(a.x+a.width>b.x for a,b in zip(boxes,boxes[1:])):return []
    return boxes if full else [Box(round(b.x+b.width*.3),round(b.y+b.height*.25),round(b.width*.4),round(b.height*.3)) for b in boxes]


@pixel_detector
def multi_choice_state(image,count):
    """Require unambiguous green/gold vertical rails on every offered card."""
    frames=discover_regions(image,count,selected_frames=True,full=True)
    if not frames:return None
    selected=[]
    for index,b in enumerate(frames):
        rails=[]
        for left in (True,False):
            x=b.x if left else b.x+round(b.width*.92)
            patch=image.crop((x,b.y+round(b.height*.28),x+max(4,round(b.width*.08)),b.y+round(b.height*.7)))
            hsv=cv2.cvtColor(np.asarray(patch),cv2.COLOR_RGB2HSV)
            valid=(hsv[:,:,1]>95)&(hsv[:,:,2]>100)
            green=np.mean(np.any(valid&(hsv[:,:,0]>=35)&(hsv[:,:,0]<=112),axis=1))
            gold=np.mean(np.any(valid&(hsv[:,:,0]>=12)&(hsv[:,:,0]<35),axis=1))
            rails.append('selected' if gold>.7 and green<.15 else 'clear' if green>.7 and gold<.15 else 'unknown')
        if len(set(rails))!=1 or rails[0]=='unknown':return None
        if rails[0]=='selected':selected.append(index)
    return frames,tuple(selected)


@pixel_detector
def branch_regions(image,count,*,include_disabled=False):
    """Choose One has compact spell frames as well as full Discover cards.

    The compact path requires both vertical rails on every card, a complete
    unique row, equal sizes and spacing. Coordinates are measured, not guessed.
    It is deliberately separate from ordinary Discover/multi-select detection.
    """
    standard=discover_regions(image,count)
    if standard:return standard
    if count not in (2,3,4):return []
    width,height=image.size;scale=min(1,1000/width)
    hsv=cv2.cvtColor(cv2.resize(np.asarray(image),None,fx=scale,fy=scale),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([12,95,100]),np.array([112,255,255]))
    if include_disabled:
        # Geometry may include grey disabled siblings. The chosen slot still
        # needs independently observed colored rails before any input.
        neutral=cv2.inRange(hsv,np.array([0,0,60]),np.array([179,45,210]))
        mask=cv2.bitwise_or(mask,neutral)
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frames=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (.075*width*scale<w<.28*width*scale and .18*height*scale<h<.63*height*scale
                and .15*height*scale<y<.52*height*scale and .3<w/h<1):continue
        if x<.08*width*scale or x+w>.92*width*scale:continue
        rails=[]
        for rx in (x,x+round(w*.9)):
            rail=mask[y+round(h*.18):y+round(h*.82),rx:rx+max(3,round(w*.1))]
            rails.append(float(np.mean(np.any(rail>0,axis=1))) if rail.size else 0)
        if min(rails)<.75:continue
        frames.append(Box(round(x/scale),round(y/scale),round(w/scale),round(h/scale)))
    frames.sort(key=lambda b:b.x)
    if len(frames)!=count:return []
    if max(b.y for b in frames)-min(b.y for b in frames)>.035*height:return []
    if max(b.width for b in frames)/min(b.width for b in frames)>1.15:return []
    if max(b.height for b in frames)/min(b.height for b in frames)>1.15:return []
    gaps=np.diff([b.point()[0] for b in frames])
    if min(gaps)<width*.12 or max(gaps)/min(gaps)>1.2:return []
    if any(a.x+a.width>=b.x for a,b in zip(frames,frames[1:])):return []
    return [Box(round(b.x+b.width*.3),round(b.y+b.height*.25),round(b.width*.4),round(b.height*.3)) for b in frames]


def branch_selection_regions(image,count,selected_index,diagnostics=None):
    """Complete row geometry AND colored rails of the legal chosen slot.

    Disabled siblings can supply geometry, never permission to select them.
    Unknown neutral art or extra frames still fail the complete-row checks.
    """
    def fail(reason):
        if diagnostics is not None:diagnostics['reason']=reason
        return []
    boxes=branch_regions(image,count,include_disabled=True)
    if len(boxes)!=count:return fail(f'row_count_{len(boxes)}_of_{count}')
    if not 0<=selected_index<count:return fail('selected_index_out_of_range')
    hit=boxes[selected_index]
    # branch_regions returns an inset hit box; reconstruct its measured frame.
    x=hit.x-hit.width*.75;y=hit.y-hit.height*(.25/.3)
    width=hit.width/.4;height=hit.height/.3
    hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([12,95,100]),np.array([112,255,255]))
    for left in (x,x+width*.9):
        rail=mask[max(0,round(y+height*.18)):round(y+height*.82),max(0,round(left)):round(left+width*.1)]
        if not rail.size or float(np.mean(np.any(rail>0,axis=1)))<.65:return fail('chosen_slot_rail_support')
    return boxes


def archive_interaction(provider,stage,action,obs,window,image,capture,**details):
    """Save first/ready/failed stage evidence, including refusal frames."""
    if not hasattr(provider,'directory'):return  # In-memory offline providers.
    from .recording import compact_observation
    stem=f'{stage}-{action.source}-{action.sub_option_id}-{time.time_ns()}'
    metadata=dict(stage=stage,action=asdict(action),details=details,
        sample=dict(observation=compact_observation(obs).to_dict(),window=window.to_dict(),capture=capture))
    archive_queue=getattr(provider,'archive_queue',None)
    if archive_queue is not None:
        try:archive_queue.put_nowait((stem,image,metadata))
        except queue.Full:
            # 证据压缩/写盘不能消耗点击证据的有效期。队列满保留元数据，绝不回退同步PNG。
            provider.audit.write('interaction_archive_backpressure',stem=stem,metadata=metadata,image_saved=False)
        return
    if hasattr(provider,'persist_evidence'):
        provider.persist_evidence(stem,image,metadata);return
    from .recording import write_json
    folder=provider.directory/'evidence';folder.mkdir(exist_ok=True)
    image.save(folder/(stem+'.png'),compress_level=1)
    write_json(folder/(stem+'.json'),metadata)


def ensure_stage_unsent(provider,name):
    ledger=getattr(provider,'checkpoint',None)
    if ledger and ledger.stage_state(name) in ('prepared','sent'):
        raise RuntimeError(name+'_stage_already_prepared_no_retry')


def capture_stage(provider,window,obs,*,kind=None,action=None):
    snapshot=getattr(provider,'client_object_snapshot',None)
    tracker=snapshot() if snapshot else None
    objects_before=tracker.proofs(obs) if tracker else None
    broker=getattr(getattr(provider,'sampler',None),'broker',None)
    if broker:
        region=None
        if kind=='confirm':region=getattr(getattr(provider,'mapper',None),'confirmation_region',lambda *a:None)(obs,window.to_dict())
        image,capture=broker.capture(window,stamp=(obs.game_id,obs.revision,obs.log_position,obs.client_task_list,obs.client_submission_position),region=region)
    else:image,capture=provider.windows.capture(window,None)
    if action is not None:
        after=provider.semantic_observation(getattr(provider.logs,'synchronize',provider.logs.observe)())
        green=getattr(provider,'uses_green_signal',False)
        def context(o):
            if not green:return stage_context(o,action,'capture')
            from .action_context import input_context_token
            return (input_context_token(o,action,ignore_playback=True),o.client_submission_position)
        if context(obs)!=context(after):
            capture={**(capture or {}),'stage_context_changed':True}
    if snapshot:
        from .client_objects import bind_capture
        tracker=snapshot()
        capture={**(capture or {}),'client_objects':bind_capture(objects_before,tracker.proofs(obs) if tracker else None)}
    return image,capture


def circular_control(image,band,*,eligible_frame=False):
    compute=lambda:_circular_control(image,band,eligible_frame=eligible_frame)
    if hasattr(image,'memo'):
        width,height=image.size
        bounds=tuple(round(v*s) for v,s in zip(band,(width,height,width,height)))
        return image.memo('circular_control',(band,eligible_frame),compute,bounds=bounds)
    return compute()


def _circular_control(image,band,*,eligible_frame=False):
    """Search a HUD slot band, accept one observed circular frame only."""
    width,height=image.size
    x0,y0,x1,y1=[round(v*s) for v,s in zip(band,(width,height,width,height))]
    scale=min(1,800/width)
    rgb=np.asarray(image.crop((x0,y0,x1,y1)))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.resize(gray,None,fx=scale,fy=scale)
    if eligible_frame:
        # Detect the independently visible eligibility rim, not an animated
        # gem/portrait inside it. The slot band alone never supplies a point.
        small=cv2.resize(rgb,None,fx=scale,fy=scale)
        hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV);h,s,v=cv2.split(hsv)
        mask=((h>=35)&(h<=100)&(s>95)&(v>110)).astype('uint8')*255
        gray=cv2.GaussianBlur(mask,(3,3),.8)
    circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1),cv2.HOUGH_GRADIENT,1,
        minDist=width*.045*scale,param1=100,param2=16 if eligible_frame else 27,
        minRadius=round(width*.024*scale),maxRadius=round(width*.048*scale))
    if circles is None or len(circles[0])!=1:return None
    if eligible_frame:
        cx,cy,radius=circles[0][0];hits=[]
        for angle in np.linspace(0,2*np.pi,48,endpoint=False):
            x=round(cx+radius*np.cos(angle));y=round(cy+radius*np.sin(angle))
            patch=mask[max(0,y-2):y+3,max(0,x-2):x+3]
            hits.append(bool(patch.size and np.any(patch)))
        # Independent support on both sides prevents one green sliver or card
        # art inside a slot band from acting as a complete circular frame.
        if np.mean(hits)<.55 or np.mean(hits[:12]+hits[36:])<.4 or np.mean(hits[12:36])<.4:return None
    x,y,r=circles[0][0]/scale
    return Box(round(x0+x-r*.4),round(y0+y-r*.4),round(r*.8),round(r*.8))


def circle_at(image,box):
    """Recheck a learned circle locally; its center never implies eligibility.

    The saved inner box is 0.8 times the detected outer radius. Probe the full
    green rim on both sides; missing/shifted frames fall back to detection.
    """
    if box is None:return None
    width,height=image.size;cx,cy=box.point();radius=(box.width+box.height)/1.6
    tolerance=max(3,round(width*.0025));pad=round(radius*1.3)+tolerance+1
    bounds=(round(cx)-pad,round(cy)-pad,round(cx)+pad+1,round(cy)+pad+1)
    if bounds[0]<0 or bounds[1]<0 or bounds[2]>width or bounds[3]>height:return None
    rgb=np.asarray(image.crop(bounds));hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    green=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=100)&(hsv[:,:,1]>95)&(hsv[:,:,2]>110)
    hits=[];inner=[];outer=[]
    for angle in np.linspace(0,2*np.pi,48,endpoint=False):
        for ratio,result in ((1,hits),(.6,inner),(1.3,outer)):
            x=round(cx-bounds[0]+radius*ratio*np.cos(angle));y=round(cy-bounds[1]+radius*ratio*np.sin(angle))
            patch=green[max(0,y-tolerance):y+tolerance+1,max(0,x-tolerance):x+tolerance+1]
            result.append(bool(patch.size and np.any(patch)))
    if np.mean(hits)<.55 or np.mean(hits[:12]+hits[36:])<.4 or np.mean(hits[12:36])<.4:return None
    # A green overlay/rectangle is not a circular rim. Conservative rejection
    # merely triggers the original detector; it never authorizes a new point.
    if np.mean(inner)>.75 or np.mean(outer)>.25:return None
    return box


def hud_mappings(obs,image,maps,*,cache=None):
    if not any(m.key=='scene:recruit' and m.status=='matched' for m in maps):return []
    result=[];visible={m.key:m for m in maps if m.status=='matched' and m.rect};seen=set()
    for a in obs.actions:
        identity=(a.kind,a.source)
        if identity in seen:continue
        seen.add(identity)
        control=next((c for c in obs.controls if c.entity_id==a.source),None)
        if not control:continue
        existing=visible.get('control:'+str(a.kind)) if a.kind in (Kind.FREEZE,Kind.HERO_POWER) else None
        if a.kind==Kind.ACTIVATE and control.card_type=='GAME_MODE_BUTTON':
            # New slots can be calibrated generically without a card-specific
            # executor. Unknown slots never inherit another button's location.
            existing=visible.get('control:game_mode_slot:'+str(control.tags.get('GAME_MODE_BUTTON_SLOT','')))
        if existing:
            if not any(m.key==f'entity:{a.source}' for m in result):
                result.append(Mapping(f'entity:{a.source}','matched',existing.rect,existing.confidence,'server_control_and_measured_hud_anchor',entity_id=a.source))
            continue
        band=None
        if a.kind==Kind.HERO_POWER:
            # Active hero power belongs to the local hero; server options exclude passives.
            linked=obs.hero.tags.get('HERO_POWER_ENTITY') if obs.hero else None
            if linked is not None and str(a.source)!=str(linked):continue
            band=(.55,.66,.66,.85)
        elif a.kind==Kind.FREEZE:band=(.62,.08,.70,.23)
        elif a.kind==Kind.ACTIVATE and control.card_type=='GAME_MODE_BUTTON' and control.tags.get('GAME_MODE_BUTTON_SLOT')=='5':
            band=(.83,.20,.92,.38)
        if band:
            identity=(obs.game_id,obs.build,image.size,str(a.kind),a.source,control.tags.get('GAME_MODE_BUTTON_SLOT'))
            frame=circle_at(image,cache.get(identity)) if cache is not None else None
            if frame is None:frame=circular_control(image,band,eligible_frame=True)
            box=frame or circular_control(image,band)
            if cache is not None:
                cache.pop(identity,None)
                if frame:cache[identity]=frame
                while len(cache)>16:cache.pop(next(iter(cache)))
            if box:
                basis=StabilityBasis.CONTROL_FRAME if frame else StabilityBasis.PIXELS
                result.append(Mapping(f'entity:{a.source}','matched',box,.85,'server_hud_slot_and_unique_observed_circular_frame',entity_id=a.source,stability_basis=basis))
                if a.kind==Kind.HERO_POWER:result.append(Mapping('control:hero_power','matched',box,.85,'local_active_power_and_observed_circular_frame',stability_basis=basis))
                if a.kind==Kind.FREEZE:result.append(Mapping('control:freeze','matched',box,.85,'server_freeze_option_and_observed_circular_frame',stability_basis=basis))
    return result


@measured_stage('followup_client_wait')
def _stage_wait(provider,executor,seconds):
    return executor.cancel.wait(seconds)


@measured_stage('followup_mouse_gesture')
def stage_click(provider,executor,action,plan,obs,window,box,*,stage,menu_recovery=False,capture=None):
    from .action_context import input_context_token
    from .interaction_timing import SAFE_TIMING
    timing=getattr(executor,'timing',SAFE_TIMING)
    x,y=box.point()
    from .green_readiness import SignalReadiness
    capture=capture if capture is not None else getattr(provider,'last_row',{}).get('capture',{})
    proof=SignalReadiness().inspect(stage_context(obs,action,stage),capture,
        (box.x,box.y,box.width,box.height),window=window.token)
    if not proof['ready']:
        error='stage_evidence_refused:'+','.join(proof['reasons'])
        if 'capture_expired' in proof['reasons'] or 'capture_reused_for_new_input' in proof['reasons']:
            raise StageEvidenceExpired(error)
        raise ValueError(error)
    needed=timing.hover+timing.click_hold+timing.release_hold+.05
    if time.time()-capture['end']+needed>.75:
        raise StageEvidenceExpired('stage_evidence_refused:capture_expiring')
    if menu_recovery:
        from .queue_recovery import menu_context
        if action.kind!=Kind.ENTER or not menu_context(provider,obs):raise RuntimeError('queue_recovery_context_changed')
    p=replace(plan,revision=obs.revision,source=tuple(client_to_screen((x,y),window.client)),destination=None,
        window_token=window.token,expires_at=min(time.time()+1,capture['end']+.75),
        state_token='' if menu_recovery else input_context_token(obs,action,ignore_playback=getattr(plan,'green_signal',False)))
    wait=getattr(executor,'wait',executor.cancel.wait)
    executor.check(p)
    prepress=getattr(provider,'before_mouse_down',None)
    # 分支、确认、奖励等按钮不属于 action.target；只有目标阶段复核目标实体。
    if prepress and stage=='target':
        park=getattr(provider,'park_neutral',None)
        if park and not park():raise RuntimeError('prepress_neutral_position_unavailable')
        # 此阶段是单击，预算来自实际节拍与余量；不能套用父动作拖拽的400ms。
        try:checked=prepress(p,target_stage=True,required_remaining=needed)
        except ValueError as exc:
            # 此子阶段尚未按下，回到原有有界目标等待；父动作绝不重发。
            raise StageEvidenceExpired('prepress_board_refresh_required:'+str(exc)) from exc
        from .native_bridge import NativePlan
        if isinstance(checked,NativePlan):p=checked
        executor.check(p)
    # 昂贵预检之后再确认完整单击预算；不延长任何捕获的750ms寿命。
    if p.expires_at-time.time()<needed:
        raise StageEvidenceExpired('stage_evidence_refused:capture_expiring_after_prepress')
    executor.transport.move(*p.source);executor.check(p);wait(timing.hover);executor.check(p)
    executor.transport.button(True);wait(timing.click_hold);executor.check(p);executor.transport.button(False)
    provider.last_input_released_at=time.time()
    wait(timing.release_hold)
    # 同英雄重选也是子阶段输入；松开并完成原节拍后立即归位，
    # 防止下一次截图被当前悬停产生的技能/卡牌浮层遮挡。
    park=getattr(provider,'park_cursor',None)
    if park:
        if park(executor) is False:raise RuntimeError('post_stage_cursor_park_failed')
        audit=getattr(provider,'audit',None)
        if audit:audit.write('stage_cursor_parked',stage=stage,mouse_released=True)


def record_stage(provider,name,state,**evidence):
    ledger=getattr(provider,'checkpoint',None)
    if ledger:ledger.stage(name,state,**evidence)


class StageEvidenceExpired(ValueError):
    """只在尚未移动/按下鼠标前抛出，允许原请求在原预算内重新取图。"""


def finish_branch(provider,executor,action,plan):
    from .interaction_protocol import branch_entities
    try:options,chosen=branch_entities(action)
    except ValueError as exc:raise RuntimeError('branch_context_invalid') from exc
    ensure_stage_unsent(provider,'branch')
    before=provider.last_obs;deadline=time.monotonic()+CHOICE_SECONDS;first=True;last_frame=None
    readiness=SignalReadiness();stage_prepared=False
    # A release point can hover one of the new choices and enlarge it, breaking
    # the complete-row proof. Park only inside the bound unobscured client.
    executor.guard()
    initial_window=provider.windows.snapshot(provider.binding.hwnd)
    if initial_window.token!=plan.window_token or initial_window.blockers:raise RuntimeError('branch_window_changed')
    if hasattr(provider,'park_cursor'):provider.park_cursor(executor)
    while time.monotonic()<deadline:
        executor.guard()
        if executor.stopped() or executor.transport.pressed(0x77):raise InterruptedError('human_stop')
        window=provider.windows.snapshot(provider.binding.hwnd)
        if window.token!=plan.window_token or window.blockers:raise RuntimeError('branch_window_changed')
        obs=provider.semantic_observation(provider.logs.observe())
        if obs.game_id!=before.game_id or obs.turn!=before.turn or obs.local_player_id!=before.local_player_id:raise RuntimeError('branch_game_or_turn_changed')
        if obs.phase not in (Phase.RECRUIT,Phase.CHOICE):raise RuntimeError('branch_phase_changed')
        if obs.interaction and (obs.interaction.mode!='choice' or tuple(c.entity_id for c in obs.choices)!=options
                or obs.interaction.owner not in (None,obs.local_player_id)):
            raise RuntimeError('unrelated_forced_choice_during_branch')
        if not obs.interaction:
            source=lambda o:next((c for c in o.hand+o.board+o.controls if c.entity_id==action.source),None)
            old,new=source(before),source(obs)
            if old is None or new is None or (old.card_id,old.controller,old.zone)!=(new.card_id,new.controller,new.zone):
                raise RuntimeError('branch_parent_entity_changed')
        image,capture=capture_stage(provider,window,obs,action=action)
        diagnostics={}
        branch=getattr(provider.mapper,'branch_slots',None) if hasattr(provider,'mapper') else None
        parent=next((c for c in before.hand+before.board+before.controls if c.entity_id==action.source),None)
        kwargs={'parent_kind':parent.card_type if parent else None} if getattr(getattr(provider,'mapper',None),'supports_parent_layout',False) else {}
        boxes=branch(obs,image,window.to_dict(),options,chosen,capture,**kwargs) if branch else branch_selection_regions(image,len(options),options.index(chosen),diagnostics)
        if capture and not capture.get('geometry_unchanged',True):raise RuntimeError('branch_capture_geometry_changed')
        last_frame=(obs,window,image,capture)
        provider.audit.write('branch_stage_observation',source=action.source,options=options,chosen=chosen,
            revision=obs.revision,regions=[asdict(b) for b in boxes],complete_row=bool(boxes),
            reason=diagnostics.get('reason'))
        if first:
            archive_interaction(provider,'branch-first',action,obs,window,image,capture,regions=[asdict(b) for b in boxes]);first=False
        shown=not obs.interaction or obs.interaction.display_state not in ('waiting','hidden')
        evidence=readiness.inspect(stage_context(obs,action,'branch'),capture,
            tuple(v for b in boxes for v in (b.x,b.y,b.width,b.height)),valid=bool(boxes) and shown,
            window=window.token)
        if evidence['ready']:
            box=boxes[options.index(chosen)]
            bounds=getattr(getattr(provider,'mapper',None),'last_branch_bounds',())
            if action.target is not None and parent and parent.card_type in ('SPELL','BATTLEGROUND_SPELL') and len(bounds)==len(options):
                from .branch_occlusion import BranchOcclusion
                provider.branch_target_occlusion=(action.key,BranchOcclusion(image,bounds[options.index(chosen)]))
            archive_interaction(provider,'branch-ready',action,obs,window,image,capture,region=asdict(box),readiness=evidence)
            if not stage_prepared:
                record_stage(provider,'branch','prepared',entity=chosen);stage_prepared=True
            try:stage_click(provider,executor,action,plan,obs,window,box,stage='branch',capture=capture)
            except StageEvidenceExpired:
                provider.audit.write('stage_evidence_recapture',stage='branch',input_sent=False)
                readiness.invalidate();continue
            record_stage(provider,'branch','sent',entity=chosen)
            provider.audit.write('branch_input_sent',source=action.source,sub_option_id=action.sub_option_id,branch_entity=chosen,revision=obs.revision)
            return
        _stage_wait(provider,executor,.04)
    if last_frame:archive_interaction(provider,'branch-failed',action,*last_frame,reason='branch_green_signal_absent')
    raise TimeoutError('branch_frame_not_confirmed_no_retry')


def target_layout(provider,obs,action,image,window,plan=None):
    """目标坐标来自审核布局；待选随从的预摆放只作私有视觉投影，不改日志。"""
    mapper=provider.mapper
    maps=mapper.map(obs,image,window.to_dict())
    source=next((c for c in obs.hand if c.entity_id==action.source),None)
    moving=(action.kind==Kind.PLAY and source and source.card_type=='MINION'
        and any(c.entity_id==action.target for c in obs.board+obs.hand)
        and not any(c.entity_id==action.source for c in obs.board)
        and not (action.interaction and action.interaction.placement=='magnetize'))
    if not moving:return obs,maps,dict(ready=True,pending_placement=False)
    position=action.position
    if position is None and not obs.board:position=1
    if position is None:
        if plan is not None and plan.destination is not None:
            point=(plan.destination[0]-window.client.x,plan.destination[1]-window.client.y)
            slots=[int(m.key.rsplit(':',1)[1]) for m in maps
                if m.key.startswith('position:board:') and m.status=='matched' and m.rect
                and max(abs(a-b) for a,b in zip(m.rect.point(),point))<=3]
            position=slots[0] if len(slots)==1 else None
        else:
            from .action_geometry import route
            key=route(action,obs,{m.key:m for m in maps}).destination
            if key and key.startswith('position:board:'):position=int(key.rsplit(':',1)[1])
    if len(obs.board)>=7 or position is None or not 1<=position<=len(obs.board)+1:
        return obs,[],dict(ready=False,reason='target_pending_placement_unknown')
    board=list(sorted(obs.board,key=lambda c:c.position))
    board.insert(position-1,replace(source,zone='PLAY'))
    visual=replace(obs,board=tuple(replace(c,position=i+1) for i,c in enumerate(board)),
        hand=tuple(replace(c,position=i+1) for i,c in enumerate(c for c in obs.hand if c.entity_id!=source.entity_id)))
    maps=mapper.map(visual,image,window.to_dict())
    bound=any(m.entity_id==action.target and m.status=='matched' and m.rect
        and m.stability_basis==StabilityBasis.GEOMETRY for m in maps)
    return visual,maps,dict(ready=bound,pending_placement=True,source=source.entity_id,
        insertion=position,visual_board_count=len(visual.board),server_board_count=len(obs.board),
        reason=None if bound else 'target_pending_layout_unreviewed')


def finish_target(provider,executor,action,plan):
    """Refresh target appearance after initial gesture, then submit once.

    原授权与当前实体证明目标合法；当前目标阶段放行后直接使用槽位坐标。
    This never retries the initial action if the target stage is uncertain.
    """
    from .live_client import input_context_token
    ensure_stage_unsent(provider,'target')
    ledger=getattr(provider,'checkpoint',None)
    if ledger and (ledger.stage_state('initial')!='sent' or
            action.sub_option_id is not None and ledger.stage_state('branch')!='sent'):
        raise RuntimeError('target_parent_input_not_sent')
    provider.target_parent_plan=(action.key,plan)
    before=provider.last_obs
    stage_prepared=False
    original=next((m for m in provider.mappings if m.key==f'entity:{action.target}' and m.status=='matched' and m.rect),None)
    if original is None:raise RuntimeError('target_mapping_missing_after_initial_input')
    executor.guard()
    initial_window=provider.windows.snapshot(provider.binding.hwnd)
    if initial_window.token!=plan.window_token or initial_window.blockers:raise RuntimeError('target_window_changed')
    provider.park_cursor(executor)
    deadline=time.monotonic()+TARGET_SECONDS;previous=None;archived=False;first=True;last_frame=None
    last_blocker='target_stage_or_coordinates_unconfirmed'
    readiness=SignalReadiness()
    # 父输入已完成；直接检查当前目标对象，不再等待红箭头或全屏变灰。
    while time.monotonic()<deadline:
        executor.guard()
        if executor.stopped() or executor.transport.pressed(0x77):raise InterruptedError('human_stop')
        window=provider.windows.snapshot(provider.binding.hwnd)
        if window.token!=plan.window_token or window.blockers:raise RuntimeError('target_window_changed')
        obs=provider.semantic_observation(provider.logs.observe())
        if obs.game_id!=before.game_id or obs.turn!=before.turn or obs.phase!=Phase.RECRUIT:raise RuntimeError('target_context_changed')
        if obs.interaction and (obs.interaction.mode!='target' or obs.interaction.owner not in (None,obs.local_player_id)
                or obs.interaction.source_entity not in (None,action.source)):
            raise RuntimeError('target_request_context_changed')
        prior=next((c for c in before.board+before.shop+before.hand if c.entity_id==action.target),before.hero if before.hero and before.hero.entity_id==action.target else None)
        current=next((c for c in obs.board+obs.shop+obs.hand if c.entity_id==action.target),obs.hero if obs.hero and obs.hero.entity_id==action.target else None)
        if not prior or not current or (prior.card_id,prior.controller,prior.zone)!=(current.card_id,current.controller,current.zone):raise RuntimeError('target_entity_left_zone')
        image,capture=capture_stage(provider,window,obs,action=action)
        if capture and not capture.get('geometry_unchanged',True):raise RuntimeError('target_capture_geometry_changed')
        last_frame=(obs,window,image,capture)
        if first:
            archive_interaction(provider,'target-first',action,obs,window,image,capture);first=False
        visual_obs,fresh_maps,projection=target_layout(provider,obs,action,image,window,plan)
        mode=provider.mapper.last_ui_state
        measured=next((m for m in fresh_maps if m.key==f'entity:{action.target}' and m.status=='matched' and m.rect),None)
        def layout(o):return tuple((c.entity_id,c.position) for c in o.board+o.shop+o.hand)
        unchanged=layout(before)==layout(obs)
        # 当前实体槽位决定坐标，目标阶段决定时机；金边不参与准入。
        box=measured.rect if measured else original.rect if unchanged and not projection.get('pending_placement') and projection['ready'] else None
        if box is None:
            last_blocker=projection.get('reason') or 'target_mapping_missing'
            readiness.invalidate();_stage_wait(provider,executor,.04);continue
        hit=box
        from .stage_admission import target_visible
        ready=projection['ready'] and target_visible(obs,mode,action.source,parent_sent=True)
        blockers=[] if ready else ['target_context_conflict']
        if capture.get('end',0)<=getattr(provider,'last_input_released_at',0):
            ready=False;blockers.append('target_capture_precedes_parent_release')
        gate=getattr(provider,'gate',None)
        if ready and callable(getattr(gate,'object_evidence',None)):
            # 后续目标和独立TARGET共用对象依赖证明，不把普通共享绿光带进模态。
            target_action=replace(action,kind=Kind.TARGET,source=None)
            kwargs={'visual_obs':visual_obs} if visual_obs is not obs else {}
            objects=gate.object_evidence(obs,target_action,image,fresh_maps,capture,window.token,str(provider.logs.session),**kwargs)
            provider.audit.write('target_object_readiness',source=action.source,target=action.target,**objects)
            ready=objects['ready']
            blockers.extend(objects.get('reasons',()))
        elif ready:
            ready=False;blockers.append('target_object_evidence_missing')
        transition=getattr(provider,'branch_target_occlusion',None)
        if ready and transition and transition[0]==action.key and transition[1].blocks(image,hit.point()):
            provider.audit.write('target_wait_branch_occlusion',source=action.source,target=action.target,point=hit.point())
            ready=False;blockers.append('target_branch_overlay')
        if ready and not archived and hasattr(provider,'archive_queue'):
            try:
                provider.archive_queue.put_nowait((f'target-stage-{action.source}-{action.target}-{obs.revision}',image,
                    dict(stage='target_selection',source=action.source,target=action.target,region=asdict(box),ui=mode,
                         sample=dict(observation=obs.to_dict(),window=window.to_dict(),capture=capture))))
                archived=True
            except queue.Full:pass
        geometry=(box.x,box.y,box.width,box.height)
        provider.audit.write('target_stage_observation',source=action.source,target=action.target,ui=mode,
            signal='parent_input_and_target_object',region=asdict(box),visual_layout=projection,
            object_ready=ready,reasons=blockers,red_arrow_required=False)
        evidence=readiness.inspect(stage_context(obs,action,'target'),capture,
            (*geometry,hit.x,hit.y,hit.width,hit.height),valid=ready,window=window.token)
        if not evidence['ready']:
            last_blocker=','.join(blockers or evidence.get('reasons',())) or 'target_object_or_stage_pending'
        if evidence['ready']:
            if not stage_prepared:
                record_stage(provider,'target','prepared',entity=action.target);stage_prepared=True
            try:stage_click(provider,executor,action,plan,obs,window,hit,stage='target',capture=capture)
            except StageEvidenceExpired as exc:
                last_blocker=str(exc)
                provider.audit.write('stage_evidence_recapture',stage='target',input_sent=False,reason=last_blocker)
                readiness.invalidate();continue
            record_stage(provider,'target','sent',entity=action.target)
            provider.audit.write('target_input_sent',source=action.source,target=action.target,revision=obs.revision,ui=mode,readiness=evidence)
            return
        _stage_wait(provider,executor,.04)
    if last_frame:archive_interaction(provider,'target-failed',action,*last_frame,reason=last_blocker)
    raise TimeoutError('target_stage_timeout_no_retry:'+last_blocker)
