"""战场对象按下前复核；只检查必要对象与已知矩形卡牌／面板遮挡，不等待全屏静止。"""
import time
from dataclasses import replace
import cv2
import numpy as np

from .model import Kind,Phase


def point_occlusion(image,point):
    return points_occlusion(image,(point,))


def hover_panel(image):
    """识别大号深灰关键词说明框，用于移开鼠标后重采；不能据此确认对象身份。"""
    from .windows import Box
    region=getattr(image,'region',Box(0,0,image.width,image.height))
    scale=min(1.,1280/region.width)
    rgb=np.asarray(image.crop((region.x,region.y,region.x+region.width,region.y+region.height)).convert('RGB'))
    if scale<1:rgb=cv2.resize(rgb,(round(region.width*scale),round(region.height*scale)))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([0,0,30]),np.array([179,65,115]))
    # 填合文字小洞，不把相邻说明框连起来；去掉字体和零散纹理。
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    width,height=(v*scale for v in image.size)
    for x,y,w,h,area in stats[1:]:
        if not width*.10<w<width*.35 or not height*.06<h<height*.25:continue
        if not 1.2<w/h<3.2 or area/(w*h)<.85:continue
        top=y+region.y*scale
        if height*.20<top<height*.70:
            return dict(reason='board_hover_panel_visible',rect=(x/scale+region.x,y/scale+region.y,w/scale,h/scale))
    return None


def points_occlusion(image,points):
    """寻找覆盖操作点的较大矩形面板；不识别卡面，不生成鼠标坐标。"""
    from .windows import Box
    region=getattr(image,'region',Box(0,0,image.width,image.height))
    points=[p for p in points if region.x<=p[0]<region.x+region.width and region.y<=p[1]<region.y+region.height]
    if not points:return None
    pixels=np.asarray(image.crop((region.x,region.y,region.x+region.width,region.y+region.height)).convert('L'))
    # 面板检测只需保留大尺度结构；一帧最多分析1280像素宽，不在每个鼠标点重复扫描。
    scale=min(1.,1280/region.width)
    if scale<1:
        pixels=cv2.resize(pixels,(round(region.width*scale),round(region.height*scale)))
        region=Box(round(region.x*scale),round(region.y*scale),pixels.shape[1],pixels.shape[0])
        points=[(x*scale,y*scale) for x,y in points]
    edges=cv2.Canny(pixels,60,140)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    width,height=(v*scale for v in image.size)
    def covered(left,top,bw,bh):return any(left<x<left+bw and top<y<top+bh for x,y in points)
    def result(rect,method='closed_panel_frame'):
        return dict(reason='board_point_covered_by_panel',rect=tuple(v/scale for v in rect),method=method)
    for contour in contours:
        bx,by,bw,bh=cv2.boundingRect(contour);left,top=bx+region.x,by+region.y
        if not covered(left,top,bw,bh):continue
        # 不把被裁断的画面边缘、正常随从边框或整个棋盘当作浮层。
        if bx<=1 or by<=1 or bx+bw>=region.width-1 or by+bh>=region.height-1:continue
        aspect=bw/bh
        large_card=.35<=aspect<=1.25 and width*.09<=bw<=width*.26 and height*.20<=bh<=height*.65
        wide_panel=1.4<=aspect<=6 and width*.14<=bw<=width*.45 and height*.045<=bh<=height*.22
        if not (large_card or wide_panel):continue
        polygon=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if not 4<=len(polygon)<=6 or cv2.contourArea(contour)<bw*bh*.80:continue
        contour_points=contour.reshape(-1,2);px,py=contour_points[:,0],contour_points[:,1]
        near=(np.minimum(px-bx,bx+bw-1-px)<=max(3,bw*.015)) | (np.minimum(py-by,by+bh-1-py)<=max(3,bh*.015))
        if float(near.mean())<.70:continue
        return result((left,top,bw,bh))
    # 卡牌边框可能与后方特效相连，闭合轮廓会丢失；同时要求两条长侧轨
    # 和上下横轨，不能把单条卡面纹理或两个随从的零散边缘当作面板。
    lines=[]
    for gap in (.0125,.0375):
        found=cv2.HoughLinesP(edges,1,np.pi/180,threshold=30,
            minLineLength=max(24,round(height*.10)),maxLineGap=max(4,round(height*gap)))
        if found is not None:lines.extend(found[:,0].tolist())
    vertical=[];horizontal=[]
    for ax,ay,bx,by in lines:
        if abs(ax-bx)<=max(2,abs(ay-by)*.02) and abs(ay-by)>=height*.15:
            vertical.append(((ax+bx)/2+region.x,min(ay,by)+region.y,max(ay,by)+region.y))
        if abs(ay-by)<=max(2,abs(ax-bx)*.02):
            horizontal.append((min(ax,bx)+region.x,max(ax,bx)+region.x,(ay+by)/2+region.y))
    for left,lt,lb in vertical:
        if not any(left<x for x,y in points):continue
        for right,rt,rb in vertical:
            if not any(left<x<right and max(lt,rt)<y<min(lb,rb) for x,y in points):continue
            bw=right-left;top=min(lt,rt);bottom=max(lb,rb);bh=bottom-top
            if not width*.09<=bw<=width*.26 or not height*.20<=bh<=height*.65 or not .35<=bw/bh<=1.25:continue
            if min(lb,rb)-max(lt,rt)<height*.20:continue
            def cap(edge):return any(min(right,b)-max(left,a)>=bw*.70 and abs(c-edge)<=height*.065 for a,b,c in horizontal)
            if cap(top) and cap(bottom):
                return result((left,top,bw,bh),'paired_large_card_rails')
    return None


def before_mouse_down(provider,plan,*,target_stage=False,required_remaining=.4,_settle_once=True):
    """只在实际操作点属于战场随从时采一帧；任何拒绝都发生于本次按下之前。"""
    if not 0<required_remaining<=.75:raise ValueError('invalid_prepress_time_budget')
    if plan.destination is not None:required_remaining=max(.4,required_remaining)
    action=getattr(provider,'pending_action',None);previous=getattr(provider,'last_obs',None)
    if action is None or previous is None:return False
    eid=action.target if target_stage or action.kind==Kind.TARGET else action.source
    if not any(c.entity_id==eid for c in previous.board):return False
    if not target_stage and action.kind in (Kind.SELL,Kind.REORDER,Kind.ACTIVATE) and previous.interaction is None:
        lease=getattr(provider,'last_input_admission',None)
        capture=(getattr(provider,'last_row',None) or {}).get('capture',{})
        if (lease and lease['object_ready'] and lease['action_key']==plan.action_key
                and lease['window_token']==plan.window_token and lease['state_token']==plan.state_token
                and lease['session']==str(provider.logs.session)
                and lease['capture_end']==capture.get('end')
                and lease['frame_sequence'] is not None and lease['frame_sequence']==capture.get('frame_sequence')
                and lease['image_identity']==id(provider.last_image)
                and lease['source']==plan.source and lease['destination']==plan.destination
                and 0<=time.time()-lease['capture_end']<=.75
                and plan.expires_at-time.time()>=required_remaining
                and lease['capture_end']>getattr(provider,'last_input_released_at',0)):
            # 统一使用实际剩余预算；另加300ms帧龄会拒掉仍有400ms手势余量
            # 的同一完整准入，反而触发更昂贵的重复取证。证据上限仍是750ms。
            provider.check_input(plan)
            if plan.expires_at-time.time()<required_remaining:raise ValueError('prepress_capture_expiring')
            provider.board_prepress_refusal=None
            provider.audit.write('board_prepress_reused_admission',entity=eid,
                capture_end=lease['capture_end'],frame_sequence=lease['frame_sequence'],
                capture_age_ms=(time.time()-lease['capture_end'])*1000,
                source=plan.source,destination=plan.destination,additional_capture=False)
            return True
    request=previous.interaction
    refusal_key=(previous.game_id,previous.turn,previous.phase,action.key,plan.window_token,
        (request.mode,request.request_id,request.source_entity) if request else None,
        tuple((c.entity_id,c.position,c.card_id) for c in previous.board))
    refusal=getattr(provider,'board_prepress_refusal',None)
    if refusal and refusal[0]==refusal_key and time.monotonic()-refusal[1]>=8:
        raise TimeoutError('board_prepress_timeout:'+refusal[2])
    if time.time()>plan.expires_at:raise ValueError('prepress_plan_expired')
    provider.check_input(plan)
    from .live_interactions import capture_stage
    from .live_client import phase_view_blockers
    from .client_readiness import logical_reasons
    from .action_context import input_context_token
    from .readiness_window import capture_blockers
    from .model import Action
    window=provider.windows.snapshot(provider.binding.hwnd)
    obs=provider.semantic_observation(getattr(provider.logs,'synchronize',provider.logs.observe)())
    prior_capture=(getattr(provider,'last_row',None) or {}).get('capture',{})
    image,capture=capture_stage(provider,window,obs,action=action)
    reasons=capture_blockers(capture)
    stamp=capture.get('end')
    if capture.get('reused_capture') or not isinstance(stamp,(int,float)) or not 0<=time.time()-stamp<=.75:
        reasons.append('prepress_capture_not_fresh')
    if isinstance(stamp,(int,float)) and stamp<=prior_capture.get('end',float('-inf')):reasons.append('prepress_capture_not_new')
    if capture.get('stage_context_changed'):reasons.append('prepress_context_changed')
    if input_context_token(obs,action,ignore_playback=plan.green_signal)!=plan.state_token:reasons.append('prepress_action_changed')
    row=dict(id='prepress',session=str(provider.logs.session),observation=obs.to_dict(),window=window.to_dict(),
        capture=capture,blockers=list(window.blockers),log_snapshot_unchanged=not capture.get('stage_context_changed',False),
        input_released_at=getattr(provider,'last_input_released_at',0))
    provider.mapper.begin_capture(row)
    visual_obs=obs;projection={'ready':True}
    if target_stage:
        from .live_interactions import target_layout
        saved=getattr(provider,'target_parent_plan',None)
        parent_plan=saved[1] if saved and saved[0]==action.key else None
        visual_obs,maps,projection=target_layout(provider,obs,action,image,window,parent_plan)
    else:maps=provider.mapper.map(obs,image,row['window'])
    ui=provider.mapper.last_ui_state
    reasons.extend(phase_view_blockers(obs,maps))
    probe=Action(Kind.TARGET,target=eid) if target_stage else action
    if target_stage:
        from .stage_admission import target_visible
        source=action.authorization_source if action.kind==Kind.TARGET else action.source
        if obs.phase!=Phase.RECRUIT or not target_visible(obs,ui,source,parent_sent=action.kind!=Kind.TARGET):reasons.append('prepress_target_stage_changed')
        if not projection['ready']:reasons.append(projection['reason'])
    else:
        reasons.extend(r for r in logical_reasons(row,action,ui) if r!='client_task_list_processing')
    matched=next((m for m in maps if m.entity_id==eid and m.status=='matched' and m.rect),None)
    point=(plan.source[0]-window.client.x,plan.source[1]-window.client.y)
    if matched is None or max(abs(a-b) for a,b in zip(matched.rect.point(),point))>3:reasons.append('prepress_source_position_changed')
    # 先区分真实遮挡，避免把被预览牌盖住的对象误报成“对象不存在”。
    if not reasons:
        points=[(p[0]-window.client.x,p[1]-window.client.y) for p in (plan.source,plan.destination) if p]
        blocked=points_occlusion(image,points)
        if blocked:reasons.append(blocked['reason'])
    gate=getattr(provider,'gate',None)
    objects=None
    if not reasons:
        if not callable(getattr(gate,'object_evidence',None)):reasons.append('prepress_object_evidence_unavailable')
        else:
            kwargs={'visual_obs':visual_obs} if visual_obs is not obs else {}
            objects=gate.object_evidence(obs,probe,image,maps,capture,window.token,str(provider.logs.session),**kwargs)
            reasons.extend(objects['reasons'])
    if not reasons:
        points=[(p[0]-window.client.x,p[1]-window.client.y) for p in (plan.source,plan.destination) if p]
        transition=getattr(provider,'branch_target_occlusion',None)
        if transition and transition[0]==action.key and any(transition[1].blocks(image,p) for p in points):
            reasons.append('prepress_branch_card_still_covering_point')
    provider.last_row,provider.last_image,provider.last_window=row,image,window
    provider.mappings=maps
    provider.audit.write('prepress_board_check',entity=eid,ready=not reasons,reasons=sorted(set(reasons)),
        previous_capture=prior_capture.get('end'),capture_end=stamp,target_stage=target_stage,objects=objects)
    # 原计划未发输入；新捕获已重新验证原身份、坐标及上下文，时效属于新捕获。
    fresh_expiry=min(time.time()+1.5,stamp+.75) if isinstance(stamp,(int,float)) else 0
    if reasons:
        if _settle_once and all(r.startswith('local_object_position_pending:') for r in reasons):
            # 仍未按下：在同一悬停位置补一帧，而不是丢掉动作并反复进出悬停。
            guard=getattr(provider,'input_guard',None)
            if guard:guard()
            if time.time()+.04<fresh_expiry:
                time.sleep(.04)
                if guard:guard()
                return before_mouse_down(provider,replace(plan,expires_at=fresh_expiry),target_stage=target_stage,
                    required_remaining=required_remaining,_settle_once=False)
        park=getattr(provider,'park_neutral',None)
        if park:park()
        if all(r.startswith(('local_object_','board_')) for r in reasons):
            if not refusal or refusal[0]!=refusal_key:
                provider.board_prepress_refusal=(refusal_key,time.monotonic(),','.join(reasons))
        provider.invalidate_readiness('prepress_board_blocked');provider.cache_until=0
        raise ValueError('native_observation_changed')
    provider.check_input(plan)
    remaining=fresh_expiry-time.time()
    provider.audit.write('prepress_budget_check',target_stage=target_stage,
        remaining_seconds=remaining,required_remaining_seconds=required_remaining,
        ready=remaining>=required_remaining,input_sent=False)
    if remaining<required_remaining:raise ValueError('prepress_capture_expiring')
    refreshed=replace(plan,created_at=time.time(),expires_at=fresh_expiry)
    provider.board_prepress_refusal=None
    provider.audit.write('prepress_plan_refreshed',capture_end=stamp,expires_at=fresh_expiry,
        previous_expires_at=plan.expires_at,source=plan.source,destination=plan.destination,input_sent=False)
    return refreshed
