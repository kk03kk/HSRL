"""按动作依赖准入：客户端实体布局、局部位置与控件资格各自取证。"""
import time
import cv2
import numpy as np
from .model import Observation,Kind
from .green_readiness import GreenReadiness,action_signal,source_geometry_ready
from .readiness_window import CaptureHistory
from .stage_admission import ORDINARY

CONTROLS=frozenset((Kind.REFRESH,Kind.FREEZE,Kind.UPGRADE,Kind.HERO_POWER))


def board_structure(image,box):
    """审核槽位内同时测量多条左右外轮廓；卡面内最强边不是对象身份。"""
    return _board_structure(image,box,False) or _board_structure(image,box,True)


def _board_structure(image,box,color):
    if box is None:return None
    pad=max(2,round(box.width*.08))
    x0=max(0,box.x-pad);x1=min(image.width,box.x+box.width+pad)
    left_end=box.x+box.width*.28;right_start=box.x+box.width*.72
    geometries=[]
    for lower,upper in ((.22,.34),(.35,.47),(.48,.60),(.61,.73)):
        y0=max(0,round(box.y+box.height*lower));y1=min(image.height,round(box.y+box.height*upper))
        patch=np.asarray(image.crop((x0,y0,x1,y1)).convert('RGB' if color else 'L'),dtype=np.float32)
        if min(patch.shape[:2])<3:return None
        # 灰度失败时补RGB两像素跨度，捕捉圣盾柔边；仍要求同强度及多行支持。
        gradient=np.max(np.abs(patch[:,2:]-patch[:,:-2]),axis=2) if color else np.abs(np.diff(patch,axis=1))
        # 多行支持抑制一行卡面细节；取外侧边而非全槽位的最强纹理。
        strength=gradient.mean(axis=0);support=(gradient>12).mean(axis=0)
        candidates=np.flatnonzero((strength>=10)&(support>=.35))+x0+(1. if color else .5)
        left=candidates[candidates<=left_end];right=candidates[candidates>=right_start]
        if not len(left) or not len(right):continue
        a,b=float(left[0]),float(right[-1])
        if not .65*box.width<=b-a<=1.18*box.width:continue
        geometries.extend((a,b))
    if len(geometries)<6:return None
    pairs=np.asarray(geometries).reshape(-1,2)
    centers=pairs.mean(axis=1)
    if np.any(np.ptp(pairs,axis=0)>box.width*.22) or abs(float(np.median(centers))-(box.x+box.width/2))>box.width*.15:return None
    # 位移取多段边界的中心，而非外圈宽度；圣盾、呼吸边框扩缩不等于对象移动。
    return dict(method='paired_board_color_structure' if color else 'paired_board_structure',geometry=(float(np.median(centers)),),
        edges=tuple(geometries),bands=len(pairs))


def choice_slot_structure(image,box):
    """候选槽位两侧的中性结构边缘；不要求绿框，不识别卡面身份。"""
    if box is None:return False
    for edge in (box.x,box.x+box.width):
        margin=max(3,round(box.width*.12));x0=max(0,edge-margin);x1=min(image.width,edge+margin)
        y0=max(0,box.y+round(box.height*.22));y1=min(image.height,box.y+round(box.height*.78))
        pixels=np.asarray(image.crop((x0,y0,x1,y1)).convert('RGB'),dtype=np.float32)
        if min(pixels.shape[:2])<3 or np.mean(np.max(np.abs(np.diff(pixels,axis=1)),axis=(1,2))>35)<.65:return False
    return True


def dependencies(obs,action):
    """接收空手牌区不依赖手牌动画；插入站位依赖当前战场布局。"""
    zones={}
    def add(eid):
        for zone in ('hand','shop','board'):
            if any(c.entity_id==eid for c in getattr(obs,zone)):
                zones.setdefault(zone,set()).add(eid);return
    if action.kind in (Kind.BUY,Kind.SELL,Kind.PLAY,Kind.REORDER,Kind.ACTIVATE):add(action.source)
    if action.target is not None:add(action.target)
    if action.kind==Kind.REORDER or action.kind==Kind.PLAY and any(c.entity_id==action.source and c.card_type=='MINION' for c in obs.hand):
        zones.setdefault('board',set()).update(c.entity_id for c in obs.board)
    return {zone:tuple(sorted(ids)) for zone,ids in zones.items()}


def local_object(mapper,obs,zone,eid,image,mappings):
    """只在已校准槽位观察结构；绝不生成或修正鼠标坐标。"""
    from .layout_motion import sample_edges
    projection=getattr(mapper,'projection',None);viewport=getattr(mapper,'viewport',None)
    spec=next((s for s in getattr(projection,'layouts',()) if s.zone==zone),None)
    cards=sorted(getattr(obs,zone),key=lambda c:c.position)
    index=next((i for i,c in enumerate(cards) if c.entity_id==eid),None)
    hit=next((m.rect for m in mappings if m.entity_id==eid and m.status=='matched' and m.rect),None)
    if spec is None or viewport is None or index is None or hit is None:return None
    body=None
    if zone=='board' or zone=='shop' and cards[index].card_type=='MINION':
        bounds=spec.probes.get('slot_bounds',{}).get(f'slot:{index+1}')
        box=viewport.box(bounds) if bounds else None
        body=board_structure(image,box) if box else None
        if zone=='board':
            if body:
                x,y=hit.point();edges=body['edges']
                body['point_covered']=bool(box.y+box.height*.22<y<box.y+box.height*.73 and
                    all(a+3<x<b-3 for a,b in zip(edges[::2],edges[1::2])))
            return body
    probe=spec.probes.get('motion',{}).get(f'slot:{index+1}')
    if probe and not (zone=='shop' and cards[index].card_type=='MINION'):
        measured=sample_edges(image,viewport,[probe])
        if measured:return dict(method='calibrated_local_edge',geometry=measured)
    if zone=='shop':
        # 酒馆随从与法术均有等级旗帜；不依赖可选金边或卡牌画面颜色。
        box=mapper._bounds(spec,viewport,index)
        if box is None:return None
        x0=max(0,box.x-round(box.width*.2));x1=min(image.width,box.x+round(box.width*1.2))
        y0=max(0,box.y-round(box.height*.25));y1=min(image.height,box.y+round(box.height*.3))
        hsv=cv2.cvtColor(np.asarray(image.crop((x0,y0,x1,y1))),cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,np.array([125,70,60]),np.array([175,255,255]))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        candidates=[r for r in stats[1:] if r[2]>=box.width*.1 and r[3]>=box.height*.07 and r[4]>=box.width*box.height*.01]
        if len(candidates)==1:
            x,y,w,h,_=candidates[0]
            flag=(float(x0+x+w/2),float(y0+y+h/2))
            if cards[index].card_type=='MINION':
                # 刷新过渡中可能已有卡体却尚无等级旗；旗帜也不能单独证明本体到位。
                if body:return {**body,'method':'shop_body_and_tier_flag',
                    'geometry':tuple(body['geometry'])+flag,'tier_flag':flag}
                return None
            return dict(method='shop_tier_flag_at_calibrated_slot',geometry=flag)
    if zone=='hand':
        from .green_readiness import hand_top_cap
        others=[m.rect.point() for m in mappings if m.entity_id in {c.entity_id for c in cards}
            and m.entity_id!=eid and m.status=='matched' and m.rect]
        cap=hand_top_cap(image,hit,others)
        if cap['ready']:return dict(method='own_hand_top_rim_presence',geometry=tuple(cap['edge_point']))
        from .hand_mapping import hand_markers
        # 只使用独立测得的绑定；不拿插值或目录坐标当成像素证据。
        measured=hand_markers(image,tuple(cards))
        point=next((box.point() for card,box in measured if card.entity_id==eid),None)
        if point:return dict(method='current_hand_marker',geometry=point)
    return None


class DependencyReadiness(GreenReadiness):
    uses_layout_evidence=True
    def __init__(self,mapper,objects,*,visual=local_object):
        super().__init__(mapper);self.objects=objects;self.visual=visual
        self.history=CaptureHistory();self.samples={};self.admitted={}

    def invalidate(self):
        # 新输入使截图失效，但未改变区域的已证布局可以保留。
        super().invalidate()

    def signal(self,obs,action,image,mappings,ui_state=None):
        if action.kind in ORDINARY and obs.interaction is None:
            if action.kind in CONTROLS or action.kind==Kind.ACTIVATE and not any(c.entity_id==action.source for c in obs.board):
                control=next((c for c in obs.controls if c.entity_id==action.source),None)
                if control and control.tags.get('LITERALLY_UNPLAYABLE')=='1':
                    return dict(ready=False,reason='control_explicitly_disabled',eligibility_state='disabled',
                        eligibility_evidence='own_control_unplayable_tag',green_observed=False)
                signal=action_signal(self.mapper,obs,action,image,mappings)
                # 坐标绑定与像素启用资格分别记录，不能把没亮说成按钮没找到。
                signal={**signal,'control_position_bound':signal.get('hit') is not None,
                    'control_entity':action.source,'control_card_id':control.card_id if control else None}
                if signal['ready']:
                    return {**signal,'eligibility_state':'enabled','eligibility_evidence':'own_control_positive_signal',
                        'green_observed':True}
                # 缺绿光只能说明启用证据缺失；不能据此声称已识别灰化或禁用。
                if signal.get('reason')=='action_green_outline_absent':
                    return {**signal,'reason':'control_readiness_unknown','eligibility_state':'unknown',
                        'eligibility_evidence':'no_positive_signal','green_observed':False}
                return {**signal,'eligibility_state':'unknown','green_observed':False}
            return dict(ready=True,signal='authorized_object_action')
        if action.kind in (Kind.CHOOSE,Kind.SELECT):
            stage=super().signal(obs,action,image,mappings,ui_state)
            if not stage['ready']:return stage
            projection=getattr(self.mapper,'projection',None)
            spec=next((s for s in getattr(projection,'layouts',()) if s.zone=='choices'),None)
            index=next((i for i,c in enumerate(obs.choices) if c.entity_id==action.source),None)
            box=self.mapper._choice_bounds(spec,self.mapper.viewport,index) if spec and index is not None else None
            present=choice_slot_structure(image,box)
            return dict(ready=present,reason=None if present else 'choice_object_pending',signal='current_candidate_structure')
        return super().signal(obs,action,image,mappings,ui_state)

    def object_evidence(self,obs,action,image,mappings,capture,window,session,*,visual_obs=None):
        from .readiness_window import capture_blockers
        # 日志读取器持有Path，采样记录持有str；同一会话必须使用同一身份。
        session=str(session)
        stamp=capture.get('end');invalid=capture_blockers(capture)
        if not isinstance(stamp,(int,float)) or not 0<=time.time()-stamp<=.75:invalid.append('capture_expired')
        if invalid:
            self.samples.clear();self.admitted.clear();self.history.invalidate()
            return dict(ready=False,reasons=invalid,dependencies=[])
        # 身份/捕获仍核对真实日志；待选战吼的本地预摆放只用于当前画面槽位。
        visual_obs=visual_obs if visual_obs is not None else obs
        deps=dependencies(obs,action);tracker=self.objects();reasons=[];details=[]
        if deps.get('board') and image is not None:
            from .board_input_guard import hover_panel
            panel=hover_panel(image)
            if panel:
                self.samples.pop('board',None);self.admitted.pop('board',None)
                return dict(ready=False,reasons=[panel['reason']],dependencies=[panel])
        context=(session,window,obs.game_id,obs.turn)
        if getattr(self,'context',None)!=context:
            self.context=context;self.samples.clear();self.admitted.clear();self.history.invalidate()
        relation=self.history.inspect(context,dict(session=session,window={'token':window},capture=capture))
        if relation.reason:
            self.samples.clear();self.admitted.clear()
            return dict(ready=False,reasons=[relation.reason],dependencies=[])
        stamp=capture.get('end',0)
        for zone,ids in deps.items():
            captured=(capture.get('client_objects') or {}).get(zone)
            if not captured or not captured.get('ready') or not captured.get('capture_consistent'):
                reasons.append('client_object_capture_pending:'+zone)
                self.admitted.pop(zone,None);self.samples.pop(zone,None)
                details.append(dict(zone=zone,captured=captured));continue
            proof=tracker.inspect(obs,zone) if tracker else dict(ready=False,reason='client_object_channel_missing',zone=zone)
            details.append(proof)
            if proof['ready'] and tuple(map(tuple,captured.get('expected',())))!=tuple(map(tuple,proof['expected'])):
                reasons.append('client_object_capture_pending:'+zone)
                self.admitted.pop(zone,None);self.samples.pop(zone,None);continue
            if not proof['ready']:
                reasons.append(proof['reason']);self.admitted.pop(zone,None);self.samples.pop(zone,None);continue
            # 区域身份/顺序改变会失效；数值变化、无关revision不使位置证据失效。
            key=tuple((c.entity_id,c.position,c.card_id) for c in getattr(visual_obs,zone))
            admitted=self.admitted.get(zone)
            geometries=[];current={};points_covered=zone=='board'
            for eid in ids:
                visible=self.visual(self.mapper,visual_obs,zone,eid,image,mappings)
                details.append(dict(zone=zone,entity=eid,visual=visible))
                if visible is None:reasons.append('local_object_absent:'+zone+':'+str(eid))
                else:
                    current[eid]=tuple(visible['geometry']);geometries.extend(current[eid])
                    points_covered=points_covered and visible.get('point_covered',False)
            if not ids:continue
            if len(geometries)==0 or any(r.startswith('local_object_absent:'+zone+':') for r in reasons):
                self.samples.pop(zone,None);self.admitted.pop(zone,None);continue
            # 布局缓存不能替代当帧像素；只有位置仍符合已证锚点才免去重新等待。
            if admitted and admitted[0]==key and all(eid in admitted[1] and (points_covered or
                    len(points)==len(admitted[1][eid]) and
                    max(abs(a-b) for a,b in zip(points,admitted[1][eid]))<=3)
                    for eid,points in current.items()):
                details.append(dict(zone=zone,layout_evidence='current_point_in_admitted_body' if points_covered else 'current_pixels_match_admitted_positions'))
                continue
            previous_positions=admitted[1] if admitted and admitted[0]==key else {}
            self.admitted.pop(zone,None)
            sample_key=(zone,key,ids);prior=self.samples.get(zone)
            covered_twice=bool(points_covered and prior and len(prior)>3 and prior[3])
            stable=bool(relation.fresh and prior and prior[0]==sample_key and len(prior[2])==len(geometries)
                and .033<=stamp-prior[1]<=2 and (covered_twice or max(abs(a-b) for a,b in zip(prior[2],geometries))<=3))
            if stable:self.admitted[zone]=(key,{**previous_positions,**current})
            else:
                moved=bool(prior and (len(prior[2])!=len(geometries) or max(abs(a-b) for a,b in zip(prior[2],geometries))>3))
                if not prior or prior[0]!=sample_key or moved or stamp-prior[1]>2:self.samples[zone]=(sample_key,stamp,tuple(geometries),points_covered)
                reasons.append('local_object_position_pending:'+zone)
        return dict(ready=not reasons,reasons=reasons,dependencies=details)

    def inspect(self,row,image,mappings,action,ui_state=None,*,proposal=None):
        result=super().inspect(row,image,mappings,action,ui_state,proposal=proposal)
        obs=Observation.from_dict({**row['observation'],'resolutions':[]})
        # 特殊面板/确认沿用独立协议；TARGET同样验证它实际依赖的对象。
        if action.kind not in ORDINARY|{Kind.TARGET}:return {**result,'protocol':'current_interaction_stage'}
        reasons=list(result['reasons'])
        if any(r not in ('action_green_outline_absent','target_stage_pending','control_readiness_unknown','control_explicitly_disabled') for r in reasons):
            self.samples.clear();self.admitted.clear()
            return {**result,'protocol':'action_dependencies','objects':dict(ready=False,reasons=['capture_or_context_invalid'])}
        if not source_geometry_ready(self.mapper,obs,action,mappings):reasons.append('action_source_geometry_unconfirmed')
        if action not in obs.actions:reasons.append('action_not_authorized')
        board_ids=dependencies(obs,action).get('board',())
        if board_ids:
            from .board_input_guard import points_occlusion
            points=[m.rect.point() for m in mappings if m.entity_id in board_ids and m.status=='matched' and m.rect]
            blocked=points_occlusion(image,points)
            if blocked:
                self.samples.pop('board',None);self.admitted.pop('board',None)
                reasons.append(blocked['reason'])
                return {**result,'ready':False,'state':'WAITING','reasons':sorted(set(reasons)),
                    'protocol':'action_dependencies','objects':dict(ready=False,reasons=[blocked['reason']])}
        proof=self.object_evidence(obs,action,image,mappings,row['capture'],row['window']['token'],row.get('session'))
        reasons.extend(proof['reasons'])
        return {**result,'ready':not reasons,'state':'READY' if not reasons else 'WAITING',
            'reasons':sorted(set(reasons)),'protocol':'action_dependencies','objects':proof,
            'green_required':action.kind in CONTROLS,'requires_another_frame':any(r.startswith('local_object_position_pending:') for r in reasons)}
