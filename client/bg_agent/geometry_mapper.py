"""Geometry-first production mapping with explicit visual recovery."""
from collections import defaultdict
from dataclasses import asdict,replace
import hashlib,json,time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .action_geometry import route
from .adaptive_spatial import AdaptiveSpatialMapper
from .capture_broker import FullCaptureRequired
from .chain_metrics import ChainMetrics
from .layout_resolver import LayoutCatalog,LayoutResolver,Viewport,choice_family
from .layout_calibration import fit_layout,normalized
from .mapping import Mapping,StabilityBasis
from .model import Kind,Phase,Action,Card
from .windows import Box


def binding_context(obs):
    ctx=obs.interaction
    return (obs.game_id,str(obs.phase),obs.build,
        tuple(tuple((c.entity_id,c.position,c.card_type,c.tags.get('LITERALLY_UNPLAYABLE')) for c in getattr(obs,z)) for z in ('hand','shop','board','choices')),
        tuple((c.entity_id,c.tags.get('GAME_MODE_BUTTON_SLOT')) for c in obs.controls),
        (ctx.request_id,ctx.source_entity,ctx.owner,ctx.task_list,ctx.selected,ctx.display_state) if ctx else None)


class GeometryMapper:
    def __init__(self,legacy,catalog=None,*,mode='geometry',directory=None,metrics=None):
        if mode not in ('geometry','legacy','compare'):raise ValueError('unknown_geometry_mode')
        self.legacy=legacy;self.mode=mode;self.directory=Path(directory) if directory else None
        self.catalog=catalog or LayoutCatalog.load(Path(__file__).parent/'data/layout_catalog.json')
        self.resolver=LayoutResolver(self.catalog);self.metrics=metrics or ChainMetrics()
        self.last_ui_state={};self.last_diagnostics={};self.last_frame_pixels=None
        self.used_geometry=False;self.used_layout_recovery=False;self.projection=None;self.viewport=None;self.context=None
        self.capture_row=None;self.session=None;self._learning=defaultdict(list);self.comparisons=[]
        self._template_cache={};self._captured_scope=None;self._motion_failures=set()
        self._choice_contours=None

    def __getattr__(self,name):return getattr(self.legacy,name)

    @property
    def cache_metrics(self):
        return {**self.legacy.cache_metrics,'layout_hits':self.resolver.hits,'layout_misses':self.resolver.misses,
                'geometry_active':self.used_geometry,'mode':self.mode}

    def begin_capture(self,row):
        self.capture_row=row
        obs=row['observation'];self.session=str(row.get('session'))+':'+obs['game_id']+':'+row['window']['token']

    def _viewport(self,image,window):
        return Viewport.from_window(image.size,window)

    def capture_region(self,obs,size,window):
        if self.mode!='geometry':return None
        viewport=Viewport.from_window(size,window)
        projection=self.resolver.resolve(obs,viewport,session=self.session)
        if obs.phase!='recruit' or not any(s.zone=='hud' for s in projection.layouts):return None
        r=viewport.render
        # Includes every pixel dependency of the reduced modal checks and Bob
        # anchor. Independently admitted hand contours extend the ROI below.
        x=r.x+round(r.width*.2);y=r.y+round(r.height*.06)
        bottom=r.y+round(r.height*.88)
        for spec in projection.layouts:
            for probe in spec.probes.get('motion',{}).values():
                if probe.get('validated'):
                    b=viewport.box(probe['rect']);x=min(x,max(0,b.x-24));y=min(y,max(0,b.y-24));bottom=max(bottom,min(size[1],b.y+b.height+24))
        return Box(x,y,r.x+r.width-x,bottom-y)

    def reject_motion(self,obs):
        if len(self._motion_failures)>64:self._motion_failures.clear()
        self._motion_failures.add(binding_context(obs))
        self.used_geometry=False
        self.metrics.count('layout_motion_mismatch_recovery')

    def motion_evidence(self,obs,action,image):
        """Required source row and affected destination row, never a cache hit."""
        if not self.used_geometry:return None
        if obs.phase==Phase.CHOICE:
            proof=self._choice_contours
            return proof[2] if proof and proof[0] is image and proof[1]==binding_context(obs) else False
        geometry=route(action,obs,{m.key:m for m in self.projection.mappings})
        probes=[];zones={}
        for key in geometry.required:
            if key in ('region:bob','region:hand'):continue
            if key.startswith(('position:board:','region:board')):
                if obs.board:
                    position=int(key.rsplit(':',1)[-1]) if key.startswith('position:board:') else len(obs.board)+1
                    zones.setdefault('board',set()).update(f'slot:{i}' for i in (position-1,position) if 1<=i<=len(obs.board))
                continue  # Empty receiving wood is separately checked by ensure.
            if key.startswith('entity:'):
                eid=int(key.split(':')[1])
                zone=next((z for z in ('hand','board','shop','choices') if any(c.entity_id==eid for c in getattr(obs,z))),None)
                if zone is None:return None
                cards=tuple(getattr(obs,zone)) if zone=='choices' else tuple(sorted(getattr(obs,zone),key=lambda c:c.position))
                position=next(i+1 for i,c in enumerate(cards) if c.entity_id==eid)
                zones.setdefault(zone,set()).add(f'slot:{position}')
            else:
                spec=next((s for s in self.projection.layouts if key in s.regions),None)
                probe=spec.probes.get('motion',{}).get(key) if spec else None
                if not probe:return None
                probes.append(probe)
        for zone,required in zones.items():
            spec=next((s for s in self.projection.layouts if s.zone==zone),None)
            if not spec or spec.probes.get('geometry_only_evidence'):return None
            admitted=spec.probes.get('motion',{})
            if not required<=admitted.keys():return None
            # Validate the required source and receiving neighbours. Particles
            # on an unrelated slot must not send this action into recovery.
            probes.extend(admitted[key] for key in sorted(required))
        from .layout_motion import sample_edges
        return sample_edges(image,self.viewport,probes) or False

    def _small_scene(self,image,viewport):
        r=viewport.render;scale=min(1,640/r.width);canvas=Image.new('RGB',(round(r.width*scale),round(r.height*scale)))
        # Recruit ribbons can contain fine disconnected purple pieces.
        # Downsampling can erase them, so this small critical region
        # keeps native pixels and the original detector thresholds.
        ribbon=image.crop((r.x+round(r.width*.27),r.y+round(r.height*.38),r.x+round(r.width*.73),r.y+round(r.height*.57)))
        hsv=cv2.cvtColor(np.asarray(ribbon),cv2.COLOR_RGB2HSV)
        purple=cv2.inRange(hsv,np.array([130,105,45]),np.array([175,255,255]))
        purple=cv2.morphologyEx(purple,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        contours,_=cv2.findContours(purple,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if any(w>r.width*.22 and r.height*.035<h<r.height*.16 and w/h>3 for x,y,w,h in (cv2.boundingRect(c) for c in contours)):
            return dict(mode='recruit_transition',protocol='native_local_recruit_ribbon')
        bounds=(r.x+round(r.width*.2),r.y+round(r.height*.06),r.x+r.width,r.y+round(r.height*.88))
        patch=image.crop(bounds)
        patch=patch.resize((round((bounds[2]-bounds[0])*scale),round((bounds[3]-bounds[1])*scale)),Image.Resampling.BOX)
        canvas.paste(patch,(round((bounds[0]-r.x)*scale),round((bounds[1]-r.y)*scale)))
        return AdaptiveSpatialMapper.ui_state(canvas)

    def _recruit_ready(self,image,viewport):
        entry=next((e for e,t in self.legacy.anchors if e['key']=='scene:recruit'),None)
        if entry is None:return False
        x,y,w,h=entry['patch_uv'];box=viewport.box((x,y,w,h));key=(box.width,box.height)
        original=next(t for e,t in self.legacy.anchors if e['key']=='scene:recruit')
        if key not in self._template_cache:self._template_cache[key]=cv2.resize(original,key)
        template=self._template_cache[key]
        patch=np.asarray(image.crop((box.x,box.y,box.x+box.width,box.y+box.height)).convert('L'))
        if patch.shape!=template.shape:return False
        score=float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
        error=float(np.mean(np.abs(patch.astype(float)-template.astype(float))))
        if score>=.87 and error<=20:return True
        # 招募中的金币粒子可能遮住旅店老板局部；分块要求多数且横纵分布，
        # 不能用一个相似角落或整个画面静止代替招募场景证据。
        height,width=patch.shape;matched=[]
        for row in range(4):
            for column in range(4):
                area=np.s_[row*height//4:(row+1)*height//4,column*width//4:(column+1)*width//4]
                part,reference=patch[area],template[area]
                if min(part.shape)<3:continue
                score=float(cv2.matchTemplate(part,reference,cv2.TM_CCOEFF_NORMED)[0,0])
                error=float(np.mean(np.abs(part.astype(float)-reference.astype(float))))
                if score>=.87 and error<=20:matched.append((row,column))
        return len(matched)>=8 and len({r for r,c in matched})>=3 and len({c for r,c in matched})>=3

    def _bounds(self,spec,viewport,index):
        raw=spec.probes.get('slot_bounds',{}).get(f'slot:{index+1}')
        return viewport.box(raw) if raw else self.resolver.regions(spec,viewport).get(f'slot:{index+1}')

    def _choice_bounds(self,spec,viewport,index):
        raw=spec.probes.get('visibility_bounds',{}).get(f'slot:{index+1}')
        return viewport.box(raw) if raw else self._bounds(spec,viewport,index)

    @staticmethod
    def _rail(image,box):
        from .choice_visibility import frame_rails
        return frame_rails(image,box)

    @staticmethod
    def _selection_rail(image,box):
        from .choice_visibility import frame_rails
        return frame_rails(image,box)

    def _choice_ready(self,obs,image,projection,viewport):
        if not obs.interaction or obs.interaction.display_state in ('waiting','hidden'):return False
        spec=next((s for s in projection.layouts if s.zone=='choices'),None)
        if spec is None:return False
        boxes=[self._choice_bounds(spec,viewport,i) for i in range(len(obs.choices))]
        if getattr(self,'uses_green_signal',False):
            from .stage_admission import panel_visible
            return spec.status!='candidate' and panel_visible(obs,image)
        if choice_family(obs)=='multi' and not any(s.zone=='hud' and s.status!='candidate' and 'control:confirm' in s.regions for s in projection.layouts):return False
        if spec.status=='candidate':
            if choice_family(obs)=='quest':
                from .live_choices import quest_links
                if not quest_links(obs):return False
            visible=all(b and self._selection_rail(image,b) for b in boxes)
        else:
            from .choice_visibility import shown
            visible=shown(image,boxes,choice_family(obs))
        if visible:
            if getattr(self,'uses_green_signal',False) and choice_family(obs)!='hero':return True
            from .choice_visibility import rail_positions
            points=rail_positions(image,boxes,choice_family(obs))
            if points:
                self._choice_contours=(image,binding_context(obs),points)
                return True
        return False

    def _verified_legacy_choices(self,obs,image,maps):
        from .choice_visibility import strip_unproven_choice_maps,STRONG_SCENES
        if obs.phase==Phase.CHOICE and not any(m.key=='scene:choice' and m.reason in STRONG_SCENES for m in maps):
            spec=next((s for s in self.projection.layouts if s.zone=='choices'),None)
            visible={m.entity_id:m.rect for m in maps if m.status=='matched' and m.rect and m.entity_id is not None}
            if spec and self._choice_ready(obs,image,self.projection,self.viewport):
                inside=True
                for i,card in enumerate(obs.choices):
                    box=visible.get(card.entity_id);bounds=self._choice_bounds(spec,self.viewport,i)
                    if box is None or bounds is None:inside=False;break
                    x,y=box.point()
                    inside &= bounds.x+bounds.width*.1<x<bounds.x+bounds.width*.9 and bounds.y+bounds.height*.1<y<bounds.y+bounds.height*.9
                if inside:
                    maps=[replace(m,reason='independent_local_choice_frames_and_measured_legacy_points') if m.key=='scene:choice' else m for m in maps]
        return strip_unproven_choice_maps(obs,maps)

    def map(self,obs,image,window,*,blocked=False):
        self.used_geometry=False;self.used_layout_recovery=False;self.context=None;self.last_frame_pixels=None
        self._choice_contours=None
        self.viewport=self._viewport(image,window)
        with self.metrics.measure('geometry_resolve'):
            projection=self.resolver.resolve(obs,self.viewport,session=self.session)
        if self.mode=='geometry' and getattr(self,'uses_green_signal',False) and obs.phase in (Phase.RECRUIT,Phase.CHOICE):
            # 坐标只来自当前日志和已审核布局；视觉仅判断界面阶段。
            self.projection=projection
            self.last_ui_state=self._small_scene(image,self.viewport)
            # 已审核布局只证明坐标；战斗画面同样可能对应下一回合日志，必须独立确认招募场景。
            valid=not blocked and (any(s.zone=='hud' for s in projection.layouts) and self._recruit_ready(image,self.viewport) if obs.phase==Phase.RECRUIT
                else self._choice_ready(obs,image,projection,self.viewport))
            self.last_diagnostics=dict(geometry=valid,layouts=[s.id for s in projection.layouts],
                missing=list(projection.missing),reason=None if valid else 'layout_or_stage_unavailable')
            self.used_geometry=valid;self.context=binding_context(obs) if valid else None
            if valid:return list(projection.mappings)
            return []
        if obs.phase==Phase.CHOICE and not any(s.zone=='choices' for s in projection.layouts) and self.mode=='geometry':
            projection=self.resolver.resolve(obs,self.viewport,session=self.session,recovery=True)
        self.projection=projection
        if obs.phase==Phase.CHOICE and obs.interaction and obs.interaction.display_state in ('waiting','hidden'):
            self.last_ui_state=dict(mode='choice_transition')
            self.last_diagnostics=dict(geometry=False,choice_layout_waiting=True,reason='client_choice_not_displayed')
            self.metrics.count('choice_layout_wait')
            return []
        if not blocked and self.mode!='legacy' and binding_context(obs) not in self._motion_failures:
            with self.metrics.measure('scene_probe'):
                if obs.phase==Phase.RECRUIT:
                    self.last_ui_state=self._small_scene(image,self.viewport)
                    valid=any(s.zone=='hud' for s in projection.layouts) and self._recruit_ready(image,self.viewport)
                    if obs.shop and not any(m.entity_id in {c.entity_id for c in obs.shop} for m in projection.mappings):valid=False
                elif obs.phase==Phase.CHOICE:
                    self.last_ui_state=self._small_scene(image,self.viewport)
                    valid=self.last_ui_state.get('mode')!='pause_menu' and self._choice_ready(obs,image,projection,self.viewport)
                elif obs.phase in (Phase.MENU,Phase.GAME_OVER):
                    self.last_ui_state={}
                    valid=self.control(obs,image,window,'enter' if obs.phase==Phase.MENU else 'dismiss') is not None
                else:valid=False
            if self.mode=='geometry' and obs.phase==Phase.CHOICE and not valid and any(s.zone=='choices' for s in projection.layouts):
                from .choice_visibility import modal_hint
                if not modal_hint(image):
                    self.last_diagnostics=dict(geometry=False,choice_layout_waiting=True,
                        layouts=[s.id for s in projection.layouts],reason='logged_choice_not_yet_visible')
                    self.metrics.count('choice_layout_wait')
                    return []
            if valid and self.mode=='geometry':
                self.used_layout_recovery=any(s.status=='candidate' for s in projection.layouts)
                self.used_geometry=not self.used_layout_recovery;self.context=binding_context(obs)
                if self.used_layout_recovery:self.metrics.count('layout_local_recovery')
                self.last_diagnostics=dict(geometry=self.used_geometry,local_recovery=self.used_layout_recovery,unverified_layouts=[s.id for s in projection.layouts if s.status=='candidate'],layouts=[s.id for s in projection.layouts],missing=list(projection.missing),badges=[])
                self.metrics.count('geometry_projection' if self.used_geometry else 'layout_local_recovery_projection');return list(projection.mappings)
        if not getattr(image,'is_full',True):raise FullCaptureRequired('geometry_probe_failed_requires_visual_recovery')
        self.metrics.count('full_localization')
        with self.metrics.measure('visual_recovery'):maps=self.legacy.map(obs,image,window,blocked=blocked)
        self.last_ui_state=self.legacy.last_ui_state;self.last_diagnostics=self.legacy.last_diagnostics
        self.last_frame_pixels=self.legacy.last_frame_pixels
        maps=self._verified_legacy_choices(obs,image,maps)
        if self.mode=='compare':
            visual={m.key:m for m in maps if m.status=='matched' and m.rect}
            differences=[]
            for m in projection.mappings:
                old=visual.get(m.key)
                if old and not (old.rect.x<=m.rect.point()[0]<=old.rect.x+old.rect.width and old.rect.y<=m.rect.point()[1]<=old.rect.y+old.rect.height):
                    differences.append(m.key)
            self.comparisons.append(dict(game_id=obs.game_id,revision=obs.revision,differences=differences))
            self.comparisons=self.comparisons[-256:]
        return maps

    def authorize_geometry(self,obs,action,point=None):
        if not self.used_geometry or self.context!=binding_context(obs):return False
        visible={m.key:m for m in self.projection.mappings if m.status=='matched' and m.rect}
        geometry=route(action,obs,visible)
        if geometry.error or any(key not in visible for key in geometry.required):return False
        if point is not None:
            source=visible.get(geometry.source)
            if source is None or max(abs(a-b) for a,b in zip(source.rect.point(),point))>2:return False
        return True

    def reuse_projection(self,obs,window):
        projection=self.resolver.resolve(obs,self.viewport,session=self.session)
        self.projection=projection;self.context=binding_context(obs);self.used_geometry=True
        self.last_diagnostics=dict(geometry=True,reused_scene_evidence=True,layouts=[s.id for s in projection.layouts],
            missing=list(projection.missing),badges=[])
        self.last_frame_pixels=None
        return list(projection.mappings)

    def target_box(self,obs,entity):
        if not self.used_geometry:return None
        if obs.hero and entity==obs.hero.entity_id:
            hud=next((s for s in self.projection.layouts if s.zone=='hud'),None)
            bounds=hud.probes.get('hero_bounds') if hud else None
            return self.viewport.box(bounds) if bounds else None
        for spec in self.projection.layouts:
            group=getattr(obs,spec.zone,())
            for index,card in enumerate(sorted(group,key=lambda c:c.position)):
                if card.entity_id==entity:
                    if spec.zone=='hand':
                        probe=spec.probes.get('motion',{}).get(f'slot:{index+1}')
                        return self.viewport.box(probe['rect']) if probe and probe.get('validated') else None
                    return self._bounds(spec,self.viewport,index)
        return None

    def target_support(self,obs,entity,image,box):
        self.last_target_evidence=None
        if any(c.entity_id==entity for c in obs.hand):
            hsv=cv2.cvtColor(np.asarray(image.crop((box.x,box.y,box.x+box.width,box.y+box.height))),cv2.COLOR_RGB2HSV)
            gold=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=36)&(hsv[:,:,1]>130)&(hsv[:,:,2]>150)
            return float(np.mean(np.any(gold,axis=1))) if gold.size else 0
        support=self.target_rim_support(image,box)
        return support

    @staticmethod
    def target_rim_support(image,box):
        """Same eligibility pixels as the legacy check, acquired locally."""
        cx,cy=box.point();radius=max(2,round(box.width*.075))
        x0=max(0,round(cx-box.width*.43)-radius-1);y0=max(0,round(cy-box.height*.43)-radius-1)
        x1=min(image.width,round(cx+box.width*.43)+radius+2);y1=min(image.height,round(cy+box.height*.43)+radius+2)
        hsv=cv2.cvtColor(np.asarray(image.crop((x0,y0,x1,y1))),cv2.COLOR_RGB2HSV);hits=[]
        for angle in np.linspace(0,2*np.pi,48,endpoint=False):
            x=round(cx+box.width*.43*np.cos(angle));y=round(cy+box.height*.43*np.sin(angle))
            patch=hsv[max(0,y-radius)-y0:min(image.height,y+radius+1)-y0,max(0,x-radius)-x0:min(image.width,x+radius+1)-x0]
            hits.append(bool(patch.size and np.any((patch[:,:,0]>=10)&(patch[:,:,0]<=36)&(patch[:,:,1]>130)&(patch[:,:,2]>150))))
        return sum(hits)/len(hits)

    def ensure(self,obs,action,image,window,maps):
        known=self.authorize_geometry(obs,action)
        # Reviewed coordinates establish where to release. Eligibility is
        # independently gated by action dependencies and stage signals. Decorative
        # colours at an already-bound point cannot revoke or move that slot.
        if known:return maps
        if self.mode=='geometry' and getattr(self,'uses_green_signal',False):
            raise ValueError('unknown_mapping:reviewed_action_layout')
        if self.mode=='legacy' or not self.used_geometry:return maps
        if not getattr(image,'is_full',True):raise FullCaptureRequired('uncalibrated_action_requires_full_capture')
        from .layout_study import visual_projection
        self.metrics.count('action_visual_recovery')
        with self.metrics.measure('visual_recovery'):result=visual_projection(self.legacy,obs,image,window)
        self.used_geometry=False;self.context=None
        self.last_ui_state=self.legacy.last_ui_state;self.last_diagnostics=self.legacy.last_diagnostics
        self.last_frame_pixels=self.legacy.last_frame_pixels
        result=self._verified_legacy_choices(obs,image,result)
        self.record_visual(obs,image,result,self.capture_row)
        return result

    @staticmethod
    def drop_ready(image,box):
        """A tiny empty receiving-gap check, never card-frame localization."""
        x,y=box.point();bounds=(round(x)-4,round(y)-8,round(x)+4,round(y)+8)
        if min(bounds[:2])<0 or bounds[2]>image.width or bounds[3]>image.height:return False
        rgb=np.asarray(image.crop(bounds));hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        h,s,v=np.median(hsv.reshape(-1,3),axis=0)
        edges=float(np.mean(cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),50,130)>0))
        return bool(8<=h<=32 and s>70 and v>65 and edges<.08)

    def record_visual(self,obs,image,maps,row):
        if self.mode=='legacy' or self.used_geometry or not row or not obs.ready or row.get('blockers'):return
        if obs.phase==Phase.RECRUIT and obs.client_task_list is not None:return
        if not getattr(image,'is_full',True) or self.last_ui_state.get('mode') in ('pause_menu','recruit_transition','targeting_suspected'):return
        from .layout_study import observations
        capture=str(row.get('session'))+':'+str(row.get('id'))+':'+str(row['capture']['end'])
        for key,value in observations(obs,image,maps,capture,obs.game_id,None,self.legacy.last_diagnostics):
            family,zone,count,variant,build,aspect=key
            if self.catalog.find(family,zone,count,variant,build,self.viewport,self.session):continue
            history=self._learning[key]
            if history and history[-1]['capture']==capture:continue
            if history and row['capture']['end']-history[-1]['time']<.05:continue
            history.append({**value,'time':row['capture']['end']});del history[:-4]
            spec=fit_layout(*key,history,session=self.session)
            if spec:
                self.metrics.count('calibration_candidate')
                if self.directory:
                    from .recording import write_json
                    candidate=asdict(spec);candidate['status']='candidate'
                    write_json(self.directory/('layout-candidate-'+spec.id+'.json'),dict(schema=1,layouts=[candidate],executable=False))

    def control(self,obs,image,window,name):
        viewport=self._viewport(image,window);projection=self.resolver.resolve(obs,viewport,session=self.session,recovery=self.used_layout_recovery)
        box=next((m.rect for m in projection.mappings if m.key=='control:'+name),None)
        if not box or self.mode!='geometry':return None
        hsv=cv2.cvtColor(np.asarray(image.crop((box.x,box.y,box.x+box.width,box.y+box.height))),cv2.COLOR_RGB2HSV)
        colour=((hsv[:,:,0]>=125)&(hsv[:,:,0]<=174)&(hsv[:,:,1]>60)&(hsv[:,:,2]>45) if name=='dismiss' else
            (hsv[:,:,0]>=80)&(hsv[:,:,0]<=125)&(hsv[:,:,1]>65)&(hsv[:,:,2]>100))
        return box if np.mean(colour)>.15 else None

    def project_interaction(self,obs,image,maps):
        if not (self.used_geometry or self.used_layout_recovery) or obs.phase!=Phase.CHOICE:return obs,maps
        family=choice_family(obs)
        if family=='hero':obs=replace(obs,warnings=tuple(w for w in obs.warnings if w!='gold_unknown'))
        if family=='quest':
            from .live_choices import quest_links
            if quest_links(obs):obs=replace(obs,interaction=replace(obs.interaction,protocol='verified_quest_pair'))
        if family=='trinket':obs=replace(obs,interaction=replace(obs.interaction,protocol='verified_trinket_select'))
        if family=='multi':
            spec=next(s for s in self.projection.layouts if s.zone=='choices');selected=[]
            for i,card in enumerate(obs.choices):
                b=self._bounds(spec,self.viewport,i);states=[]
                for x in (b.x,b.x+round(b.width*.92)):
                    hsv=cv2.cvtColor(np.asarray(image.crop((x,b.y+round(b.height*.28),x+max(4,round(b.width*.08)),b.y+round(b.height*.7)))),cv2.COLOR_RGB2HSV)
                    valid=(hsv[:,:,1]>95)&(hsv[:,:,2]>100)
                    green=np.mean(np.any(valid&(hsv[:,:,0]>=35)&(hsv[:,:,0]<=112),axis=1))
                    gold=np.mean(np.any(valid&(hsv[:,:,0]>=12)&(hsv[:,:,0]<35),axis=1))
                    states.append('selected' if gold>.7 and green<.15 else 'clear' if green>.7 and gold<.15 else 'unknown')
                if len(set(states))!=1 or states[0]=='unknown':
                    return replace(obs,ready=False,actions=(),warnings=obs.warnings+('multi_choice_selection_state_unknown',)),maps
                if states[0]=='selected':selected.append(card.entity_id)
            selected=tuple(selected);ctx=replace(obs.interaction,selected=selected,protocol='verified_multi_select')
            actions=[]
            if len(selected)<ctx.max_select:
                actions=[Action(Kind.SELECT,c.entity_id,choice_id=ctx.request_id,selected=selected+(c.entity_id,),evidence='server_choices') for c in obs.choices if c.entity_id not in selected]
            if ctx.min_select<=len(selected)<=ctx.max_select and self.control(obs,image,{'client':{}},'confirm'):
                actions.append(Action(Kind.CONFIRM,choice_id=ctx.request_id,selected=selected,evidence='server_choices'))
            obs=replace(obs,interaction=ctx,actions=tuple(actions),warnings=tuple(w for w in obs.warnings if w!='multi_select_choice_requires_adapter'))
        self.context=binding_context(obs)
        return obs,maps

    supports_parent_layout=True

    def branch_slots(self,obs,image,window,options,chosen,capture,*,parent_kind=None):
        """A separate family; ordinary Discover calibration never substitutes."""
        from .live_interactions import branch_selection_regions
        viewport=self._viewport(image,window);count=len(options)
        family={'MINION':'branch_minion','SPELL':'branch_spell','BATTLEGROUND_SPELL':'branch_spell'}.get(parent_kind,'branch')
        spec=self.catalog.find(family,'choices',count,'default',obs.build,viewport,self.session)
        if self.mode=='geometry' and getattr(self,'uses_green_signal',False):
            from .stage_admission import panel_visible
            if not spec or spec.status=='candidate':return []
            boxes=self.resolver.regions(spec,viewport)
            self.last_branch_bounds=tuple(self._bounds(spec,viewport,i) for i in range(count))
            ctx=obs.interaction
            if ctx and ctx.display_state in ('hidden','waiting'):return []
            from .dependency_readiness import choice_slot_structure
            if not choice_slot_structure(image,self.last_branch_bounds[options.index(chosen)]):return []
            if not panel_visible(obs,image):
                # 无标题的抉择以任一候选绿光证明分支阶段；不要求选中项自身亮起。
                from .green_readiness import green_outline
                hits=[boxes[f'slot:{i+1}'] for i in range(count)]
                if not any(green_outline(image,b,bounds=self._bounds(spec,viewport,i),zone='choices',
                    other_hits=[h.point() for j,h in enumerate(hits) if j!=i])['ready'] for i,b in enumerate(hits)):return []
            return [boxes[f'slot:{i+1}'] for i in range(count)]
        if self.mode=='geometry' and spec:
            boxes=self.resolver.regions(spec,viewport)
            if self._rail(image,self._bounds(spec,viewport,options.index(chosen))):
                return [boxes[f'slot:{i+1}'] for i in range(count)]
        boxes=branch_selection_regions(image,count,options.index(chosen))
        self.metrics.count('branch_visual_recovery')
        if boxes and self.mode!='legacy' and self.session:
            key=('branch','choices',count,'default',obs.build,round(viewport.aspect,6))
            history=self._learning[key];stamp=capture['end']
            if not history or stamp-history[-1]['time']>=.05:
                bounds={}
                for i,b in enumerate(boxes):
                    bounds[f'slot:{i+1}']=normalized(Box(round(b.x-b.width*.75),round(b.y-b.height*.25/.3),round(b.width/.4),round(b.height/.3)),image.size)
                history.append(dict(capture=str(stamp),time=stamp,group=obs.game_id,sha256=None,
                    regions={f'slot:{i+1}':normalized(b,image.size) for i,b in enumerate(boxes)},bounds=bounds,locked=[False]*count))
                del history[:-4]
                learned=fit_layout(*key,history,session=self.session)
                if learned and self.directory:
                    from .recording import write_json
                    candidate=asdict(learned);candidate['status']='candidate'
                    write_json(self.directory/('layout-candidate-'+learned.id+'.json'),dict(schema=1,layouts=[candidate],executable=False))
        return boxes

    def confirmation(self,obs,image,window):
        """Return local selection evidence only when its own family is known."""
        family=choice_family(obs);viewport=self._viewport(image,window)
        projection=self.resolver.resolve(obs,viewport,session=self.session)
        spec=next((s for s in projection.layouts if s.zone=='choices'),None)
        confirm=next((m.rect for m in projection.mappings if m.key=='control:confirm'),None)
        if self.mode!='geometry' or not spec or not confirm:return None
        if family=='hero':
            # Detect the selection halo inside each independently calibrated
            # slot; this never searches the screen for card locations.
            selected=[]
            for i in range(len(obs.choices)):
                b=self._bounds(spec,viewport,i);pad=round(b.height*.35)
                x0=max(0,b.x-round(b.width*.2));y0=max(0,b.y-pad)
                patch=image.crop((x0,y0,min(image.width,b.x+round(b.width*1.2)),min(image.height,b.y+round(b.height*1.5))))
                hsv=cv2.cvtColor(np.asarray(patch),cv2.COLOR_RGB2HSV)
                mask=cv2.inRange(hsv,np.array([35,160,160]),np.array([90,255,255]))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    x,y,w,h=cv2.boundingRect(c);x+=x0;y+=y0
                    if (.65*b.width<w<1.45*b.width and .55*b.height<h<1.5*b.height and
                        b.x-.2*b.width<x<b.x+.15*b.width and b.y-.35*b.height<y<b.y+.12*b.height and
                        x+w>b.x+.55*b.width and y+h>b.y+.55*b.height and cv2.contourArea(c)>.025*b.width*b.height):
                        selected.append(obs.choices[i].entity_id);break
            return self.control(obs,image,window,'confirm'),tuple(selected)
        if family=='trinket':
            probes=spec.probes.get('local',{})
            if any(f'selection:{i+1}' not in probes for i in range(len(obs.choices))):return None
            selected=[]
            for i,card in enumerate(obs.choices):
                b=viewport.box(probes[f'selection:{i+1}']);unit=b.height/1.2
                hsv=cv2.cvtColor(np.asarray(image.crop((b.x,b.y,b.x+b.width,b.y+b.height))),cv2.COLOR_RGB2HSV)
                green=((hsv[:,:,0]>40)&(hsv[:,:,0]<85)&(hsv[:,:,1]>150)&(hsv[:,:,2]>180)).astype('uint8')
                _,_,stats,_=cv2.connectedComponentsWithStats(green)
                span=sum(int(r[2]) for r in stats[1:] if r[2]>unit*.7 and r[4]>unit*2)/b.width
                if span>.6:selected.append(card.entity_id)
            return self.control(obs,image,window,'confirm'),tuple(selected)
        return None

    def confirmation_region(self,obs,window):
        if self.mode!='geometry':return None
        viewport=Viewport.from_window((window['client']['width'],window['client']['height']),window)
        projection=self.resolver.resolve(obs,viewport,session=self.session)
        family=choice_family(obs);spec=next((s for s in projection.layouts if s.zone=='choices'),None)
        confirm=next((m.rect for m in projection.mappings if m.key=='control:confirm'),None)
        if not spec or not confirm or family not in ('hero','trinket'):return None
        boxes=[confirm]
        if family=='hero':
            for i in range(len(obs.choices)):
                b=self._bounds(spec,viewport,i)
                x0=max(0,b.x-round(b.width*.2));y0=max(0,b.y-round(b.height*.35))
                x1=min(viewport.width,b.x+round(b.width*1.2));y1=min(viewport.height,b.y+round(b.height*1.5))
                boxes.append(Box(x0,y0,x1-x0,y1-y0))
        else:
            probes=spec.probes.get('local',{})
            if any(f'selection:{i+1}' not in probes for i in range(len(obs.choices))):return None
            boxes.extend(viewport.box(probes[f'selection:{i+1}']) for i in range(len(obs.choices)))
        x=min(b.x for b in boxes);y=min(b.y for b in boxes)
        return Box(x,y,max(b.x+b.width for b in boxes)-x,max(b.y+b.height for b in boxes)-y)
