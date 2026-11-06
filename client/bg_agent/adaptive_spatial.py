"""Post-holdout regression improvements; frozen spatial.py stays reproducible.

Choice rows are grouped by observed badge scale/alignment rather than a fixed
vertical band. A single spell hand slot needs both a generic currency marker
and an attached card-top edge, never an art/card-id template.
"""
import json
import hashlib
from pathlib import Path
from dataclasses import replace
import cv2
import numpy as np
from PIL import Image
from .spatial import SpatialMapper
from .mapping import Mapping,StabilityBasis
from .windows import Box
from .geometry_cache import EvidenceCache,AnchorCache,freeze
from .geometry import CenteredRow
from .hand_fan import diagnose_fan
from .hand_cap import portrait_cap
from .trinket_mapping import TrinketMapper


class _FramePixels:
    """One immutable RGB array per mapping call, never reused across captures.

    PIL's array interface serializes the entire image on every np.asarray().
    Delegate image operations, but let independent detectors share the exact
    same captured pixels without repeating that full-frame conversion.
    """
    def __init__(self,image,evidence_cache=None):
        self.image=image;self.pixels=np.asarray(image)
        self.evidence_cache=evidence_cache
        self._fingerprints={}
        self._rim_support_cache={}
        self._hand_frame_cache={}
        self._hsv=None

    def memo(self,name,context,compute,*,bounds=None):
        if self.evidence_cache is None:return compute()
        bounds=bounds or (0,0,self.size[0],self.size[1])
        if bounds not in self._fingerprints:
            x0,y0,x1,y1=bounds
            pixels=np.ascontiguousarray(self.pixels[y0:y1,x0:x1])
            self._fingerprints[bounds]=hashlib.sha256(pixels).digest()
        key=(name,self.size,bounds,freeze(context),self._fingerprints[bounds])
        return self.evidence_cache.resolve(key,compute)

    @property
    def hsv_pixels(self):
        if self._hsv is None:self._hsv=cv2.cvtColor(self.pixels,cv2.COLOR_RGB2HSV)
        return self._hsv

    @property
    def __array_interface__(self):return self.pixels.__array_interface__

    def __getattr__(self,name):return getattr(self.image,name)


class AdaptiveSpatialMapper(SpatialMapper):
    @property
    def cache_metrics(self):
        return dict(pixel_hits=self.evidence_cache.hits,pixel_misses=self.evidence_cache.misses,
            pixel_entries=len(self.evidence_cache.entries),anchor_patch_hits=self.anchor_cache.hits,
            anchor_searches=self.anchor_cache.searches,hud_locations=len(self.hud_locations))

    @staticmethod
    def target_rim_support(image,box):
        """Eligible target frames are orange/gold, not ordinary green rims."""
        hsv=getattr(image,'hsv_pixels',None)
        if hsv is None:hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
        cx,cy=box.point();hits=[]
        radius=max(2,round(box.width*.075))
        for angle in np.linspace(0,2*np.pi,48,endpoint=False):
            x=round(cx+box.width*.43*np.cos(angle));y=round(cy+box.height*.43*np.sin(angle))
            patch=hsv[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
            hits.append(bool(patch.size and np.any((patch[:,:,0]>=10)&(patch[:,:,0]<=36)&(patch[:,:,1]>130)&(patch[:,:,2]>150))))
        return sum(hits)/len(hits)

    def __init__(self,profile,hand_marker=None):
        super().__init__(profile)
        self.hand_marker=Path(hand_marker) if hand_marker else None
        self.marker=None
        self.last_ui_state={}
        self.evidence_cache=EvidenceCache()
        self.anchor_cache=AnchorCache()
        self.hud_locations={}
        self.geometry_scope=None
        trinket_profile=self.profile_path.parent.parent/'v05_trinket'/'profile.json'
        self.trinkets=TrinketMapper(trinket_profile) if trinket_profile.exists() else None
        if self.hand_marker:
            self.marker_spec=json.loads(self.hand_marker.read_text(encoding='utf-8'))
            self.marker=np.asarray(Image.open(self.hand_marker.parent/self.marker_spec['template']).convert('L'))

    @staticmethod
    def ui_state(image):
        pixels=np.asarray(image);height,width=pixels.shape[:2]
        hsv=getattr(image,'hsv_pixels',None)
        if hsv is None:hsv=cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV)
        from .modal_detection import pause_menu,targeting_arrow
        if pause_menu(hsv):return dict(mode='pause_menu',protocol='game_menu_geometry',evidence='three aligned gold buttons with red top control; refusal only')
        # Recruit-start ribbon can be static while the HUD behind it is blurred
        # and unclickable. Pixel stability must not authorize this modal.
        ribbon=hsv[round(height*.38):round(height*.57),round(width*.27):round(width*.73)]
        purple=cv2.inRange(ribbon,np.array([130,105,45]),np.array([175,255,255]))
        purple=cv2.morphologyEx(purple,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        contours,_=cv2.findContours(purple,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        recruit_banner=any(w>width*.22 and height*.035<h<height*.16 and w/h>3
            for x,y,w,h in (cv2.boundingRect(c) for c in contours))
        if recruit_banner:return dict(mode='recruit_transition',protocol='recruit_ribbon',evidence='wide central purple recruit ribbon; HUD not actionable')
        region=hsv[round(height*.3):round(height*.7),round(width*.2):round(width*.85)]
        low_chroma=float(np.mean(region[:,:,1]<75))
        hud=hsv[round(height*.06):round(height*.25),round(width*.30):round(width*.72)]
        hud_low_chroma=float(np.mean(hud[:,:,1]<75))
        band=hsv[round(height*.4):round(height*.72),round(width*.2):round(width*.85)]
        red=(((band[:,:,0]<8)|(band[:,:,0]>170))&(band[:,:,1]>130)&(band[:,:,2]>110)).astype('uint8')
        _,_,stats,_=cv2.connectedComponentsWithStats(red)
        largest=float(max(stats[1:,4],default=0)/(width*height))
        # Every legal minion stays colored in target mode. On a full board,
        # desaturation of the non-targetable HUD is the stronger scene cue.
        arrow=targeting_arrow(hsv)
        suspected=((low_chroma>.65 or low_chroma>.35 and hud_low_chroma>.85) and largest>.001) or arrow
        return dict(mode='targeting_suspected' if suspected else 'unknown',low_chroma_fraction=low_chroma,
                    hud_low_chroma_fraction=hud_low_chroma,
                    targeting_arrow_visible=arrow,
                    largest_red_component_fraction=largest,protocol='unknown',
                    evidence='desaturated scene with red component or aligned red pointer dashes; refusal signal only')

    @staticmethod
    def shop_badge_regions(obs,image,bob,badges):
        """A complete row of equally sized shields, excluding purple card art."""
        cards=sorted(obs.shop,key=lambda c:c.position)
        if not cards or [c.position for c in cards]!=list(range(1,len(cards)+1)):return []
        unit=bob.height;width,height=image.size
        candidates=[Box(**b) for b in badges]
        candidates=[b for b in candidates if bob.y+unit*.8<b.y<bob.y+unit*1.45
                    and bob.x-unit*3<b.x<bob.x+unit*3
                    and .10*unit<b.width<.28*unit and .12*unit<b.height<.4*unit]
        rows={}
        for seed in candidates:
            row=sorted((b for b in candidates if abs(b.y-seed.y)<unit*.06
                        and .8<b.width/seed.width<1.25 and .8<b.height/seed.height<1.25),key=lambda b:b.x)
            if len(row)!=len(cards):continue
            if len(row)>1:
                gaps=np.diff([b.point()[0] for b in row])
                if min(gaps)<unit*.45 or max(gaps)>unit*1.4 or max(gaps)/min(gaps)>1.2:continue
            boxes=[Box(round(b.x-b.width*.8),b.y,round(b.width*2.6),round(b.height*3.7)) for b in row]
            if any(b.x<0 or b.y<0 or b.x+b.width>width or b.y+b.height>height for b in boxes):continue
            rows[tuple((b.x,b.y,b.width,b.height) for b in row)]=boxes
        return next(iter(rows.values())) if len(rows)==1 else []

    def choice_regions(self,badges,count,image):
        if count<2:return []
        width,height=image.size
        candidates=[b for b in badges if .04*height<b.height<.14*height and .64<b.width/b.height<1.06]
        groups=[]
        for seed in candidates:
            group=sorted([b for b in candidates if abs(b.y-seed.y)<height*.015 and .85< b.height/seed.height<1.18],key=lambda b:b.x)
            if len(group)!=count or group[-1].x-group[0].x<width*.25:continue
            gaps=np.diff([b.x for b in group])
            if len(gaps)>1 and max(gaps)/min(gaps)>1.18:continue
            if any(abs(seed.y-g[0].y)<height*.015 for g in groups):continue
            groups.append(group)
        if len(groups)!=1:return []
        return [Box(b.x,round(b.y-b.height*.1),round(b.width*4.2),round(b.height*6.3)) for b in groups[0]]

    def single_spell_region(self,image):
        if self.marker is None:return None,{}
        width,height=image.size;spec=self.marker_spec
        gray=np.asarray(image.convert('L'))
        template=cv2.resize(self.marker,(max(8,round(self.marker.shape[1]*width/spec['width'])),max(8,round(self.marker.shape[0]*height/spec['height']))))
        # Ignore the patch corners, which include variable neighboring card art.
        margin=max(2,round(template.shape[1]*.12));core=template[margin:-margin,margin:-margin]
        ox,oy=round(width*.25),round(height*.84)
        roi=gray[oy:,ox:round(width*.65)]
        if min(roi.shape)<max(template.shape):return None,{}
        scores=cv2.matchTemplate(roi,core,cv2.TM_CCOEFF_NORMED);hits=[]
        for _ in range(8):
            _,score,_,(x,y)=cv2.minMaxLoc(scores)
            if score<.93:break
            patch=roi[y:y+core.shape[0],x:x+core.shape[1]]
            error=float(np.mean(np.abs(patch.astype(float)-core)))
            if error<=16:hits.append((x+ox-margin,y+oy-margin,float(score)))
            cv2.rectangle(scores,(max(0,x-template.shape[1]//2),max(0,y-template.shape[0]//2)),(x+template.shape[1]//2,y+template.shape[0]//2),-1,-1)
        if len(hits)!=1:return None,dict(marker_candidates=len(hits))
        x,y,score=hits[0];w,h=template.shape[1],template.shape[0]
        # Verify a near-horizontal card frame attached to the marker's right.
        edge_roi=gray[max(0,round(y+h*.15)):round(y+h*.8),round(x+w*.9):round(x+w*4.1)]
        lines=cv2.HoughLinesP(cv2.Canny(edge_roi,50,130),1,np.pi/180,threshold=15,minLineLength=max(12,round(w*.65)),maxLineGap=5)
        attached=lines is not None and any(abs(y2-y1)<max(3,abs(x2-x1)*.18) for x1,y1,x2,y2 in lines[:,0,:])
        rect=Box(round(x+w*.15),round(y+h*.65),round(w*4),height-round(y+h*.65))
        inside=rect.x>=0 and rect.y>=0 and rect.x+rect.width<=width and rect.height>h
        return (rect if attached and inside else None),dict(marker=[x,y,w,h],score=score,attached_top_edge=bool(attached))

    @staticmethod
    def append_gap(obs,image,result):
        if obs.phase!='recruit' or not 0<len(obs.board)<7:return None,{}
        regions={m.entity_id:m for m in result if m.status=='matched' and m.rect}
        if any(c.entity_id not in regions for c in obs.board):return None,{}
        boxes=[regions[c.entity_id].rect for c in obs.board]
        right=max(boxes,key=lambda b:b.x+b.width);width,height=image.size
        cx=right.x+right.width*1.35;cy=right.y+right.height*.47
        gap=Box(round(cx-right.width*.09),round(cy-right.height*.14),max(8,round(right.width*.18)),max(8,round(right.height*.28)))
        if gap.x+gap.width>width*.85 or gap.y<height*.4 or gap.y+gap.height>height*.7:return None,{}
        patch=np.asarray(image.crop((gap.x,gap.y,gap.x+gap.width,gap.y+gap.height)))
        hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV);h,s,v=np.median(hsv.reshape(-1,3),axis=0)
        gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
        edge_fraction=float(np.mean(cv2.Canny(gray,50,130)>0))
        empty_wood=8<=h<=32 and s>70 and v>65 and edge_fraction<.08
        return (gap if empty_wood else None),dict(empty_wood=bool(empty_wood),median_hsv=[float(h),float(s),float(v)],edge_fraction=edge_fraction)

    @staticmethod
    def insertion_gaps(obs,image,result):
        if obs.phase!='recruit' or not obs.board:return []
        visible={m.entity_id:m.rect for m in result if m.status=='matched' and m.rect and m.entity_id is not None}
        cards=sorted(obs.board,key=lambda c:c.position)
        if [c.position for c in cards]!=list(range(1,len(cards)+1)) or any(c.entity_id not in visible for c in cards):return []
        boxes=[visible[c.entity_id] for c in cards]
        if any(a.point()[0]>=b.point()[0] for a,b in zip(boxes,boxes[1:])):return []
        result=[];width,height=image.size
        for index in range(len(boxes)+1):
            left=boxes[index-1] if index else None;right=boxes[index] if index<len(boxes) else None
            ref=left or right
            x=(left.x+left.width+right.x)/2 if left and right else right.x-ref.width*.22 if right else left.x+left.width+ref.width*.22
            y=ref.point()[1];box=Box(round(x)-4,round(y)-8,8,16)
            if not (width*.15<box.x and box.x+box.width<width*.86 and height*.4<box.y<height*.7):continue
            if any(b.x<=x<=b.x+b.width and b.y<=y<=b.y+b.height for b in boxes):continue
            pixels=np.asarray(image.crop((box.x,box.y,box.x+box.width,box.y+box.height)))
            hsv=cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV);h,s,v=np.median(hsv.reshape(-1,3),axis=0)
            edges=float(np.mean(cv2.Canny(cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY),50,130)>0))
            if 8<=h<=32 and s>70 and v>65 and edges<.08:
                result.append(Mapping(f'position:board:{index+1}','matched',box,.85,'observed_log_order_and_empty_insertion_gap',stability_basis=StabilityBasis.GEOMETRY))
        return result

    def map(self,obs,image,window,*,blocked=False):
        # Entity/slot identities are intentionally NOT inferred from cached
        # coordinates. Pure visual results can be rebound only by current logs.
        controls=tuple((c.entity_id,c.card_id,c.tags.get('GAME_MODE_BUTTON_SLOT')) for c in obs.controls)
        scope=(obs.game_id,obs.build,str(obs.phase),window.get('token'),image.size,controls)
        if blocked or scope!=self.geometry_scope:
            self.evidence_cache.clear();self.anchor_cache.clear()
            self.hud_locations.clear()
        self.geometry_scope=scope
        self.last_frame_pixels=_FramePixels(image,self.evidence_cache) if not blocked and obs.phase in ('recruit','choice') else None
        return self._map(obs,self.last_frame_pixels if self.last_frame_pixels is not None else image,window,blocked=blocked)

    def _anchors(self,image):
        compute=lambda:self.anchor_cache.resolve(image,self.anchors,self.profile_path)
        # Identical current pixels also prove unchanged absent controls; avoid
        # repeatedly searching the whole frame for a button hidden in this UI.
        return image.memo('hud_anchors',(),compute) if hasattr(image,'memo') else compute()

    def components(self,image):
        compute=lambda: super(AdaptiveSpatialMapper,self).components(image)
        return image.memo('components',(),compute) if hasattr(image,'memo') else compute()

    def _map(self,obs,image,window,*,blocked=False):
        # Menu/result adapters measure their own controls. Combat and unknown
        # scenes cannot execute recruit actions; searching hidden card rows and
        # the entire screen for five missing HUD templates adds no evidence.
        if obs.phase not in ('recruit','choice'):
            self.last_ui_state={}
            self.last_diagnostics=dict(components=[],badges=[],groups={z:[] for z in ('shop','board','hand','choices')})
            return []
        self.last_ui_state={} if blocked else (image.memo('ui_state',(),lambda:self.ui_state(image)) if hasattr(image,'memo') else self.ui_state(image))
        result=super().map(obs,image,window,blocked=blocked)
        if blocked:return result
        scene=next((m for m in result if m.key=='scene:recruit' and m.status=='matched' and m.rect),None)
        if obs.phase=='recruit' and scene:
            boxes=self.shop_badge_regions(obs,image,scene.rect,self.last_diagnostics['badges'])
            if boxes:
                keys={f'entity:{c.entity_id}' for c in obs.shop}
                result=[m for m in result if m.key not in keys]
                result.extend(Mapping(f'entity:{c.entity_id}','matched',b,.9,
                    'complete_size_aligned_shop_shield_row',str(self.profile_path),c.entity_id,
                    stability_basis=StabilityBasis.GEOMETRY)
                    for c,b in zip(sorted(obs.shop,key=lambda c:c.position),boxes))
                self.last_diagnostics['shop_shield_row']=[vars(b) for b in boxes]
                # Card-art false positives also polluted the base mapper's
                # shop pitch, shifting its board lattice. Rebuild it from the
                # proven shield row, still requiring each board rim in pixels.
                if len(boxes)>1 and obs.board and sorted(c.position for c in obs.board)==list(range(1,len(obs.board)+1)):
                    pitch=float(np.median(np.diff([b.point()[0] for b in boxes])))
                    bob=scene.rect;unit=bob.height;board=[]
                    row=CenteredRow(bob.point()[0],bob.y+unit*2.76,pitch)
                    for box in row.boxes(len(obs.board),unit*.72,unit*.95):
                        if self.rim_support(image,box)>=.38:board.append(box)
                    if len(board)==len(obs.board):
                        keys={f'entity:{c.entity_id}' for c in obs.board}
                        result=[m for m in result if m.key not in keys]
                        result.extend(Mapping(f'entity:{c.entity_id}','matched',b,.9,
                            'shop_shield_pitch_and_independent_board_rim',str(self.profile_path),c.entity_id,
                            stability_basis=StabilityBasis.GEOMETRY)
                            for c,b in zip(sorted(obs.board,key=lambda c:c.position),board))
        is_trinket=bool(obs.choices) and all(c.card_type=='BATTLEGROUND_TRINKET' for c in obs.choices)
        if obs.phase=='choice' and is_trinket and self.trinkets:
            boxes,title,evidence=self.trinkets.detect(image,len(obs.choices))
            self.last_diagnostics['trinket_choice']=evidence
            if boxes:
                self.last_ui_state.update(modal='trinket_choice',
                    selected_position_inferred=evidence['selected_position_inferred'],
                    confirm_appearance=evidence['confirm_appearance'],protocol='unknown')
                keys={f'entity:{c.entity_id}' for c in obs.choices}|{'scene:choice','control:confirm'}
                result=[m for m in result if m.key not in keys]
                result.append(Mapping('scene:choice','matched',title,.94,'generic_trinket_title_and_four_paired_frame_rails',str(self.trinkets.path)))
                for card,box in zip(obs.choices,boxes):
                    result.append(Mapping(f'entity:{card.entity_id}','matched',box,.88,
                        'log_choice_order_and_observed_trinket_frame_rails; protocol_unknown',str(self.trinkets.path),card.entity_id))
                if evidence['confirm_region']:
                    result.append(Mapping('control:confirm','matched',Box(**evidence['confirm_region']),.85,
                        evidence['confirm_appearance']+'; interactability_unknown',str(self.trinkets.path)))
        if obs.phase=='recruit' and len(obs.hand)>=3 and any(m.key=='scene:recruit' and m.status=='matched' for m in result):
            badges=[Box(**b) for b in self.last_diagnostics['badges']]
            fan=diagnose_fan(obs.hand,image,badges)
            self.last_diagnostics['hand_fan']=fan
            if fan['status']=='inferred' and self.last_ui_state.get('mode')!='targeting_suspected':
                caps=[];pixels=np.asarray(image)
                for assignment in fan['assignments']:
                    rect,evidence=portrait_cap(image,assignment,fan['assignments'],fan['evidence']['pitch'],pixels=pixels)
                    caps.append(dict(entity_id=assignment['entity_id'],rect=rect,evidence=evidence))
                    key=f"entity:{assignment['entity_id']}"
                    if rect and not any(m.key==key and m.status=='matched' for m in result):
                        result=[m for m in result if m.key!=key]
                        result.append(Mapping(key,'matched',Box(**rect),.78,
                            'unique_log_tier_sequence_and_observed_portrait_rim; interactability_unknown',
                            str(self.profile_path),assignment['entity_id'],stability_basis=StabilityBasis.GEOMETRY))
                self.last_diagnostics['hand_caps']=caps
        if obs.phase=='choice' and not is_trinket and not any(m.key=='scene:choice' and m.status=='matched' for m in result):
            badges=[Box(**b) for b in self.last_diagnostics['badges']]
            boxes=self.choice_regions(badges,len(obs.choices),image)
            width,height=image.size
            if boxes and all(b.x>=0 and b.y>=0 and b.x+b.width<=width and b.y+b.height<=height for b in boxes):
                keys={f'entity:{c.entity_id}' for c in obs.choices}|{'scene:choice'}
                result=[m for m in result if m.key not in keys]
                result.append(Mapping('scene:choice','matched',boxes[0],.9,'scale_aligned_choice_badge_group',str(self.profile_path)))
                for c,b in zip(obs.choices,boxes):result.append(Mapping(f'entity:{c.entity_id}','matched',b,.9,'log_choice_order_and_scale_aligned_badges',str(self.profile_path),c.entity_id))
        if obs.phase=='recruit' and len(obs.hand)==1 and obs.hand[0].position==1 and obs.hand[0].card_type in ('SPELL','BATTLEGROUND_SPELL'):
            card=obs.hand[0];key=f'entity:{card.entity_id}'
            if any(m.key=='scene:recruit' and m.status=='matched' for m in result) and not any(m.key==key and m.status=='matched' for m in result):
                rect,evidence=self.single_spell_region(image);self.last_diagnostics['single_spell_evidence']=evidence
                if rect:
                    result=[m for m in result if m.key!=key]
                    result.append(Mapping(key,'matched',rect,evidence['score'],'log_single_spell_and_currency_marker_with_attached_frame',str(self.hand_marker),card.entity_id))
        if self.last_ui_state.get('mode')!='targeting_suspected':
            result+=self.insertion_gaps(obs,image,result)
            gap,evidence=self.append_gap(obs,image,result);self.last_diagnostics['append_gap']=evidence
            if gap:
                result=[m for m in result if m.key not in ('region:board',f'position:board:{len(obs.board)+1}')]
                result.append(Mapping('region:board','matched',gap,.85,'observed_rightmost_frame_and_empty_board_patch',str(self.profile_path),stability_basis=StabilityBasis.GEOMETRY))
                result.append(Mapping(f'position:board:{len(obs.board)+1}','matched',gap,.85,'observed_append_slot_only',str(self.profile_path),stability_basis=StabilityBasis.GEOMETRY))
        return result
