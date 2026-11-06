"""Appearance-independent card frames + calibrated generic UI anchors.

Uses log identities/order, never card_id templates or frame-id lookups. Pixel
regions come from image components. Generic UI anchors can be calibrated from
development screenshots; no per-card coordinates are loaded by this mapper.
"""
from dataclasses import replace
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from .mapping import Mapping,StabilityBasis
from .windows import Box


class SpatialMapper:
    def __init__(self,profile: Path):
        self.profile_path=Path(profile)
        self.profile=json.loads(self.profile_path.read_text(encoding="utf-8"))
        self.anchors=[]
        for e in self.profile["entries"]:
            if e["key"].startswith("control:") or e["key"]=="scene:recruit":
                template=np.asarray(Image.open(self.profile_path.parent/e["template"]).convert("L"))
                self.anchors.append((e,template))
        self.last_diagnostics={}

    def _anchors(self,image):
        width,height=image.size;factor=min(1,1000/width)
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
        small=cv2.resize(gray,None,fx=factor,fy=factor)
        found={}
        for e,t in self.anchors:
            # Size follows the client image; location is searched, never copied.
            px,py,pw,ph=e["patch_uv"];rx,ry,rw,rh=e["rect_uv"]
            templ=cv2.resize(t,(max(8,round(pw*width*factor)),max(8,round(ph*height*factor))))
            if templ.shape[0]>small.shape[0] or templ.shape[1]>small.shape[1]:continue
            # Search near normalized HUD layout first; coordinates still come
            # from a measured match, with full-image fallback for layout drift.
            ex,ey=round(px*width*factor),round(py*height*factor);margin=round(width*factor*.035)
            x0,y0=max(0,ex-margin),max(0,ey-margin)
            roi=small[y0:min(small.shape[0],ey+templ.shape[0]+margin),x0:min(small.shape[1],ex+templ.shape[1]+margin)]
            result=cv2.matchTemplate(roi,templ,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(result);loc=(loc[0]+x0,loc[1]+y0)
            if score<.9:
                result=cv2.matchTemplate(small,templ,cv2.TM_CCOEFF_NORMED)
                _,score,_,loc=cv2.minMaxLoc(result)
            patch=small[loc[1]:loc[1]+templ.shape[0],loc[0]:loc[0]+templ.shape[1]]
            if score<.9 or np.mean(np.abs(patch.astype(float)-templ))>18:continue
            dx=loc[0]/factor-px*width;dy=loc[1]/factor-py*height
            box=Box(round(rx*width+dx),round(ry*height+dy),round(rw*width),round(rh*height))
            found[e["key"]]=Mapping(e["key"],"matched",box,score,"generic_ui_anchor",str(self.profile_path))
        return found

    def components(self,image):
        width,height=image.size;factor=min(1,1000/width)
        rgb=cv2.resize(np.asarray(image),None,fx=factor,fy=factor)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        h,s,v=cv2.split(hsv)
        # Cyan/green/gold outlines, with small morphological joins. The art
        # interiors are ignored; component topology and log count must agree.
        mask=((h>=37)&(h<=111)&(s>95)&(v>135)).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for contour in contours:
            x,y,w,ht=cv2.boundingRect(contour)
            if .035*width*factor<w<.25*width*factor and .07*height*factor<ht<.5*height*factor:
                boxes.append(Box(round(x/factor),round(y/factor),round(w/factor),round(ht/factor)))
        # Purple tavern shields independent of minion art/cost/tribe.
        badge_mask=((h>=130)&(h<=169)&(s>110)&(v>75)).astype(np.uint8)*255
        badge_mask=cv2.morphologyEx(badge_mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(badge_mask)
        badges=[]
        for x,y,w,ht,area in stats[1:]:
            if .01*width*factor<w<.08*width*factor and .012*height*factor<ht<.12*height*factor and .35<w/ht<1.5 and area>w*ht*.35:
                badges.append(Box(round(x/factor),round(y/factor),round(w/factor),round(ht/factor)))
        return boxes,badges

    def map(self,obs,image,window,*,blocked=False):
        # A forced-choice modal hides the recruit HUD; those template searches
        # cannot authorize any action in this phase and need not run at all.
        anchors=self._anchors(image) if not blocked and obs.phase!='choice' else {}
        boxes,badges=self.components(image) if not blocked else ([],[])
        width,height=image.size
        scene=anchors.get("scene:recruit")
        results=list(anchors.values())
        groups={z:[] for z in ("shop","board","hand","choices")}
        if scene:
            # Bob's portrait supplies scale and rows; components supply x/y.
            bob=scene.rect;unit=bob.height
            shop_badges=[b for b in badges if bob.y+unit*.8<b.y<bob.y+unit*1.45 and bob.x-unit*3<b.x<bob.x+unit*3]
            if shop_badges:
                center_y=float(np.median([b.y for b in shop_badges]))
                shop_badges=[b for b in shop_badges if abs(b.y-center_y)<unit*.12]
                for b in sorted(shop_badges,key=lambda b:b.x):
                    groups["shop"].append(Box(round(b.x-b.width*.8),b.y,round(b.width*2.6),round(b.height*3.7)))
            groups["board"]=[b for b in boxes if bob.y+unit*2.1<b.y<bob.y+unit*3 and b.height<unit*1.25 and .45<b.width/b.height<1.05 and bob.x-unit*3<b.x<bob.x+unit*3]
            groups["hand"]=[b for b in boxes if b.y>bob.y+unit*4.3 and b.y+b.height>=height*.94 and b.width<unit*1.5]
            # A row lattice is measured from detected shop shields, not from
            # screenshot coordinates. Every predicted board slot must also have
            # an independent colored-frame rim; log count alone cannot pass it.
            xs=sorted(b.point()[0] for b in shop_badges)
            pitch=float(np.median(np.diff(xs))) if len(xs)>1 else unit*.8
            proposals=[];supports=[]
            for index in range(len(obs.board)):
                cx=bob.point()[0]+(index-(len(obs.board)-1)/2)*pitch;cy=bob.y+unit*2.76
                w,h=unit*.72,unit*.95
                proposal=Box(round(cx-w/2),round(cy-h/2),round(w),round(h))
                support=self.rim_support(image,proposal)
                supports.append(support)
                if support>=.38:proposals.append(proposal)
            if len(proposals)==len(obs.board)>0:groups["board"]=proposals
            hand_badges=[b for b in badges if b.y>height*.82 and b.x>width*.25 and b.x<width*.75 and b.height<height*.04]
            if hand_badges and len(hand_badges)==len(obs.hand):
                hand_boxes=[]
                for b in hand_badges:
                    rect=Box(b.x,round(b.y-b.height*.4),round(b.height*4.6),min(round(b.height*6.5),height-round(b.y-b.height*.4)))
                    if any(rect.x<=q.point()[0]<=rect.x+rect.width for q in groups["hand"]):hand_boxes.append(rect)
                if len(hand_boxes)==len(obs.hand):groups["hand"]=hand_boxes
            for key in ("region:hand","region:board","region:bob"):
                # Drop regions remain geometric proposals, not proven acceptance.
                if key=="region:bob":rect=bob
                elif key=="region:board":rect=Box(round(bob.x+unit*1.6),round(bob.y+unit*2.3),round(unit*.7),round(unit*.6))
                else:rect=Box(round(bob.x+unit*.4),round(height-unit*.35),round(unit*.6),round(unit*.3))
                results.append(Mapping(key,"matched",rect,scene.confidence,"ui_anchor_relative_region",str(self.profile_path),stability_basis=StabilityBasis.RECEIVING_REGION))
        # Discover cards are large, vertically aligned components. Underlying
        # board/shop entities are not visible through a choice modal.
        choice_badges=[b for b in badges if b.height>height*.052 and .15*height<b.y<.28*height and .12*width<b.x<.85*width and .6<b.width/b.height<.97]
        if choice_badges:
            y=float(np.median([b.y for b in choice_badges]))
            groups["choices"]=[Box(b.x,round(b.y-b.height*.1),round(b.width*4.2),round(b.height*6.3))
                               for b in choice_badges if abs(b.y-y)<height*.025]
        for zone in groups:
            unique=[]
            distance=scene.rect.height*.42 if scene else width*.045
            for box in sorted(groups[zone],key=lambda b:b.width*b.height,reverse=True):
                cx,cy=box.point()
                if not any(abs(cx-u.point()[0])<distance and abs(cy-u.point()[1])<distance for u in unique):unique.append(box)
            groups[zone]=sorted(unique,key=lambda b:b.x)
        self.last_diagnostics=dict(components=[vars(b) for b in boxes],badges=[vars(b) for b in badges],groups={z:[vars(b) for b in bs] for z,bs in groups.items()})
        if obs.phase=="choice" and len(groups["choices"])==len(obs.choices)>0:
            b=groups["choices"][0]
            results.append(Mapping("scene:choice","matched",b,.9,"choice_card_frame_group",str(self.profile_path)))
        for zone in groups:
            cards=getattr(obs,zone)
            valid_order=zone=="choices" or sorted(c.position for c in cards)==list(range(1,len(cards)+1))
            candidates=groups[zone]
            reason="visual_count_or_frame_topology_mismatch"
            if blocked:reason="blocked_frame"
            elif not valid_order:reason="log_slot_order_unknown"
            elif obs.phase=="choice" and zone!="choices":reason="hidden_by_choice_modal"
            elif not scene and zone!="choices":reason="recruit_ui_unconfirmed"
            else:
                if len(cards)==len(candidates) and valid_order:
                    for c,b in zip(sorted(cards,key=lambda c:c.position) if zone!="choices" else cards,candidates):
                        inside=b.x>=0 and b.y>=0 and b.x+b.width<=width and b.y+b.height<=height
                        results.append(Mapping(f"entity:{c.entity_id}","matched" if inside else 'unknown',b if inside else None,.9 if inside else 0,
                                               "log_order_and_card_frame" if inside else 'frame_out_of_client',str(self.profile_path),c.entity_id,stability_basis=StabilityBasis.GEOMETRY))
                    continue
            for c in cards:results.append(Mapping(f"entity:{c.entity_id}","unknown",None,0,reason,entity_id=c.entity_id))
        return results

    @staticmethod
    def rim_support(image,box):
        cache=getattr(image,'_rim_support_cache',None)
        key=(box.x,box.y,box.width,box.height)
        if cache is not None and key in cache:return cache[key]
        rgb=np.asarray(image);cx,cy=box.point();hits=[]
        radius=max(2,round(box.width*.075))
        for angle in np.linspace(0,2*np.pi,48,endpoint=False):
            x=round(cx+box.width*.43*np.cos(angle));y=round(cy+box.height*.43*np.sin(angle))
            patch=rgb[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
            if not patch.size:hits.append(False);continue
            r,g,b=patch[:,:,0].astype(int),patch[:,:,1].astype(int),patch[:,:,2].astype(int)
            hits.append(bool(np.any((g>r+25)&(g>110)&((b>r+15)|(g>b+25)))))
        support=sum(hits)/len(hits)
        if cache is not None:cache[key]=support
        return support
