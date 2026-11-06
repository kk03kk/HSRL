"""Local, family-specific evidence that a choice panel is actually displayed.

Saturated wood is not a card rail. Neither a log BEGIN nor a stable source
patch proves that an input handler exists at a predicted choice position.
"""
import cv2
import numpy as np


STRONG_SCENES=frozenset((
    'independent_local_choice_frames_and_measured_legacy_points',
    'generic_trinket_title_and_four_paired_frame_rails',
    'hero_portrait_group_with_server_hero_choices',
    'observed_green_power_card_group','complete_paired_quest_panel_row',
    'complete_aligned_discover_frames','server_multi_choice_and_unambiguous_frame_states'))


def frame_rails(image,box,*,family='discover'):
    """Only two narrow exterior strips; never search or identify card faces."""
    supports=[]
    purple=family in ('hero','trinket')
    for x in (box.x,box.x+box.width):
        margin=max(3,round(box.width*(.20 if family=='hero' else .12)))
        y0=box.y+round(box.height*(.28 if purple else .18));y1=box.y+round(box.height*(.72 if purple else .88))
        patch=image.crop((max(0,x-margin),max(0,y0),min(image.width,x+margin),min(image.height,y1)))
        if min(patch.size)<2:return False
        hsv=cv2.cvtColor(np.asarray(patch),cv2.COLOR_RGB2HSV)
        h,s,v=cv2.split(hsv)
        mask=((h>=125)&(h<=175)&(s>60)&(v>55)) if purple else ((h>=35)&(h<=112)&(s>105)&(v>125))
        rows=(np.any(mask,axis=1)&np.any(~mask,axis=1)).astype('uint8')
        if purple:supports.append(float(rows.mean()))
        else:
            # Separate shop and board glows cannot form one tall card rail.
            # Close only tiny particle/text gaps, not gaps between two rows.
            size=max(3,round(len(rows)*.02))
            rows=cv2.morphologyEx(rows.reshape(-1,1),cv2.MORPH_CLOSE,np.ones((size,1),np.uint8)).ravel()
            runs=np.diff(np.flatnonzero(np.diff(np.r_[0,rows,0])))
            supports.append(float(max(runs[::2],default=0)/len(rows)))
    # Hero armour shields legitimately cover much of the right portrait rail;
    # the independent wide purple banner remains mandatory for this family.
    return min(supports)>=(.35 if family=='hero' else .5 if purple else .70)


def hero_banner(image):
    """Purple header must span the middle, unlike isolated upgrade controls."""
    w,h=image.size;x0=round(w*.27);y0=round(h*.045)
    patch=image.crop((x0,y0,round(w*.73),round(h*.25)))
    hsv=cv2.cvtColor(np.asarray(patch),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([125,75,65]),np.array([175,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return any(cw>w*.20 and ch>h*.035 for x,y,cw,ch in (cv2.boundingRect(c) for c in contours))


def shown(image,boxes,family):
    if not boxes or any(b is None for b in boxes):return False
    if family=='hero' and not hero_banner(image):return False
    return all(frame_rails(image,b,family=family) for b in boxes)


def neutral_discover_frames(image,boxes):
    return bool(neutral_discover_positions(image,boxes))


def neutral_discover_positions(image,boxes):
    """已校准发现布局的无绿光边框；完整行与独立弹层标志共同成立。"""
    if not 2<=len(boxes)<=4 or any(b is None for b in boxes) or not modal_hint(image):return False
    ordered=sorted(boxes,key=lambda b:b.x)
    if any(a.x+a.width>=b.x for a,b in zip(ordered,ordered[1:])):return False
    points=[]
    for box in boxes:
        if not image.width*.08<=box.width<=image.width*.3 or not image.height*.2<=box.height<=image.height*.65:return False
        for edge in (box.x,box.x+box.width):
            margin=max(3,round(box.width*.08))
            crop=image.crop((max(0,edge-margin),box.y+round(box.height*.2),
                min(image.width,edge+margin),box.y+round(box.height*.85))).convert('L')
            pixels=np.asarray(crop,dtype=np.float32)
            if pixels.shape[1]<3 or not len(pixels):return False
            # 两侧均须有贯穿大部分高度的窄边缘，不接受整块绿色/纯色卡面。
            gradient=np.abs(np.diff(pixels,axis=1))
            supported=gradient.max(axis=1)>25
            support=supported.mean()
            if support<.75:return False
            points.append(float(max(0,edge-margin)+np.median(gradient[supported].argmax(axis=1))))
    return tuple(points)


def rail_positions(image,boxes,family):
    """Relative motion in the same admitted strips, not new card locations."""
    points=[];purple=family in ('hero','trinket')
    for box in boxes:
        for x in (box.x,box.x+box.width):
            margin=max(3,round(box.width*(.20 if family=='hero' else .12)))
            x0=max(0,x-margin);x1=min(image.width,x+margin)
            y0=max(0,box.y+round(box.height*(.28 if purple else .18)))
            y1=min(image.height,box.y+round(box.height*(.72 if purple else .88)))
            hsv=cv2.cvtColor(np.asarray(image.crop((x0,y0,x1,y1))),cv2.COLOR_RGB2HSV)
            h,s,v=cv2.split(hsv)
            mask=((h>=125)&(h<=175)&(s>60)&(v>55)) if purple else ((h>=35)&(h<=112)&(s>105)&(v>125))
            profile=mask.mean(axis=0);peak=float(profile.max())
            if peak<=0:return None
            core=np.where(profile>=peak*.90,profile,0)
            points.append(float(x0+np.average(np.arange(len(core)),weights=core)))
            vertical=mask[:,core>0].sum(axis=1).astype(float)
            accumulated=np.cumsum(vertical)
            # Track vertical entry as well as horizontal expansion. Quantiles
            # ignore isolated particles outside the admitted exterior rail.
            for fraction in (.05,.95):
                points.append(float(y0+np.searchsorted(accumulated,accumulated[-1]*fraction)))
    return tuple(points)


def modal_hint(image):
    """Allows recovery scanning, never clicking: a modal header or Hide button."""
    if hero_banner(image):return True
    w,h=image.size
    patch=image.crop((round(w*.18),round(h*.65),round(w*.36),round(h*.86)))
    hsv=cv2.cvtColor(np.asarray(patch),cv2.COLOR_RGB2HSV)
    blue=cv2.inRange(hsv,np.array([85,70,130]),np.array([125,255,255]))
    blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return any(cw>w*.06 and ch>h*.03 and .7<cw/ch<5 for x,y,cw,ch in (cv2.boundingRect(c) for c in contours))


def strip_unproven_choice_maps(obs,maps):
    """Generic badge/contour groups can be the shop beneath a pending modal.

Recovery adapters add independently measured, family-specific scene evidence
after this boundary. Known geometry is admitted separately by shown().
"""
    if str(obs.phase)!='choice':return maps
    hidden=obs.interaction and obs.interaction.display_state in ('waiting','hidden')
    strong=not hidden and any(m.key=='scene:choice' and m.status=='matched' and m.reason in STRONG_SCENES for m in maps)
    if strong:return maps
    keys={'scene:choice','control:confirm'}|{f'entity:{c.entity_id}' for c in obs.choices}
    return [m for m in maps if m.key not in keys]
