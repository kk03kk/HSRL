"""Pixel evidence for choice layouts, independent of policy and input APIs."""
import cv2
import numpy as np
from .windows import Box
from .geometry_cache import pixel_detector
from .interaction_protocol import quest_reward_links as quest_links


def ui_build_error(obs,profile):
    expected=profile.get('build')
    if expected is None or str(obs.phase) not in ('recruit','choice','combat'):return None
    if str(obs.build)!=str(expected):
        return f'client_ui_build_unvalidated: observed={obs.build}, calibration={expected}'
    return None


@pixel_detector
def power_card_row(image,count):
    """Complete aligned hero-power card row, including split green frames.

    Normalize morphology to client scale, then join only nearly touching,
    equally tall fragments. Portrait/cost artwork and missing neighbours cannot
    substitute for two independently observed outer vertical rails per card.
    """
    if count not in (2,3,4):return []
    width,height=image.size;scale=min(1,1000/width)
    rgb=cv2.resize(np.asarray(image),None,fx=scale,fy=scale)
    hh,ww=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([35,95,100]),np.array([100,255,255]))
    joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    parts=sorted((cv2.boundingRect(c) for c in contours),key=lambda b:b[0])
    parts=[b for b in parts if .035*ww<b[2]<.30*ww and .26*hh<b[3]<.66*hh
           and .18*hh<b[1]<.5*hh and .15<b[2]/b[3]<1.3]
    frames=[]
    for x,y,w,h in parts:
        if frames:
            px,py,pw,ph=frames[-1]
            right=max(px+pw,x+w);bottom=max(py+ph,y+h)
            if (-.003*ww<=x-(px+pw)<=.012*ww and abs(y-py)<.025*hh and
                .8<h/ph<1.25 and right-px<.30*ww):
                frames[-1]=(px,min(py,y),right-px,bottom-min(py,y));continue
        frames.append((x,y,w,h))
    if len(frames)!=count:return []
    if any(not (.10*ww<w<.30*ww and .4<w/h<1.3) for x,y,w,h in frames):return []
    if max(y for x,y,w,h in frames)-min(y for x,y,w,h in frames)>.045*hh:return []
    if max(w for x,y,w,h in frames)/min(w for x,y,w,h in frames)>1.18:return []
    if max(h for x,y,w,h in frames)/min(h for x,y,w,h in frames)>1.15:return []
    centers=[x+w/2 for x,y,w,h in frames];gaps=np.diff(centers)
    if min(gaps)<.12*ww or max(gaps)/min(gaps)>1.2:return []
    if abs((centers[0]+centers[-1])/2-ww/2)>.10*ww:return []
    if any(ax+aw>=bx for (ax,ay,aw,ah),(bx,by,bw,bh) in zip(frames,frames[1:])):return []
    for x,y,w,h in frames:
        for rx in (x,x+round(w*.9)):
            rail=mask[y+round(h*.28):y+round(h*.78),rx:rx+max(3,round(w*.1))]
            if not rail.size or np.mean(np.any(rail>0,axis=1))<.60:return []
    return [Box(round((x+w*.3)/scale),round((y+h*.35)/scale),
                round(w*.4/scale),round(h*.3/scale)) for x,y,w,h in frames]


@pixel_detector
def selected_hero(image, reference_regions):
    """Match one green selection halo to a previously measured hero slot.

    The complete initial row supplies slot identity even if a tooltip covers a
    neighbour after selection. A blue confirm button alone supplies no identity.
    """
    if not 2<=len(reference_regions)<=4:return None
    hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([35,160,160]),np.array([90,255,255]))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    matches=set()
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        for index,hit in enumerate(reference_regions):
            # Invert hero_regions' measured inset, never invent row spacing.
            fx=hit.x-.75*hit.width;fy=hit.y-.8*hit.height
            fw=hit.width/.4;fh=hit.height/.25
            if (.65*fw<w<1.45*fw and .55*fh<h<1.5*fh and
                fx-.20*fw<x<fx+.15*fw and fy-.35*fh<y<fy+.12*fh and
                x+w>fx+.55*fw and y+h>fy+.55*fh and
                cv2.contourArea(contour)>.025*fw*fh):
                matches.add(index)
    return next(iter(matches)) if len(matches)==1 else None


@pixel_detector
def quest_regions(image,count):
    """A complete row of short green quest panels below paired reward cards.

    Ordinary full-height Discover and the blue Hide button cannot satisfy this
    geometry. Source quest/reward links are checked separately by the provider.
    """
    if count not in (2,3,4):return []
    width,height=image.size;scale=min(1,1000/width)
    hsv=cv2.cvtColor(cv2.resize(np.asarray(image),None,fx=scale,fy=scale),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([35,160,160]),np.array([85,255,255]))
    # Remove thin spell particles before joining the actual panel rails.
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for contour in contours:
        x,y,w,h=[v/scale for v in cv2.boundingRect(contour)]
        if not (.1*width<w<.27*width and .10*height<h<.25*height and
                .50*height<y<.76*height and 1.1<w/h<2.8):continue
        boxes.append(Box(round(x),round(y),round(w),round(h)))
    boxes.sort(key=lambda b:b.x)
    if len(boxes)!=count:return []
    if max(b.y for b in boxes)-min(b.y for b in boxes)>.025*height:return []
    if max(b.width for b in boxes)/min(b.width for b in boxes)>1.15:return []
    if max(b.height for b in boxes)/min(b.height for b in boxes)>1.15:return []
    if any(a.x+a.width>=b.x for a,b in zip(boxes,boxes[1:])):return []
    centers=[b.point()[0] for b in boxes]
    if abs((centers[0]+centers[-1])/2-width/2)>.12*width:return []
    if count>2:
        gaps=np.diff(centers)
        if max(gaps)/min(gaps)>1.2:return []
    return [Box(round(b.x+.25*b.width),round(b.y+.25*b.height),
                round(.5*b.width),round(.5*b.height)) for b in boxes]


