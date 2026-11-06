"""Measured locked-prefix mixed hands; no card-art identities or guessed slots.

Only a complete tier-badge suffix and an independently observed chained,
circular lock badge authorize this fallback. Each playable body also needs its
own measured rim. Incomplete or ambiguous rows produce no replacement points.
"""
from dataclasses import asdict
import cv2
import numpy as np

from .hand_cap import portrait_cap
from .hand_fan import tier_markers
from .interaction_regions import allowed, mask_zone
from .windows import Box


def spell_body(image, assignment, assignments, pitch, *, pixels):
    """Require the exposed top border of a rectangular spell beside its badge."""
    b=assignment['badge'];bw,bh=b['width'],b['height']
    centers=np.array([[a['badge']['x']+a['badge']['width']/2,
                       a['badge']['y']+a['badge']['height']/2] for a in assignments])
    mid=float(centers[:,0].mean());w,h=image.size
    curve=np.polyfit((centers[:,0]-mid)/w,centers[:,1]/h,2)
    x,y=b['x']+bw/2,b['y']+bh/2
    theta=np.arctan(np.polyval(np.polyder(curve),(x-mid)/w)*h/w)
    us=np.linspace(.7,3.2,200);vs=np.linspace(-1.7,1,180)
    uu,vv=np.meshgrid(us*bw,vs*bh)
    mx=x+uu*np.cos(theta)-vv*np.sin(theta)
    my=y+uu*np.sin(theta)+vv*np.cos(theta)
    patch=cv2.remap(pixels,mx.astype('float32'),my.astype('float32'),cv2.INTER_LINEAR)
    hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    edge=cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY),(3,3),0),35,100)
    material=((hsv[:,:,1]<125)&(hsv[:,:,2]>80)) | (
        (hsv[:,:,0]>=12)&(hsv[:,:,0]<=111)&(hsv[:,:,1]>90)&(hsv[:,:,2]>100))
    mask=((edge>0)&material&(vv/bh<-.15)).astype('uint8')
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,3),np.uint8))
    count,labels,_,_=cv2.connectedComponentsWithStats(mask)
    support=False
    for label in range(1,count):
        py,px=np.where(labels==label);u,v=us[px],vs[py]
        if len(px)<35 or np.ptp(u)<1.1 or min(u)>1.25 or max(u)<2.3:continue
        bx=[];by=[]
        for start in np.arange(1.0,2.51,.15):
            selected=(u>=start)&(u<start+.15)
            if np.count_nonzero(selected)>=2:
                bx.append(float(np.median(u[selected])));by.append(float(np.median(v[selected])))
        if len(bx)<8:continue
        fit=np.polyfit(bx,by,1)
        if max(abs(np.polyval(fit,bx)-by))<.15 and -1.6<float(np.polyval(fit,1.8))<-.2:
            support=True;break
    point=np.array([x,y])+np.array([np.cos(theta),np.sin(theta)])*1.8*bw+np.array([-np.sin(theta),np.cos(theta)])*.35*bh
    radius=max(3,round(min(bw,bh)*.12));px,py=map(round,point)
    if not support or not allowed((px,py),image.size,'hand') or px+radius>=x+pitch-bw*.4:return None
    if not (radius<px<w-radius and radius<py<h-radius):return None
    texture=pixels[py-radius:py+radius,px-radius:px+radius]
    if texture.mean()<=25 or texture.std()<=5:return None
    return Box(px-radius,py-radius,radius*2,radius*2)


def locked_prefix_binding(image,cards,*,pixels=None,hsv=None):
    """Bind 4–10 mixed slots with one locked leading card and a tiered suffix.

    The locked card occupies a real observed slot but is never given a drag
    point. Body rectangles returned for the playable suffix are independent
    of nominal costs, including white zero-cost spell badges.
    """
    cards=sorted(cards,key=lambda c:c.position)
    if not 4<=len(cards)<=10 or [c.position for c in cards]!=list(range(1,len(cards)+1)):return []
    if cards[0].tags.get('LITERALLY_UNPLAYABLE') not in ('1',1):return []
    suffix=cards[1:]
    if any(c.card_type not in ('MINION','SPELL','BATTLEGROUND_SPELL') or c.tier not in range(1,8)
           or c.tags.get('LITERALLY_UNPLAYABLE') in ('1',1) for c in suffix):return []
    pixels=np.asarray(image) if pixels is None else pixels
    hsv=cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV) if hsv is None else hsv
    w,h=image.size
    purple=cv2.inRange(hsv,np.array([130,100,65]),np.array([169,255,255]));mask_zone(purple,'hand')
    purple=cv2.morphologyEx(purple,cv2.MORPH_CLOSE,np.ones((max(3,round(h*.0056)),3),np.uint8))
    contours,_=cv2.findContours(purple,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    badges=[Box(*cv2.boundingRect(c)) for c in contours]
    markers=[m for m in tier_markers(image,badges) if m['y']<h*.925]
    if len(markers)>24:return []
    fits={}
    for a in markers:
        for b in markers:
            if suffix[0].card_type=='MINION' and a['tier']!=suffix[0].tier:continue
            if suffix[-1].card_type=='MINION' and b['tier']!=suffix[-1].tier:continue
            pitch=(b['x']-a['x'])/(len(suffix)-1)
            if not w*.018<pitch<w*.10:continue
            chosen=[]
            for i,card in enumerate(suffix):
                # Spell-star artwork can split into several gold components;
                # require a real tier shield, but only minions use star counts
                # as an additional identity-order check.
                matches=[m for m in markers if (card.card_type!='MINION' or m['tier']==card.tier)
                         and abs(m['x']-a['x']-i*pitch)<pitch*.12]
                if len(matches)!=1:break
                chosen.append(matches[0])
            if len(chosen)!=len(suffix) or len({m['x'] for m in chosen})!=len(suffix):continue
            fits[tuple(m['x'] for m in chosen)]=(chosen,pitch)
    if len(fits)!=1:return []
    chosen,pitch=next(iter(fits.values()))
    # A circular golden badge must occupy the missing leading slot. Its
    # adjacent cyan chain must have two long components, not a resource HUD.
    gold=cv2.inRange(hsv,np.array([12,105,140]),np.array([36,255,255]));mask_zone(gold,'hand')
    gold=cv2.morphologyEx(gold,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(gold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locks=[]
    expected=chosen[0]['x']-pitch
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour)
        if not (.010*w<bw<.032*w and .016*h<bh<.052*h and .7<bw/bh<1.3):continue
        if abs(x+bw/2-expected)>pitch*.18 or cv2.contourArea(contour)<bw*bh*.45:continue
        left,top,right,bottom=max(0,x),max(0,y),min(w,x+round(4.5*bw)),min(h,y+round(3.7*bh))
        chain=cv2.inRange(hsv[top:bottom,left:right],np.array([80,90,85]),np.array([112,255,255]))
        chain=cv2.morphologyEx(chain,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        cc,_=cv2.findContours(chain,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rails=[c for c in cc if cv2.boundingRect(c)[2]>bw*1.6 and cv2.contourArea(c)>bw*bh*.12]
        if len(rails)>=2:locks.append(Box(x,y,bw,bh))
    if len(locks)!=1:return []
    assignments=[dict(entity_id=cards[0].entity_id,badge=asdict(locks[0]))]+[
        dict(entity_id=c.entity_id,badge=m['rect']) for c,m in zip(suffix,chosen)]
    centers=np.array([[a['badge']['x']+a['badge']['width']/2,a['badge']['y']+a['badge']['height']/2] for a in assignments])
    xs=(centers[:,0]-centers[:,0].mean())/w;ys=centers[:,1]/h
    curve=np.polyfit(xs,ys,2)
    if not .02<curve[0]<8 or max(abs(np.polyval(curve,xs)-ys))>.009:return []
    result=[]
    for c,a in zip(suffix,assignments[1:]):
        if c.card_type=='MINION':
            rect,_=portrait_cap(image,a,assignments,pitch,pixels=pixels,include_eligible_rim=True)
            hit=Box(**rect) if rect else None
        else:hit=spell_body(image,a,assignments,pitch,pixels=pixels)
        if hit is None:continue
        badge=Box(**a['badge'])
        if not allowed(hit.point(),image.size,'hand') or badge.x<=hit.point()[0]<=badge.x+badge.width:return []
        result.append((c,badge,hit))
    return result
