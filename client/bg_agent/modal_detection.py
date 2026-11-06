"""Measured modal cues; these never authorize an action on their own."""
import cv2
import numpy as np


def targeting_arrow(hsv):
    """Three or more similar, collinear red dashes, independent of desaturation.

    Some choose-one spells keep the whole tavern colored. Their dashed pointer
    is still visible after parking the cursor, unlike ordinary red card art.
    Normalize component geometry so the cue follows client scale.
    """
    height,width=hsv.shape[:2]
    scale=min(1,1000/width)
    small=cv2.resize(hsv,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    height,width=small.shape[:2];h,s,v=cv2.split(small)
    mask=(((h<8)|(h>170))&(s>150)&(v>110)).astype('uint8')
    mask[:round(height*.25)]=0;mask[round(height*.88):]=0
    mask[:,:round(width*.25)]=0
    _,_,stats,centers=cv2.connectedComponentsWithStats(mask)
    parts=[(np.array(c),w,ht) for (x,y,w,ht,area),c in zip(stats[1:],centers[1:])
           if .025*width<w<.12*width and .015*height<ht<.14*height
           and .65<w/ht<2.8 and area>.42*w*ht]
    for i,(origin,w,h) in enumerate(parts):
        for other,ow,oh in parts[i+1:]:
            direction=other-origin;distance=np.linalg.norm(direction)
            if distance<w*.6:continue
            direction=direction/distance
            row=sorted(float(np.dot(c-origin,direction)) for c,cw,ch in parts
                       if .65<cw/w<1.55 and .65<ch/h<1.55
                       and abs(float(np.dot(c-origin,(-direction[1],direction[0]))))<max(4,height*.012))
            if len(row)<3:continue
            gaps=np.diff(row)
            if row[-1]-row[0]>width*.12 and min(gaps)>w*.6 and max(gaps)<w*2 and max(gaps)/min(gaps)<1.8:
                return True
    return False


def pause_menu(hsv):
    height,width=hsv.shape[:2]
    # Three equal gold controls, with a red top control, identify the game menu
    # independently of language or card art. A red card alone cannot match.
    roi=hsv[round(height*.20):round(height*.68),round(width*.30):round(width*.70)]
    h,s,v=cv2.split(roi)
    gold=((h>=15)&(h<=40)&(s>40)&(s<200)&(v>170)).astype('uint8')
    gold=cv2.morphologyEx(gold,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(gold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[cv2.boundingRect(c) for c in contours]
    boxes=[b for b in boxes if .07*width<b[2]<.22*width and .018*height<b[3]<.065*height and b[2]/b[3]>3]
    for first in boxes:
        x,y,w,hh=first;cx=x+w/2
        row=sorted((b for b in boxes if abs(b[0]+b[2]/2-cx)<width*.01 and .85<b[2]/w<1.15 and .7<b[3]/hh<1.3),key=lambda b:b[1])
        if len(row)!=3 or first!=row[0]:continue
        gaps=np.diff([b[1] for b in row])
        if min(gaps)<height*.06 or max(gaps)>height*.14 or max(gaps)/min(gaps)>1.15:continue
        patch=roi[y:y+hh,x:x+w];ph,ps,pv=cv2.split(patch)
        if np.mean(((ph<8)|(ph>170))&(ps>130)&(pv>70))>.25:return True
    return False
