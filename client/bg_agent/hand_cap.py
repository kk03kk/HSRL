"""Observed portrait-rim evidence for overlapping minion hands.

The fan proposes a small visible portrait point. A measured silver rim above
that point must independently support it; no card-art template is involved.
This geometry cannot establish UI hit-testing or a successful drag.
"""
import cv2
import numpy as np


def portrait_cap(image, assignment, assignments, pitch, *, pixels=None,include_eligible_rim=False):
    compute=lambda:_portrait_cap(image,assignment,assignments,pitch,pixels=pixels,include_eligible_rim=include_eligible_rim)
    # np.asarray(frame) is a zero-copy view, not necessarily the same ndarray
    # object. Alternate pixels must bypass this cache; equivalent immutable
    # views should share the rim analysis between mapper and execution checks.
    same_pixels=pixels is None or hasattr(image,'pixels') and pixels.shape==image.pixels.shape and pixels.strides==image.pixels.strides and np.shares_memory(pixels,image.pixels)
    if hasattr(image,'memo') and same_pixels:
        return image.memo('portrait_cap',(assignment,assignments,pitch,include_eligible_rim),compute)
    return compute()


def _portrait_cap(image, assignment, assignments, pitch, *, pixels=None,include_eligible_rim=False):
    width,height=image.size
    centers=np.array([[a['badge']['x']+a['badge']['width']/2,
                       a['badge']['y']+a['badge']['height']/2] for a in assignments])
    mid=float(np.mean(centers[:,0]))
    # A small hand can expose only one minion badge (other cards may be
    # spells). There is no independently measured fan slope in that case.
    curve=np.polyfit((centers[:,0]-mid)/width,centers[:,1]/height,min(2,len(centers)-1)) if len(centers)>1 else np.array([0.])
    b=assignment['badge'];bw,bh=b['width'],b['height']
    x,y=b['x']+bw/2,b['y']+bh/2
    theta=np.arctan(np.polyval(np.polyder(curve),(x-mid)/width)*height/width)
    # Analyze a normalized local frame; coordinates derive from this image's
    # actual badge and fan slope, not from any screenshot resolution or id.
    cols,rows=240,200
    us=np.linspace(.4,3.3,cols);vs=np.linspace(-1.7,1,rows)
    uu,vv=np.meshgrid(us*bw,vs*bh)
    mx=x+uu*np.cos(theta)-vv*np.sin(theta)
    my=y+uu*np.sin(theta)+vv*np.cos(theta)
    pixels=np.asarray(image) if pixels is None else pixels
    patch=cv2.remap(pixels,mx.astype('float32'),my.astype('float32'),cv2.INTER_LINEAR)
    hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    edge=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),35,100)
    silver=(hsv[:,:,1]<125)&(hsv[:,:,2]>80)
    gold=(hsv[:,:,0]>=12)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>90)&(hsv[:,:,2]>100)
    green=(hsv[:,:,0]>=37)&(hsv[:,:,0]<=111)&(hsv[:,:,1]>95)&(hsv[:,:,2]>135)
    allowed=(vv/bh < -.12)&(uu/bw>.8)&(uu/bw<2.9)
    regions=[]
    for material in (silver,gold)+((green,) if include_eligible_rim else ()):
        mask=((edge>0)&material&allowed).astype('uint8')
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        regions.extend(np.where(labels==label) for label in range(1,count))
    supports=[]
    for py,px in regions:
        area=len(px)
        u=us[px];v=vs[py]
        if np.ptp(u)<.85 or min(u)>1.42 or max(u)<2.18 or area<35:continue
        # Median bins follow the border, rather than its many parallel pixels.
        bx=[];by=[]
        for start in np.arange(1.1,2.51,.14):
            selected=(u>=start)&(u<start+.14)
            if np.count_nonzero(selected)>=2:
                bx.append(float(np.median(u[selected])));by.append(float(np.median(v[selected])))
        if len(bx)<7:continue
        fit=np.polyfit(bx,by,2)
        residual=float(np.max(np.abs(np.polyval(fit,bx)-by)))
        rim=float(np.polyval(fit,1.8))
        # A straight background trim is not a portrait arc. Insufficient
        # curvature is refused even when a silver horizontal edge is visible.
        if residual>.14 or not .15<fit[0]<1.5 or not .4<.15-rim<1.9:continue
        supports.append(dict(span=float(np.ptp(u)),residual=residual,rim_v=rim,
                             curvature=float(fit[0]),pixels=int(area)))
    point=np.array([x,y])+np.array([np.cos(theta),np.sin(theta)])*1.8*bw+np.array([-np.sin(theta),np.cos(theta)])*.15*bh
    # Reserve space before the next slot, including spells without a badge.
    next_boundary=x+pitch-bw*.4
    radius=max(3,round(min(bw,bh)*.12))
    inside=radius<point[0]<width-radius and radius<point[1]<height-radius
    before_neighbor=point[0]+radius<next_boundary
    point_patch=np.asarray(image.crop((round(point[0])-radius,round(point[1])-radius,
                                       round(point[0])+radius,round(point[1])+radius)))
    patch_valid=float(np.mean(point_patch))>25 and float(np.std(point_patch))>5
    evidence=dict(rim_supports=supports,angle_degrees=float(theta*180/np.pi),
                  before_next_slot=bool(before_neighbor),inside_client=bool(inside),
                  point_patch_has_visible_texture=bool(patch_valid),
                  proposed_point=[float(v) for v in point],interactability='unknown')
    if not supports or not inside or not before_neighbor or not patch_valid:
        return None,dict(evidence,reason='portrait_rim_or_visible_slot_not_confirmed')
    return dict(x=round(point[0])-radius,y=round(point[1])-radius,width=radius*2,height=radius*2),dict(evidence,reason='observed_rim_above_slot_interior')
