"""Bounded caches of evidence, not of input authorization.

Detector results require the same input pixels AND parameters. HUD locations
are rechecked against their actual template in every capture before reuse.
Neither cache creates captures, stable-frame counts or successful receipts.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields,is_dataclass
from functools import wraps

import cv2
import numpy as np

from .mapping import Mapping
from .windows import Box


def freeze(value):
    if is_dataclass(value):return tuple((f.name,freeze(getattr(value,f.name))) for f in fields(value))
    if isinstance(value,dict):return tuple(sorted((k,freeze(v)) for k,v in value.items()))
    if isinstance(value,(list,tuple)):return tuple(freeze(v) for v in value)
    return value


def pixel_detector(function):
    """Memoize pure image detectors; parameters include count/selection mode.

    Full-image evidence is deliberate for general detectors: a contour can
    connect to pixels outside its nominal search band. Narrow ROI caches are
    used only by detectors whose implementation actually crops that ROI first.
    """
    @wraps(function)
    def wrapped(image,*args,**kwargs):
        compute=lambda:function(image,*args,**kwargs)
        if hasattr(image,'memo'):
            return image.memo(function.__module__+'.'+function.__name__,(args,kwargs),compute)
        return compute()
    return wrapped


class EvidenceCache:
    def __init__(self,limit=64):
        self.limit=limit;self.entries=OrderedDict();self.hits=0;self.misses=0;self.scope=None

    def begin(self,scope):
        if scope!=self.scope:self.clear();self.scope=scope

    def clear(self):self.entries.clear()

    def resolve(self,key,compute):
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key)
            return deepcopy(self.entries[key])
        self.misses+=1;result=compute()
        self.entries[key]=deepcopy(result)
        while len(self.entries)>self.limit:self.entries.popitem(last=False)
        return result


class AnchorCache:
    """Cache scale and location, while requiring fresh patch evidence."""
    def __init__(self):
        self.templates={};self.locations={};self.size=None;self.hits=0;self.searches=0

    def clear(self):self.locations.clear()

    @staticmethod
    def match(small,templ,loc):
        x,y=loc;h,w=templ.shape
        patch=small[y:y+h,x:x+w]
        if x<0 or y<0 or patch.shape!=templ.shape:return None
        score=float(cv2.matchTemplate(patch,templ,cv2.TM_CCOEFF_NORMED)[0,0])
        if score<.9 or np.mean(np.abs(patch.astype(float)-templ.astype(float)))>18:return None
        return score

    def resolve(self,image,anchors,profile_path):
        width,height=image.size;factor=min(1,1000/width)
        if self.size!=image.size:
            self.size=image.size;self.templates.clear();self.clear()
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
        small=cv2.resize(gray,None,fx=factor,fy=factor);found={}
        for entry,original in anchors:
            key=entry['key'];px,py,pw,ph=entry['patch_uv'];rx,ry,rw,rh=entry['rect_uv']
            if key not in self.templates:
                self.templates[key]=cv2.resize(original,(max(8,round(pw*width*factor)),max(8,round(ph*height*factor))))
            templ=self.templates[key];th,tw=templ.shape
            if th>small.shape[0] or tw>small.shape[1]:continue
            # Bob is a reference for several derived rows, not a fixed hit
            # button. Measure its best local alignment afresh on changed
            # frames instead of propagating an acceptable two-pixel offset.
            loc=self.locations.get(key) if key!='scene:recruit' else None
            score=self.match(small,templ,loc) if loc is not None else None
            if score is not None:self.hits+=1
            else:
                self.searches+=1
                ex,ey=round(px*width*factor),round(py*height*factor);margin=round(width*factor*.035)
                x0,y0=max(0,ex-margin),max(0,ey-margin)
                roi=small[y0:min(small.shape[0],ey+th+margin),x0:min(small.shape[1],ex+tw+margin)]
                if roi.shape[0]<th or roi.shape[1]<tw:self.locations.pop(key,None);continue
                _,candidate,_,local=cv2.minMaxLoc(cv2.matchTemplate(roi,templ,cv2.TM_CCOEFF_NORMED))
                loc=(local[0]+x0,local[1]+y0)
                if candidate<.9:
                    _,_,_,loc=cv2.minMaxLoc(cv2.matchTemplate(small,templ,cv2.TM_CCOEFF_NORMED))
                score=self.match(small,templ,loc)
            if score is None:
                self.locations.pop(key,None);continue
            self.locations[key]=loc
            dx=loc[0]/factor-px*width;dy=loc[1]/factor-py*height
            box=Box(round(rx*width+dx),round(ry*height+dy),round(rw*width),round(rh*height))
            if box.x<0 or box.y<0 or box.x+box.width>width or box.y+box.height>height:continue
            found[key]=Mapping(key,'matched',box,score,'generic_ui_anchor',str(profile_path))
        return found
