"""Four-choice geometry from generic UI title and paired observed frame rails.

No trinket art, entity ids, price digits or precomputed card coordinates are
templates. Gray confirm appearance is recorded separately from clickability.
Selection/commit protocol stays unknown until independently evidenced.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .windows import Box


class TrinketMapper:
    def __init__(self,profile):
        self.path=Path(profile)
        self.spec=json.loads(self.path.read_text(encoding='utf-8'))
        self.templates={key:np.asarray(Image.open(self.path.parent/name).convert('L'))
                        for key,name in self.spec['templates'].items()}

    def anchor(self,image,key):
        width,height=image.size;factor=min(1,1280/width)
        gray=cv2.resize(np.asarray(image.convert('L')),None,fx=factor,fy=factor)
        source=self.templates[key]
        template=cv2.resize(source,(round(source.shape[1]*width/self.spec['width']*factor),
                                    round(source.shape[0]*height/self.spec['height']*factor)))
        if min(template.shape)<8 or any(a>b for a,b in zip(template.shape,gray.shape)):return None,{}
        scores=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        error=float(np.mean(np.abs(gray[y:y+template.shape[0],x:x+template.shape[1]].astype(float)-template)))
        if score<.94 or error>14:return None,dict(score=score,error=error)
        # Distinct, equally plausible copies of the title/control are ambiguous.
        cv2.rectangle(scores,(max(0,x-template.shape[1]),max(0,y-template.shape[0])),
                      (x+template.shape[1],y+template.shape[0]),-1,-1)
        second=float(cv2.minMaxLoc(scores)[1])
        if second>.9:return None,dict(score=score,error=error,ambiguous=True)
        return Box(round(x/factor),round(y/factor),round(template.shape[1]/factor),round(template.shape[0]/factor)),dict(score=score,error=error,second=second)

    def detect(self,image,count):
        evidence=dict(protocol='unknown',interactability='unknown',candidate_rows=[])
        if count!=4:return [],None,evidence
        title,match=self.anchor(image,'title');evidence['title_match']=match
        if title is None:return [],None,evidence
        evidence['title']=vars(title)
        width,height=image.size;unit=title.height;cx=title.x+title.width/2
        left=max(0,round(cx-title.width*3));right=min(width,round(cx+title.width*3))
        hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
        h,s,v=cv2.split(hsv)
        purple=(h>125)&(h<175)&(s>65)&(v>50)
        # Sweep below the detected title; actual rails determine x coordinates.
        # A band must independently contain eight narrow vertical frame rails.
        candidates=[]
        for top in np.arange(title.y+unit*4.5,title.y+unit*7.4,unit*.25):
            y0=round(top);y1=round(top+unit*2)
            if y0<0 or y1>height:continue
            projection=np.sum(purple[y0:y1,left:right],axis=0)>(y1-y0)*.45
            indices=np.flatnonzero(projection)
            runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            rails=[(int(run[0]+left),int(run[-1]+left)) for run in runs
                   if len(run) and unit*.065<len(run)<unit*.3]
            if len(rails)!=count*2:continue
            centers=[(a+b)/2 for a,b in rails]
            spans=np.array([centers[i+1]-centers[i] for i in range(0,8,2)])
            gaps=np.array([centers[i+2]-centers[i+1] for i in range(0,6,2)])
            if min(spans)<unit*2 or max(spans)/min(spans)>1.12:continue
            if min(gaps)<unit*.4 or max(gaps)/min(gaps)>1.2 or max(gaps)>min(spans)*.5:continue
            # Use an inset of the observed lower-body rail band, not a hidden
            # artwork/tooltip region or predicted full-card bounding box.
            boxes=[Box(round(centers[i]+spans[i//2]*.18),round(y0+unit*.6),
                       round(spans[i//2]*.64),round(unit*.8)) for i in range(0,8,2)]
            candidates.append((boxes,rails,y0,y1))
        evidence['candidate_rows']=[dict(y0=y0,y1=y1,rails=rails) for _,rails,y0,y1 in candidates]
        if not candidates:return [],None,evidence
        reference=candidates[0][1]
        if any(max(abs(a[0]-b[0]) for a,b in zip(reference,rails))>unit*.15 for _,rails,_,_ in candidates):
            evidence['reason']='conflicting_frame_rows';return [],None,evidence
        boxes,rails,y0,y1=candidates[len(candidates)//2]
        evidence['selected_band']=[y0,y1];evidence['rails']=rails
        confirm,cm=self.anchor(image,'confirm_gray');evidence['confirm_match']=cm
        if confirm and not (confirm.y>y1 and abs(confirm.x+confirm.width/2-cx)<title.width*.7):confirm=None
        if confirm:
            pixels=hsv[confirm.y:confirm.y+confirm.height,confirm.x:confirm.x+confirm.width]
            low_chroma=float(np.mean(pixels[:,:,1]<55))
            evidence['confirm_low_chroma_fraction']=low_chroma
            if low_chroma<.85:confirm=None
        evidence['confirm_appearance']='gray_unselected_like' if confirm else 'unknown'
        if not confirm:
            # A colored button must be observed independently; its location is
            # not copied from the gray template or a fixed client coordinate.
            low=max(0,y1);high=min(height,round(y1+unit*2.5))
            x0=max(0,round(cx-title.width*.8));x1=min(width,round(cx+title.width*.8))
            cyan=((h[low:high,x0:x1]>80)&(h[low:high,x0:x1]<111)&
                  (s[low:high,x0:x1]>60)&(v[low:high,x0:x1]>170)).astype('uint8')
            cyan=cv2.morphologyEx(cyan,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            _,_,stats,_=cv2.connectedComponentsWithStats(cyan)
            buttons=[Box(int(x+x0),int(y+low),int(w),int(ht)) for x,y,w,ht,area in stats[1:]
                     if title.width*.45<w<title.width*1.15 and unit*.6<ht<unit*2
                     and area>w*ht*.2 and abs(x+x0+w/2-cx)<title.width*.25]
            if len(buttons)==1:
                confirm=buttons[0];evidence['confirm_appearance']='cyan_selected_like'
        green=(h>40)&(h<85)&(s>150)&(v>180)
        highlighted=[];strengths=[]
        for index in range(4):
            a,b=rails[index*2:index*2+2]
            yy0=max(0,round(title.y+unit*1.8));yy1=min(height,round(title.y+unit*3))
            patch=green[yy0:yy1,a[0]:b[1]].astype('uint8')
            _,_,stats,_=cv2.connectedComponentsWithStats(patch)
            # Ignore small green price digits. Wide pieces either side of a
            # price badge support a highlighted frame, but remain inference.
            wide=[row for row in stats[1:] if row[2]>unit*.7 and row[4]>unit*2]
            span=sum(int(row[2]) for row in wide)/(b[1]-a[0])
            strengths.append(span)
            if span>.6:highlighted.append(index+1)
        evidence['highlight_width_fractions']=strengths
        evidence['selected_position_inferred']=(highlighted[0] if len(highlighted)==1 and evidence['confirm_appearance']=='cyan_selected_like' else None)
        evidence['confirm_region']=vars(confirm) if confirm else None
        return boxes,title,evidence
