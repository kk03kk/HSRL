"""移植旧执行器的抉择框/遮挡算法，适配PIL和主项目坐标；无策略选择。

来源：HearthstoneAI的bg_agent/live_interactions.py与branch_occlusion.py。
只复用完整卡框测量及局部模板遮挡，不复用ClickBot的试点分支策略。
OpenCV沿用现有隔离的live-runtime-deps，由实机入口加载。
"""
import numpy as np
from PIL import Image
from .contracts import Blocked


def regions(image,count=2,*,include_disabled=False):
    """旧branch_regions核心：完整、对齐、等尺寸的一行卡框，保留全框。"""
    import cv2
    if count!=2:return []
    width,height=image.size;scale=min(1,1000/width)
    hsv=cv2.cvtColor(cv2.resize(np.asarray(image.convert('RGB')),None,fx=scale,fy=scale),cv2.COLOR_RGB2HSV)
    # 原实现先走完整Discover卡的绿色轮廓；直接混入金/木色会把
    # 两张牌和酒馆木框连成一个巨大轮廓，真实联盟旗帜就会漏检。
    green=cv2.inRange(hsv,np.array([35,95,100]),np.array([112,255,255]))
    contours,_=cv2.findContours(cv2.morphologyEx(green,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    standard=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if .12*width*scale<w<.30*width*scale and .29*height*scale<h<.63*height*scale and .18*height*scale<y<.48*height*scale and .3<w/h<1:
            standard.append((x/(width*scale),y/(height*scale),w/(width*scale),h/(height*scale)))
    standard.sort()
    if (len(standard)==count and max(b[1] for b in standard)-min(b[1] for b in standard)<.045
            and standard[1][0]+standard[1][2]/2-standard[0][0]-standard[0][2]/2>.15
            and standard[0][0]+standard[0][2]<standard[1][0]):return standard
    mask=cv2.inRange(hsv,np.array([12,95,100]),np.array([112,255,255]))
    if include_disabled:mask=cv2.bitwise_or(mask,cv2.inRange(hsv,np.array([0,0,60]),np.array([179,45,210])))
    contours,_=cv2.findContours(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frames=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (.075*width*scale<w<.28*width*scale and .18*height*scale<h<.63*height*scale
                and .15*height*scale<y<.52*height*scale and .3<w/h<1):continue
        if x<.08*width*scale or x+w>.92*width*scale:continue
        supports=[]
        for rx in (x,x+round(w*.9)):
            rail=mask[y+round(h*.18):y+round(h*.82),rx:rx+max(3,round(w*.1))]
            supports.append(float(np.mean(np.any(rail>0,axis=1))) if rail.size else 0)
        if min(supports)<.75:continue
        frames.append((x/(width*scale),y/(height*scale),w/(width*scale),h/(height*scale)))
    frames.sort()
    if len(frames)!=count:return []
    if max(b[1] for b in frames)-min(b[1] for b in frames)>.035:return []
    if max(b[2] for b in frames)/min(b[2] for b in frames)>1.15:return []
    if max(b[3] for b in frames)/min(b[3] for b in frames)>1.15:return []
    if frames[1][0]-frames[0][0]<.12 or frames[0][0]+frames[0][2]>=frames[1][0]:return []
    return frames


def inspect(image,profile,data,pending):
    if not pending or not pending[0].binding.branches:return {}
    d,before,_=pending;b=d.binding
    if data['game_id']!=d.game_id or data['turn']!=before['client']['turn']:return {}
    source=next((c for z in ('hand','board','controls') for c in before['client'].get(z,[]) if c['entity_id']==b.source),None)
    current_source=next((c for z in ('hand','board','controls') for c in data.get(z,[]) if c['entity_id']==b.source),None)
    current={c['entity_id']:c for c in data.get('branches',[])};ids=[row[0] for row in b.branches]
    if (not source or not current_source or len(ids)!=2 or any(e not in current for e in ids)
            or any(source.get(k)!=current_source.get(k) for k in ('card_id','controller','card_type'))):return {}
    from .practical import PracticalGeometry
    synthetic={**data,'choices':[current[e] for e in ids],'interaction':dict(source_entity=b.source)}
    points,_=PracticalGeometry(profile).locations(dict(client=synthetic,view=dict(pending=3)))
    boxes=regions(image) or regions(image,include_disabled=True)
    evidence='legacy_complete_branch_row_at_calibrated_hits'
    if not boxes:
        # Background highlights can join a branch frame and distort its outer
        # contour. Require independent long rails around each pinned hit.
        import cv2
        from .choice_ui import paired_rails
        width,height=image.size;scale=min(1,1000/width)
        hsv=cv2.cvtColor(cv2.resize(np.asarray(image.convert('RGB')),None,fx=scale,fy=scale),cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,np.array([35,95,100]),np.array([112,255,255]))
        proof=paired_rails(mask,[points['entity:'+str(e)] for e in ids],width,height,scale)
        if proof['visible']:boxes=proof['boxes'];evidence='paired_straight_branch_rails_at_pinned_hits'
    # 继续使用既有校准命中点；完整行只证明当前这些位置确有卡框。
    if len(boxes)!=2:return dict(visible=False,boxes=[])
    for eid,(x,y,w,h) in zip(ids,boxes):
        px,py=points['entity:'+str(eid)]
        if not x+w*.15<=px<=x+w*.85 or not y+h*.15<=py<=y+h*.85:return dict(visible=False,boxes=[])
    import cv2
    hsv=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2HSV)
    colored=cv2.inRange(hsv,np.array([12,95,100]),np.array([112,255,255]));selectable=[]
    for x,y,w,h in boxes:
        supports=[]
        for left in (x,x+w*.9):
            rail=colored[round((y+h*.18)*image.height):round((y+h*.82)*image.height),round(left*image.width):round((left+w*.1)*image.width)]
            supports.append(float(np.mean(np.any(rail>0,axis=1))) if rail.size else 0)
        selectable.append(min(supports)>=.65)
    return dict(visible=True,boxes=boxes,selectable=selectable,source_card_id=source['card_id'],entities=ids,
                evidence=evidence)


class Occlusion:
    """旧BranchOcclusion局部多尺度匹配；只阻挡实际被大牌覆盖的目标。"""
    def __init__(self,image,box):
        import cv2
        self.size=(min(960,image.width),round(image.height*min(960,image.width)/image.width))
        image=image.resize(self.size,Image.Resampling.BILINEAR)
        x,y,w,h=box;self.box=tuple(round(v) for v in (x*image.width,y*image.height,w*image.width,h*image.height))
        x,y,w,h=self.box;self.offset=(round(w*.2),round(h*.15));dx,dy=self.offset
        self.template=cv2.cvtColor(np.asarray(image.crop((x+dx,y+dy,x+round(w*.8),y+round(h*.45)))),cv2.COLOR_RGB2GRAY)
        if not self.template.size or self.template.std()<10:raise Blocked('branch occlusion template has insufficient texture')
    def blocks(self,image,point):
        import cv2
        image=image.resize(self.size,Image.Resampling.BILINEAR);px,py=point[0]*image.width,point[1]*image.height
        x,y,w,h=self.box;padx=round(w*.25);pady=round(h*.25);x0=max(0,x-padx);y0=max(0,y-pady)
        roi=cv2.cvtColor(np.asarray(image.crop((x0,y0,min(image.width,x+w+padx),min(image.height,y+h+pady)))),cv2.COLOR_RGB2GRAY)
        for scale in (.9,1.,1.1):
            t=cv2.resize(self.template,None,fx=scale,fy=scale)
            if min(roi.shape[0]-t.shape[0],roi.shape[1]-t.shape[1])<0:continue
            _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(roi,t,cv2.TM_CCOEFF_NORMED))
            if score<.80:continue
            left=x0+loc[0]-self.offset[0]*scale;top=y0+loc[1]-self.offset[1]*scale
            if left<=px<=left+w*scale and top<=py<=top+h*scale:return True
        return False
