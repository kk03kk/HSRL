"""只识别已点击分支的卡图遮挡，不对普通动作施加动画/稳定条件。"""
import cv2
import numpy as np


class BranchOcclusion:
    def __init__(self,image,box):
        self.box=box
        self.offset=(round(box.width*.2),round(box.height*.15))
        x,y=self.offset
        self.template=cv2.cvtColor(np.asarray(image.crop((box.x+x,box.y+y,
            box.x+round(box.width*.8),box.y+round(box.height*.45)))),cv2.COLOR_RGB2GRAY)

    def blocks(self,image,point):
        b=self.box;padx=round(b.width*.25);pady=round(b.height*.25)
        x0=max(0,b.x-padx);y0=max(0,b.y-pady)
        roi=cv2.cvtColor(np.asarray(image.crop((x0,y0,min(image.width,b.x+b.width+padx),
            min(image.height,b.y+b.height+pady)))),cv2.COLOR_RGB2GRAY)
        if self.template.std()<10:return False
        for scale in (.9,1.,1.1):
            t=cv2.resize(self.template,None,fx=scale,fy=scale)
            if min(roi.shape[0]-t.shape[0],roi.shape[1]-t.shape[1])<0:continue
            _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(roi,t,cv2.TM_CCOEFF_NORMED))
            if score<.80:continue
            left=x0+loc[0]-self.offset[0]*scale;top=y0+loc[1]-self.offset[1]*scale
            if left<=point[0]<=left+b.width*scale and top<=point[1]<=top+b.height*scale:return True
        return False
