"""Local contour displacement. Neither card artwork matching nor input authority."""
import numpy as np
from .layout_resolver import Viewport


def observe_edge(image,rect):
    x,y,w,h=rect
    pixels=np.asarray(image.crop((x,y,x+w,y+h)).convert('L'),dtype=np.float32)
    if min(pixels.shape)<3:return None
    profile=np.abs(np.diff(pixels,axis=1)).mean(axis=0)
    index=int(np.argmax(profile));strength=float(profile[index])
    return (x+index+.5,strength)


def fit_edge(train,held,rect,*,tolerance=3):
    """Fit training only, admit only a separate captured edge observation."""
    a=observe_edge(train,rect);b=observe_edge(held,rect)
    if not a or not b or min(a[1],b[1])<12 or abs(a[0]-b[0])>tolerance:return None
    return dict(rect=[rect[0]/2560,rect[1]/1600,rect[2]/2560,rect[3]/1600],
        expected=a[0]/2560,minimum_contrast=max(8,min(a[1],b[1])*.45),tolerance=tolerance,
        validated=True,method='independent_external_vertical_contour')


def sample_edges(image,viewport,probes):
    values=[]
    for probe in probes:
        if not probe.get('validated'):return None
        box=viewport.box(probe['rect']);sample=observe_edge(image,(box.x,box.y,box.width,box.height))
        if not sample or sample[1]<probe['minimum_contrast']:return None
        expected=viewport.render.x+probe['expected']*viewport.render.width
        if abs(sample[0]-expected)>probe['tolerance']:return None
        values.append(sample[0])
    return tuple(values) if values else None
