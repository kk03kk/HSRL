"""Positive hero-row evidence at calibrated slots; no input or policy decisions.

Server EntityChoices may arrive while the matching wheel is still on screen.
Observe all portrait rails plus the confirm control, then independently identify
the selected portrait's outer green halo. Artwork alone is not selection proof.
"""
import numpy as np
from PIL import Image
from .vision import crop_box


def inspect(image,profile,data):
    choices=data.get('choices',[])
    if len(choices)!=4 or any(c['card_type']!='HERO' for c in choices):
        return dict(visible=False,selected_index=None,button_enabled=False,slots=[])
    from .practical import PracticalGeometry
    points,_=PracticalGeometry(profile).locations(dict(client=data,view=dict(pending=1)))
    width=min(960,image.width)
    rgb=np.asarray(image.convert('RGB').resize((width,round(image.height*width/image.width)),Image.Resampling.BILINEAR),dtype=np.float32)
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    purple=(r>g*1.10)&(b>g*1.15)&(r>45)&(b>55)
    green=(g>150)&(g>r*1.40)&(g>b*1.15)
    def patch(array,rect):
        x0,y0,x1,y1=crop_box(rect,(rgb.shape[1],rgb.shape[0]));return array[y0:y1,x0:x1]
    def rail(mask,rect):return float(np.mean(np.any(patch(mask,rect),axis=1)))
    slots=[]
    for c in choices:
        x,y=points['entity:'+str(c['entity_id'])]
        # Upper portrait rails avoid the armor shield and lower nameplate.
        left=[x-.09,y-.07,.035,.09];right=[x+.055,y-.07,.04,.09]
        supports=[rail(purple|green,rect) for rect in (left,right)]
        halo=[rail(green,rect) for rect in (left,right)]
        portrait=patch(rgb,[x-.045,y-.06,.09,.12])
        texture=float(portrait.std(axis=(0,1)).mean())
        textured=bool(texture>10 and portrait.mean()>25)
        slots.append(dict(visible=min(supports)>.25 and textured,rails=supports,halo=min(halo),portrait_std=texture))
    rect=next(l['regions']['control:confirm'] for l in profile['layouts'] if l['family']=='hero' and l['zone']=='hud')
    button=patch(rgb,rect);br,bg,bb=button[:,:,0],button[:,:,1],button[:,:,2]
    neutral=(button.max(axis=2)-button.min(axis=2)<35)&(button.mean(axis=2)>80)
    cyan=(bb>130)&(bg>100)&(bb>br*1.12)&(bg>br*1.08)
    neutral_fraction=float(neutral.mean());cyan_fraction=float(cyan.mean())
    # The confirm control is deliberately not part of row visibility: before a
    # hero is selected the control can be absent, hidden by the hover tooltip,
    # or neutral. It is post-selection evidence only.
    visible=bool(all(s['visible'] for s in slots))
    selected=[i for i,s in enumerate(slots) if s['halo']>.40]
    return dict(visible=visible,selected_index=selected[0] if visible and len(selected)==1 else None,
        button_enabled=visible and cyan_fraction>.12,button_neutral=visible and neutral_fraction>.25,
        button_cyan_fraction=cyan_fraction,button_neutral_fraction=neutral_fraction,slots=slots)
