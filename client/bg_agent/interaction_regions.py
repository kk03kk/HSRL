"""Semantic region allowlists and HUD exclusions; never manufacture a point."""
from functools import lru_cache
import json
from .resources import calibration_root


@lru_cache(maxsize=1)
def layout():
    data=json.loads((calibration_root()/'v05_ui/interaction_regions.json').read_text(encoding='utf-8'))
    if data.get('schema')!=1:raise ValueError('interaction_region_schema_unknown')
    return data


def contains(bounds,point,size):
    x,y=point;w,h=size;a,b,c,d=bounds
    return a*w<=x<c*w and b*h<=y<d*h


def allowed(point,size,zone=None):
    data=layout()
    return not any(contains(b,point,size) for b in data['forbidden'].values()) and (
        zone is None or zone in data['zones'] and contains(data['zones'][zone],point,size))


def mask_zone(mask,zone):
    """Crop candidates before contour search, including explicit HUD holes."""
    h,w=mask.shape[:2];a,b,c,d=layout()['zones'][zone]
    mask[:round(b*h)]=0;mask[round(d*h):]=0
    mask[:,:round(a*w)]=0;mask[:,round(c*w):]=0
    for a,b,c,d in layout()['forbidden'].values():mask[round(b*h):round(d*h),round(a*w):round(c*w)]=0
    return mask
