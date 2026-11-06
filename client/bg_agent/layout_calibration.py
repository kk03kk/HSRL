"""Fit safe geometry from independent visual observations, never predictions."""
from dataclasses import asdict
import hashlib
import statistics

from .layout_resolver import LayoutSpec


def normalized(box,size):
    w,h=size
    return [box.x/w,box.y/h,box.width/w,box.height/h]


def fit_layout(family,zone,count,variant,build,aspect,observations,*,session=None):
    """Every proposed hit rectangle must lie inside every observed safe box.

    A verified disk calibration also needs independent capture groups. A live
    session calibration needs two distinct captures; it cannot cross sessions.
    Only the calibration groups fit parameters; holdout groups validate them.
    """
    unique={row['capture']:row for row in observations};rows=list(unique.values())
    if not session:
        seen=set();independent=[]
        for row in rows:
            identity=row.get('sha256') or row['capture']
            if identity not in seen:independent.append(row);seen.add(identity)
        rows=independent
    if len(rows)<2:return None
    groups=sorted({row['group'] for row in rows})
    train=[row for row in rows if row['group']!=groups[-1]] if len(groups)>1 else rows
    holdout=[row for row in rows if row['group']==groups[-1]] if len(groups)>1 else []
    keys=set.union(*(set(row['regions']) for row in rows)) if zone=='hud' else set.intersection(*(set(row['regions']) for row in rows))
    if zone in ('shop','board','hand','choices') and any(f'slot:{i+1}' not in keys for i in range(count)):
        # Locked cards are physical slots without a legal source point.
        if not all(f'slot:{i+1}' in keys or all(row.get('locked',[])[i] for row in rows) for i in range(count)):
            return None
    regions={}
    for key in sorted(keys):
        available=[row for row in rows if key in row['regions']]
        if len(available)<2 or not session and len({row['group'] for row in available})<2:continue
        boxes=[row['regions'][key] for row in available]
        left=max(b[0] for b in boxes);top=max(b[1] for b in boxes)
        right=min(b[0]+b[2] for b in boxes);bottom=min(b[1]+b[3] for b in boxes)
        if right-left<.0007 or bottom-top<.0007:
            if key.startswith('slot:'):return None
            continue
        regions[key]=[left,top,right-left,bottom-top]
    if not regions:return None
    model={}
    if zone in ('board','shop','choices') and count:
        points=[]
        for row in train:
            points.append([(row['regions'][f'slot:{i+1}'][0]+row['regions'][f'slot:{i+1}'][2]/2,
                            row['regions'][f'slot:{i+1}'][1]+row['regions'][f'slot:{i+1}'][3]/2) for i in range(count)])
        cx=statistics.median(sum(p[0] for p in row)/count for row in points)
        cy=statistics.median(p[1] for row in points for p in row)
        pitch=statistics.median(row[i+1][0]-row[i][0] for row in points for i in range(count-1)) if count>1 else .08
        margins=[]
        for i in range(count):
            x=cx+(i-(count-1)/2)*pitch;b=regions[f'slot:{i+1}']
            margins.append((min(x-b[0],b[0]+b[2]-x),min(cy-b[1],b[1]+b[3]-cy)))
        if min(min(x,y) for x,y in margins)>.0004:
            size=[min(.04,min(x for x,y in margins)*1.8),min(.05,min(y for x,y in margins)*1.8)]
            model=dict(kind='centered_row',cx=cx,cy=cy,pitch=pitch,size=size)
            for i in range(count):regions.pop(f'slot:{i+1}',None)
    status='session_calibrated' if session else 'verified' if holdout and train else 'candidate'
    identity=f'{family}:{zone}:{count}:{variant}:{build}:{aspect:.5f}'
    provenance=tuple(dict(capture=row['capture'],group=row['group'],sha256=row.get('sha256'),
                          role='validation' if row in holdout else 'calibration') for row in rows)
    bounds={}
    for key in keys:
        values=[row.get('bounds',{}).get(key) for row in train]
        values=[v for v in values if v]
        if values:bounds[key]=[statistics.median(v[i] for v in values) for i in range(4)]
    local_probes={}
    for key in set.intersection(*(set(row.get('local_probes',{})) for row in rows)):
        values=[row['local_probes'][key] for row in rows]
        x=max(v[0] for v in values);y=max(v[1] for v in values)
        right=min(v[0]+v[2] for v in values);bottom=min(v[1]+v[3] for v in values)
        if right>x and bottom>y:local_probes[key]=[x,y,right-x,bottom-y]
    return LayoutSpec(hashlib.sha256(identity.encode()).hexdigest()[:16],family,zone,count,build,aspect,
                      regions,variant,status,provenance,model,dict(slot_bounds=bounds,local=local_probes),session=session)


def spec_dict(spec):return asdict(spec)
