"""Read-only historical screenshot geometry. Crops are never treated as full clients."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .contracts import Blocked


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


@dataclass
class ArchivedFrame:
    image_path: Path
    metadata_path: Path
    image_size: tuple
    client_size: tuple
    region: tuple
    observation: dict
    metadata: dict

    @classmethod
    def load(cls,image_path,metadata_path=None):
        from PIL import Image
        image_path=Path(image_path);metadata_path=Path(metadata_path or image_path.with_suffix('.json'))
        d=json.loads(metadata_path.read_text(encoding='utf-8-sig'));sample=d.get('sample',d)
        obs=sample.get('observation')
        if not isinstance(obs,dict):raise Blocked('archive lacks observation metadata')
        capture=sample.get('capture',{});window=sample.get('window',{})
        client=window.get('client') or capture.get('window_after',{}).get('client')
        region=capture.get('region_client')
        if not client or not region:raise Blocked('archive lacks explicit client/crop coordinate transform')
        if capture.get('geometry_unchanged') is False:raise Blocked('capture crossed a window geometry change')
        with Image.open(image_path) as im:size=im.size
        archived=d.get('image',{})
        if archived.get('sha256') and archived['sha256']!=file_sha(image_path):raise Blocked('archive image SHA differs')
        if archived.get('size') and list(size)!=archived['size']:raise Blocked('archive image size differs')
        cw,ch=int(client['width']),int(client['height'])
        x,y,w,h=(int(region[k]) for k in ('x','y','width','height'))
        if not (cw>0 and ch>0 and w>0 and h>0 and 0<=x<x+w<=cw and 0<=y<y+h<=ch):
            raise Blocked('archive crop lies outside client')
        if abs(size[0]/size[1]-w/h)>.005:raise Blocked('archive image was stretched or crop metadata differs')
        return cls(image_path,metadata_path,size,(cw,ch),(x,y,w,h),obs,d)

    def point(self,normalized):
        x,y,w,h=self.region;cw,ch=self.client_size;px,py=normalized[0]*cw,normalized[1]*ch
        if not x<=px<x+w or not y<=py<y+h:raise Blocked('coordinate lies outside archived crop')
        return ((px-x)*self.image_size[0]/w,(py-y)*self.image_size[1]/h)

    def evidence(self):
        sample=self.metadata.get('sample',self.metadata)
        return dict(image=str(self.image_path),metadata=str(self.metadata_path),
            image_sha256=file_sha(self.image_path),metadata_sha256=file_sha(self.metadata_path),
            image_size=self.image_size,client_size=self.client_size,region_client=self.region,
            build=self.observation.get('build'),phase=self.observation.get('phase'),
            dpi=sample.get('window',{}).get('dpi'),historical_only=True,
            current_client_authorized=False)


def overlay(frame,profile,destination):
    from PIL import Image,ImageDraw
    from .vision import Geometry
    data=frame.observation
    if any(z not in data for z in ('hand','board','shop','choices')):raise Blocked('archive zone list missing')
    if abs(frame.client_size[0]/frame.client_size[1]-profile['aspect'])>.005:
        raise Blocked('client aspect differs from reference')
    # This selects a geometry family, not the model's strategic pending state.
    pending=1 if data['choices'] and all(c['card_type']=='HERO' for c in data['choices']) else 2 if data['choices'] else 0
    points,zones=Geometry(profile).locations(dict(client=data,view=dict(pending=pending)))
    with Image.open(frame.image_path) as source:image=source.convert('RGB')
    draw=ImageDraw.Draw(image);visible={};outside=[]
    for name,p in points.items():
        try:x,y=frame.point(p)
        except Blocked:outside.append(name);continue
        visible[name]=[x,y];draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
        draw.text((x+6,y),name,fill='yellow')
    image.save(destination)
    return dict(points_visible=visible,points_outside_crop=outside,geometry_evidence_only=True)
