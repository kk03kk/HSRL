"""Conservative log-first mapping using evidence-backed local image templates.

No built-in Hearthstone pixel positions. Unknown layouts/cards remain unknown.
Calibrations are explicit annotations on synchronized real screenshots, and each
new frame must independently match their appearance in the declared slot area.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from enum import StrEnum

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .windows import Box


class StabilityBasis(StrEnum):
    PIXELS='pixels'
    GEOMETRY='geometry'
    CONTROL_FRAME='control_frame'
    RECEIVING_REGION='receiving_region'


@dataclass(frozen=True)
class Mapping:
    key: str
    status: str
    rect: Box | None
    confidence: float
    reason: str
    calibration: str | None = None
    entity_id: int | None = None
    stability_basis: StabilityBasis = StabilityBasis.PIXELS


def calibrate(sample_path: Path,annotations_path: Path,out: Path):
    """Build an appearance profile from reviewed annotations, never self-label."""
    row=json.loads(sample_path.read_text(encoding="utf-8"))
    annotations=json.loads(annotations_path.read_text(encoding="utf-8"))
    if row["blockers"] or not row["log_snapshot_unchanged"]:raise ValueError("calibration sample must be synchronized and stable")
    frame=Path(row["capture"]["path"])
    if hashlib.sha256(frame.read_bytes()).hexdigest()!=row["capture"]["sha256"]:raise ValueError("source frame hash mismatch")
    image=Image.open(frame).convert("RGB");width,height=image.size
    if annotations.get("reviewed") is not True:raise ValueError("annotations require explicit visual review")
    out.mkdir(parents=True,exist_ok=False)
    obs=row["observation"];entries=[]
    for index,a in enumerate(annotations["regions"]):
        key=a["key"];entry=dict(a)
        if a.get("zone") in ("shop","hand","board","choices"):
            cards=obs[a["zone"]]
            card=next((c for c in cards if c["entity_id"]==a.get("entity_id")),None)
            if not card:raise ValueError("annotation entity not in observed zone")
            entry.update(card_id=card["card_id"],position=card["position"] if a["zone"]!="choices" else cards.index(card)+1,
                         count=len(cards))
        rect=a["rect"]
        x,y,w,h=rect
        if min(w,h)<8 or min(x,y)<0 or x+w>width or y+h>height:raise ValueError("invalid annotation box")
        patch=a.get("patch",rect);px,py,pw,ph=patch
        anchor=next((r for r in annotations["regions"] if r["key"]==a.get("anchor")),None)
        ax,ay,aw,ah=anchor["rect"] if anchor else rect
        if a.get("anchor") and anchor is None:raise ValueError("missing reviewed anchor")
        if min(pw,ph)<8 or px<ax or py<ay or px+pw>ax+aw or py+ph>ay+ah:raise ValueError("template must lie within target or explicit reviewed anchor")
        name=f"template-{index:03d}.png"
        image.crop((px,py,px+pw,py+ph)).save(out/name)
        entry.update(template=name,rect_uv=[x/width,y/height,w/width,h/height],
                     patch_uv=[px/width,py/height,pw/width,ph/height],key=key)
        entries.append(entry)
    profile=dict(schema=1,source_sample=str(sample_path.resolve()),source_sha256=row["capture"]["sha256"],
                 source_window=row["window"]["identity"],phase=obs["phase"],build=obs["build"],
                 width=width,height=height,entries=entries,review_note=annotations.get("note",""))
    (out/"profile.json").write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding="utf-8")
    return profile


class EntityMapper:
    def __init__(self,profiles=()):
        self.profiles=[]
        for path in profiles:
            path=Path(path)
            data=json.loads(path.read_text(encoding="utf-8"))
            for entry in data["entries"]:
                # Pillow handles non-ASCII Windows paths (e.g. the user's home).
                template=np.asarray(Image.open(path.parent/entry["template"]).convert("L"))
                self.profiles.append((path,data,entry,template))

    def map(self,obs,image,window,*,blocked=False):
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
        width,height=image.size
        results=[]
        requested=[]
        for zone in ("shop","hand","board","choices"):
            group=getattr(obs,zone)
            positions=[c.position for c in group]
            valid=zone=="choices" or (len(set(positions))==len(positions) and all(p>0 for p in positions))
            for index,card in enumerate(group):
                requested.append((f"entity:{card.entity_id}",zone,index+1 if zone=="choices" else card.position,len(group),card.card_id,card.entity_id,valid))
        # Controls/destinations are only mapped if explicitly calibrated.
        for key in ("scene:recruit","control:refresh","control:upgrade","control:hero_power","control:freeze","region:hand","region:board","region:bob"):
            requested.append((key,None,None,None,None,None,True))
        for key,zone,position,count,card_id,eid,valid in requested:
            if blocked or not valid:
                results.append(Mapping(key,"unknown",None,0,"blocked_frame" if blocked else "log_slot_order_unknown",entity_id=eid));continue
            candidates=[]
            for path,profile,entry,template in self.profiles:
                if profile["phase"]!=str(obs.phase) or (profile.get("build") and profile["build"]!=obs.build):continue
                if abs((width/height)/(profile["width"]/profile["height"])-1)>.015:continue
                if zone:
                    if any(entry.get(k)!=v for k,v in (("zone",zone),("position",position),("count",count),("card_id",card_id))):continue
                elif entry["key"]!=key:continue
                rx,ry,rw,rh=entry["rect_uv"];px,py,pw,ph=entry["patch_uv"]
                tw,th=max(8,round(pw*width)),max(8,round(ph*height))
                templ=cv2.resize(template,(tw,th))
                margin_x,margin_y=round(width*.012),round(height*.012)
                x0=max(0,round(px*width)-margin_x);y0=max(0,round(py*height)-margin_y)
                x1=min(width,round(px*width)+tw+margin_x);y1=min(height,round(py*height)+th+margin_y)
                roi=gray[y0:y1,x0:x1]
                if roi.shape[0]<th or roi.shape[1]<tw or np.std(templ)<8:continue
                scores=cv2.matchTemplate(roi,templ,cv2.TM_CCOEFF_NORMED)
                _,score,_,loc=cv2.minMaxLoc(scores)
                if score<.92:continue
                matched=roi[loc[1]:loc[1]+th,loc[0]:loc[0]+tw]
                # Normalized correlation alone also matches dimmed modal backdrops.
                if np.mean(np.abs(matched.astype(np.float32)-templ.astype(np.float32)))>18:continue
                dx=x0+loc[0]-px*width;dy=y0+loc[1]-py*height
                box=Box(round(rx*width+dx),round(ry*height+dy),round(rw*width),round(rh*height))
                if box.x<0 or box.y<0 or box.x+box.width>width or box.y+box.height>height:continue
                candidates.append((score,box,str(path.resolve())))
            if candidates:
                score,box,path=max(candidates,key=lambda c:c[0])
                # Incompatible reviewed profiles must not silently disagree.
                if any(abs(box.x-b.x)>width*.03 or abs(box.y-b.y)>height*.03 for _,b,_ in candidates):
                    results.append(Mapping(key,"unknown",None,score,"conflicting_profiles",entity_id=eid))
                else:results.append(Mapping(key,"matched",box,score,"reviewed_slot_and_appearance_match",path,eid))
            else:results.append(Mapping(key,"unknown",None,0,"no_reviewed_matching_layout_or_appearance",entity_id=eid))
        return results


def overlay(image,mappings,plan,path):
    image=image.copy();draw=ImageDraw.Draw(image)
    for m in mappings:
        if m.rect:
            b=m.rect;color="#58dbbd"
            draw.rectangle((b.x,b.y,b.x+b.width,b.y+b.height),outline=color,width=4)
            draw.text((b.x,b.y-14),f"{m.key} {m.confidence:.3f}",fill=color)
    route=[]
    for index,step in enumerate(plan.get("steps",[]),1):
        p=step.get("point_client")
        if p:
            x,y=p;route.append((x,y));draw.ellipse((x-10,y-10,x+10,y+10),outline="#ffcb66",width=5)
            label=f"{index}: {step['kind']} {step.get('region','')} (proposed point)"
            tx=max(8,min(x+16,image.width-380));ty=max(50,y-26)
            draw.rectangle((tx-3,ty-3,min(image.width,tx+370),ty+16),fill="#122036")
            draw.text((tx,ty),label,fill="#ffcb66")
            region=next((m for m in mappings if m.key==step.get('region') and m.rect),None)
            if region:
                b=region.rect
                draw.rectangle((b.x,b.y,b.x+b.width,b.y+b.height),outline="#ffcb66",width=6)
    if len(route)>1:draw.line(route,fill="#ffcb66",width=3)
    draw.rectangle((0,0,image.width,44),fill="#122036")
    action=plan['action']
    text=f"SHADOW ONLY / {action['kind']} source={action.get('source')} target={action.get('target')} / {plan['status']} / rev {plan['revision']} / NO INPUT"
    draw.text((12,12),text,fill="#ffffff")
    if path is not None:image.save(path,format="PNG",compress_level=1)
    return image
