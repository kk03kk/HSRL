"""Offline inventory and calibration using the unchanged visual adapters."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict,replace
import hashlib,json,time
from pathlib import Path

from PIL import Image

from .adaptive_spatial import AdaptiveSpatialMapper
from .hand_mapping import hand_markers,hand_card_frames,hand_frame_binding
from .layout_calibration import fit_layout,normalized,spec_dict
from .layout_resolver import choice_family,hand_variant
from .live_client import menu_button,result_panel,hero_regions,confirm_button,power_choice_regions
from .live_interactions import discover_regions,hud_mappings
from .live_choices import quest_regions
from .model import Observation,Kind
from .resources import calibration_root


def visual_projection(mapper,obs,image,window):
    maps=mapper.map(obs,image,window)
    def replace_entities(cards,boxes):
        keys={f'entity:{c.entity_id}' for c in cards}
        maps[:]=[m for m in maps if m.key not in keys]
        from .mapping import Mapping
        maps.extend(Mapping(f'entity:{c.entity_id}','matched',b,.9,'independent_visual_calibration',entity_id=c.entity_id) for c,b in zip(cards,boxes))
    if obs.phase=='recruit':
        measured=hand_markers(getattr(mapper,'last_frame_pixels',None) or image,obs.hand,
            fan_hint=mapper.last_diagnostics.get('hand_fan'))
        if measured:replace_entities([c for c,b in measured],[b for c,b in measured])
        frames=hand_frame_binding(image,obs.hand)
        if frames:
            # The independent frame proof contains a safe horizontal interval,
            # not just the old detector's arbitrarily selected 10-pixel dot.
            from .windows import Box
            boxes=[];margin=max(3,image.width*.002)
            for i,(card,body,hit) in enumerate(frames):
                left=max(body.x+body.width*.25,frames[i-1][1].x+frames[i-1][1].width+margin if i else body.x)
                right=min(body.x+body.width*.8,frames[i+1][1].x-margin if i+1<len(frames) else body.x+body.width)
                if right-left<2:break
                boxes.append(Box(round(left),round(body.y+body.height*.28),round(right-left),max(2,round(body.height*.3))))
            if len(boxes)==len(frames):replace_entities([c for c,b,h in frames],boxes)
        maps.extend(hud_mappings(obs,image,maps))
    if obs.phase=='choice':
        family=choice_family(obs);boxes=[]
        if family=='hero':boxes=hero_regions(image,len(obs.choices))
        elif family=='power':boxes=power_choice_regions(image,len(obs.choices))
        elif family=='quest':boxes=quest_regions(image,len(obs.choices))
        elif family=='discover':boxes=discover_regions(image,len(obs.choices))
        elif family=='multi':
            from .live_interactions import multi_choice_state
            from .windows import Box
            state=multi_choice_state(image,len(obs.choices))
            if state:boxes=[Box(round(b.x+b.width*.3),round(b.y+b.height*.25),round(b.width*.4),round(b.height*.3)) for b in state[0]]
        if boxes:replace_entities(obs.choices,boxes)
        elif family!='trinket':
            keys={f'entity:{c.entity_id}' for c in obs.choices}
            maps[:]=[m for m in maps if m.key not in keys]
    return maps


def observations(obs,image,maps,capture,group,sha256,diagnostics=None):
    visible={m.key:m.rect for m in maps if m.status=='matched' and m.rect}
    rows=[]
    def add(family,zone,cards,variant='default'):
        regions={};bounds={};locked=[]
        for index,card in enumerate(cards):
            lock=card.tags.get('LITERALLY_UNPLAYABLE') in ('1',1);locked.append(lock)
            box=visible.get(f'entity:{card.entity_id}')
            if box is None and not lock:return
            if not lock:
                key=f'slot:{index+1}';regions[key]=normalized(box,image.size)
                outer=box
                if zone=='choices':
                    from .windows import Box
                    # Invert the independent adapter's documented safe inset;
                    # these are probe bounds, not another source locator.
                    factors={'hero':(.3,.2,.4,.25),'discover':(.3,.25,.4,.3),
                             'power':(.3,.35,.4,.3),'quest':(.25,.25,.5,.5),
                             'multi':(.3,.25,.4,.3)}
                    if family in factors:
                        x,y,w,h=factors[family]
                        outer=Box(round(box.x-box.width*x/w),round(box.y-box.height*y/h),round(box.width/w),round(box.height/h))
                    elif family=='trinket':
                        from .windows import Box
                        outer=Box(round(box.x-box.width*.18/.64),round(box.y-box.height*.6/.8),round(box.width/.64),round(box.height*2/.8))
                bounds[key]=normalized(outer,image.size)
        if not cards:
            if zone=='board':
                regions={key:normalized(box,image.size) for key,box in visible.items() if key=='region:board'}
            else:
                regions={key:normalized(box,image.size) for key,box in visible.items() if key.startswith(('control:','scene:','region:')) and key!='region:board'}
            # Controls discovered through an Options entity still belong to a
            # game-mode slot, never to this particular entity id or card art.
            for c in obs.controls if zone=='hud' else ():
                box=visible.get(f'entity:{c.entity_id}');slot=c.tags.get('GAME_MODE_BUTTON_SLOT')
                if box and slot is not None:regions['control:game_mode_slot:'+str(slot)]=normalized(box,image.size)
        elif zone=='board':
            # Minion centres do not prove insertion gaps. Preserve only the
            # independently observed wood/gap evidence from the teacher.
            regions.update({key:normalized(box,image.size) for key,box in visible.items()
                if key.startswith('position:board:') or key=='region:board'})
        if not regions:return
        key=(family,zone,len(cards),variant,obs.build,round(image.width/image.height,6))
        probes={}
        evidence=(diagnostics or {}).get('trinket_choice',{})
        if family=='trinket' and zone=='choices' and evidence.get('title') and evidence.get('rails'):
            from .windows import Box
            title=Box(**evidence['title']);unit=title.height;rails=evidence['rails']
            for i in range(len(cards)):
                a,b=rails[2*i:2*i+2]
                probes[f'selection:{i+1}']=normalized(Box(a[0],round(title.y+unit*1.8),b[1]-a[0],round(unit*1.2)),image.size)
        row=dict(capture=capture,group=group,sha256=sha256,regions=regions,locked=locked,bounds=bounds,local_probes=probes)
        rows.append((key,row))
    if obs.phase=='recruit' and 'scene:recruit' in visible:
        add('recruit','hud',())
        for zone in ('shop','board','hand'):
            cards=tuple(sorted(getattr(obs,zone),key=lambda c:c.position))
            if cards and [c.position for c in cards]==list(range(1,len(cards)+1)):
                add('recruit',zone,cards,hand_variant(cards) if zone=='hand' else 'default')
            elif zone=='board' and not cards:add('recruit','board',())
    elif obs.phase=='choice' and obs.choices:
        add(choice_family(obs),'choices',obs.choices)
        box=visible.get('control:confirm') if choice_family(obs)=='trinket' else confirm_button(image)
        if box:
            rows.append(((choice_family(obs),'hud',0,'default',obs.build,round(image.width/image.height,6)),
                dict(capture=capture,group=group,sha256=sha256,regions={'control:confirm':normalized(box,image.size)},locked=[])))
    elif obs.phase in ('menu','game_over'):
        box=menu_button(image) if obs.phase=='menu' else result_panel(image)
        if box:
            name='enter' if obs.phase=='menu' else 'dismiss'
            rows.append(((str(obs.phase),'hud',0,'default',obs.build,round(image.width/image.height,6)),
                dict(capture=capture,group=group,sha256=sha256,regions={'control:'+name:normalized(box,image.size)},locked=[])))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--catalog',type=Path)
    args=parser.parse_args();root=args.root.resolve();args.out.mkdir(parents=True,exist_ok=False)
    mapper=AdaptiveSpatialMapper(calibration_root()/'v05_ui/profile.json',calibration_root()/'v05_hand/profile.json')
    groups=defaultdict(list);inventory=[];errors=[];counts=defaultdict(Counter);started=time.perf_counter()
    sources=[root/'Backup/development/tests/fixtures',root/'Backup/development/runtime']
    sources+=sorted((root/'runtime').glob('live-play-*'))
    for source in sources:
        for metadata in sorted(source.rglob('*.json')):
            image_path=next((metadata.with_suffix(ext) for ext in ('.png','.jpg') if metadata.with_suffix(ext).is_file()),None)
            if image_path is None:continue
            try:
                data=json.loads(metadata.read_text(encoding='utf-8'));sample=data.get('sample',data)
                if not isinstance(sample,dict) or 'observation' not in sample:continue
                region=sample.get('capture',{}).get('region_client')
                if region and region!=dict(x=0,y=0,width=sample.get('window',{}).get('client',{}).get('width'),height=sample.get('window',{}).get('client',{}).get('height')):continue
                obs=Observation.from_dict(sample['observation'])
                if obs.phase=='recruit' and (not obs.ready or obs.client_task_list is not None):continue
                if obs.phase=='recruit' and str(data.get('stage','')).startswith(('branch','target','choice','confirm')):continue
                if any(reason in sample.get('blockers',()) for reason in ('recruit_action_processing','client_task_list_processing','log_changed_during_capture')):continue
                with Image.open(image_path) as original:image=original.convert('RGB')
                relative=str(metadata.relative_to(root))
                parts=metadata.parent.relative_to(source).parts
                group=source.name if source.name.startswith('live-play-') else '/'.join(parts[:2] if parts[0]=='bridge' else parts[:1])
                digest=hashlib.sha256(image_path.read_bytes()).hexdigest()
                maps=visual_projection(mapper,obs,image,sample.get('window',{}))
                rows=observations(obs,image,maps,relative,group,digest,mapper.last_diagnostics)
                for key,row in rows:groups[key].append(row)
                inventory.append(dict(metadata=relative,image=str(image_path.relative_to(root)),group=group,
                    sha256=digest,build=obs.build,size=image.size,phase=str(obs.phase),layouts=[list(k) for k,r in rows]))
                for zone in ('hand','board','shop','choices'):counts[zone][len(getattr(obs,zone))]+=1
            except (ValueError,KeyError,TypeError,OSError) as exc:errors.append(dict(file=str(metadata),reason=str(exc)))
    # Archived fixtures and live-run folders can contain the same captures.
    # Such aliases are one holdout group, never independent validation games.
    parents={row['group']:row['group'] for row in inventory};seen={}
    def canonical(value):
        while parents[value]!=value:
            parents[value]=parents[parents[value]];value=parents[value]
        return value
    for row in inventory:
        previous=seen.setdefault(row['sha256'],row['group'])
        a,b=canonical(previous),canonical(row['group'])
        if a!=b:parents[max(a,b)]=min(a,b)
    for row in inventory:row['group']=canonical(row['group'])
    for rows in groups.values():
        for row in rows:row['group']=canonical(row['group'])
    layouts=[];gaps=[]
    for key,rows in groups.items():
        spec=fit_layout(*key,rows)
        if spec:layouts.append(spec_dict(spec))
        if spec is None or spec.status!='verified':gaps.append(dict(layout=list(key),frames=len(rows),groups=sorted({r['group'] for r in rows})))
    report=dict(schema=1,frames=len(inventory),counts={k:dict(sorted(v.items())) for k,v in counts.items()},
        errors=errors,gaps=gaps,verified_layouts=sum(s['status']=='verified' for s in layouts),
        elapsed_seconds=time.perf_counter()-started,frames_inventory=inventory)
    (args.out/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    catalog=dict(schema=1,reference='independent_visual_calibration_and_cross_run_validation',layouts=layouts)
    target=args.catalog or args.out/'catalog.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('frames_inventory','errors','gaps')},ensure_ascii=True))


if __name__=='__main__':main()
