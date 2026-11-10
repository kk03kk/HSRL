"""Protocol-bound UI integration for explicitly approximate autoplay."""
import copy
import hashlib
import time
import re
import numpy as np
from .contracts import Blocked
from .interactions import choice_context
from .vision import Geometry,crop_box,stable

class PracticalGeometry(Geometry):
    def __init__(self,profile):
        p=copy.deepcopy(profile);p['review_state']='inherited_geometry'
        layouts=p['layouts']
        def existing(f,z,n):return next((r for r in layouts if (r['family'],r['zone'],r['count'])==(f,z,n)),None)
        for family in ('quest','power','timewarp','branch_minion','branch_spell'):
            if not existing(family,'hud',0):
                item=copy.deepcopy(existing('discover','hud',0));item['family']=family
                item['provenance']='inherited HUD controls, not independent scene qualification';layouts.append(item)
        for family in ('hero','trinket'):
            for n in range(1,4):
                if not existing(family,'choices',n):
                    item=copy.deepcopy(existing(family,'choices',4));item.update(count=n,
                        provenance='inherited centered card spacing; real layout qualification pending');layouts.append(item)
        for n in range(1,5):
            if not existing('timewarp','choices',n):
                item=copy.deepcopy(existing('discover','choices',n));item.update(family='timewarp',
                    provenance='discover-family projection; real timewarp qualification pending');layouts.append(item)
        for n in (5,6):
            if not existing('timewarp','choices',n):
                layouts.append(dict(family='timewarp',zone='choices',count=n,regions={},
                    model=dict(kind='centered_row',cx=.5,cy=.413,pitch=.18,size=[.05,.13]),
                    provenance='runtime-derived centered row; actual rail proof required before input'))
        for n in (5,6):
            if not existing('discover','choices',n):
                # 时空扭曲等大卡面板在 255790 实测会出现 5 个选项；旧表只标定
                # 1..4，输入因此整整暂停 15 s。这里给出可审计的运行时投影，
                # 但点击仍必须通过 choice_ui 的实际绿色/金色横梁证明。
                layouts.append(dict(family='discover',zone='choices',count=n,regions={},
                    model=dict(kind='centered_row',cx=.5,cy=.413,pitch=.18,size=[.05,.13]),
                    provenance='runtime-derived centered row; actual rail proof required before input'))
        if not existing('recruit','shop',8):
            item=existing('recruit','shop',7)
            points=[(r[0]+r[2]/2,r[1]+r[3]/2) for k,r in sorted(item['regions'].items()) if k.startswith('slot:')]
            xs=sorted(x for x,y in points)
            layouts.append(dict(family='recruit',zone='shop',count=8,regions={},
                model=dict(kind='centered_row',cx=sum(xs)/len(xs),cy=sum(y for x,y in points)/len(points),pitch=(xs[-1]-xs[0])/6),
                provenance='seven-slot row spacing extended to eight; real layout qualification pending'))
        super().__init__(p)
    def family(self,snapshot):
        pending=snapshot['view']['pending']
        if pending==3:
            source=(snapshot['client'].get('interaction') or {}).get('source_entity')
            card=next((c for z in ('hand','board','controls') for c in snapshot['client'].get(z,[]) if c['entity_id']==source),None)
            if card:return 'branch_minion' if card['card_type']=='MINION' else 'branch_spell'
        if pending==5:return 'power'
        return super().family(snapshot)
    def locations(self,snapshot):
        preview=snapshot.get('target_preview')
        if preview:
            client=copy.deepcopy(snapshot['client'])
            # Observation.to_dict() publishes zone containers as tuples; the
            # preview needs an ordered mutable copy. Mutating the published
            # tuple raised AttributeError at a real targeted battlecry and the
            # run died while the client waited for its target.
            board=list(client['board'])
            # The preview source is shared with the pinned pre-action snapshot;
            # renumbering it in place would corrupt that pinned identity.
            source=copy.deepcopy(preview['source']);position=preview.get('position')
            if position is None or not 0<=position<=len(board):raise Blocked('target preview insertion is unlocated')
            board.insert(position,source)
            for i,c in enumerate(board):c['position']=i+1
            client['board']=board
            client['hand']=[c for c in client['hand'] if c['entity_id']!=source['entity_id']]
            for i,c in enumerate(client['hand']):c['position']=i+1
            snapshot={**snapshot,'client':client}
        points,zones=super().locations(snapshot)
        # A derived 5/6-card row is only a search window. Once the frame gate
        # has measured the real card rectangles, click their centres rather
        # than the provisional model coordinates.
        measured=(snapshot.get('visual') or {}).get('choice_ui') or {}
        boxes=measured.get('boxes') or []
        entities=measured.get('entities') or []
        choices=snapshot.get('client',{}).get('choices') or []
        if (measured.get('visible') and len(boxes)==len(choices)
                and entities==[c['entity_id'] for c in choices]):
            for card,(x,y,w,h) in zip(choices,boxes):
                points['entity:'+str(card['entity_id'])]=(x+w/2,y+h/2)
        # Container drop areas exist with both empty and occupied zones.
        points.setdefault('region:board',(.5,.55));points.setdefault('region:hand',(.5,.925))
        points.setdefault('region:bob',(.5,.185))
        for c in snapshot.get('target_heroes',[]):
            points['entity:'+str(c['entity_id'])]=(.5,.76) if c['is_local'] else (.5,.185)
        return points,zones
    def plan(self,decision,snapshot):
        b=decision.binding
        if b.kind=='choose' and all(c['card_type']=='HERO' for c in snapshot['client']['choices']):
            p,_=self.locations(snapshot)
            return [('click',list(p['entity:'+str(b.source)])),('move_pointer',[.96,.91])]
        if b.kind=='play' and b.target is not None:
            source=next(c for c in snapshot['client']['hand'] if c['entity_id']==b.source)
            if source['card_type'] in ('SPELL','BATTLEGROUND_SPELL'):
                p,_=self.locations(snapshot)
                return [('drag',list(p['entity:'+str(b.source)]),list(p['entity:'+str(b.target)])),('move_pointer',[.96,.91])]
        if b.kind=='magnetize':
            p,_=self.locations(snapshot);x,y=p['entity:'+str(b.target)]
            return [('drag',list(p['entity:'+str(b.source)]),[x-self.profile['board_pitch']*.35,y]),('move_pointer',[.96,.91])]
        result=super().plan(decision,snapshot)
        if result:result.append(('move_pointer',[.96,.91]))
        return result

def choice_green_scores(image,profile,data):
    if not data.get('choices'):return []
    pending=1 if all(c['card_type']=='HERO' for c in data['choices']) else 2
    points,_=PracticalGeometry(profile).locations(dict(client=data,view=dict(pending=pending)))
    scores=[]
    for c in data['choices']:
        x,y=points['entity:'+str(c['entity_id'])]
        a=np.asarray(image.crop(crop_box([max(0,x-.075),max(0,y-.19),.15,.38],image.size)).convert('RGB'),np.float32)
        mask=(a[:,:,1]>150)&(a[:,:,1]>a[:,:,0]*1.3)&(a[:,:,1]>a[:,:,2]*1.12)
        scores.append(float(mask.mean()))
    return scores

def visual_state(backend,profile,data,pending=None):
    if data.get('choices') and all(c['card_type']=='HERO' for c in data['choices']) and data.get('turn') is None:
        data={**data,'turn':0}
    first,window=backend.capture();time.sleep(.22);image,after=backend.capture()
    if window.token()!=after.token():raise Blocked('window changed between frames')
    if list(image.size)!=profile['client_size'] or window.dpi!=profile['dpi']:raise Blocked('client geometry differs from autoplay profile')
    if not stable(first,image,crop_box([.23,.27,.55,.55],image.size),10.):raise Blocked('card interaction animation is still active')
    return interpret_frame(image,after,profile,data,pending),image


def confirm_control_rect(profile,data):
    """Calibrated confirm control of this choice panel's own family."""
    cards=data.get('choices',[])
    family=('trinket' if cards and all(c['card_type']=='BATTLEGROUND_TRINKET' for c in cards)
            else 'hero' if cards and all(c['card_type']=='HERO' for c in cards)
            else 'power' if cards and all(c['card_type']=='HERO_POWER' for c in cards)
            else 'quest' if any(c.get('tags',{}).get('QUEST')=='1' for c in cards)
            else 'discover')
    for layout in profile['layouts']:
        if layout['family']==family and layout['zone']=='hud':
            return layout['regions'].get('control:confirm')
    return None


def confirm_button_stats(image,profile,data):
    """Appearance of the confirm control; only *changes* authorize a click."""
    rect=confirm_control_rect(profile,data)
    if not rect:return None
    pixels=np.asarray(image.crop(crop_box(rect,image.size)).convert('RGB'),dtype=np.float32)
    if not pixels.size:return None
    red,green,blue=pixels[:,:,0],pixels[:,:,1],pixels[:,:,2]
    cyan=float(np.mean((blue>130)&(green>100)&(blue>red*1.12)&(green>red*1.08)))
    return dict(mean=round(float(pixels.mean()),2),
                spread=round(float((pixels.max(axis=2)-pixels.min(axis=2)).mean()),2),
                cyan=round(cyan,4))


def confirm_button_changed(before,now):
    """A neutral/grey control turning highlighted is the selection proof."""
    if not before or not now:return False
    return (abs(before['mean']-now['mean'])>8 or abs(before['spread']-now['spread'])>8
            or abs(before['cyan']-now['cyan'])>.05)


def interpret_frame(image,after,profile,data,pending=None):
    if data.get('choices') and all(c['card_type']=='HERO' for c in data['choices']) and data.get('turn') is None:
        data={**data,'turn':0}
    choices=data.get('choices',[])
    phase='hero' if choices and all(c['card_type']=='HERO' for c in choices) else 'discover' if choices else data['phase']
    result=dict(window_token=after.token(),phase=phase,target_open=False,confirm_enabled=False,selection=None,
        available_races_shown=False,interaction_evidence='window identity + real server phase + stable card region')
    if phase=='hero':
        from .hero_ui import inspect
        proof=inspect(image,profile,data);result['hero_ui']=proof
        result['choice_green_scores']=[slot['halo'] for slot in proof['slots']]
        if pending:
            d,before,_=pending;index=proof['selected_index']
            same=(data['game_id']==before['client']['game_id'] and data['turn']==before['client']['turn']
                  and (data.get('interaction') or {}).get('request_id')==d.binding.choice_id)
            if (same and d.binding.kind=='choose' and index is not None
                    and choices[index]['entity_id']==d.binding.source and proof['button_enabled']):
                result['selection']=dict(entity_id=d.binding.source,context=choice_context(data),
                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),evidence='measured_hero_halo_and_confirm')
                result['confirm_enabled']=True
        return result
    try:result['choice_green_scores']=choice_green_scores(image,profile,data)
    except Blocked:result['choice_green_scores']=[]
    button=confirm_button_stats(image,profile,data)
    if button is not None:result['confirm_button']=button
    if pending:
        d,before,_=pending
        same=data['game_id']==before['client']['game_id'] and data['turn']==before['client']['turn']
        if same and d.binding.targets:result['target_open']=True
        if same and d.binding.branches:
            from .branch_ui import inspect
            proof=inspect(image,profile,data,pending);result['branch_ui']=proof
            if proof.get('visible') and d.binding.sub_option_id is None:
                result.update(branch_open=True,branch_source_card_id=proof['source_card_id'],branch_entities=proof['entities'])
        interaction=data.get('interaction') or {};old=before.get('visual',{}).get('choice_green_scores',[]);new=result['choice_green_scores']
        if d.binding.kind=='choose' and interaction.get('request_id')==d.binding.choice_id and len(old)==len(new) and new:
            deltas=[b-a for a,b in zip(old,new)];index=int(np.argmax(new))
            second=max((x for i,x in enumerate(new) if i!=index),default=0)
            if (deltas[index]>.012 or new[index]>.04 and new[index]-second>.02) and choices[index]['entity_id']==d.binding.source:
                result['selection']=dict(entity_id=d.binding.source,context=choice_context(data),
                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),evidence='persistent_outline_after_pointer_exit')
                result['confirm_enabled']=True
        # Trinket purchases need an explicit 确认 after the card is selected. The
        # 21:17 and 19:48 runs left the shop open until the operator clicked it,
        # because the green-outline proof never fires on purple/gold trinkets.
        # Positive proof: the calibrated control's own appearance changed after
        # our click while the same authorized choice is still open. Once only.
        # Restricted to trinket panels: an ordinary discover resolves on click,
        # and its same-named region sits over other controls (e.g. 隐藏).
        trinket_panel=bool(choices and all(c['card_type']=='BATTLEGROUND_TRINKET' for c in choices))
        if (button is not None and trinket_panel and not result.get('confirm_enabled')
                and d.binding.kind=='choose'
                and interaction.get('request_id')==d.binding.choice_id
                and any(c['entity_id']==d.binding.source for c in choices)
                and confirm_button_changed((before.get('visual') or {}).get('confirm_button'),button)):
            result['selection']=dict(entity_id=d.binding.source,context=choice_context(data),
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                evidence='confirm_control_changed_after_selection')
            result['confirm_enabled']=True
    return result


def autoplay(root,args):
    from pathlib import Path
    from .runtime import exclusive
    with exclusive(Path(root)/'artifacts/runs/live-client.lock'):
        args.client_lock_owned=True
        return _autoplay_locked(root,args)


def _autoplay_locked(root,args):
    """Run an explicit number of games (zero means until STOP/F8).

    A batch is a *sequence of independent matches*, not one long transaction:
    a failed match is classified, archived and (within budget) followed by the
    next one. Only an unsafe class (S0) or an exhausted failure budget stops the
    batch — one flaky match must not cost the whole night, and one systemic bug
    must not burn matches forever. Classification is deterministic and lives in
    ``supervisor.py``; it never touches the input path.
    """
    from pathlib import Path
    from .runtime import save,load
    from .supervisor import FailureBudget,append_incident,classify_run_end
    count=getattr(args,'games',1)
    if count<0:raise Blocked('games must be zero or positive')
    if count==1:return _one_game(root,args)
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise Blocked('autoplay requires a new output directory')
    out.mkdir(parents=True,exist_ok=True);results=[];i=0
    budget=FailureBudget(max_total=getattr(args,'max_failures',3),
                         max_same=getattr(args,'max_same_failure',2))
    while count==0 or i<count:
        if (out/'STOP').exists():break
        i+=1;child=copy.copy(args);child.out=str(out/f'match-{i:04d}');child.games=1
        child.stop_file=str(out/'STOP');child.no_queue=args.no_queue if i==1 else False
        if i>1:child.no_auto_focus=True  # Startup activation, never a focus tug-of-war.
        failure=None
        try:code=_one_game(root,child)
        except Blocked as exc:
            # Lobby/queue/preflight failures happen before a run exists, so the
            # reason must be carried to the classifier explicitly.
            failure=str(exc);code=2
        p=Path(child.out)/'result.json'
        # Reuse the pinned, immutable inference model between matches. The
        # reader, controller and visual evidence remain new per match.
        if getattr(child,'preloaded_policy',None):args.preloaded_policy=child.preloaded_policy
        result=load(p) if p.exists() else None
        incident=classify_run_end(Path(child.out),code,result,reason=failure)
        budget.record(incident);append_incident(out,incident,match=i)
        results.append(dict(match=i,exit_code=code,result=result,severity=incident['severity'],
                            signature=incident.get('signature',''),reason=incident.get('reason','')))
        save(out/'matches.json',results)
        allowed,why=budget.allow(incident)
        if not allowed:
            save(out/'batch-stop.json',dict(at=__import__('time').time(),match=i,severity=incident['severity'],
                reason=incident.get('reason',''),why=why,budget=budget.snapshot(),results=results))
            return code or 2
        if (out/'STOP').exists():break
    save(out/'batch-stop.json',dict(at=__import__('time').time(),match=i,severity='ok',why='batch finished',
        budget=budget.snapshot(),results=results))
    return 0


def _one_game(root,args):
    """One explicit operator-started match; all strategic choices stay model-only."""
    from pathlib import Path
    from .runtime import run
    from .startup import prepare_input,wait_for_match
    args.practical=True;args.enable_input=not args.shadow
    if args.facts:raise Blocked('autoplay does not accept manually forged state sidecars')
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise Blocked('autoplay requires a new output directory')
    if not args.shadow:
        backend,profile=prepare_input(root,args,out)
        if not args.no_queue:wait_for_match(root,args,backend,profile,out)
    return run(root,args)
