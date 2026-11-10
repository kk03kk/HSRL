"""Bounded, action-local two-frame evidence. All backends are injected."""
import time
import numpy as np
from .contracts import Blocked,digest,state_stamp
from .vision import crop_box,stable
from .practical import PracticalGeometry,interpret_frame


def patch_correlation(first,second):
    """Brightness-normalised similarity of two same-size patches.

    Card art is unchanged while a choice panel dims the non-hovered cards or
    adds a gold rim, so a correlation is the right continuity signal; absolute
    colour thresholds are not. Different cards score far below 0.5.
    """
    a=np.asarray(first.convert('L'),dtype=np.float32);b=np.asarray(second.convert('L'),dtype=np.float32)
    if a.shape!=b.shape or a.size==0:return 0.
    a=a-a.mean();b=b-b.mean()
    denominator=float(np.sqrt((a*a).sum())*np.sqrt((b*b).sum()))
    if denominator<1e-6:return 0.
    return float((a*b).sum()/denominator)


def layout_key(snapshot):
    data=snapshot['client']
    synthetic=(data.get('interaction') or {}).get('authority') in ('pinned_server_option','pinned_server_subOptions')
    return digest(dict(game=snapshot['game_id'],turn=data.get('turn'),phase=data['phase'],pending=0 if synthetic else snapshot['view']['pending'],
        interaction=None if synthetic else data.get('interaction'),
        zones={z:[tuple(c.get(k) for k in ('entity_id','card_id','position','controller'))
                  for c in ([] if synthetic and z=='choices' else data.get(z,[]))]
               for z in ('hand','board','shop','choices','branches')}))


def dependency_regions(profile,snapshot,binding=None):
    data=snapshot['client']
    regions={'board':[.20,.46,.63,.23],'hand':[.20,.78,.61,.21],'shop':[.20,.27,.49,.20]}
    kind=binding.kind if binding else None
    if data.get('choices') and snapshot['view']['pending']!=4:
        points,_=PracticalGeometry(profile).locations(snapshot);result=[]
        for c in data['choices']:
            x,y=points['entity:'+str(c['entity_id'])]
            x0=max(0,x-.08);y0=max(0,y-.18)
            result.append([x0,y0,min(.16,1-x0),min(.36,1-y0)])
        result.append([.43,.74,.14,.10])
        return result
    # 旧执行器按动作对象准入。HUD按键不依赖整排随从的持续光效；
    # 仍核验真实Options、按钮局部双帧、遮罩及悬停后状态。
    controls=kind in ('refresh','freeze','upgrade','hero_power','dark_gift') or (kind=='activate' and binding.source in {c['entity_id'] for c in data.get('controls',[])})
    zones=() if controls else {'buy':('shop','hand'),'play':('hand','board'),'sell':('board','hand'),
        'move':('board',),'magnetize':('hand','board'),'refresh':('shop',),'freeze':('shop',),
        'upgrade':('shop',),'target':('hand','board','shop')}.get(kind,('hand','board','shop'))
    # A minion insertion only depends on the two neighbours that define the
    # destination slot. Requiring the *whole* board strip to be pixel-idle
    # charged ~2 s after every play (battlecry/effect animations anywhere in the
    # row). The neighbours' own boxes still catch a real reflow, which is the
    # hazard this guard exists for.
    insertion=bool(binding and kind=='play' and binding.target is None and any(
        c.get('entity_id')==binding.source and c.get('card_type')=='MINION'
        for c in data.get('hand',[])))
    if insertion:zones=('hand',)
    result=[regions[z] for z in zones]
    # Source/target controls and insertion points must also remain still.
    if binding and kind!='end':
        from .contracts import Decision,state_stamp
        plan=PracticalGeometry(profile).plan(Decision(snapshot['game_id'],snapshot['revision'],state_stamp(snapshot),
            binding.action,binding,'',''),snapshot)
        # A targeted spell dropped on an existing minion animates exactly at its
        # destination (blood gems: the +1/+1 sparkle). Requiring that small box
        # to be pixel-idle blocked every second gem for ~1.9 s. The destination
        # identity/order is pinned by the exact semantic stamp re-checked around
        # this capture, and the broad board region below still catches a real
        # row reflow, so only the animated destination box is exempt.
        animated_destination=bool(kind=='play' and binding.target is not None and any(
            c.get('entity_id')==binding.source and c.get('card_type') in ('SPELL','BATTLEGROUND_SPELL')
            for c in data.get('hand',[])))
        for row in plan:
            if row[0]=='move_pointer':continue
            points=list(row[1:])
            if animated_destination and row[0]=='drag':points=points[:1]
            for x,y in points:
                x0=max(0,x-.04);y0=max(0,y-.06)
                result.append([x0,y0,min(.08,1-x0),min(.12,1-y0)])
        if insertion:
            located,_=PracticalGeometry(profile).locations(snapshot)
            board=data.get('board',[])
            for index in {binding.position-1,binding.position}:
                if not 0<=index<len(board):continue
                point=located.get('entity:'+str(board[index]['entity_id']))
                if point is None:continue
                x,y=point;x0=max(0,x-.03);y0=max(0,y-.05)
                result.append([x0,y0,min(.06,1-x0),min(.10,1-y0)])
    return result


class FrameGate:
    def __init__(self,backend,profile,*,clock=time.monotonic,sleep=time.sleep,min_interval=.06,max_age=.45,out=None):
        self.backend=backend;self.profile=profile;self.clock=clock
        self.min_interval=min_interval;self.max_age=max_age;self.previous=None;self.captures=0
        self.admitted=None;self.sleep=sleep
        self.interruptions=None
        self.out=out;self.saved_blocks=set();self.last_frame=None
        self.last_hero_probe=None;self.last_branch_probe=None;self.last_hero_wait_saved=-float('inf')
        self.last_choice_probe=None
        self.last_choice_wait_saved=-float('inf')
        self.branch_occlusion=None;self.admitted_binding=None;self.last_motion=None
        self.dispatch_saves=0;self.dispatch_saved_key=None
        # A hovered card lifts for a moment. The press recheck used to inspect
        # that mid-animation frame, cancel, re-admit and hover again, which
        # produced dozens of cancelled presses per discovery panel (the hover
        # animation re-starts identically, so the same phase failed every time).
        self.recheck_settle=.14;self.recheck_attempts=3
        # Rail-support bar for the bounded continuity fallback used when a
        # hover/animation artefact hides one rail of an already authorized row.
        self.recheck_min_support=.42
        # Timestamp of the newest capture that passed local visual checks. The
        # press deadline is measured from this, not from the earlier admission
        # frame: semantic/OCR rechecks themselves take real time.
        self.evidence_at=None
        # Optional asynchronous frame writer (GameTrace).  Encoding a
        # 2560x1600 JPEG costs tens of milliseconds and used to sit directly
        # before the press; the image is evidence, not an input precondition.
        self.frame_writer=None
        self.end_ocr=None
    def save_dispatch(self,snapshot,binding):
        """Two rotating diagnostic frames, separate from input authorization."""
        if self.out is None or self.last_frame is None or binding is None:return None
        from pathlib import Path
        from dataclasses import asdict
        from .runtime import save
        key=(snapshot['game_id'],snapshot['client'].get('turn'),state_stamp(snapshot),digest(asdict(binding)))
        if key==self.dispatch_saved_key:return self.dispatch_saved_path
        folder=Path(self.out)/'dispatch';folder.mkdir(parents=True,exist_ok=True)
        self.dispatch_saves+=1;name=str(self.dispatch_saves%2)
        path=folder/(name+'.jpg')
        metadata=dict(sequence=self.dispatch_saves,game_id=key[0],turn=key[1],binding=asdict(binding),
            log_position=snapshot['client'].get('log_position'),evidence='compressed final admission frame before hover')
        if self.frame_writer is not None:
            self.frame_writer(path,self.last_frame,metadata,quality=85)
        else:
            self.last_frame.convert('RGB').save(path,quality=85)
            save(folder/(name+'.json'),metadata)
        self.dispatch_saved_key=key;self.dispatch_saved_path=str(path)
        return str(path)
    def observe(self,snapshot,pending=None,binding=None,*,revalidate=False):
        self.last_frame=None;self.last_hero_probe=None;self.last_branch_probe=None;self.last_choice_probe=None
        try:return self._observe(snapshot,pending,binding,revalidate=revalidate)
        except Blocked as exc:
            if snapshot['client'].get('choices') and not all(c['card_type']=='HERO' for c in snapshot['client']['choices']):
                self.save_choice_wait(snapshot,str(exc),self.last_frame)
            if self.out is not None and self.last_frame is not None and len(self.saved_blocks)<12:
                key=(snapshot['game_id'],snapshot['client'].get('turn'),str(exc))
                if key not in self.saved_blocks:
                    from pathlib import Path
                    from .runtime import save
                    self.saved_blocks.add(key);folder=Path(self.out);folder.mkdir(parents=True,exist_ok=True)
                    name=f'{len(self.saved_blocks):02d}'
                    data=snapshot['client']
                    metadata=dict(reason=str(exc),game_id=snapshot['game_id'],
                        turn=data.get('turn'),phase=data.get('phase'),interaction=data.get('interaction'),
                        choices=[{k:c.get(k) for k in ('entity_id','card_id','card_type')} for c in data.get('choices',[])],
                        interruption_probe=getattr(self.interruptions,'last_probe',None),hero_probe=self.last_hero_probe,
                        branch_probe=self.last_branch_probe,choice_probe=self.last_choice_probe,
                        motion=self.last_motion)
                    if self.frame_writer is not None:
                        self.frame_writer(folder/(name+'.jpg'),self.last_frame,metadata,quality=90)
                    else:
                        self.last_frame.convert('RGB').save(folder/(name+'.jpg'),quality=90)
                        save(folder/(name+'.json'),metadata)
                if str(exc).startswith('waiting for visible hero') and self.clock()-self.last_hero_wait_saved>=2:
                    from pathlib import Path
                    from .runtime import save
                    folder=Path(self.out);folder.mkdir(parents=True,exist_ok=True)
                    # Keep the first evidence above and refresh this explicitly
                    # named latest sample while loading. It is never a source
                    # artifact or an authorization frame for a pending input.
                    latest=dict(reason=str(exc),game_id=snapshot['game_id'],
                        interaction=snapshot['client'].get('interaction'),hero_probe=self.last_hero_probe)
                    if self.frame_writer is not None:
                        self.frame_writer(folder/'hero-wait-last.jpg',self.last_frame,latest,quality=90)
                    else:
                        self.last_frame.convert('RGB').save(folder/'hero-wait-last.jpg',quality=90)
                        save(folder/'hero-wait-last.json',latest)
                    self.last_hero_wait_saved=self.clock()
            raise
    def _observe(self,snapshot,pending=None,binding=None,*,revalidate=False):
        if revalidate and self.previous:
            # Model computation overlaps this minimum frame interval. There is
            # only one bounded residual wait, rather than two full settle loops.
            self.sleep(max(0,self.min_interval-(self.clock()-self.previous[3])))
        image,window=self.backend.capture();self.captures+=1;now=self.clock();self.last_frame=image
        if list(image.size)!=self.profile['client_size'] or window.dpi!=self.profile['dpi']:
            raise Blocked('client geometry differs from autoplay profile')
        choice_regions=[];data=snapshot['client'];interaction=data.get('interaction') or {}
        if data.get('choices') and interaction.get('mode')=='choice' and interaction.get('owner')==data.get('local_player_id'):
            points,_=PracticalGeometry(self.profile).locations(snapshot)
            for c in data['choices']:
                x,y=points['entity:'+str(c['entity_id'])]
                choice_regions.append([max(0,x-.025),max(0,y-.065),.05,.13])
        def interpret(frame,current_window):
            visual=interpret_frame(frame,current_window,self.profile,snapshot['client'],pending)
            self.last_hero_probe=visual.get('hero_ui')
            self.last_branch_probe=visual.get('branch_ui')
            if visual['phase']=='hero' and not visual.get('hero_ui',{}).get('visible'):
                self.previous=None;self.admitted=None
                raise Blocked('waiting for visible hero row; no hero input sent')
            data=snapshot['client'];interaction=data.get('interaction') or {}
            if data.get('choices') and interaction.get('mode')=='choice' and visual['phase']!='hero':
                from .choice_ui import inspect
                visual['choice_ui']=inspect(frame,self.profile,data)
                self.last_choice_probe=visual['choice_ui']
                if not visual['choice_ui']['visible']:
                    self.previous=None;self.admitted=None
                    # Warm the interruption scan during the unavoidable card
                    # entrance, rather than adding full-screen OCR after the
                    # row is ready. This frame never authorizes any input.
                    if self.interruptions:self.interruptions.check(frame,data,pending,choice_regions=choice_regions,
                                                                   choice_visible=False)
                    raise Blocked('waiting for complete visible choice row; no choice input sent')
            return visual
        visual=interpret(image,window)
        if pending and pending[0].binding.branches:
            d=pending[0];key=(d.game_id,d.binding.source,d.stamp)
            if self.branch_occlusion and self.branch_occlusion[0]!=key:self.branch_occlusion=None
            proof=visual.get('branch_ui') or {}
            if (not proof.get('visible') and d.binding.sub_option_id is None and not data.get('choices')
                    and data.get('game_id')==d.game_id and data.get('turn')==pending[1]['client']['turn']):
                self.previous=None;self.admitted=None
                if self.interruptions:self.interruptions.check(image,data,pending,choice_regions=choice_regions,
                    choice_visible=bool((visual.get('choice_ui') or {}).get('visible') or (visual.get('hero_ui') or {}).get('visible')))
                raise Blocked('waiting for complete visible branch row; no branch input sent')
            if proof.get('visible') and not self.branch_occlusion:
                from .branch_ui import Occlusion
                self.branch_occlusion=(key,[Occlusion(image,box) for box in proof['boxes']])
            if binding and binding.kind=='branch' and not proof.get('visible'):
                raise Blocked('waiting for complete visible branch row; no branch input sent')
            if binding and binding.kind=='branch' and binding.source not in proof.get('entities',[]):
                raise Blocked('model-selected branch is absent from the visible row')
            if binding and binding.kind=='branch' and not proof['selectable'][proof['entities'].index(binding.source)]:
                raise Blocked('model-selected branch is not visibly enabled')
            self.check_branch_occlusion(image,snapshot,pending,binding)
        else:self.branch_occlusion=None
        def check(frame):
            if self.interruptions:self.interruptions.check(frame,data,pending,choice_regions=choice_regions,
                choice_visible=bool((visual.get('choice_ui') or {}).get('visible') or (visual.get('hero_ui') or {}).get('visible')))
            if binding and binding.kind=='end':
                # A visible countdown means the server has already accepted
                # the end of recruiting; there is no button to press until it
                # reaches zero. Treat that as an explicit wait, not as an OCR
                # failure that burns the whole admission interval.
                probe=self.end_control_probe(frame,snapshot)
                visual['end_control']=probe
                if probe.get('countdown') and not probe.get('ready'):
                    raise Blocked('recruiting timer still running; no end-turn input sent')
        check(image)
        # OCR can be slower than the evidence age budget. Refresh afterwards;
        # never timestamp an old pre-OCR frame as a fresh mouse admission.
        if self.clock()-now>self.max_age:
            token=window.token();image,window=self.backend.capture();self.captures+=1;now=self.clock();self.last_frame=image
            if window.token()!=token:raise Blocked('window changed during interruption recognition')
            if list(image.size)!=self.profile['client_size'] or window.dpi!=self.profile['dpi']:
                raise Blocked('client geometry changed during interruption recognition')
            check(image)
            if self.clock()-now>self.max_age:raise Blocked('interruption recognition exceeded fresh-frame budget')
            visual=interpret(image,window)
            if binding and binding.kind=='branch' and not visual.get('branch_ui',{}).get('visible'):
                raise Blocked('branch row changed during interruption recognition')
            self.check_branch_occlusion(image,snapshot,pending,binding)
        key=layout_key(snapshot);old=self.previous
        if binding is None and not revalidate:
            # Read-only speculation can precede physical readiness. Native input
            # is forbidden until check_action obtains two-frame local admission.
            if old is None or old[0]!=key or old[2]!=window.token() or now-old[3]>self.max_age:
                self.previous=(key,image,window.token(),now,visual.get('choice_ui'));self.admitted=None
            visual.update(observed_at=now,admitted=False,evidence='model candidate only; input admission pending')
            return visual,image
        self.previous=(key,image,window.token(),now,visual.get('choice_ui'))
        if old is None or old[0]!=key or old[2]!=window.token() or now-old[3]>self.max_age:
            self.admitted=None;raise Blocked('collecting fresh layout stability evidence')
        reuse=bool(revalidate and self.admitted and self.admitted[:2]==(key,window.token()) and now-self.admitted[2]<=self.max_age)
        if now-old[3]<self.min_interval and not reuse:
            # Keep the older baseline until enough real frame time elapsed.
            self.previous=old
            raise Blocked('waiting for second stability frame')
        from .interactions import branch_snapshot,target_snapshot
        stage=target_snapshot(branch_snapshot(snapshot,pending,visual),pending,visual,allow_public_hero=True)
        current_choice=visual.get('choice_ui') or {};old_choice=old[4] if len(old)>4 else None
        measured_choice=bool(binding and binding.kind=='choose' and current_choice.get('visible') and old_choice
            and old_choice.get('visible') and current_choice['entities']==old_choice['entities']
            and len(current_choice['boxes'])==len(old_choice['boxes']))
        if measured_choice:
            # The old executor checks measured card geometry instead of waiting
            # for all art/particles to stop. Keep two fresh complete rows, stable
            # physical boxes and a quiet calibrated hit patch for this source.
            # Rail maxima can move inside a thick glowing rim while the hit
            # and artwork stay still. Allow a small resolution-scaled rim band;
            # the source hit patch below must still pass the pixel comparison.
            for previous_box,current_box in zip(old_choice['boxes'],current_choice['boxes']):
                if max(abs(a-b)*scale for a,b,scale in zip(previous_box,current_box,
                        (image.width,image.height,image.width,image.height)))>(max(4,image.width*.008) if
                            current_choice.get('evidence')=='paired_straight_discovery_rails_at_every_calibrated_hit'
                            and old_choice.get('evidence')==current_choice.get('evidence') else 4):
                    self.admitted=None;raise Blocked('choice row geometry is still moving')
            x,y=PracticalGeometry(self.profile).locations(stage)[0]['entity:'+str(binding.source)]
            boxes=[[x-.02,y-.03,.04,.06]]
            if visual.get('confirm_enabled'):boxes.append([.43,.74,.14,.10])
        else:boxes=dependency_regions(self.profile,stage,binding)
        if not all(stable(old[1],image,crop_box(box,image.size),10.) for box in boxes):
            # Name the boxes that moved: the next run can then say whether it
            # was the source card, a neighbour or a whole-zone animation.
            from PIL import ImageChops,ImageStat
            failed=[]
            for index,box in enumerate(boxes):
                region=crop_box(box,image.size)
                if stable(old[1],image,region,10.):continue
                stat=ImageStat.Stat(ImageChops.difference(old[1].crop(region).convert('RGB'),
                                                         image.crop(region).convert('RGB')))
                failed.append(dict(box=index,mean=round(max(stat.mean),1),
                                   rect=[round(value,3) for value in box]))
            self.admitted=None;self.last_motion=dict(failed=failed,boxes=len(boxes))
            detail=','.join('box%d=%s'%(row['box'],row['mean']) for row in failed[:3])
            raise Blocked('action-dependent card layout is still moving ('+detail+')')
        self.admitted=(key,window.token(),now)
        self.evidence_at=now
        if binding and binding.kind=='branch':
            from .branch_ui import Occlusion
            # 用最终准入帧刷新模板，避免保留抉择入场时较小的卡图。
            d=pending[0];self.branch_occlusion=((d.game_id,d.binding.source,d.stamp),
                [Occlusion(image,box) for box in visual['branch_ui']['boxes']])
        visual.update(observed_at=now,admitted=True,region_count=len(boxes),evidence='two fresh frames; exact window/layout; action dependencies')
        return visual,image
    def end_control_probe(self,image,snapshot):
        """Classify the calibrated end-turn neighbourhood without blind clicks.

        Returns either ``ready`` with the exact phrase proof, or a countdown
        observation. The caller must keep waiting in either case until the
        server exposes the start-combat control.
        """
        from .lobby import normalized
        from .vision import OCR
        data=snapshot['client'];interaction=data.get('interaction') or {}
        if data.get('phase')!='recruit' or interaction or not data.get('ready'):
            raise Blocked('end-turn control is not available in a ready recruit phase')
        points,_=PracticalGeometry(self.profile).locations(snapshot)
        point=points.get('control:end')
        if point is None:raise Blocked('end-turn control is not calibrated')
        x,y=point
        left=max(0,min(x-.055,.78));top=max(0,min(y-.045,.405))
        right=min(1,max(x+.055,.90));bottom=min(1,max(y+.05,.51))
        box=crop_box([left,top,right-left,bottom-top],image.size)
        crop=image.crop(box)
        if self.end_ocr is None:
            self.end_ocr=getattr(self.interruptions,'ocr',None)
            if self.end_ocr is None:
                self.end_ocr=OCR(max_width=1024)
                if self.interruptions is not None:self.interruptions.ocr=self.end_ocr
        rows=self.end_ocr.read(crop)
        found=[text for text,score in rows if score>=.9 and normalized(text)=='开始战斗']
        countdown=[text for text,score in rows if score>=.9 and
                   ('秒' in text or any(ch.isdigit() for ch in text))]
        roi=[round(v,6) for v in (box[0]/image.width,box[1]/image.height,
            (box[2]-box[0])/image.width,(box[3]-box[1])/image.height)]
        if len(found)==1:
            return dict(visible=True,ready=True,text='开始战斗',point=list(point),roi=roi,
                evidence='OCR exact phrase in calibrated end-turn neighbourhood')
        if countdown:
            return dict(visible=True,ready=False,countdown=True,text=countdown[0],point=list(point),
                roi=roi,evidence='OCR recruiting countdown; waiting for start-combat')
        return dict(visible=True,ready=False,countdown=False,point=list(point),roi=roi,
            evidence='end-turn control not yet ready')

    def start_combat_proof(self,image,snapshot):
        """Require the real '开始战斗' control before dispatching EndTurn.

        BG exposes EndTurn as a right-side client button, not a server option.
        The calibrated point alone is insufficient: when the button is absent
        or replaced by a countdown, a blind click could hit unrelated UI. OCR
        only the calibrated control neighbourhood and require one high-
        confidence exact phrase. This is local evidence, not a model estimate.
        """
        probe=self.end_control_probe(image,snapshot)
        if not probe.get('ready'):
            raise Blocked('waiting for real start-combat button; no end-turn input sent')
        return probe
    def _recheck_capture(self,snapshot,reason):
        """One fresh press-time frame, still bound to the admitted window."""
        image,window=self.backend.capture();self.captures+=1;self.evidence_at=self.clock()
        if (list(image.size)!=self.profile['client_size'] or window.dpi!=self.profile['dpi']
                or window.token()!=snapshot['visual']['window_token']):
            raise Blocked(reason)
        return image,window
    def recheck_hero(self,snapshot,pending=None):
        """Hero UI has no semantic log for local selection; check it at press.

        Hovering a portrait can itself add the outer halo. That is the intended
        hero becoming selected, not a lost row. Only a different hero (or a lost
        row/confirmation) cancels the press. The portrait also animates for a
        moment after the pointer lands, so bounded fresh retries are allowed
        before the press is cancelled.
        """
        intended=None
        if pending is not None:
            source=pending[0].binding.source
            intended=next((i for i,c in enumerate(snapshot['client'].get('choices',[]))
                           if c['entity_id']==source),None)
            if intended is None:
                raise Blocked('model hero is absent from the visible row; no button down sent')
        old=snapshot['visual'].get('hero_ui') or {}
        error='hero row disappeared during hover; no button down sent'
        for attempt in range(self.recheck_attempts):
            image,window=self._recheck_capture(snapshot,'hero window/geometry changed during hover')
            visual=interpret_frame(image,window,self.profile,snapshot['client'],pending)
            new=visual.get('hero_ui') or {}
            selected=new.get('selected_index')
            if not new.get('visible'):
                error='hero row disappeared during hover; no button down sent'
            elif pending is not None and selected is not None and selected!=intended:
                error='a different hero is selected during hover; no button down sent'
            elif pending is None and selected!=old.get('selected_index'):
                error='hero row/selection changed during hover; no button down sent'
            elif snapshot['visual'].get('confirm_enabled') and not visual.get('confirm_enabled'):
                error='hero confirmation proof lost during hover'
            else:
                if attempt:print('Press: hero row confirmed after hover settle retry %d.'%(attempt+1),flush=True)
                return
            if attempt+1<self.recheck_attempts:self.sleep(self.recheck_settle)
        raise Blocked(error)
    def highlight_moved(self,snapshot,image):
        """True when a slot that was *not* gold-only at admission is gold-only now.

        Gold frames are legitimate (golden cards), so the detector accepts them.
        A card turning gold after admission is the client highlighting something
        else (stale hover); the only slot allowed to change is the authorized
        one. This keeps the 10/28 "gold elsewhere is not a row" guard without
        rejecting legitimate golden cards.
        """
        from .choice_ui import slot_colors
        binding=self.admitted_binding
        if not (self.previous and self.admitted and binding and binding.kind=='choose'):return False
        if self.previous[0]!=layout_key(snapshot):return False
        try:
            before=slot_colors(self.previous[1],self.profile,snapshot['client'])
            now=slot_colors(image,self.profile,snapshot['client'])
        except Blocked:return False
        if len(before)!=len(now) or not before:return False
        def gold_only(row):
            # A "gold highlight" is a slot whose frame was *already* mostly
            # gold at admission, then remains gold on both sides. Green
            # discover rails legitimately change brightness during the hover
            # animation; only a material gold-vs-green transition is a real
            # stale-highlight signal.
            return min(row['gold'])>self.recheck_min_support
        # Only a slot that was NOT gold-only at admission and is gold-only now
        # is evidence that some other object became highlighted. Trinket
        # panels legitimately have all four cards gold; comparing set
        # differences there made every retry fail with the exact same fixed
        # golden set. The admitted frame is the baseline, so a card that was
        # already gold never looks like a new highlight.
        changed={row['entity'] for old,row in zip(before,now)
                 if not gold_only(old) and gold_only(row)
                 and min(row['gold'])>max(self.recheck_min_support,min(row['base'])+.15)}
        return bool(changed-{binding.source})

    def recheck_choice(self,snapshot):
        if not snapshot['visual'].get('choice_ui'):return
        from .choice_ui import inspect
        binding=self.admitted_binding
        hovered=binding.source if binding and binding.kind=='choose' else None
        expected=snapshot['visual']['choice_ui']['entities']
        image=None;proof=None
        for attempt in range(self.recheck_attempts):
            image,window=self._recheck_capture(snapshot,'choice window changed during hover')
            proof=inspect(image,self.profile,snapshot['client'],hovered_entity=hovered)
            self.last_choice_probe=proof
            if proof['visible'] and proof['entities']==expected:
                if self.highlight_moved(snapshot,image):
                    raise Blocked('another slot was highlighted after admission; no button down sent')
                if attempt:print('Press: choice row confirmed after hover settle retry %d.'%(attempt+1),flush=True)
                return
            if attempt+1<self.recheck_attempts:self.sleep(self.recheck_settle)
        # Bounded continuity fallback. The row identity is pinned by the server
        # choice and the exact semantic stamp was re-checked around this capture,
        # so only the rail-support bar is relaxed to tolerate hover shading,
        # card lift or a partial frame glitch. A different panel still fails the
        # four calibrated slots and the candidate identity comparison.
        image,window=self._recheck_capture(snapshot,'choice window changed during hover')
        relaxed=inspect(image,self.profile,snapshot['client'],hovered_entity=hovered,
                        min_support=self.recheck_min_support)
        self.last_choice_probe=relaxed
        if relaxed['visible'] and relaxed['entities']==expected:
            if self.highlight_moved(snapshot,image):
                raise Blocked('another slot was highlighted after admission; no button down sent')
            print('Press: choice row confirmed by relaxed frame continuity (support %s).'
                  %([[round(x,2) for x in pair] for pair in relaxed.get('rails',[])]),flush=True)
            return
        # Last resort: the panel can dim its non-hovered cards or lift the
        # hovered one, which defeats every absolute colour threshold while the
        # very same art is still displayed at the calibrated slots. Compare with
        # the admitted frame's patches; a different card/panel cannot correlate.
        if self.previous and self.admitted and self.previous[0]==layout_key(snapshot):
            admitted=self.previous[1];points,_=PracticalGeometry(self.profile).locations(snapshot)
            scores=[]
            for index,row in enumerate(snapshot['client'].get('choices',[])):
                point=points.get('entity:'+str(row['entity_id']))
                if point is None:scores.append(1.);continue
                x,y=point;box=crop_box([x-.05,y-.10,.10,.20],image.size)
                scores.append(round(patch_correlation(admitted.crop(box),image.crop(box)),2))
            chosen=next((i for i,row in enumerate(snapshot['client'].get('choices',[]))
                         if row['entity_id']==hovered),None)
            if (chosen is not None and scores
                    and all(score>=.5 for i,score in enumerate(scores) if i!=chosen)
                    and scores[chosen]>=.4):
                if self.highlight_moved(snapshot,image):
                    raise Blocked('another slot was highlighted after admission; no button down sent')
                print('Press: choice row confirmed by admitted-frame continuity (correlation %s).'%scores,flush=True)
                return
        self.save_choice_wait(snapshot,'choice row disappeared during hover; no button down sent',image)
        raise Blocked('choice row disappeared during hover; no button down sent')
    def save_choice_wait(self,snapshot,reason,image):
        """Keep a recent failed panel, rather than only its entrance animation."""
        if self.out is None or image is None or self.clock()-self.last_choice_wait_saved<2:return
        from pathlib import Path
        from .runtime import save
        folder=Path(self.out);folder.mkdir(parents=True,exist_ok=True)
        metadata=dict(reason=reason,game_id=snapshot['game_id'],
            turn=snapshot['client'].get('turn'),client={k:v for k,v in snapshot['client'].items()
                if k not in ('resolutions','actions')},choice_probe=self.last_choice_probe)
        if self.frame_writer is not None:
            self.frame_writer(folder/'choice-wait-last.jpg',image,metadata,quality=85)
        else:
            image.convert('RGB').save(folder/'choice-wait-last.jpg',quality=85)
            save(folder/'choice-wait-last.json',metadata)
        self.last_choice_wait_saved=self.clock()
    def check_branch_occlusion(self,image,snapshot,pending,binding):
        if not binding or binding.kind!='target' or not pending or not pending[0].binding.branches:return
        d=pending[0];key=(d.game_id,d.binding.source,d.stamp)
        if self.branch_occlusion and self.branch_occlusion[0]==key:
            point=PracticalGeometry(self.profile).locations(snapshot)[0]['entity:'+str(binding.target)]
            if any(guard.blocks(image,point) for guard in self.branch_occlusion[1]):
                raise Blocked('target is covered by the previous branch card; input paused')
    def recheck_interaction(self,snapshot,pending):
        """与旧执行器相同：悬停后还需确认当前分支/目标未被遮挡。"""
        mode=(snapshot['client'].get('interaction') or {}).get('mode')
        if mode not in ('branch','target'):return
        if mode=='branch':
            binding=self.admitted_binding
            for attempt in range(self.recheck_attempts):
                image,window=self._recheck_capture(snapshot,'interaction window changed during hover')
                visual=interpret_frame(image,window,self.profile,snapshot['client'],pending)
                proof=visual.get('branch_ui') or {}
                if (proof.get('visible') and binding is not None and binding.source in proof.get('entities',[])
                        and proof['selectable'][proof['entities'].index(binding.source)]):return
                if attempt+1<self.recheck_attempts:self.sleep(self.recheck_settle)
            raise Blocked('branch row disappeared or disabled during hover; no button down sent')
        image,window=self._recheck_capture(snapshot,'interaction window changed during hover')
        if self.admitted_binding:self.check_branch_occlusion(image,snapshot,pending,self.admitted_binding)
