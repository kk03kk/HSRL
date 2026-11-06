"""Log-order hand geometry; card type does not dictate its rendered badge."""
from dataclasses import asdict
from copy import deepcopy
import cv2
import numpy as np
from .windows import Box
from .interaction_regions import mask_zone,allowed


def hand_card_frames(image,*,hsv=None):
    """Measured lower-card outlines, independent of gold cost/HUD artwork.

    Frame topology is assigned to entities only by a complete ordered spell
    row. Both side rails must persist; a colored badge is not a card body.
    """
    width,height=image.size
    cache=getattr(image,'_hand_frame_cache',None)
    if cache is not None and hsv is not None and hsv is not image.hsv_pixels:cache=None
    if cache is not None and 'frames' in cache:return list(cache['frames'])
    if cache is not None and hasattr(image,'memo'):
        # The mask zeroes everything outside the configured hand zone before
        # contours; those pixels cannot affect frame topology.
        from .interaction_regions import layout
        a,b,c,d=layout()['zones']['hand']
        bounds=(round(a*width),round(b*height),round(c*width),round(d*height))
        frames=image.memo('hand_card_frames',(),lambda:_measure_hand_frames(image,hsv=hsv),bounds=bounds)
        cache['frames']=tuple(frames)
        return list(frames)
    return _measure_hand_frames(image,hsv=hsv)


def _measure_hand_frames(image,*,hsv=None):
    width,height=image.size
    cache=getattr(image,'_hand_frame_cache',None)
    if cache is not None and hsv is not None and hsv is not image.hsv_pixels:cache=None
    if hsv is None:hsv=getattr(image,'hsv_pixels',None)
    if hsv is None:hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([35,95,100]),np.array([112,255,255]))
    mask_zone(mask,'hand')
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frames=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (.035*width<w<.48*width and .055*height<h<.21*height and
                .81*height<y<.94*height and y+h>=.95*height):continue
        rails=[mask[y+round(h*.2):y+round(h*.8),left:left+max(3,round(w*.2))]
               for left in (x,x+round(w*.8))]
        if any(not rail.size or np.mean(np.any(rail>0,axis=1))<.65 for rail in rails):continue
        frames.append(Box(x,y,w,h))
    frames=sorted(frames,key=lambda b:b.x)
    if cache is not None:cache['frames']=tuple(frames)
    return frames


def spell_frame_slots(image,cards,*,frames=None):
    """Bind a complete small spell row, including unplayable occupied slots.

    White cost badges and gold artwork must not change entity order. Every
    point is inside its measured frame and outside other slots' x extents.
    Missing, merged or densely overlapping frames provide no inferred points.
    """
    if not 1<=len(cards)<=3 or any(c.card_type not in ('SPELL','BATTLEGROUND_SPELL') for c in cards):return []
    cards=sorted(cards,key=lambda c:c.position)
    if [c.position for c in cards]!=list(range(1,len(cards)+1)):return []
    frames=hand_card_frames(image) if frames is None else frames
    width,height=image.size
    if len(frames)!=len(cards) or any(b.width>=width*.14 for b in frames):return []
    frames=sorted(frames,key=lambda b:b.x)
    centers=[b.point()[0] for b in frames]
    if any(not width*.025<b-a<width*.14 for a,b in zip(centers,centers[1:])):return []
    result=[];margin=max(3,width*.002)
    for i,(card,b) in enumerate(zip(cards,frames)):
        left=max(b.x+b.width*.25,frames[i-1].x+frames[i-1].width+margin if i else b.x)
        right=min(b.x+b.width*.8,frames[i+1].x-margin if i+1<len(frames) else b.x+b.width)
        if right-left<8:return []
        point=((left+right)/2,b.y+b.height*.45)
        if not allowed(point,image.size,'hand'):return []
        result.append((card,Box(round(point[0])-5,round(point[1])-5,10,10)))
    return result


def hand_frame_binding(image,cards,*,frames=None,max_cards=10):
    """Complete measured card-frame row for ANY hand size (execution boundary).

    Deliberately not a mapper: it never proposes a replacement point, it only
    proves whether every logged hand slot maps left-to-right onto exactly one
    measured card frame with plausible, evenly spaced geometry. Missing,
    merged, duplicated, out-of-bounds or implausibly scaled rows return [].

    Vertical position inside a frame is not part of the proof: the client's
    card body extends above the measured lower frame, and hand-zone/HUD
    checks already bound it. The horizontal slot is the decisive evidence,
    because a badge or neighbouring card can never share it.
    """
    if not cards or len(cards)>max_cards:return []
    cards=sorted(cards,key=lambda c:c.position)
    if [c.position for c in cards]!=list(range(1,len(cards)+1)):return []
    frames=sorted(hand_card_frames(image) if frames is None else frames,key=lambda b:b.x)
    width,height=image.size
    if len(frames)!=len(cards):return []
    if any(b.width<=0 or b.height<=0 or b.x<0 or b.y<0 or b.x+b.width>width or b.y+b.height>height for b in frames):return []
    # One physical row has one scale. A merged blob next to a real card is
    # rejected by size, not repaired by interpolation.
    if max(b.width for b in frames)>min(b.width for b in frames)*1.6:return []
    if max(b.height for b in frames)>min(b.height for b in frames)*1.5:return []
    if max(b.point()[1] for b in frames)-min(b.point()[1] for b in frames)>height*.06:return []
    centers=[b.point()[0] for b in frames]
    gaps=[b-a for a,b in zip(centers,centers[1:])]
    if gaps and (min(gaps)<width*.012 or max(gaps)/min(gaps)>1.4):return []
    result=[];margin=max(3,width*.002)
    for index,(card,b) in enumerate(zip(cards,frames)):
        left=max(b.x+b.width*.25,frames[index-1].x+frames[index-1].width+margin if index else b.x)
        right=min(b.x+b.width*.8,frames[index+1].x-margin if index+1<len(frames) else b.x+b.width)
        if right-left<8:return []
        point=((left+right)/2,b.y+b.height*.45)
        if not allowed(point,image.size,'hand'):return []
        result.append((card,b,Box(round(point[0])-5,round(point[1])-5,10,10)))
    return result


def hand_markers(image,cards,*,tracker=None,fan_hint=None,window_token=None):
    """Reuse unchanged hand evidence while preserving tracking state changes."""
    if not cards:return []
    def compute():
        result=_hand_markers(image,cards,tracker=tracker,fan_hint=fan_hint,window_token=window_token)
        return result,deepcopy(tracker)
    if hasattr(image,'memo'):
        result,state=image.memo('hand_markers',(cards,tracker,fan_hint,window_token),compute)
    else:result,state=compute()
    if tracker is not None:
        tracker.clear();tracker.update(state)
    return result


def _hand_markers(image,cards,*,tracker=None,fan_hint=None,window_token=None):
    """Require measured card frames or a complete cost/tier marker sequence."""
    if not cards or len(cards)>10 or sorted(c.position for c in cards)!=list(range(1,len(cards)+1)):return []
    width,height=image.size;rgb=np.asarray(image);hsv=getattr(image,'hsv_pixels',None)
    if hsv is None:hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    frames=hand_card_frames(image,hsv=hsv)
    complete=hand_frame_binding(image,cards,frames=frames)
    if complete:return [(c,hit) for c,_,hit in complete]
    from .mixed_hand import locked_prefix_binding
    mixed=locked_prefix_binding(image,cards,pixels=rgb,hsv=hsv)
    if mixed:return [(c,hit) for c,_,hit in mixed]
    complete_spells=spell_frame_slots(image,cards,frames=frames)
    if complete_spells:return complete_spells
    if len(cards)==1 and cards[0].card_type in ('SPELL','BATTLEGROUND_SPELL'):
        # Zero-cost spells can have a white/neutral cost badge. Never substitute
        # the only gold component on screen (often the resource HUD) for it.
        candidates=[b for b in frames if b.width<width*.14]
        if len(candidates)==1:
            b=candidates[0];x,y=b.point(.5,.45)
            return [(cards[0],Box(round(x)-5,round(y)-5,10,10))]
    masks=[('spell',cv2.inRange(hsv,np.array([12,105,140]),np.array([36,255,255]))),
           ('minion',cv2.inRange(hsv,np.array([130,100,65]),np.array([169,255,255])))]
    markers=[]
    for kind,mask in masks:
        mask[:int(height*.84)]=0;mask[int(height*.96):]=0
        mask[:,:int(width*.18)]=0;mask[:,int(width*.82):]=0
        mask_zone(mask,'hand')
        kernel=np.ones((max(3,round(height*.0056)),3),np.uint8) if kind=='minion' else np.ones((3,3),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if .010*width<w<.032*width and .016*height<h<.052*height and .45<w/h<1.5:
                # A cost badge attaches to the frame's upper-left corner;
                # currency or gold artwork inside the card is not a new slot.
                if kind=='spell' and not any(b.x-w<=x<=b.x+b.width*.2 and b.y-h<=y<=b.y+b.height*.25 for b in frames):continue
                markers.append((kind,Box(x,y,w,h)))
    markers.sort(key=lambda p:p[1].x)
    cards=sorted(cards,key=lambda c:c.position)
    fallback=[]
    measured_tiers=None
    # The wider 10-card search includes both artwork currency fragments and
    # shields. Bound the search without rejecting a legitimate mixed row just
    # because the unused marker family has extra components.
    if len(markers)>40:return []
    dense_minions=sum(c.card_type=='MINION' for c in cards)>=4
    if dense_minions:
        # Use full-resolution shields for a curved fan. Downsampling can split
        # the small shields into star-sized fragments and lose the unique row.
        from .hand_fan import tier_markers,match_fan
        from .hand_cap import portrait_cap
        minion_ids={c.entity_id for c in cards if c.card_type=='MINION'}
        reusable=bool(fan_hint and fan_hint.get('status')=='inferred' and {a['entity_id'] for a in fan_hint['assignments']}==minion_ids)
        if reusable:fan=fan_hint
        else:
            measured_tiers=tier_markers(image,[b for kind,b in markers if kind=='minion'])
            fan=match_fan(cards,measured_tiers,image.size)
        signature=(window_token,tuple((c.entity_id,c.card_id,c.position,c.card_type,c.tier) for c in cards))
        if tracker is not None:
            if tracker.get('signature')!=signature:tracker.clear();tracker['signature']=signature
            tracker['used_tracking']=False
        if fan['status']=='inferred':
            if tracker is not None:tracker['fan']=fan
        elif tracker is not None and tracker.get('fan'):
            prior=tracker['fan'];assignments=[];seen=set()
            for a in prior['assignments']:
                b=Box(**a['badge']);cx,cy=b.point()
                matches=[r for kind,r in markers if kind=='minion' and abs(r.point()[0]-cx)<min(prior['evidence']['pitch']*.20,b.width*.7)
                         and abs(r.point()[1]-cy)<b.height*.4 and .6<r.width/b.width<1.7 and .6<r.height/b.height<1.7]
                if len(matches)==1:assignments.append({**a,'badge':asdict(matches[0])});seen.add(a['entity_id'])
                else:assignments.append(a)
            fan={**prior,'assignments':assignments};tracker['used_tracking']=True
            # Only actually re-observed badges may supply an input point.
            fan={**fan,'observed_ids':seen}
        if fan['status']=='inferred':
            found=[]
            by_id={c.entity_id:c for c in cards}
            for assignment in fan['assignments']:
                if 'observed_ids' in fan and assignment['entity_id'] not in fan['observed_ids']:continue
                card=by_id[assignment['entity_id']]
                rect,_=portrait_cap(image,assignment,fan['assignments'],fan['evidence']['pitch'],pixels=rgb)
                if rect:found.append((card,Box(**rect)))
            fallback=found
            if len(fallback)==len(cards):return fallback
    from .hand_fan import tier_markers
    shields=measured_tiers if measured_tiers is not None else tier_markers(image,[b for kind,b in markers if kind=='minion'])
    shield_xs={m['rect']['x'] for m in shields}
    def expected(card):return ('minion',) if card.card_type=='MINION' else ('spell','minion')
    fits=[]
    if len(cards)==1:
        fits=[markers] if len(markers)==1 and (markers[0][0]=='spell' or markers[0][1].x in shield_xs) else []
    else:
        for left in markers:
            for right in markers:
                if left[0] not in expected(cards[0]) or right[0] not in expected(cards[-1]):continue
                origin=left[1].point()[0];pitch=(right[1].point()[0]-origin)/(len(cards)-1)
                if not .018*width<pitch<.10*width:continue
                selected=[]
                for i,card in enumerate(cards):
                    candidates=[m for m in markers if m[0] in expected(card) and abs(m[1].point()[0]-(origin+i*pitch))<pitch*.20]
                    # Tiered spells can also contain gold artwork. A purple
                    # shield is their slot badge; gold components must not
                    # turn an otherwise complete tier-badge row ambiguous.
                    badges=[m for m in candidates if m[0]=='minion' and (card.tier or m[1].x in shield_xs)]
                    if badges:candidates=badges
                    if len(candidates)!=1:break
                    selected.append(candidates[0])
                if len(selected)==len(cards) and len({m[1].x for m in selected})==len(cards) and selected not in fits:
                    boxes=[m[1] for m in selected]
                    # A complete marker row is also aligned and similarly
                    # scaled. Purple portrait art below the badges is not a
                    # second equally plausible hand row.
                    if len(boxes)>=4:
                        xs=np.array([b.point()[0] for b in boxes])/width
                        ys=np.array([b.point()[1] for b in boxes])/height
                        curve=np.polyfit(xs-xs.mean(),ys,2)
                        if not .02<curve[0]<8 or max(abs(np.polyval(curve,xs-xs.mean())-ys))>.009:continue
                    elif max(b.point()[1] for b in boxes)-min(b.point()[1] for b in boxes)>height*.035:continue
                    if max(b.height for b in boxes)>min(b.height for b in boxes)*1.5:continue
                    if max(b.width for b in boxes)>min(b.width for b in boxes)*1.5:continue
                    fits.append(selected)
    if len(fits)!=1:return fallback
    markers=fits[0]
    result=[]
    for i,(c,(kind,b)) in enumerate(zip(cards,markers)):
        if kind not in expected(c):return []
        if dense_minions and c.card_type=='MINION':continue  # Require its exposed portrait cap.
        # Use a small exposed strip below the measured marker, before the
        # next marker. No interpolation fills missing markers.
        x=b.x+b.width*1.4;y=b.y+b.height*1.25
        if len(markers)==2:
            # For a two-card hand, badge centers remain fixed while gold stars
            # split the purple mask into slightly different component bounds.
            # The observed inter-badge pitch gives a stable interior offset.
            pitch=markers[1][1].x-markers[0][1].x
            x=b.x+pitch*.25;y=b.y+pitch*.25
        if i+1<len(markers):x=min(x,markers[i+1][1].x-b.width*.35)
        if c.card_type=='MINION' and b.x<=x<=b.x+b.width and b.y<=y<=b.y+b.height:
            # Inter-badge pitch in a mixed small hand can be much smaller than
            # the minion badge itself. The tier shield is not the draggable
            # card body. Require its actual portrait arc instead of dragging
            # the shield, adding a magic offset or replaying a failed gesture.
            from .hand_cap import portrait_cap
            assignment=dict(entity_id=c.entity_id,badge=asdict(b))
            boundary=markers[i+1][1].x-b.x if i+1<len(markers) else width
            rect,_=portrait_cap(image,assignment,[assignment],boundary,pixels=rgb,include_eligible_rim=True)
            if rect and allowed(Box(**rect).point(),image.size,'hand'):
                result.append((c,Box(**rect)))
            continue
        if not b.x<x<width or not 0<y<height*.99:return []
        box=Box(round(x)-5,round(y)-5,10,10)
        if not allowed(box.point(),image.size,'hand'):continue
        pixels=np.asarray(image.crop((box.x,box.y,box.x+box.width,box.y+box.height)))
        if float(pixels.mean())<=20 or float(pixels.std())<=5:continue
        result.append((c,box))
    # Independently measured portrait caps are preferable for minions. A
    # missing spell no longer discards proven minion slots, or vice versa.
    proven={c.entity_id:(c,b) for c,b in result}
    proven.update({c.entity_id:(c,b) for c,b in fallback})
    return [proven[c.entity_id] for c in cards if c.entity_id in proven]

