"""Positive ordinary discovery evidence, independent of server availability.

Adapted from the read-only HearthstoneAI reference's bg_agent/live_interactions.py
discover_regions. Full row geometry is bound to existing calibrated hit points;
it does not identify cards from their art or choose an alternative action.
"""
import numpy as np


def power_fragments(mask,count):
    """Join near-touching, equally tall power-card fragments, then measure rails.

    Adapted from the read-only reference's live_choices.power_card_row. This
    returns outer boxes for the caller's calibrated-hit checks, not new clicks.
    """
    import cv2
    height,width=mask.shape
    joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    parts=sorted((cv2.boundingRect(c) for c in contours),key=lambda b:b[0])
    frames=[]
    for x,y,w,h in parts:
        if not (.035*width<w<.30*width and .26*height<h<.66*height
                and .18*height<y<.50*height and .15<w/h<1.3):continue
        if frames:
            px,py,pw,ph=frames[-1];right=max(px+pw,x+w);bottom=max(py+ph,y+h)
            if (-.003*width<=x-px-pw<=.012*width and abs(y-py)<.025*height
                    and .8<h/ph<1.25 and right-px<.30*width):
                frames[-1]=(px,min(py,y),right-px,bottom-min(py,y));continue
        frames.append((x,y,w,h))
    if len(frames)!=count or any(not (.10*width<w<.30*width and .4<w/h<1.3) for x,y,w,h in frames):return []
    return [(x/width,y/height,w/width,h/height) for x,y,w,h in frames]


def paired_rails(mask,hits,width,height,scale,min_support=.55):
    """Measured long rails for every pinned hit; artwork connectivity is irrelevant."""
    xs=[p[0] for p in hits];supports=[];boxes=[]
    for index,(px,py) in enumerate(hits):
        pair=[];observed=[]
        # Stay inside this hit's own slot; a neighboring card's opposite edge
        # must never stand in for the missing rail of this candidate.
        left=max(.06,(xs[index-1]+px)/2 if index else px-.18)
        right=min(.94,(px+xs[index+1])/2 if index+1<len(hits) else px+.18)
        for x0,x1 in ((left,px-.055),(px+.055,right)):
            y0=max(0,py-.10);y1=min(1,py+.15)
            patch=mask[round(y0*height*scale):round(y1*height*scale),
                       round(x0*width*scale):round(x1*width*scale)]
            columns=np.mean(patch>0,axis=0) if patch.size else np.array([0.])
            pair.append(float(columns.max()))
            observed.append(float((round(x0*width*scale)+int(np.argmax(columns)))/(width*scale)))
        supports.append(pair);boxes.append((observed[0],y0,observed[1]-observed[0],y1-y0))
    valid=bool(hits) and all(min(pair)>min_support for pair in supports) and all(.12<box[2]<.30 for box in boxes)
    valid=valid and all(a[0]+a[2]<b[0] for a,b in zip(boxes,boxes[1:]))
    return dict(visible=valid,boxes=boxes,rails=supports,evidence='paired_straight_discovery_rails_at_every_calibrated_hit')


def adaptive_rails(mask,hits,width,height,scale,min_support=.65):
    """Last-resort rail finder for layout variants (large/small card rows).

    The calibrated hit is only guaranteed to be *inside* the card; some panels
    (e.g. the big "进行选择" spell row) place the cards 3 % off the inherited
    spacing, which made every fixed-offset strip miss one rail and blocked the
    row for ~11 s. Search a bounded window per side, then require a plausible
    card width and that the hit lies between the two rails.
    """
    supports=[];boxes=[]
    for px,py in hits:
        y0=max(0,py-.10);y1=min(1,py+.15)
        band=mask[round(y0*height*scale):round(y1*height*scale)]
        if not band.size:return dict(visible=False,boxes=[],rails=[],evidence='adaptive_rails_unavailable')
        columns=np.mean(band>0,axis=0)
        def best(lo,hi):
            a=max(0,round(lo*width*scale));b=min(columns.size,round(hi*width*scale))
            if b<=a:return 0.,None
            index=a+int(np.argmax(columns[a:b]))
            return float(columns[a:b].max()),float(index/(width*scale))
        left_support,left_x=best(px-.125,px-.025)
        right_support,right_x=best(px+.025,px+.125)
        supports.append([left_support,right_support])
        boxes.append((left_x,right_x))
    widths=[right-left for left,right in boxes if left is not None and right is not None]
    valid=(len(widths)==len(hits) and all(min(pair)>min_support for pair in supports)
           and all(.10<=width<=.22 for width in widths))
    if valid and len(boxes)>1:
        valid=all(pair[1][0]<pair[0][0] for pair in zip(boxes,boxes[1:]))
    framed=[(left,y0,right-left,y1-y0) for (left,right) in boxes] if valid else []
    return dict(visible=valid,boxes=framed,rails=supports,
                evidence='adaptive_bounded_card_edges_at_calibrated_hits')


def slot_colors(image,profile,data):
    """Per-slot (purple/green, gold) rail support for highlight-difference checks."""
    import cv2
    from .practical import PracticalGeometry
    cards=data.get('choices',[]);count=len(cards)
    if not 1<=count<=6:return []
    kind=5 if all(c['card_type']=='HERO_POWER' for c in cards) else 6 if all(
        c['card_type']=='BATTLEGROUND_TRINKET' for c in cards) else 7 if any(c.get('tags',{}).get('QUEST')=='1' for c in cards) else 2
    points,_=PracticalGeometry(profile).locations(dict(client=data,view=dict(pending=kind)))
    width,height=image.size;scale=min(1,1000/width)
    hsv=cv2.cvtColor(cv2.resize(np.asarray(image.convert('RGB')),None,fx=scale,fy=scale),cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    base=np.where((h>125)&(h<175)&(s>65)&(v>50),True,False) if kind==6 else \
        (cv2.inRange(hsv,np.array([35,95,100]),np.array([112,255,255]))>0)
    gold=(h>=12)&(h<35)&(s>95)&(v>100)
    rows=[]
    for card in cards:
        px,py=points['entity:'+str(card['entity_id'])];pair_b=[];pair_g=[]
        for left,right in ((px-.09,px-.04),(px+.04,px+.09)):
            y0=round((py-.045)*height*scale);y1=round((py+.045)*height*scale)
            x0=round(left*width*scale);x1=round(right*width*scale)
            for pair,mask in ((pair_b,base),(pair_g,gold)):
                patch=mask[y0:y1,x0:x1]
                pair.append(float(np.mean(np.any(patch,axis=1))) if patch.size else 0)
        rows.append(dict(entity=card['entity_id'],base=pair_b,gold=pair_g))
    return rows


def gold_only_slots(rows,min_support=.65):
    """Slots whose only frame evidence is gold (a golden card or a highlight)."""
    return {row['entity'] for row in rows
            if min(row['gold'])>min_support and min(row['base'])<=min_support}


def inspect(image,profile,data,*,hovered_entity=None,min_support=.55):
    import cv2
    from .practical import PracticalGeometry
    cards=data.get('choices',[]);count=len(cards)
    kind=5 if cards and all(c['card_type']=='HERO_POWER' for c in cards) else 6 if cards and all(
        c['card_type']=='BATTLEGROUND_TRINKET' for c in cards) else 7 if any(c.get('tags',{}).get('QUEST')=='1' for c in cards) else 2
    # Time-warp/discover panels in 255790 can show five options.  The rail
    # detector already accepts 1..6 ordinary cards; keep the family choice
    # independent from the old four-card assumption.
    result=dict(visible=False,boxes=[],entities=[c['entity_id'] for c in cards],kind=kind)
    if not 1<=count<=6:return result
    points,_=PracticalGeometry(profile).locations(dict(client=data,view=dict(pending=kind)))
    width,height=image.size;scale=min(1,1000/width)
    hsv=cv2.cvtColor(cv2.resize(np.asarray(image.convert('RGB')),None,fx=scale,fy=scale),cv2.COLOR_RGB2HSV)
    if kind==6:
        # Trinkets have purple rails; a *golden* trinket paints them gold (the
        # 13:55 run offered 三联奖品 with a gold right rail; the 19:48 run
        # offered two fully golden trinkets). Gold is therefore a legitimate
        # frame colour and never a reason to reject the row by itself; the
        # stale-highlight guard moved to the press-time comparison against the
        # admitted frame (frame_gate / slot_colors).
        h,s,v=cv2.split(hsv);purple=(h>125)&(h<175)&(s>65)&(v>50)
        gold=(h>=12)&(h<35)&(s>95)&(v>100)
        supports=[];box_rows=[]
        if count!=4:return result
        for card in cards:
            px,py=points['entity:'+str(card['entity_id'])];pair_p=[];pair_g=[]
            # Rails sit ~0.066 from the hit; the neighbouring card's rail is at
            # ~0.094. Keep both strips inside that band so a slot can never
            # borrow its neighbour's edge.
            for left,right in ((px-.085,px-.045),(px+.045,px+.085)):
                y0=round((py-.045)*height*scale);y1=round((py+.045)*height*scale)
                x0=round(left*width*scale);x1=round(right*width*scale)
                for pair,mask in ((pair_p,purple),(pair_g,gold)):
                    patch=mask[y0:y1,x0:x1]
                    pair.append(float(np.mean(np.any(patch,axis=1))) if patch.size else 0)
            supports.append(pair_p);box_rows.append(pair_g)
        # Gold alone also matches brown shop/board decorations on a recruit
        # screen, so a real trinket row must still show a purple frame on both
        # sides of at least one calibrated slot (all measured real panels do,
        # including the one with two fully golden trinkets; the recruit frames
        # that used to be false positives show none).
        purple_sides=sum(1 for pair in supports if min(pair)>=min_support)
        visible=purple_sides>=1 and all(min(max(p,g) for p,g in zip(supports[i],box_rows[i]))>min_support
                                        for i in range(count))
        boxes=[(points['entity:'+str(c['entity_id'])][0]-.09,points['entity:'+str(c['entity_id'])][1]-.045,.18,.09)
               for c in cards]
        if not visible:
            adaptive=adaptive_rails(purple|gold,[points['entity:'+str(c['entity_id'])] for c in cards],
                                    width,height,scale,min_support=min_support)
            if adaptive['visible']:
                return dict(result,visible=True,boxes=adaptive['boxes'],rails=supports,gold_rails=box_rows,
                            evidence=adaptive['evidence'])
        return dict(result,visible=visible,boxes=boxes,rails=supports,gold_rails=box_rows,
                    evidence='four_paired_purple_or_golden_trinket_rails_at_calibrated_hits')
    # Ordinary frames are green, but a *golden* (premium) card is painted gold
    # and the 21:17 run's spell/discover rows were blocked for up to 11 s by a
    # single golden card. Gold is frame evidence; the stale-highlight guard is
    # the press-time comparison with the admitted frame.
    mask=cv2.inRange(hsv,np.array([35,95,100]),np.array([112,255,255]))
    h,s,v=cv2.split(hsv)
    gold_mask=(h>=12)&(h<35)&(s>95)&(v>100)
    if hovered_entity in result['entities']:
        # Ordinary and power choices gain a golden hover outline. Keep all
        # siblings green; constrain the extra color to this source's own slot.
        index=result['entities'].index(hovered_entity)
        xs=[points['entity:'+str(c['entity_id'])][0] for c in cards]
        left=(xs[index-1]+xs[index])/2 if index else max(.06,xs[index]-.18)
        right=(xs[index]+xs[index+1])/2 if index+1<count else min(.94,xs[index]+.18)
        gold=cv2.inRange(hsv,np.array([12,95,100]),np.array([34,255,255]))
        x0,x1=round(left*width*scale),round(right*width*scale)
        mask[:,x0:x1]=cv2.bitwise_or(mask[:,x0:x1],gold[:,x0:x1])
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for contour in contours:
        x,y,w,h=[v/scale for v in cv2.boundingRect(contour)]
        ordinary=(.12*width<w<.30*width and .29*height<h<.63*height and .18*height<y<.48*height and .3<w/h<1)
        if kind==5:
            ordinary=(.10*width<w<.30*width and .26*height<h<.66*height and .18*height<y<.50*height and .4<w/h<1.3)
        quest=(.10*width<w<.27*width and .10*height<h<.25*height and .50*height<y<.76*height and 1.1<w/h<2.8)
        if quest if kind==7 else ordinary:boxes.append((x/width,y/height,w/width,h/height))
    boxes.sort()
    if len(boxes)!=count and kind==5:boxes=power_fragments(mask,count)
    if len(boxes)==count and (
            max(b[1] for b in boxes)-min(b[1] for b in boxes)>.045
            or max(b[2] for b in boxes)/min(b[2] for b in boxes)>1.18
            or max(b[3] for b in boxes)/min(b[3] for b in boxes)>1.15):
        # Legendary decorations can split/reshape one contour. In that case
        # require fresh paired rails for the entire row, not looser box ratios.
        boxes=[]
    if len(boxes)!=count:
        # Attached reward panels, golden cards and card artwork can split or
        # hide the outer contour. Require both rails at every calibrated card
        # instead of treating a missing closed contour as a missing panel.
        if kind not in (2,5):return result
        hits=[points['entity:'+str(c['entity_id'])] for c in cards]
        frame_mask=cv2.bitwise_or(mask,(gold_mask.astype(np.uint8)*255))
        proof=paired_rails(frame_mask,hits,width,height,scale,min_support=min_support)
        if proof['visible']:
            return dict(result,**{**proof,'evidence':proof['evidence']+';golden frames allowed'})
        # Layout variants (large-card rows) place cards off the inherited
        # spacing; search bounded windows before declaring the row absent.
        adaptive=adaptive_rails(frame_mask,hits,width,height,scale,min_support=min_support)
        if adaptive['visible']:return dict(result,**adaptive)
        return dict(result,**proof)
    if max(b[1] for b in boxes)-min(b[1] for b in boxes)>.045:return result
    if max(b[2] for b in boxes)/min(b[2] for b in boxes)>1.18:return result
    if max(b[3] for b in boxes)/min(b[3] for b in boxes)>1.15:return result
    centers=[x+w/2 for x,y,w,h in boxes]
    if abs((centers[0]+centers[-1])/2-.5)>.12:return result
    if count>1:
        gaps=np.diff(centers)
        if min(gaps)<.12 or max(gaps)/min(gaps)>1.2:return result
        if any(a[0]+a[2]>=b[0] for a,b in zip(boxes,boxes[1:])):return result
    for card,(x,y,w,h) in zip(cards,boxes):
        px,py=points['entity:'+str(card['entity_id'])]
        if not x+w*.15<=px<=x+w*.85 or not y+h*.15<=py<=y+h*.85:return result
        for left in (x,x+w*.9):
            rail=mask[round((y+h*.25)*height*scale):round((y+h*.75)*height*scale),
                      round(left*width*scale):round((left+w*.1)*width*scale)]
            if not rail.size or np.mean(np.any(rail>0,axis=1))<min_support:return result
    return dict(result,visible=True,boxes=boxes,evidence='complete_colored_choice_row_at_calibrated_hits')
