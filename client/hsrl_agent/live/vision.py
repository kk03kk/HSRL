"""Offline-testable geometry, visual stability and optional Chinese/English OCR."""
from pathlib import Path
import re
import numpy as np
from PIL import Image,ImageChops,ImageStat
from .contracts import Blocked

TRIBES={'野兽':'BEAST','恶魔':'DEMON','龙':'DRAGON','元素':'ELEMENTAL','机械':'MECHANICAL',
        '鱼人':'MURLOC','海盗':'PIRATE','野猪人':'QUILBOAR','亡灵':'UNDEAD','畸变怪':'ABERRATION','纳迦':'NAGA'}

def stable(first,second,box,threshold=3.):
    if first.size!=second.size: return False
    diff=ImageChops.difference(first.crop(box).convert('RGB'),second.crop(box).convert('RGB'))
    return max(ImageStat.Stat(diff).mean)<=threshold

def crop_box(rect,size):
    w,h=size;x,y,rw,rh=rect
    if any(not np.isfinite(v) for v in rect) or not 0<=x<x+rw<=1 or not 0<=y<y+rh<=1:
        raise Blocked('invalid normalized image region')
    return tuple(round(v) for v in (x*w,y*h,(x+rw)*w,(y+rh)*h))

def template_score(image,template):
    a=np.asarray(image.convert('RGB').resize(template.size),np.float32)
    b=np.asarray(template.convert('RGB'),np.float32)
    return float(np.mean(np.abs(a-b))/255.)

class OCR:
    def __init__(self,max_width=1280):
        # Optional dependencies live in a separate directory and never modify
        # training requirements/site-packages. No import until explicitly used.
        from rapidocr import RapidOCR
        model_dir=Path(__file__).resolve().parents[3]/'artifacts/client/ocr'
        self.engine=RapidOCR(params={'Global.model_root_dir':str(model_dir),'EngineConfig.onnxruntime.intra_op_num_threads':2,
                                     'EngineConfig.onnxruntime.inter_op_num_threads':1})
        # Lobby/choice text is large; half-width OCR keeps every calibrated
        # phrase but avoids the 1.5-2.3 s full-screen pass seen at 2560 px.
        self.max_width=max_width
    def read(self,image):
        output=self.engine(np.asarray(image.convert('RGB')))
        if output is None or output.txts is None: return []
        return [(text,float(score)) for text,score in zip(output.txts,output.scores)]
    def sample(self,image):
        if not isinstance(self.max_width,int) or self.max_width<320 or image.width<=self.max_width:return image
        width=self.max_width;height=max(1,round(image.height*width/image.width))
        return image.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
    def locate(self,image):
        sample=self.sample(image)
        output=self.engine(np.asarray(sample.convert('RGB')))
        if output is None or output.txts is None or output.boxes is None:return []
        # Normalized coordinates are scale invariant; use the OCR sample size.
        return [dict(text=str(text),confidence=float(score),point=(np.asarray(box).mean(axis=0)/np.array(sample.size)).tolist())
                for text,score,box in zip(output.txts,output.scores,output.boxes)]


class AsyncOCR:
    """Build the OCR engine in a background thread during the model load.

    Engine construction takes ~1.0 s on this machine (measured offline) and the
    operator waited for it *after* the window switch. Starting it while torch and
    the checkpoint load hides that cost without touching the game.
    """

    def __init__(self,max_width=1024):
        import threading
        self.max_width=max_width;self._engine=None;self._error=None
        self._thread=threading.Thread(target=self._build,name='lobby-ocr-prewarm',daemon=True)
        self._thread.start()

    def _build(self):
        try:self._engine=OCR(max_width=self.max_width)
        except BaseException as exc:self._error=exc

    def get(self):
        self._thread.join()
        if self._error is not None:raise self._error
        return self._engine

def races_from_text(rows):
    texts=[t for t,score in rows if score>=.9]
    found=set()
    for text in texts:
        # ROI must contain only the "available" list, not both available/banned.
        for word,name in TRIBES.items():
            if word in text: found.add(name)
        for name in TRIBES.values():
            if re.search(r'\b'+name+r'\b',text.upper()): found.add(name)
    if len(found)!=5: raise Blocked('OCR did not establish exactly five available tribes')
    return sorted(found)

class Geometry:
    def __init__(self,profile): self.profile=profile
    def family(self,snapshot):
        data=snapshot['client'];pending=snapshot['view']['pending']
        if pending==4:return 'recruit'
        return 'hero' if pending==1 else {6:'trinket',7:'quest',8:'timewarp'}.get(pending,'discover') if data['choices'] else 'recruit'
    def locations(self,snapshot):
        data=snapshot['client'];pending=snapshot['view']['pending']
        family=self.family(snapshot)
        points={};zones={};layouts=self.profile['layouts']
        for zone in ('hand','board','shop','choices'):
            if zone=='choices' and pending==4: continue
            zone_family=family if zone=='choices' else 'recruit'
            cards=data[zone];count=len(cards)
            if not count:
                for layout in layouts:
                    if layout['family']==zone_family and layout['zone']==zone and layout['count']==0:
                        if self.profile.get('review_state')=='reviewed' and zone_family+'/'+zone+'/0' not in self.profile.get('reviewed_layouts',[]):
                            raise Blocked('empty zone has not been visually reviewed:'+zone)
                        for key,rect in layout.get('regions',{}).items():
                            if key.startswith('region:'):
                                x,y,w,h=rect;points[key]=(x+w/2,y+h/2)
                continue
            matches=[v for v in layouts if v['family']==zone_family and v['zone']==zone and v['count']==count]
            key=zone_family+'/'+zone+'/'+str(count)
            if len(matches)!=1: raise Blocked('uncalibrated layout:'+key)
            if self.profile.get('review_state')=='reviewed' and key not in self.profile.get('reviewed_layouts',[]):
                raise Blocked('layout has not been visually reviewed:'+key)
            layout=matches[0];regions=layout.get('regions',{});model=layout.get('model',{})
            # Board insertion slots are a first-class calibrated geometry, not
            # a value inferred from whichever cards currently happen to occupy
            # the row.  Their keys are also used by Play/Move drops below.
            if zone=='board':
                for key,rect in regions.items():
                    if key.startswith('position:'):
                        x,y,w,h=rect;points[key]=(x+w/2,y+h/2)
            for i,c in enumerate(cards):
                rect=regions.get('slot:'+str(i+1))
                if rect:
                    x,y,w,h=rect;point=(x+w/2,y+h/2)
                elif model.get('kind')=='centered_row':
                    point=(model['cx']+(i-(count-1)/2)*model['pitch'],model['cy'])
                else: raise Blocked('slot has no calibrated position')
                points['entity:'+str(c['entity_id'])]=point;zones[c['entity_id']]=zone
        hud=[v for v in layouts if v['family']==family and v['zone']=='hud']
        if len(hud)!=1: raise Blocked('HUD layout missing/ambiguous')
        if self.profile.get('review_state')=='reviewed' and family+'/hud/'+str(hud[0]['count']) not in self.profile.get('reviewed_layouts',[]):
            raise Blocked('HUD controls have not been visually reviewed:'+family)
        for key,rect in hud[0]['regions'].items():
            x,y,w,h=rect;points[key]=(x+w/2,y+h/2)
        return points,zones
    def plan(self,decision,snapshot):
        b=decision.binding
        points,zones=self.locations(snapshot)
        def point(key):
            if key not in points: raise Blocked('unlocated UI object:'+key)
            x,y=points[key]
            if not 0<x<1 or not 0<y<1: raise Blocked('UI coordinate out of client bounds')
            return [x,y]
        if b.kind=='cancel': return [('right_click',point('entity:'+str(b.source)))] if b.source else [('right_click',[.5,.5])]
        if b.kind=='end': return [('click',point('control:end'))]
        if b.kind in ('refresh','freeze','upgrade'): return [('click',point('control:'+b.kind))]
        if b.kind in ('hero_power','dark_gift'):
            key='control:hero_power' if b.kind=='hero_power' else 'control:game_mode_slot:5'
            powers=snapshot['view'].get('power_entities') or []
            # A staged binding whose power entity is no longer in the current
            # view is an unavailable target, never an IndexError that ends the run.
            if b.kind=='hero_power' and b.source not in powers:raise Blocked('hero power entity is not in the current view')
            if b.kind=='hero_power' and powers.index(b.source)>0:
                key='control:hero_power:1'
            return [('click',point(key))]
        source=point('entity:'+str(b.source))
        if b.kind in ('choose','branch','target','activate'): return [('click',source)]
        if b.kind=='buy': target=point('region:hand')
        elif b.kind=='sell': target=point('region:bob')
        elif b.kind=='magnetize':
            spec=self.profile.get('magnetize_drop') or {}
            if not spec.get('qualified') or spec.get('mode')!='left_of_target':
                raise Blocked('magnetize drop geometry is not qualified')
            x,y=point('entity:'+str(b.target))
            offset=spec.get('pitch_fraction')
            if not isinstance(offset,(int,float)) or not .2<=offset<=.5:raise Blocked('invalid magnetic insertion offset')
            target=[x-self.profile['board_pitch']*offset,y]
        elif b.kind=='move':
            board=snapshot['client']['board'];pos=b.position
            if not board or not 0<=pos<len(board):raise Blocked('invalid final move position')
            # Training Move(src,dst) means final 0-based slot dst.  Unity's
            # SendOption position is the calibrated drop slot dst+1.  Use that
            # immutable slot, not a midpoint recomputed after deleting the
            # source: in the real 00:45:44 game deleting source slot 0 shifted
            # that midpoint one slot right, so dst=5 was submitted as slot 7.
            target=list(point('position:board:'+str(pos+1)))
        elif b.kind=='play':
            card=next(c for c in snapshot['client']['hand'] if c['entity_id']==b.source)
            if card['card_type']=='MINION':
                board=snapshot['client']['board'];pos=b.position
                # Inserting N+1 positions differs from moving an existing card.
                if not 0<=pos<=len(board): raise Blocked('invalid insertion')
                if board:
                    # The calibrated position slots already encode insertion
                    # before card N (1-based N) and the trailing slot N+1.
                    target=list(point('position:board:'+str(pos+1)))
                else: target=point('region:board')
            else: target=point('region:board')
        else: raise Blocked('unhandled model action:'+b.kind)
        if not all(0<v<1 for v in target): raise Blocked('destination outside client')
        return [('drag',source,target)]
