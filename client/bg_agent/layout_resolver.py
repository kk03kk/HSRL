"""Pixel-free layout resolution. Catalog entries are independent calibrations.

Identity, artwork, stats and glow never select coordinates. Live readiness and
server authorization are intentionally outside this module.
"""
from dataclasses import dataclass,field
import json
import math
from pathlib import Path

from .mapping import Mapping,StabilityBasis
from .model import Phase
from .windows import Box


def hand_variant(cards):
    if any(c.tags.get('LITERALLY_UNPLAYABLE') in ('1',1) for c in cards):return 'locked'
    if len(cards)<=3 and all(c.card_type in ('SPELL','BATTLEGROUND_SPELL') for c in cards):return 'spell_row'
    return 'fan'


def choice_family(obs):
    if not obs.choices:return 'unknown'
    if obs.interaction and obs.interaction.max_select>1:return 'multi'
    if any(c.tags.get('QUEST')=='1' for c in obs.choices):return 'quest'
    if all(c.card_type=='MINION' and c.tags.get('HAS_DARK_GIFT')=='1' and c.tags.get('DARK_GIFT_ENTITY') for c in obs.choices):return 'dark_gift'
    kinds={c.card_type for c in obs.choices}
    return {frozenset(('HERO',)):'hero',frozenset(('HERO_POWER',)):'power',
            frozenset(('BATTLEGROUND_TRINKET',)):'trinket'}.get(frozenset(kinds),'discover')


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int
    origin_x: int=0
    origin_y: int=0
    render: Box|None=None

    def __post_init__(self):
        r=self.render or Box(0,0,self.width,self.height)
        if self.width<=0 or self.height<=0 or r.width<=0 or r.height<=0 or min(r.x,r.y)<0 or r.x+r.width>self.width or r.y+r.height>self.height:
            raise ValueError('invalid_render_viewport')
        object.__setattr__(self,'render',r)

    @property
    def aspect(self):return self.render.width/self.render.height

    def box(self,uv):
        x,y,w,h=uv;r=self.render
        if not all(math.isfinite(v) for v in uv) or min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:
            raise ValueError('invalid_normalized_layout_region')
        return Box(round(r.x+x*r.width),round(r.y+y*r.height),max(2,round(w*r.width)),max(2,round(h*r.height)))

    @classmethod
    def from_window(cls,size,window):
        client=window.get('client',{})
        render=window.get('render_rect')
        return cls(*size,client.get('x',0),client.get('y',0),Box(**render) if render else None)


@dataclass(frozen=True)
class LayoutSpec:
    id: str
    family: str
    zone: str
    count: int
    build: str|None
    aspect: float
    regions: dict
    variant: str='default'
    status: str='candidate'
    provenance: tuple=()
    model: dict=field(default_factory=dict)
    probes: dict=field(default_factory=dict)
    session: str|None=None

    @classmethod
    def from_dict(cls,data):
        data=dict(data);data['provenance']=tuple(data.get('provenance',()))
        spec=cls(**data)
        if spec.count<0 or spec.count>10 or spec.status not in ('candidate','verified','session_calibrated'):
            raise ValueError('invalid_layout_spec')
        if spec.status!='candidate' and len(spec.provenance)<2:raise ValueError('layout_requires_independent_evidence')
        for value in spec.regions.values():
            if value is not None:Viewport(1000,1000).box(value)
        for value in spec.probes.get('slot_bounds',{}).values():Viewport(1000,1000).box(value)
        return spec


@dataclass(frozen=True)
class GeometryProjection:
    mappings: tuple
    layouts: tuple
    missing: tuple


class LayoutCatalog:
    def __init__(self,entries=()):self.entries=list(entries);self.generation=0;self._matches={}

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema')!=1:raise ValueError('layout_catalog_schema_unknown')
        return cls(LayoutSpec.from_dict(row) for row in data['layouts'])

    def add(self,spec):
        if spec.status=='candidate':return
        self.entries=[s for s in self.entries if s.id!=spec.id]+[spec];self.generation+=1;self._matches.clear()

    def find(self,family,zone,count,variant,build,viewport,session=None):
        key=(family,zone,count,variant,build,viewport.aspect,session,self.generation)
        if key in self._matches:return self._matches[key]
        candidates=[s for s in self.entries if s.status!='candidate' and
            (s.family,s.zone,s.count,s.variant)==(family,zone,count,variant) and s.build==build and
            abs(s.aspect/viewport.aspect-1)<=.005 and
            (s.status!='session_calibrated' or s.session==session)]
        if zone=='hand':
            candidates += [s for s in self.entries if s.status=='verified' and s.probes.get('local_profile') and s.probes.get('unified_hand') and
                (s.family,s.zone,s.count,s.variant)==(family,zone,count,'default') and s.build==build and abs(s.aspect/viewport.aspect-1)<=.005]
        if build is None and family=='menu':
            candidates += [s for s in self.entries if s.status=='verified' and s.probes.get('local_profile') and
                (s.family,s.zone,s.count,s.variant)==(family,zone,count,variant) and abs(s.aspect/viewport.aspect-1)<=.005]
        result=max(enumerate(candidates),key=lambda pair:(bool(pair[1].probes.get('local_profile')),pair[0]))[1] if candidates else None
        if len(self._matches)>=512:self._matches.clear()
        self._matches[key]=result
        return result

    def recovery(self,family,zone,count,variant,build,viewport):
        return next((s for s in reversed(self.entries) if s.status=='candidate' and s.probes.get('controlled_recovery') and
            (s.family,s.zone,s.count,s.variant,s.build)==(family,zone,count,variant,build) and abs(s.aspect/viewport.aspect-1)<=.005),None)


class LayoutResolver:
    def __init__(self,catalog):
        self.catalog=catalog;self._geometry={};self.hits=0;self.misses=0

    def regions(self,spec,viewport):
        r=viewport.render;key=(self.catalog.generation,spec.id,viewport.width,viewport.height,r.x,r.y,r.width,r.height)
        if key in self._geometry:self.hits+=1;return self._geometry[key]
        self.misses+=1
        regions={key:viewport.box(uv) for key,uv in spec.regions.items() if uv is not None}
        model=spec.model
        if model.get('kind')=='centered_row':
            for index in range(spec.count):
                x=model['cx']+(index-(spec.count-1)/2)*model['pitch']
                w,h=model['size'];regions[f'slot:{index+1}']=viewport.box((x-w/2,model['cy']-h/2,w,h))
        self._geometry[key]=regions
        if len(self._geometry)>128:self._geometry.pop(next(iter(self._geometry)))
        return regions

    def resolve(self,obs,viewport,*,session=None,family=None,cards=None,recovery=False):
        maps=[];used=[];missing=[]
        if family is not None:requests=[(family,'choices',cards or obs.choices,'default')]
        elif obs.phase==Phase.RECRUIT:
            requests=[('recruit','hud',(), 'default')]+[('recruit',z,getattr(obs,z),hand_variant(obs.hand) if z=='hand' else 'default') for z in ('shop','board','hand')]
        elif obs.phase==Phase.CHOICE:
            kind=choice_family(obs);requests=[(kind,'choices',obs.choices,'default'),(kind,'hud',(),'default')]
        else:requests=[(str(obs.phase),'hud',(),'default')]
        for family,zone,group,variant in requests:
            count=len(group)
            if zone not in ('hud','choices') and sorted(c.position for c in group)!=list(range(1,count+1)):
                missing.append(zone+':log_order');continue
            spec=self.catalog.find(family,zone,count,variant,obs.build,viewport,session)
            if spec is None and recovery:spec=self.catalog.recovery(family,zone,count,variant,obs.build,viewport)
            if spec is None:
                if count or zone=='hud':missing.append(zone+':uncalibrated')
                continue
            used.append(spec)
            regions=self.regions(spec,viewport)
            basis=StabilityBasis.PIXELS if spec.status=='candidate' else StabilityBasis.GEOMETRY
            for key,box in regions.items():
                if not key.startswith('slot:'):
                    maps.append(Mapping(key,'matched',box,.95,'calibrated_layout:'+spec.id,spec.id,stability_basis=basis))
            ordered=tuple(group) if zone=='choices' else sorted(group,key=lambda c:c.position)
            for index,card in enumerate(ordered):
                box=regions.get(f'slot:{index+1}')
                if box is not None:
                    maps.append(Mapping(f'entity:{card.entity_id}','matched',box,.95,'calibrated_layout:'+spec.id,spec.id,card.entity_id,basis))
        visible={m.key:m.rect for m in maps}
        for control in obs.controls:
            slot=control.tags.get('GAME_MODE_BUTTON_SLOT')
            key='control:game_mode_slot:'+str(slot)
            if key in visible:
                maps.append(Mapping(f'entity:{control.entity_id}','matched',visible[key],.95,
                    'calibrated_layout:control_slot',entity_id=control.entity_id,stability_basis=StabilityBasis.GEOMETRY))
        if obs.hero and 'region:hero' in visible:
            maps.append(Mapping(f'entity:{obs.hero.entity_id}','matched',visible['region:hero'],.95,
                'calibrated_layout:local_hero',entity_id=obs.hero.entity_id,stability_basis=StabilityBasis.GEOMETRY))
        if obs.phase==Phase.CHOICE and any(s.zone=='choices' for s in used):
            box=next((m.rect for m in maps if m.entity_id is not None),None)
            if box:maps.append(Mapping('scene:choice','matched',box,.95,'calibrated_layout:choice_scene',stability_basis=StabilityBasis.GEOMETRY))
        return GeometryProjection(tuple(maps),tuple(used),tuple(missing))
