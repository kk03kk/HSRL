"""Reviewed, device-bound geometry. No capture, image search or input APIs."""
from dataclasses import asdict,replace
import hashlib,json,os
from pathlib import Path
from functools import lru_cache

from .layout_resolver import LayoutCatalog,LayoutSpec,Viewport
from .windows import Box


DEFAULT_PATH=Path(__file__).parent/'data/local_layout_2560x1600.json'
DISPLAY_KEYS=('graphicswidth','graphicsheight','graphicsfullscreen','graphicsquality')


def display_settings(path=None):
    path=Path(path) if path else Path(os.environ.get('LOCALAPPDATA',''))/'Blizzard/Hearthstone/options.txt'
    if not path.is_file():return {}
    stat=path.stat()
    return dict(_settings(str(path),stat.st_mtime_ns,stat.st_size))


@lru_cache(maxsize=8)
def _settings(path,mtime,size):
    result={}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        key,sep,value=line.partition('=')
        if sep and key.strip().lower() in DISPLAY_KEYS:result[key.strip().lower()]=value.strip()
    return tuple(result.items())


@lru_cache(maxsize=4)
def _binary_hash(path,mtime,size):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def executable_hash(path):
    path=Path(path);stat=path.stat()
    return _binary_hash(str(path.resolve()),stat.st_mtime_ns,stat.st_size)


def environment_errors(environment,window,build,settings,binary_hash=None):
    errors=[];client=window.client if hasattr(window,'client') else Box(**window['client'])
    dpi=window.dpi if hasattr(window,'dpi') else window.get('dpi')
    if (client.width,client.height)!=(environment['width'],environment['height']):errors.append('local_render_size_mismatch')
    if environment.get('dpi') is not None and dpi!=environment['dpi']:errors.append('local_dpi_mismatch')
    if build is not None and environment.get('build') is not None and str(build)!=str(environment['build']):errors.append('local_build_mismatch')
    if environment.get('executable_sha256') is None or binary_hash!=environment.get('executable_sha256'):errors.append('local_executable_mismatch')
    if build is None and binary_hash is None:errors.append('local_build_unknown')
    expected=environment.get('display_settings',{})
    if not expected or any(settings.get(k)!=v for k,v in expected.items()):errors.append('local_display_settings_mismatch')
    if settings.get('graphicsfullscreen','').lower()!='true':errors.append('local_fullscreen_required')
    return errors


def coverage(entries):
    """Ordinary release gate. Special families remain explicit recovery gaps."""
    entries=[s for s in entries if s.status=='verified'];missing=[]
    for zone,counts in (('hand',range(1,11)),('shop',range(1,8)),('board',range(8))):
        for count in counts:
            specs=[s for s in entries if (s.family,s.zone,s.count)==('recruit',zone,count)]
            if zone=='hand':specs=[s for s in specs if s.variant=='default' and s.probes.get('unified_hand')]
            if not specs:missing.append(f'{zone}:{count}');continue
            if zone in ('shop','board') and count and not any(
                    all(f'slot:{i}' in s.probes.get('slot_bounds',{}) for i in range(1,count+1)) for s in specs):
                # 鼠标命中点不是对象检测范围；不能把只有6×6命中框登记为覆盖完成。
                missing.append(f'{zone}:{count}:object_bounds')
            if zone=='board':
                for index in range(1,count+2):
                    if count==0:key='region:board'
                    else:key=f'position:board:{index}'
                    if not any(key in s.regions for s in specs):missing.append(f'board:{count}:{key}')
    for family,key in [('menu','control:enter')]+[('recruit','control:'+name) for name in ('refresh','upgrade','freeze','hero_power','game_mode_slot:1','game_mode_slot:5')]:
        if not any(s.family==family and s.zone=='hud' and key in s.regions for s in entries):missing.append(f'{family}:{key}')
    return missing


class LocalLayoutProfile:
    def __init__(self,data):
        if data.get('schema')!=1:raise ValueError('local_layout_schema_unknown')
        self.data=data;self.environment=data['environment'];self.status=data['status']
        self.entries=[LayoutSpec.from_dict(row) for row in data.get('layouts',())]
        self.missing=coverage(self.entries)

    @classmethod
    def load(cls,path=DEFAULT_PATH):return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def require_ready(self,window,build,settings,binary_hash=None):
        errors=environment_errors(self.environment,window,build,settings,binary_hash)
        if self.status!='verified':errors.append('local_calibration_incomplete')
        errors.extend('local_coverage_missing:'+key for key in self.missing)
        if errors:raise ValueError(';'.join(errors))

    def catalog(self,base):
        if self.status!='verified':raise ValueError('local_calibration_incomplete')
        local=[replace(s,probes={**s.probes,'local_profile':True}) for s in self.entries]
        return LayoutCatalog(list(base.entries)+local)


def compile_reviewed(records,environment):
    """Fit only annotated training captures; validate against separate captures.

    Rectangles are native physical client pixels. Publish only a small inset
    inside training safety boxes; held-out data never fits the point.
    """
    groups={}
    for record in records:
        if not record.get('reviewed'):raise ValueError('annotation_review_required')
        if record.get('role') not in ('calibration','validation'):raise ValueError('annotation_role_required')
        if record.get('predicted'):raise ValueError('predictions_are_not_calibration')
        if record.get('environment')!=environment:raise ValueError('annotation_environment_mismatch')
        if record.get('blocked') or record.get('synchronized') is not True:raise ValueError('annotation_not_synchronized')
        key=(record['family'],record['zone'],record['count'],record.get('variant','default'))
        groups.setdefault(key,[]).append(record)
    result=[];w,h=environment['width'],environment['height']
    for key,rows in groups.items():
        train=[r for r in rows if r['role']=='calibration'];held=[r for r in rows if r['role']=='validation']
        if not train or not held:raise ValueError('independent_holdout_required:'+str(key))
        identities=lambda rs:{r['capture'] for r in rs}
        if identities(train)&identities(held):raise ValueError('same_capture_cannot_validate_itself')
        if {r['sha256'] for r in train}&{r['sha256'] for r in held}:raise ValueError('duplicate_image_is_not_holdout')
        if any(abs(a['captured_at']-b['captured_at'])<.033 for a in train for b in held):raise ValueError('captures_too_close')
        family,zone,count,variant=key
        keys=set.intersection(*(set(r['regions']) for r in rows))
        if zone in ('hand','board','shop','choices') and any(f'slot:{i+1}' not in keys for i in range(count)):
            raise ValueError('physical_slot_annotation_missing')
        regions={}
        for name in keys:
            boxes=[Box(**r['regions'][name]) for r in train]
            x=max(b.x for b in boxes);y=max(b.y for b in boxes)
            right=min(b.x+b.width for b in boxes);bottom=min(b.y+b.height for b in boxes)
            if right-x<8 or bottom-y<8:raise ValueError('safe_region_too_small:'+name)
            cx=(x+right)/2;cy=(y+bottom)/2
            box=Box(round(cx)-3,round(cy)-3,6,6)
            for r in held:
                b=Box(**r['regions'][name])
                if not (b.x<=box.x and b.y<=box.y and box.x+6<=b.x+b.width and box.y+6<=b.y+b.height):raise ValueError('holdout_position_mismatch:'+name)
            regions[name]=[box.x/w,box.y/h,box.width/w,box.height/h]
        # Local stability probes are manually selected external contours, not
        # source-card artwork. They must recur in calibration and validation.
        probes={};probe_names=set.intersection(*(set(r.get('edge_probes',{})) for r in rows))
        for name in probe_names:
            values=[r['edge_probes'][name] for r in rows]
            if all(v==values[0] for v in values):
                p=values[0];b=Box(**p['rect']);Viewport(w,h).box([b.x/w,b.y/h,b.width/w,b.height/h])
                probes[name]=dict(rect=[b.x/w,b.y/h,b.width/w,b.height/h],axis=p['axis'])
        provenance=tuple({k:r[k] for k in ('capture','sha256','captured_at','role','source_environment','geometry_only_review') if k in r} for r in rows)
        identity=hashlib.sha256(json.dumps([key,environment],sort_keys=True).encode()).hexdigest()[:16]
        geometry_only=any(r.get('geometry_only_review') for r in rows)
        extra=dict(local_profile=True,unified_hand=zone=='hand',
            geometry_only_evidence=geometry_only,edge_probes={} if geometry_only else probes)
        result.append(LayoutSpec.from_dict(dict(id='local-'+identity,family=family,zone=zone,count=count,
            variant=variant,build=environment['build'],aspect=w/h,status='verified',regions=regions,provenance=provenance,probes=extra)))
    return result
