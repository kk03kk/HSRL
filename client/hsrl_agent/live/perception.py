"""Template-classified selection identity, bound to a fresh choice request.

No green-outline/hover heuristic can authorize confirmation. A qualified profile
must distinguish unselected, hover, transition and selected examples. Historical
template matching alone never establishes current client identity.
"""
import hashlib
from pathlib import Path
from PIL import Image
from .contracts import Blocked
from .interactions import choice_context
from .vision import crop_box,template_score


def classify(image,profile,spec):
    qualification=spec.get('qualification',{})
    if (not qualification.get('independent_review') or
        set(('none','hover','transition','selected'))-set(qualification.get('tested_states',[]))):
        raise Blocked('selection classifier has no independent negative-control qualification')
    rows=[]
    for cls in spec['classes']:
        path=(Path(profile['_directory'])/cls['template']).resolve()
        if not path.is_relative_to(Path(profile['_directory']).resolve()):raise Blocked('selection template outside profile')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cls['sha256']:raise Blocked('selection template changed')
        with Image.open(path) as template:
            error=template_score(image.crop(crop_box(spec['rect'],image.size)),template)
        rows.append((error,cls))
    rows.sort(key=lambda x:x[0])
    if len(rows)<2 or rows[0][0]>spec.get('max_error',.025) or rows[1][0]-rows[0][0]<spec.get('margin',.01):
        raise Blocked('selected identity is visually ambiguous')
    return rows[0]


def selection(image,profile,data):
    choices=data.get('choices',[])
    if not choices:return None
    family='hero' if all(c['card_type']=='HERO' for c in choices) else 'discover'
    spec=profile.get('selection_guards',{}).get(family+'/'+str(len(choices)))
    if not spec:return None
    error,label=classify(image,profile,spec);index=label.get('selected_index')
    if label.get('state')!='selected':return None
    if type(index) is not int or not 0<=index<len(choices):raise Blocked('selected index is not a current candidate')
    return dict(entity_id=choices[index]['entity_id'],context=choice_context(data),
        image_sha256=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),error=error,
        evidence='qualified_template_selection',template_sha256=label['sha256'])


def branch(image,profile,data,pending):
    if not pending or not pending[0].binding.branches:return {}
    d,before,_=pending
    source=next((c for z in ('hand','board','controls') for c in before['client'].get(z,[]) if c['entity_id']==d.binding.source),None)
    if not source:return {}
    spec=profile.get('branch_guards',{}).get(source['card_id'])
    if not spec:return {}
    error,label=classify(image,profile,spec)
    if label.get('state')!='selected':return {}
    current={c['entity_id']:c for c in data.get('branches',[])}
    ids=[r[0] for r in d.binding.branches]
    if any(e not in current for e in ids):raise Blocked('branch entities no longer exist')
    if [current[e]['card_id'] for e in ids]!=label.get('branch_card_ids'):
        raise Blocked('branch template identities differ from server candidates')
    return dict(branch_open=True,branch_source_card_id=source['card_id'],branch_entities=ids,
                branch_visual_error=error,branch_template_sha256=label['sha256'])
