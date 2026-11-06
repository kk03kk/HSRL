"""One coordinate contract for capability checks, planning and execution.

Routes describe geometry, never authorize an action. Server legality, modality,
fresh-frame readiness and result verification remain independent requirements.
"""
from dataclasses import dataclass

from .model import Kind


@dataclass(frozen=True)
class GeometryRoute:
    gesture: str
    source: str | None = None
    destination: str | None = None
    required: tuple[str, ...] = ()
    source_zone: str | None = None
    error: str | None = None


CONTROL_KINDS = frozenset((Kind.ENTER, Kind.DISMISS, Kind.REFRESH,
    Kind.UPGRADE, Kind.FREEZE, Kind.HERO_POWER, Kind.CONFIRM, Kind.CANCEL))


def play_destination_key(action,obs,mapped):
    """Translate placement intent, never substitute another explicit slot."""
    source=next((c for c in obs.hand if c.entity_id==action.source),None)
    if action.position is not None:
        if source and source.card_type=='MINION' and not obs.board and action.position==1:return 'region:board'
        return f'position:board:{action.position}'
    if source and obs.board and (source.card_type=='MINION' or action.sub_option_id is not None):
        from .magnetic import magnetic
        right=f'position:board:{len(obs.board)+1}'
        if magnetic(source):return right
        if right in mapped:return right
        for position in range(1,len(obs.board)+1):
            key=f'position:board:{position}'
            if key in mapped:return key
    return 'region:board'


def reorder_destination_key(action,obs):
    """Final position of an existing card is NOT an N+1 insertion index."""
    cards=sorted(obs.board,key=lambda c:c.position)
    ids=[c.entity_id for c in cards];position=action.position
    if action.scope!='board' or action.source not in ids or position is None:return None
    if [c.position for c in cards]!=list(range(1,len(cards)+1)):return None
    if len(ids)!=len(set(ids)) or not 1<=position<=len(ids):return None
    order=[eid for eid in ids if eid!=action.source];order.insert(position-1,action.source)
    if tuple(order)!=action.order or order==ids:return None
    return f'position:board:{position}'


def route(action,obs,mapped=()):
    """Resolve logical intent to named regions; no pixel or screen constants."""
    kind=action.kind;source=f'entity:{action.source}';destination=None;extra=[];zone=None
    if kind in CONTROL_KINDS:
        source=f'control:{kind}';gesture='click'
    elif kind in (Kind.CHOOSE,Kind.SELECT,Kind.ACTIVATE):gesture='click'
    elif kind==Kind.TARGET:source=f'entity:{action.target}';gesture='click_target'
    elif kind in (Kind.BUY,Kind.SELL,Kind.PLAY,Kind.REORDER):
        gesture='drag';zone={Kind.BUY:'shop',Kind.SELL:'board',Kind.PLAY:'hand',Kind.REORDER:'board'}[kind]
        if kind==Kind.BUY:destination='region:hand'
        elif kind==Kind.SELL:destination='region:bob'
        elif kind==Kind.REORDER:
            destination=reorder_destination_key(action,obs)
            if destination is None:return GeometryRoute(gesture,source,source_zone=zone,error='invalid_single_drag_reorder')
        else:
            card=next((c for c in obs.hand if c.entity_id==action.source),None)
            if action.interaction and action.interaction.placement=='magnetize':
                target=next((c for c in obs.board if c.entity_id==action.target),None)
                if target is None:return GeometryRoute(gesture,source,source_zone=zone,error='magnetic_target_unknown')
                destination=f'position:board:{target.position}'
            elif card and card.card_type in ('SPELL','BATTLEGROUND_SPELL') and action.target is not None and action.sub_option_id is None:
                if any(c.entity_id==action.target for c in obs.hand):gesture='click';destination=None
                else:destination=f'entity:{action.target}'
            else:destination=play_destination_key(action,obs,mapped)
    else:return GeometryRoute('none',error='unsupported_interaction:'+str(kind))
    # Deferred targets must already be locatable, but are clicked only after a
    # new target-stage observation. A route never flattens a multi-stage action.
    if action.target is not None:extra.append(f'entity:{action.target}')
    required=tuple(dict.fromkeys(k for k in (source,destination,*extra) if k))
    return GeometryRoute(gesture,source,destination,required,zone)
