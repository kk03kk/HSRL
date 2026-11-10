from dataclasses import dataclass,asdict
import hashlib
import json
import time


def replace_file(source,target):
    """Atomic publication, tolerating brief Windows reader/AV sharing locks."""
    for attempt in range(8):
        try:
            source.replace(target);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.02*2**min(attempt,4))


def remove_file(path):
    for attempt in range(8):
        try:path.unlink(missing_ok=True);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.02*2**min(attempt,4))

class Blocked(RuntimeError):
    """Unavailable evidence pauses a decision; it never selects an alternative policy."""


class InputNotSent(Blocked):
    """Execution stopped before the first button-down; pointer movement only."""


class FocusUnavailable(Blocked):
    """Temporary loss of the verified foreground, never permission to click."""
    def __init__(self,message,*,details=None):
        super().__init__(message);self.details=details or {}

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

@dataclass(frozen=True)
class Binding:
    action: int
    kind: str
    source: int | None = None
    target: int | None = None
    position: int | None = None
    option_id: int | None = None
    choice_id: int | str | None = None
    targets: tuple = ()
    scope: str = ''
    sub_option_id: int | None = None
    branches: tuple = ()  # (entity, subOption id, error, authorized targets)

@dataclass(frozen=True)
class Decision:
    game_id: str
    revision: int
    stamp: str
    action: int
    binding: Binding
    checkpoint_sha256: str
    logits_sha256: str
    observation_sha256: str = ''
    mask_sha256: str = ''

def state_stamp(snapshot):
    if snapshot.get('comparison')=='semantic-v1':
        return digest(decision_semantics(snapshot))
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()
            if k not in ('captured_at','source_time','source_updated_at','log_event_at','image','visual')}
        if isinstance(value,(tuple,list)): return [clean(v) for v in value]
        return value
    return digest(clean(snapshot))


def decision_semantics(snapshot):
    """Policy bytes plus exact executable identities/authorizations.

    Log cursors and old receipts remain on the original snapshot for verification;
    they cannot invalidate an otherwise identical decision. Legacy strict callers
    retain the original state_stamp behavior.
    """
    from .actions import bind_actions
    data=snapshot['client']
    def identity(card):
        return {k:card.get(k) for k in ('entity_id','card_id','card_type','controller','zone','position','is_local')}
    preview=snapshot.get('target_preview')
    turn=data.get('turn')
    if turn is None and data.get('choices') and all(c['card_type']=='HERO' for c in data['choices']):turn=0
    return dict(game_id=snapshot['game_id'],build=snapshot.get('build',data.get('build')),
        view=snapshot['view'],phase=data.get('phase'),turn=turn,
        gates={k:data.get(k) for k in ('mode','local_player_id','ready','warnings','options_pending','client_task_list')},
        interaction=data.get('interaction'),
        zones={z:[identity(c) for c in data.get(z,[])] for z in ('board','hand','shop','choices','controls')},
        hero=identity(data['hero']) if data.get('hero') else None,
        target_heroes=[identity(c) for c in snapshot.get('target_heroes',[])],
        preview=dict(source=identity(preview['source']),position=preview['position']) if preview else None,
        bindings=[asdict(b) for b in sorted(bind_actions(snapshot),key=lambda b:b.action)])


def execution_semantics(snapshot):
    """Exact UI/action identity, excluding unrelated model-only context.

    The press-time check used to hash the complete model input.  A public
    opponent tier/health update while a choice panel is open therefore
    cancelled an otherwise unchanged click, and the whole two-frame gate had
    to run again.  This stamp keeps everything that can make the physical
    gesture wrong (phase, turn, interaction, entity identities, authorization
    and executable bindings) while excluding opponent summaries and other
    model-only features that do not alter the pending gesture.
    """
    from .actions import bind_actions
    data=snapshot['client']
    def identity(card):
        return {k:card.get(k) for k in ('entity_id','card_id','card_type','controller','zone','position','is_local')}
    preview=snapshot.get('target_preview')
    turn=data.get('turn')
    if turn is None and data.get('choices') and all(c['card_type']=='HERO' for c in data['choices']):turn=0
    view=snapshot.get('view') or {}
    return dict(game_id=snapshot['game_id'],build=snapshot.get('build',data.get('build')),
        phase=data.get('phase'),turn=turn,pending=view.get('pending'),pending_source=view.get('pending_source'),
        gates={k:data.get(k) for k in ('mode','local_player_id','ready','warnings','options_pending','client_task_list')},
        interaction=data.get('interaction'),
        zones={z:[identity(c) for c in data.get(z,[])] for z in ('board','hand','shop','choices','controls','branches')},
        hero=identity(data['hero']) if data.get('hero') else None,
        target_heroes=[identity(c) for c in snapshot.get('target_heroes',[])],
        preview=dict(source=identity(preview['source']),position=preview['position']) if preview else None,
        bindings=[asdict(b) for b in sorted(bind_actions(snapshot),key=lambda b:b.action)])


def execution_stamp(snapshot):
    return digest(execution_semantics(snapshot))


def compact_snapshot(snapshot):
    result=dict(snapshot);result['client']={**snapshot['client'],'resolutions':[]}
    result['receipt_history_omitted']=True
    return result
