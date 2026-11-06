from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import lru_cache
from enum import StrEnum
from typing import Protocol


@lru_cache(maxsize=32)
def _wire_fields(kind):
    return tuple(f.name for f in fields(kind))


def wire(value):
    """Detach mutable containers without deepcopy of immutable scalar leaves.

    Same tuple/list wire shape as asdict; receipt post-states remain detached.
    This is deliberately not an object cache: a new playback snapshot must
    never inherit old task/submission metadata under the same semantic revision.
    """
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {k:wire(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(wire(v) for v in value)
    if isinstance(value,list):return [wire(v) for v in value]
    if is_dataclass(value):return {k:wire(getattr(value,k)) for k in _wire_fields(type(value))}
    raise TypeError('unsupported_observation_wire_type:'+type(value).__name__)


class FrozenTags(dict):
    """Published log tags remain JSON/dict compatible but cannot be edited."""
    def __init__(self,*a,**kw):
        dict.__init__(self,*a,**kw);self.signature=tuple(sorted(self.items()))
    def _readonly(self,*a,**kw):raise TypeError('published_log_tags_are_read_only')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_readonly
    def __deepcopy__(self,memo):return self


class Phase(StrEnum):
    UNKNOWN = "unknown"
    MENU = "menu"
    CHOICE = "choice"
    RECRUIT = "recruit"
    COMBAT = "combat"
    GAME_OVER = "game_over"


class Kind(StrEnum):
    WAIT = "wait"
    ENTER = "enter"
    CHOOSE = "choose"
    BUY = "buy"
    SELL = "sell"
    PLAY = "play"
    UPGRADE = "upgrade"
    REFRESH = "refresh"
    FREEZE = "freeze"
    HERO_POWER = "hero_power"
    ACTIVATE = "activate"
    REORDER = "reorder"
    END = "end"
    DISMISS = "dismiss"
    SELECT = "select"
    TARGET = "target"
    CONFIRM = "confirm"
    CANCEL = "cancel"


@dataclass(frozen=True)
class InteractionStage:
    mode: str
    operation: str
    target: int | None = None
    timeout_ms: int = 1500
    requires_new_observation: bool = True


@dataclass(frozen=True)
class InteractionSpec:
    stages: tuple[InteractionStage, ...] = ()
    placement: str = "normal"  # normal / magnetize / insertion
    min_select: int = 1
    max_select: int = 1
    mandatory: bool = False


@dataclass(frozen=True)
class InteractionContext:
    mode: str
    request_id: int | None = None
    min_select: int = 1
    max_select: int = 1
    selected: tuple[int, ...] = ()
    protocol: str = "unknown"
    source_entity: int | None = None
    task_list: int | None = None
    owner: int | None = None
    choice_type: str | None = None
    display_state: str = 'unknown'  # Explicit ChoiceCardMgr waiting/shown/hidden only.


@dataclass(frozen=True)
class Card:
    entity_id: int
    card_id: str = ""
    name: str = ""
    controller: int | None = None
    zone: str = ""
    position: int = 0
    card_type: str = ""
    attack: int | None = None
    health: int | None = None
    tier: int | None = None
    cost: int | None = None  # Raw COST tag, not necessarily shop purchase price.
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    kind: Kind
    source: int | None = None
    target: int | None = None
    option_id: int | None = None
    choice_id: int | None = None
    order: tuple[int, ...] = ()
    evidence: str = "server_options"
    sub_option_id: int | None = None
    position: int | None = None
    scope: str | None = None
    selected: tuple[int, ...] = ()
    interaction: InteractionSpec | None = None
    resource_cost: int | None = None
    authorization_source: int | None = None

    @property
    def key(self) -> str:
        base=f"{self.kind}:{self.source}:{self.target}:{self.choice_id}:{self.order}:{self.sub_option_id}:{self.position}:{self.scope}"
        if self.selected or self.interaction:
            import json
            base+=":"+json.dumps(dict(selected=self.selected,interaction=asdict(self.interaction) if self.interaction else None),sort_keys=True)
        return base


@dataclass(frozen=True)
class Resolution:
    at_revision: int
    kind: Kind
    source: int | None = None
    target: int | None = None
    sub_option_id: int | None = None
    scope: str | None = None
    origin: str = "server_power"
    block_type: str = "PLAY"
    complete: bool = True
    started_at_revision: int | None = None
    log_position: int | None = None
    started_at_log_position: int | None = None
    zone_transitions: tuple[tuple[int,str,str], ...] = ()
    entity_changes: tuple[tuple[int,str,str,str], ...] = ()
    post_state: dict | None = None  # Root-bound snapshot, excludes receipts/actions.
    choice_owner: int | None = None


@dataclass(frozen=True)
class Observation:
    schema_version: int = 1
    revision: int = 0
    game_id: str = ""
    mode: str = "unknown"
    phase: Phase = Phase.UNKNOWN
    local_player_id: int | None = None
    turn: int | None = None
    gold: int | None = None
    tavern_tier: int | None = None
    upgrade_cost: int | None = None
    hero: Card | None = None
    hand: tuple[Card, ...] = ()
    shop: tuple[Card, ...] = ()
    board: tuple[Card, ...] = ()
    choices: tuple[Card, ...] = ()
    actions: tuple[Action, ...] = ()
    warnings: tuple[str, ...] = ()
    source: str = "power_log"
    captured_at: float = 0.0
    source_updated_at: float | None = None
    build: str | None = None
    scene: str | None = None
    resolutions: tuple[Resolution, ...] = ()
    ready: bool = True  # Adapter says the snapshot is actionable, not mid-animation.
    log_event_at: float | None = None
    interaction: InteractionContext | None = None
    controls: tuple[Card, ...] = ()
    ui_revision: int = 0  # Local UI selection changes, separate from Power revisions.
    log_position: int | None = None  # Stable Power-log line cursor for replay/recovery.
    branches: tuple[Card, ...] = ()  # Server sub-option entities, ordered in each action's InteractionSpec.
    choice_rewards: tuple[Card, ...] = ()  # Quest/reward links supplied by entity tags.
    client_task_list: int | None = None  # Active visual playback, separate from server revision.
    client_submission_position: int | None = None  # Last client SendOption/SendChoices, not a server ack.
    client_playback_position: int | None = None  # Matched EndCurrentTaskList cursor, not a semantic revision.
    options_pending: bool = False  # Local state invalidated Options; waiting for a new server snapshot.

    def to_dict(self) -> dict:
        return wire(self)

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        data["phase"]=Phase(data["phase"])
        for zone in ("hand","shop","board","choices","controls","branches","choice_rewards"):
            data[zone]=tuple(Card(**c) for c in data.get(zone,()))
        if data.get("hero"):data["hero"]=Card(**data["hero"])
        actions=[]
        for raw in data.get("actions",()):
            a={**raw,"kind":Kind(raw["kind"]),"order":tuple(raw.get("order",())),"selected":tuple(raw.get("selected",()))}
            if a.get("interaction"):
                spec=dict(a["interaction"]);spec["stages"]=tuple(InteractionStage(**s) for s in spec.get("stages",()))
                a["interaction"]=InteractionSpec(**spec)
            actions.append(Action(**a))
        data["actions"]=tuple(actions)
        if data.get("interaction"):
            data["interaction"]=InteractionContext(**{**data["interaction"],"selected":tuple(data["interaction"].get("selected",()))})
        data["resolutions"]=tuple(Resolution(**{**r,"kind":Kind(r["kind"]),"zone_transitions":tuple(tuple(x) for x in r.get('zone_transitions',())),"entity_changes":tuple(tuple(x) for x in r.get('entity_changes',()))}) for r in data.get("resolutions",()))
        data["warnings"]=tuple(data.get("warnings",()))
        return cls(**data)


@dataclass(frozen=True)
class Decision:
    action: Action
    revision: int
    reason: str


class Policy(Protocol):
    def decide(self, observation: Observation) -> Decision: ...


class ObservationProvider(Protocol):
    def observe(self) -> Observation: ...


class Backend(ObservationProvider, Protocol):
    def perform(self, action: Action, revision: int) -> None: ...

    def acknowledged(self, action: Action, before: Observation, after: Observation) -> bool: ...
