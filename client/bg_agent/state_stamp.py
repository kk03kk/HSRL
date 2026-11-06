"""Cheap structural revisions without recursively copying receipt snapshots."""
from dataclasses import fields,is_dataclass
from functools import lru_cache

from .model import Observation,Resolution,FrozenTags


@lru_cache(maxsize=16)
def names(kind):
    omitted={'captured_at','source_updated_at','log_position','log_event_at'} if kind is Observation else {'post_state'} if kind is Resolution else set()
    return tuple(f.name for f in fields(kind) if f.name not in omitted)


def state_stamp(value):
    if isinstance(value,FrozenTags):return value.signature
    if is_dataclass(value):return tuple(state_stamp(getattr(value,name)) for name in names(type(value)))
    if isinstance(value,dict):return tuple(sorted((key,state_stamp(item)) for key,item in value.items()))
    if isinstance(value,(tuple,list)):return tuple(state_stamp(item) for item in value)
    return value
