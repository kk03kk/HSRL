"""Semantic execution contract. A policy never sees pixels or transport state."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .model import Action


class Delivery(StrEnum):
    ACCEPTED = "accepted"  # Input accepted, effect still requires verification.
    NOT_SENT = "not_sent"  # Transport proves no input was dispatched.
    REJECTED = "rejected"  # Receiver explicitly rejected without applying effects.
    AMBIGUOUS = "ambiguous"  # Never retry automatically.


@dataclass(frozen=True)
class Dispatch:
    delivery: Delivery
    reason: str


class Executor(Protocol):
    def execute(self, action: Action, revision: int) -> Dispatch: ...


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def point(self, u=.5, v=.5):
        if self.width <= 0 or self.height <= 0 or not 0 <= u <= 1 or not 0 <= v <= 1:
            raise ValueError("invalid window/relative coordinate")
        return self.x + self.width*u, self.y + self.height*v


@dataclass(frozen=True)
class InputPlan:
    revision: int
    action_key: str
    window: Rect
    source: tuple[float, float]
    destination: tuple[float, float] | None
    layout_token: str

    def validate(self):
        for point in (self.source, self.destination):
            if point is None:
                continue
            x, y = point
            if not (self.window.x <= x < self.window.x+self.window.width and
                    self.window.y <= y < self.window.y+self.window.height):
                raise ValueError("input outside the bound window")


class ReadOnlyExecutor:
    def execute(self, action, revision):
        return Dispatch(Delivery.NOT_SENT, "real Hearthstone inputs are disabled")
