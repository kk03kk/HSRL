"""Versioned live action budgets, separate from the server legal mask."""
from dataclasses import dataclass

PEER_R1 = 'peer-r1'
PLAYER150 = 'player150'


@dataclass(frozen=True)
class ActionProfile:
    actions: int
    moves: int
    freezes: int | None
    repeat_move: int | None


PROFILES = {
    None: ActionProfile(80, 5, None, 2),
    PEER_R1: ActionProfile(300, 10, 3, None),
    PLAYER150: ActionProfile(150, 3, 2, None),
}


def peer_actions(snapshot):
    return snapshot.get('action_profile') in (PEER_R1, PLAYER150)
