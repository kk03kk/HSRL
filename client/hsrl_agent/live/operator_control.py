"""Local desktop operator mailbox and transaction boundaries."""
import json
from pathlib import Path
import time

from .contracts import Blocked
from .runtime import save


class OperatorStop(Blocked):
    """A stop reached a safe boundary before gameplay began."""


def read_state(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


class OperatorControl:
    def __init__(self, folder, *, clock=time.monotonic, wall_clock=time.time):
        self.folder = Path(folder)
        self.clock = clock
        self.wall_clock = wall_clock
        self.command = 'stop'
        self.sequence = -1
        self.deadline = 0.
        self.next_poll = 0.
        self.match = 0
        self.completed = 0
        self.last_status = None
        self.has_command = False

    def poll(self, *, force=False):
        now = self.clock()
        if force or now >= self.next_poll:
            self.next_poll = now + .05
            data = read_state(self.folder / 'command.json')
            if isinstance(data, dict):
                sequence = data.get('sequence')
                command = data.get('command')
                timestamp = data.get('at')
                if (type(sequence) is int and sequence >= self.sequence
                        and (sequence > self.sequence or command == self.command)
                        and command in ('run', 'pause', 'stop')
                        and type(timestamp) in (int, float)
                        and -2 <= self.wall_clock() - timestamp < 12
                        and (not self.has_command or self.command != 'stop')):
                    self.sequence = sequence
                    self.command = command
                    self.deadline = now + 12 - max(0., self.wall_clock() - timestamp)
                    self.has_command = True
        if self.has_command and now >= self.deadline:
            self.command = 'stop'
        return self.command

    def publish(self, state, detail='', *, pending=False, turn=None):
        payload = dict(state=state, detail=detail, pending=bool(pending), turn=turn,
                       sequence=self.sequence, match=self.match, completed=self.completed)
        if payload != self.last_status:
            save(self.folder / 'control-status.json', dict(payload, at=self.wall_clock()))
            self.last_status = payload

    def require_new_input(self):
        if self.poll(force=True) != 'run':
            raise Blocked('operator request arrived before a new transaction; no input sent')


def reset_admission(controller, pipeline):
    controller.candidate = None
    if pipeline:
        pipeline.admission = None
        pipeline.gate.previous = None
        pipeline.gate.admitted = None
        pipeline.gate.admitted_binding = None
