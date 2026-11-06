"""Read-only GameNetLogger lifecycle. Silence is never a disconnect signal."""
import hashlib,re
from pathlib import Path
from .connection import IncrementalLog,latest_log_file


def blocks_game_input(connection,phase):
    return connection in ('disconnected','connecting','reconnecting','restoring') or (
        connection=='closed' and str(phase) in ('recruit','choice','combat'))


class NetworkState:
    def __init__(self):
        self.session=None;self.tail=None;self.connection='unknown';self.match=None;self.expected_end=False;self.epoch=0
        self.queue_cancelled=False;self.queue_failure_epoch=0;self.queue_attempt=0
        self.queue_failure_events=set()

    def feed(self,line):
        if 'GameMgr.ChangeFindGameState()' in line:
            state=re.search(r' - state: (\w+)',line)
            if state and state[1]=='CLIENT_STARTED':
                self.queue_attempt+=1;self.queue_cancelled=False
            elif state and state[1]=='SERVER_GAME_CANCELED':
                if not self.queue_cancelled and line not in self.queue_failure_events:self.queue_failure_epoch+=1
                self.queue_cancelled=True;self.queue_failure_events.add(line)
        if 'Network.GotoGameServe()' in line and 'game=' in line:
            identity=re.search(r'address=\s*([^,]+),\s*game=(\d+),\s*client=(\d+)',line)
            if identity:self.match=hashlib.sha256('|'.join(identity.groups()).encode()).hexdigest()[:24]
            self.connection='reconnecting' if 'reconnecting=True' in line else 'connecting';self.expected_end=False
        elif 'Network.DisconnectFromGameServer()' in line and 'Reason: EndGameScreen' in line:
            self.expected_end=True;self.connection='closed'
        elif 'Network.OnGameServerDisconnectEvent()' in line:
            self.connection='closed' if self.expected_end else 'disconnected';self.epoch+=1
        elif 'Network.OnGameServerConnectEvent()' in line and 'error code' in line:
            if 'ERROR_OK' in line:self.connection='restoring' if self.connection in ('reconnecting','disconnected') else 'connected'
            else:self.connection='disconnected'
        elif 'GameMgr.OnGameSetup()' in line:self.connection='connected';self.expected_end=False;self.queue_cancelled=False

    def inspect(self,session):
        if session!=self.session:
            self.session=session;self.tail=None;self.connection='unknown';self.expected_end=False;self.match=None
            self.queue_cancelled=False;self.queue_failure_epoch=0;self.queue_attempt=0
            self.queue_failure_events.clear()
        newest=latest_log_file(session,'GameNetLogger*.log')
        if newest and (self.tail is None or self.tail.path!=newest):
            # Follow a new log-family file without inventing a new match or
            # losing the identity of an in-flight input in the same session.
            self.tail=IncrementalLog(newest);self.epoch+=1
        if self.tail:
            reset,lines=self.tail.poll()
            if reset:
                self.connection='unknown';self.match=None;self.expected_end=False;self.epoch+=1;self.queue_cancelled=False
            for line in lines:self.feed(line)
        return dict(connection=self.connection,match_token=self.match,connection_epoch=self.epoch,
            queue_cancelled=self.queue_cancelled,queue_failure_epoch=self.queue_failure_epoch,queue_attempt=self.queue_attempt,
            network_provenance='GameNetLogger events' if self.tail else 'unavailable')
