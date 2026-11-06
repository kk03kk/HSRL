"""Incremental byte logs, complete lines only, explicit reset/session boundaries."""
import datetime as dt
from functools import lru_cache
import os
from pathlib import Path
import re
import time
from .contracts import Blocked
from .legacy_protocol.power import PowerState

class RawState(PowerState):
    def _actions(self,*args): return []  # Own action mapper replaces legacy policy-facing actions.
    def feed(self,line,*,source=None):
        from .tags import symbolic_line
        super().feed(symbolic_line(line,self.build),source=source)
    def _options(self,body):
        super()._options(body)
        # Preserve denied target identity for observation/price reconstruction;
        # denied targets never enter the legal `targets` list.
        match=re.match(r'target \d+ entity=(.*?) error=(\w+)',body)
        recipient=self.current_sub_option if self.current_sub_option is not None else self.current_option
        if match and recipient is not None:
            eid=self.resolve(match[1])
            if eid is not None:recipient.setdefault('target_rows',[]).append(dict(entity=eid,error=match[2]))

class Tail:
    def __init__(self,path):
        self.path=Path(path);self.offset=0;self.pending=b'';self.identity=None;self.first_line=True
    def poll(self):
        with self.path.open('rb') as stream:
            stat=os.fstat(stream.fileno());identity=(stat.st_dev,stat.st_ino)
            reset=self.identity is not None and (self.identity!=identity or stat.st_size<self.offset)
            if reset:self.offset=0;self.pending=b'';self.first_line=True
            self.identity=identity
            stream.seek(self.offset);chunk=stream.read(4*1024*1024);self.offset=stream.tell()
            size=os.fstat(stream.fileno()).st_size
        rows=(self.pending+chunk).split(b'\n');self.pending=rows.pop()
        if self.first_line and rows:
            rows[0]=rows[0].removeprefix(b'\xef\xbb\xbf');self.first_line=False
        current=self.path.stat()
        same=(current.st_dev,current.st_ino)==identity and current.st_size==size
        return reset,[r.rstrip(b'\r').decode('utf-8',errors='strict') for r in rows],same and self.offset==size and not self.pending

@lru_cache(maxsize=256)
def _second_candidates(anchor,h,m,s):
    base=dt.datetime.fromtimestamp(anchor).replace(hour=h,minute=m,second=s,microsecond=0)
    return tuple((base+dt.timedelta(days=d)).timestamp() for d in (-1,0,1))


def line_time(line,anchor):
    match=re.match(r'^[DWE] (\d{2}):(\d{2}):(\d{2})\.(\d+)',line)
    if not match: return None
    h,m,s=map(int,match.group(1,2,3))
    fraction=int(match[4][:6].ljust(6,'0'))/1_000_000
    return min((v+fraction for v in _second_candidates(anchor,h,m,s)),key=lambda v:abs(v-anchor))

class LogReader:
    def __init__(self,game_dir,player_id=None):
        self.game_dir=Path(game_dir);self.player_id=player_id;self.session=None
        self.state=RawState(player_id);self.tails={};self.epoch=0;self.latest_time=None
        self.history_complete=False;self.caught_up=False
        self.power_uses={};self.seen_boards={};self.last_opponent=None;self._generation=0;self._receipt=None;self.trinket_slots={}
        self.combat_seen=False
        self.connection='unknown'
        self.spectating=False
        self.start_latest=False;self.game_offset=0
        self.last_complete_poll_at=None
    def feed(self,line,source='power',anchor=None):
        if 'Spectating' in line or 'Spectator Game' in line:self.spectating=True
        if source=='network':
            if 'Network.OnGameServerDisconnectEvent()' in line:self.connection='disconnected'
            elif 'Network.GotoGameServe()' in line:self.connection='connecting'
            elif 'Network.OnGameServerConnectEvent()' in line:
                self.connection='connected' if 'ERROR_OK' in line else 'disconnected'
            elif 'GameMgr.OnGameSetup()' in line:self.connection='connected'
            return
        if 'CREATE_GAME' in line and ('GameState.' in line or source=='power'):
            self.history_complete=True
        self.state.feed(line,source=source)
        if self._generation!=self.state.generation:
            self._generation=self.state.generation;self.power_uses={};self.seen_boards={};self.last_opponent=None;self._receipt=None;self.combat_seen=False;self.trinket_slots={}
        if source=='power' and 'GameState.DebugPrintPower()' in line:
            slot=re.search(r'FULL_ENTITY - Creating ID=(\d+) CardID=BG30_Trinket_(1st|2nd)(?:\s|$)',line)
            if slot:self.trinket_slots[int(slot[1])]=0 if slot[2]=='1st' else 1
        game=self.state.entities.get(self.state.game_entity)
        if game and game.tags.get('BACON_IN_COMBAT_PHASE')=='1': self.combat_seen=True
        if self.state.resolutions:
            r=self.state.resolutions[-1]
            key=(self.state.generation,r.log_position)
            if key!=self._receipt:
                self._receipt=key
                if r.complete and r.kind in ('hero_power','activate'):
                    self.power_uses[r.source]=self.power_uses.get(r.source,0)+1
        when=line_time(line,anchor if anchor is not None else time.time())
        if when is not None and source=='power': self.latest_time=when if self.latest_time is None else max(self.latest_time,when)
    def poll(self):
        # A failed/partial read must revoke the previous poll's completeness.
        # Retry only temporary sharing/access failures; never skip bytes or
        # classify malformed protocol data as a successful acquisition.
        self.caught_up=False;self.last_complete_poll_at=None
        try:return self._poll()
        except PermissionError as exc:
            raise Blocked('temporary log access unavailable; input paused') from exc
        except OSError as exc:
            if getattr(exc,'winerror',None) in (32,33):
                raise Blocked('temporary log sharing lock; input paused') from exc
            raise
    def _poll(self):
        self.last_complete_poll_at=None
        sessions=list((self.game_dir/'Logs').glob('Hearthstone_*'))
        if not sessions: raise Blocked('no Hearthstone log session; enable client logs before testing')
        session=max(sessions,key=lambda p:p.name)
        if session!=self.session:
            self.session=session;self.state=RawState(self.player_id);self.epoch+=1
            self.tails={};self.latest_time=None;self.history_complete=False
            self.power_uses={};self.seen_boards={};self.last_opponent=None;self._generation=0;self._receipt=None;self.trinket_slots={}
            self.combat_seen=False;self.connection='unknown';self.spectating=False
        all_done=True
        # Power.log already carries Options, choices and client playback. Do not
        # interleave duplicate GameState/PowerTaskList semantic projections.
        networks=sorted(session.glob('GameNetLogger*.log'),key=lambda p:p.stat().st_mtime)
        files=[('LoadingScreen.log','scene'),('Power.log','power')]
        if networks:files.append((networks[-1].name,'network'))
        for name,source in files:
            path=session/name
            if not path.exists():
                if source=='power': raise Blocked('Power.log missing')
                continue
            if name not in self.tails:
                tail=Tail(path)
                if source=='power' and self.start_latest:
                    self.game_offset=latest_game_info(path)['offset'];tail.offset=self.game_offset
                self.tails[name]=tail
            tail=self.tails[name]
            reset,rows,done=tail.poll()
            if reset:
                self.state=RawState(self.player_id);self.epoch+=1
                self.history_complete=False;self.latest_time=None
                self._generation=0;self.power_uses={};self.seen_boards={};self.last_opponent=None
                self._receipt=None;self.combat_seen=False;self.connection='unknown';self.caught_up=False;self.trinket_slots={}
                # Reset every cursor, not just the replaced file.
                self.tails={};raise Blocked('log reset: full session reconstruction required')
            all_done &= done
            anchor=path.stat().st_mtime
            # GameState resumes asynchronous discoveries with indentation reset
            # to zero, while BLOCK_START/END remain balanced. Practical mode
            # follows that semantic stack; legacy strict replay stays unchanged.
            self.state.balanced_async_blocks=self.start_latest
            for row in rows: self.feed(row,source,anchor)
        self.caught_up=all_done
        if all_done:self.last_complete_poll_at=time.monotonic()
        obs=self.state.observe(source_updated_at=self.latest_time)
        data=obs.to_dict()
        prefix=f'{session.name}:game@{self.game_offset}' if self.start_latest else session.name
        data['game_id']=f'{prefix}:{self.epoch}:{data["game_id"]}'
        return data
    def live_tail_current(self):
        """Acquisition freshness, distinct from the time of last game activity."""
        if (not self.start_latest or not self.caught_up or not self.history_complete or self.connection!='connected'
                or self.last_complete_poll_at is None or time.monotonic()-self.last_complete_poll_at>.25):return False
        tail=self.tails.get('Power.log')
        if not tail or tail.pending:return False
        try:stat=tail.path.stat()
        except OSError:return False
        return (stat.st_dev,stat.st_ino)==tail.identity and stat.st_size==tail.offset


_LATEST_CACHE={}


def latest_game_info(path):
    """Find a complete current-game history without parsing older matches.

    The lobby loop asks this on every iteration; an mmap scan of a 10-70 MB
    Power.log is cheap per call but not free, so the result is cached on the
    exact file identity/size/mtime that produced it.
    """
    path=Path(path)
    try:stat=path.stat()
    except OSError:return dict(offset=0,complete=False)
    key=(str(path.resolve()),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    cached=_LATEST_CACHE.get(key)
    if cached is not None:return dict(cached)
    import mmap
    with path.open('rb') as f:
        if not f.seek(0,2):return dict(offset=0,complete=False)
        f.seek(0)
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
            marker=raw.rfind(b'GameState.DebugPrintPower() - CREATE_GAME')
            offset=raw.rfind(b'\n',0,marker)+1 if marker>=0 else 0
            terminal=raw.rfind(b'tag=STATE value=COMPLETE')
            result=dict(offset=offset,complete=marker>=0 and terminal>marker)
    if len(_LATEST_CACHE)>8:_LATEST_CACHE.clear()
    _LATEST_CACHE[key]=result
    return dict(result)
