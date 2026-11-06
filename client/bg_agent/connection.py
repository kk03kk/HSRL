from __future__ import annotations

from pathlib import Path
import time
import threading
import re
from datetime import datetime, timedelta

from .model import Action
from .power import PowerState
from .chain_metrics import measured_stage


def observation_version(obs):
    """Input-relevant publication version; render-only traffic is not a change."""
    return (obs.game_id,obs.revision,obs.ready,obs.warnings,obs.options_pending,
            obs.client_task_list,obs.client_submission_position,obs.client_playback_position)


def latest_log_file(directory,pattern):
    candidates=[]
    for path in Path(directory).glob(pattern):
        try:stamp=path.stat().st_mtime_ns
        except FileNotFoundError:continue  # Rotation between directory/stat.
        candidates.append((stamp,path.name,path))
    return max(candidates)[2] if candidates else None


class IncrementalLog:
    """Byte cursor; incomplete UTF-8 lines are retained until newline arrives."""
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.pending = b""
        self.signature = None
        self.last_event_at = None

    def poll(self):
        stat = self.path.stat()
        sig = (stat.st_dev, stat.st_ino)
        reset = self.signature is not None and (sig != self.signature or stat.st_size < self.offset)
        if reset:
            self.offset = 0
            self.pending = b""
            self.last_event_at = None
        self.signature = sig
        if not reset and stat.st_size==self.offset:return False,[]
        with self.path.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        rows = (self.pending + chunk).split(b"\n")
        self.pending = rows.pop()
        decoded=[r.rstrip(b"\r").decode("utf-8", errors="replace") for r in rows]
        # Source logs have local time-of-day, not a UTC date. Anchor to file mtime
        # and select the nearest day (including midnight rollover).
        for row in reversed(decoded):
            match=re.match(r"^[DWE] (\d{2}):(\d{2}):(\d{2})\.(\d+)",row)
            if match:
                h,m,s=map(int,match.group(1,2,3))
                us=int(match[4][:6].ljust(6,"0"))
                base=datetime.fromtimestamp(stat.st_mtime).replace(hour=h,minute=m,second=s,microsecond=us)
                self.last_event_at=min((base+timedelta(days=d) for d in (-1,0,1)),key=lambda t:abs(t.timestamp()-stat.st_mtime)).timestamp()
                break
        return reset, decoded


def find_power_logs(game_dir: Path) -> list[Path]:
    root = game_dir / "Logs"
    paths = list(root.glob("Hearthstone_*/*Power*.log")) if root.exists() else []
    return sorted(paths, key=lambda p: (p.parent.name, p.stat().st_mtime_ns, p.name == "Power.log"))


def latest_session(game_dir: Path):
    sessions = sorted((game_dir / "Logs").glob("Hearthstone_*"))
    return sessions[-1] if sessions else None


class HearthstoneLogs:
    """Read-only real-client adapter. No input, process injection or networking."""
    def __init__(self, game_dir: Path, player_id=None,*,keep_options_during_silence=False):
        self.game_dir = game_dir
        self.player_id = player_id
        self.keep_options_during_silence=keep_options_during_silence
        self.state = PowerState(player_id)
        self.tail: IncrementalLog | None = None
        self.scene_tail: IncrementalLog | None = None
        self.updated_at = None
        self.error = None
        self.session = None
        self.lock = threading.RLock()
        self._cached_state = None
        self._cached_observation = None
        self._receipt_view_key=None;self._receipt_view=()
        self._projected_state=None;self.revision_offset=0;self.last_revision=-1
        self._condition=threading.Condition();self._reader=None;self._reader_stop=threading.Event()
        self._published=None;self._request=0;self._completed=0;self._sequence=0

    def start_reader(self,interval=.02):
        """One I/O owner; consumers share immutable versioned observations."""
        if self._reader is not None:return
        if not .005<=interval<=1:raise ValueError('invalid_log_reader_interval')
        self._published=self._observe_sync()
        def run():
            while not self._reader_stop.is_set():
                with self._condition:requested=self._request
                with self.lock:
                    observation=self._observe_sync()
                    with self._condition:
                        old=self._published
                        if old is None or (old.game_id,old.revision,old.log_position,old.client_task_list,old.client_submission_position,old.client_playback_position,old.warnings)!=(observation.game_id,observation.revision,observation.log_position,observation.client_task_list,observation.client_submission_position,observation.client_playback_position,observation.warnings):self._sequence+=1
                        self._published=observation;self._completed=requested;self._condition.notify_all()
                with self._condition:
                    if self._request==requested:self._condition.wait(interval)
        self._reader=threading.Thread(target=run,name='hearthstone-log-publisher',daemon=True)
        self._reader.start()

    def close(self):
        self._reader_stop.set()
        with self._condition:self._condition.notify_all()
        if self._reader:self._reader.join(timeout=3)

    def synchronize(self):
        """A capture/input boundary requires an actual poll after this call."""
        if self._reader is None:return self._observe_sync()
        with self._condition:
            self._request+=1;requested=self._request;self._condition.notify_all()
            ok=self._condition.wait_for(lambda:self._completed>=requested or not self._reader.is_alive(),timeout=3)
            if not ok or self._completed<requested:raise TimeoutError('log_publisher_unavailable')
            self.last_observed_sequence=self._sequence
            return self._published

    def wait_for_update(self,sequence,timeout=.02):
        with self._condition:
            self._condition.wait_for(lambda:self._sequence!=sequence or self._reader_stop.is_set(),timeout=timeout)
            return self._sequence

    def observe(self):
        if self._reader is None:return self._observe_sync()
        from dataclasses import replace
        with self._condition:
            if not self._reader.is_alive():raise RuntimeError('log_publisher_stopped')
            self.last_observed_sequence=self._sequence
            return replace(self._published,captured_at=time.time())

    def publication_version(self):
        """Peek without consuming a wakeup or authorizing input from old pixels."""
        if self._reader is None:return observation_version(self._observe_sync())
        with self._condition:
            if not self._reader.is_alive():raise RuntimeError('log_publisher_stopped')
            return observation_version(self._published)

    def poll(self):
        session = latest_session(self.game_dir)
        if session != self.session:
            self.session = session
            self.state = PowerState(self.player_id)
            self.tail = None
            scene=latest_log_file(session,'LoadingScreen*.log') if session else None
            self.scene_tail = IncrementalLog(scene) if scene else None
            self.updated_at = None
        scene=latest_log_file(session,'LoadingScreen*.log') if session else None
        if scene and (self.scene_tail is None or self.scene_tail.path!=scene):self.scene_tail=IncrementalLog(scene)
        paths = [p for p in find_power_logs(self.game_dir) if p.parent == session]
        if not paths:
            if self.scene_tail:
                _, lines = self.scene_tail.poll()
                for line in lines: self.state.feed(line,source='scene')
            self.error = "当前客户端会话尚无 Power 日志；菜单阶段可能正常，等待进入对局"
            return
        path = paths[-1]
        if self.tail is None or self.tail.path != path:
            self.state = PowerState(self.player_id)
            self.tail = IncrementalLog(path)
            scene=latest_log_file(path.parent,'LoadingScreen*.log')
            self.scene_tail = IncrementalLog(scene) if scene else None
        try:
            reset, lines = self.tail.poll()
            if reset:
                self.state = PowerState(self.player_id)
            for line in lines:
                self.state.feed(line)
            scene=latest_log_file(session,'LoadingScreen*.log')
            if scene and (self.scene_tail is None or self.scene_tail.path!=scene):self.scene_tail=IncrementalLog(scene)
            if self.scene_tail:
                _, scenes = self.scene_tail.poll()
                for line in scenes:
                    self.state.feed(line,source='scene')
            self.updated_at = path.stat().st_mtime
            self.error = None
        except (OSError, ValueError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @measured_stage('log_read_and_projection')
    def _observe_sync(self):
        with self.lock:
            try:
                self.poll()
            except OSError as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            from dataclasses import replace
            if self._projected_state is not self.state:
                if self._projected_state is not None:self.revision_offset=self.last_revision+1
                self._projected_state=self.state
            if self._cached_state is not self.state or self._cached_observation is None or self._cached_observation.revision!=self.state.revision:
                if getattr(self,'metrics',None):self.metrics.count('state_rebuild')
                self._cached_state=self.state
                self._cached_observation=self.state.observe()
            receipt_key=(id(self.state),self._cached_observation.revision,self.revision_offset)
            if receipt_key!=self._receipt_view_key:
                self._receipt_view_key=receipt_key
                self._receipt_view=self._cached_observation.resolutions if not self.revision_offset else tuple(
                    replace(r,at_revision=r.at_revision+self.revision_offset,
                        started_at_revision=r.started_at_revision+self.revision_offset if r.started_at_revision is not None else None)
                    for r in self._cached_observation.resolutions)
            obs = replace(self._cached_observation,captured_at=time.time(),source_updated_at=self.updated_at,
                          # Render-only lines do not change semantic revision,
                          # but their cursor/playback/submission data are fresh.
                          log_position=self.state.power_lines,client_task_list=self.state.client_task_list,
                          client_submission_position=self.state.client_submission_position,
                          client_playback_position=self.state.client_playback_position,
                          log_event_at=self.tail.last_event_at if self.tail else None,
                          revision=self._cached_observation.revision+self.revision_offset,
                          resolutions=self._receipt_view)
            self.last_revision=obs.revision
            if self.error:
                obs = replace(obs, actions=(), warnings=obs.warnings+("connection_not_ready: advice_paused",))
            if self.updated_at and time.time()-self.updated_at > 30 and not self.keep_options_during_silence:
                obs = replace(obs, actions=(), warnings=obs.warnings+("power_log_older_than_30s: advice_paused",))
            return obs

    def perform(self, action: Action, revision: int):
        raise PermissionError("真实客户端适配器为只读；自动输入尚未验收，未启用")

    def acknowledged(self, action, before, after):
        return False  # Read-only transport cannot claim an executed game action.


def replay(path: Path, player_id=None, limit=None):
    state = PowerState(player_id)
    import gzip
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if limit is not None and i > limit:
                break
            state.feed(line)
    return state
