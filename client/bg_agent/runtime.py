from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
import uuid

from .model import Kind


class AuditLog:
    @staticmethod
    def segments(path):
        """Retained segments in chronological order, including the active file."""
        path=Path(path)
        backups=[p for p in path.parent.glob(path.name+'.*') if p.name[len(path.name)+1:].isdigit()]
        return sorted(backups,key=lambda p:int(p.name[len(path.name)+1:]),reverse=True)+([path] if path.exists() else [])

    def __init__(self, path: Path, *, max_bytes=None, backup_count=0):
        if backup_count < 0 or (backup_count and (max_bytes is None or max_bytes < 256)):
            raise ValueError('rotating audit requires a positive segment budget >= 256 bytes')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.max_bytes=max_bytes
        self.backup_count=backup_count
        self.rotations=0;self.omitted_records=0
        self.bytes_written=path.stat().st_size if path.exists() else 0
        self._context=threading.local()

    @contextmanager
    def context(self,**fields):
        """Bind synchronous stage records to their action, including children."""
        previous=getattr(self._context,'fields',{})
        self._context.fields={**previous,**fields}
        try:yield
        finally:self._context.fields=previous

    def write(self, event: str, **fields):
        row = {"time": time.time(), "event": event, **getattr(self._context,'fields',{}), **fields}
        payload=(json.dumps(row, ensure_ascii=False, default=str) + "\n").encode('utf-8')
        with self.lock:
            if self.max_bytes is not None and self.bytes_written+len(payload)>self.max_bytes:
                if not self.backup_count:raise OSError('audit_storage_limit:'+self.path.name)
                # Diagnostic history is a bounded ring, independent of the
                # durable action checkpoint. Preserve recent complete JSON rows.
                if len(payload)>self.max_bytes:
                    payload=(json.dumps(dict(time=row['time'],event='audit_record_omitted',
                        omitted_event=event,serialized_bytes=len(payload),reason='record_exceeds_segment_budget'))+'\n').encode('utf-8')
                    # An arbitrary caller-provided event name can itself be huge.
                    if len(payload)>self.max_bytes:
                        payload=b'{"event":"audit_record_omitted","reason":"record_exceeds_segment_budget"}\n'
                    self.omitted_records+=1
                if self.bytes_written+len(payload)>self.max_bytes:
                    for index in range(self.backup_count,0,-1):
                        source=self.path if index==1 else self.path.with_name(self.path.name+f'.{index-1}')
                        if source.exists():
                            from .recording import atomic_replace
                            atomic_replace(source,self.path.with_name(self.path.name+f'.{index}'))
                    self.bytes_written=0;self.rotations+=1
            with self.path.open('ab') as f:f.write(payload)
            self.bytes_written+=len(payload)


class Controller:
    """Single in-flight action, revision check, acknowledgement and stop token.

    An ambiguous timeout or transport exception latches the controller stopped.
    Create a fresh controller only after independently checking the outcome.
    """
    def __init__(self, backend, policy, audit: AuditLog, stop_file: Path | None = None):
        self.backend, self.policy, self.audit = backend, policy, audit
        self.stop_file = stop_file
        self.stopped = threading.Event()
        self.lock = threading.Lock()

    def stop(self):
        self.stopped.set()
        self.audit.write("stopped")

    def is_stopped(self):
        return self.stopped.is_set() or bool(self.stop_file and self.stop_file.exists())

    def step(self, timeout=2.0):
        if not self.lock.acquire(blocking=False):
            return {"status": "busy"}
        aid = uuid.uuid4().hex
        start = time.monotonic()
        try:
            if self.is_stopped():
                return {"status": "stopped"}
            before = self.backend.observe()
            decision = self.policy.decide(before)
            self.audit.write("decision", action_id=aid, revision=before.revision,
                             phase=before.phase, decision=asdict(decision))
            if decision.action.kind == Kind.WAIT:
                return {"status": "wait", "reason": decision.reason}
            if decision.revision != before.revision or decision.action not in before.actions:
                raise ValueError("decision is stale or not in advertised actions")
            if self.is_stopped():
                return {"status": "stopped"}
            self.backend.perform(decision.action, decision.revision)
            while time.monotonic() - start < timeout:
                if self.is_stopped():
                    return {"status": "stopped"}
                after = self.backend.observe()
                if self.backend.acknowledged(decision.action, before, after):
                    elapsed = (time.monotonic() - start) * 1000
                    self.audit.write("acknowledged", action_id=aid, elapsed_ms=elapsed, revision=after.revision)
                    return {"status": "acknowledged", "elapsed_ms": elapsed}
                self.stopped.wait(0.02)
            self.audit.write("timeout", action_id=aid, elapsed_ms=(time.monotonic() - start) * 1000,
                             recovery="latched_stop_manual_inspection_required")
            self.stopped.set()
            return {"status": "timeout"}
        except Exception as exc:
            self.stopped.set()
            self.audit.write("error", action_id=aid, error_type=type(exc).__name__, error=str(exc))
            return {"status": "error", "error": str(exc)}
        finally:
            self.lock.release()
