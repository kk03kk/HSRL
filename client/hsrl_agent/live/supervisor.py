"""Deterministic incident classification, batch budget and heartbeat (no LLM, no input).

The live runner owns the input hot path. This module only *reads* what the runner
already wrote (``status.json`` / ``exit.json`` / ``result.json``) and decides
whether a batch may continue. It never sends input and never touches a running
game, so it is safe to use from the batch loop and from an out-of-band watcher.

Severity contract (see docs/LIVE_V1_1_AUDIT.md §6 and the D-phase plan):

* ``ok``  — the match finished and produced a result.
* ``S2``  — cosmetic/noise; record only.
* ``S1``  — degraded but safe (code fault, internal pause, deadline reached).
* ``S3``  — environment (lobby/matchmaking/client), recoverable by restart.
* ``S0``  — unsafe/unknown delivery; the batch must stop.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

SEVERITY_RANK = {'ok': 0, 'S2': 1, 'S3': 2, 'S1': 3, 'S0': 4}

#: S0 stops the batch immediately; every other class may continue while the
#: budget allows it. Repeating the same signature is what stops a bad batch:
#: one flaky match must not cost a whole night, and one systemic bug must not
#: burn matches forever.
STOP_ON = frozenset(('S0',))

_S0_PATTERNS = (
    'input outcome unknown',
    'observed action outcome differs',
    'context changed with unresolved input',
    'closed-game retirement lacks',
    'model decision is inconsistent',
)
_S3_PATTERNS = (
    'matchmaking', 'lobby', 'queue', 'autoplay stopped before queue',
    'unresolved prior input', 'reconnect', 'connection',
)
_S1_PATTERNS = (
    'internal ', 'time limit', 'halted', 'layout', 'uncalibrated',
    'no successful game result', 'receipt', 'blocked',
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:240]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def signature(reason: str, *, phase: str = "") -> str:
    """Stable identity for "the same problem", not the same sentence.

    Digits/ids are dropped so that ``turn=8`` and ``turn=11`` collapse; the
    first three words keep the module/verb that actually failed.
    """
    text = re.sub(r"[0-9a-f]{8,}", "#", _norm(reason).lower())
    text = re.sub(r"\d+", "#", text)
    words = [w for w in re.split(r"[^a-z\u4e00-\u9fff#]+", text) if w][:4]
    return (phase + ":" if phase else "") + " ".join(words)


def classify_run_end(out_dir: Path, exit_code: int, result=None, *, status=None, reason: str | None = None) -> dict:
    """Classify one finished match from the files the runner left behind.

    ``reason`` overrides the files for failures that happened before the run
    directory had a ``status.json`` (lobby/queue/preflight).
    """
    out_dir = Path(out_dir)
    status = status or _read_json(out_dir / "status.json") or {}
    exitj = _read_json(out_dir / "exit.json") or {}
    reason = _norm(reason or status.get("reason") or exitj.get("error") or "")
    phase = str((status.get("client") or {}).get("phase") or "")
    lowered = reason.lower()

    if result and not reason:
        return dict(severity="ok", reason="", signature="", exit_code=exit_code,
                    phase=phase, action="continue")

    if exit_code == 0 and result and not any(p in lowered for p in ('halted', 'unknown')):
        return dict(severity="ok", reason=reason, signature="", exit_code=exit_code,
                    phase=phase, action="continue")

    for pattern in _S0_PATTERNS:
        if pattern in lowered:
            return dict(severity="S0", reason=reason, signature=signature(reason, phase=phase),
                        exit_code=exit_code, phase=phase, action="stop_batch")
    for pattern in _S3_PATTERNS:
        if pattern in lowered:
            return dict(severity="S3", reason=reason, signature=signature(reason, phase=phase),
                        exit_code=exit_code, phase=phase, action="recover_then_continue")
    for pattern in _S1_PATTERNS:
        if pattern in lowered:
            return dict(severity="S1", reason=reason, signature=signature(reason, phase=phase),
                        exit_code=exit_code, phase=phase, action="continue")
    if reason:
        return dict(severity="S1", reason=reason, signature=signature(reason, phase=phase),
                    exit_code=exit_code, phase=phase, action="continue")
    return dict(severity="S2", reason="exit %s without a reason" % exit_code, signature="",
                exit_code=exit_code, phase=phase, action="continue")


class FailureBudget:
    """How many failures a batch may absorb before it stops.

    Defaults: one match may fail twice for *different* reasons and three times in
    total; the same signature twice in a row stops the batch (systemic bug, more
    matches would only resample it).
    """

    def __init__(self, *, max_total: int = 3, max_same: int = 2, max_consecutive_s0: int = 1):
        self.max_total = max(0, int(max_total))
        self.max_same = max(1, int(max_same))
        self.max_consecutive_s0 = max(1, int(max_consecutive_s0))
        self.total = 0
        self.by_signature: dict[str, int] = {}
        self.consecutive_s0 = 0

    def record(self, incident: dict) -> None:
        severity = incident.get("severity", "S2")
        if severity == "ok":
            self.consecutive_s0 = 0
            return
        self.total += 1
        sig = incident.get("signature") or severity
        self.by_signature[sig] = self.by_signature.get(sig, 0) + 1
        self.consecutive_s0 = self.consecutive_s0 + 1 if severity == "S0" else 0

    def allow(self, incident: dict) -> tuple[bool, str]:
        severity = incident.get("severity", "S2")
        if severity in STOP_ON:
            return False, "severity %s forbids another match" % severity
        if self.consecutive_s0 >= self.max_consecutive_s0:
            return False, "consecutive S0"
        if self.total > self.max_total:
            return False, "failure budget exhausted (%d)" % self.total
        sig = incident.get("signature") or severity
        if self.by_signature.get(sig, 0) > self.max_same:
            return False, "signature repeated: " + sig
        return True, ""

    def snapshot(self) -> dict:
        return dict(total=self.total, by_signature=dict(self.by_signature),
                    max_total=self.max_total, max_same=self.max_same)


def append_incident(batch_dir: Path, incident: dict, *, match: int) -> Path:
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    record = dict(at=time.time(), match=match, **incident)
    path = batch_dir / "incidents.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def health_payload(*, reason: str, data=None, controller=None, waits=None, extra=None) -> dict:
    """Cheap heartbeat: enough to see a stall without another screenshot."""
    data = data or {}
    client = data if isinstance(data, dict) else {}
    hero = (client.get("hero") or {}) if isinstance(client.get("hero"), dict) else {}
    payload = dict(at=time.time(), reason=_norm(reason), game_id=client.get("game_id"),
                   turn=client.get("turn"), phase=client.get("phase"),
                   gold=client.get("gold"), tier=client.get("tavern_tier"),
                   hero_health=hero.get("health"),
                   pending=bool(controller.pending) if controller is not None else None,
                   halted=getattr(controller, "halted", None) if controller is not None else None,
                   log_position=client.get("log_position"))
    if waits is not None:
        payload["elapsed_seconds_by_category"] = dict(getattr(waits, "totals", {}) or {})
    if extra:
        payload.update(extra)
    return payload
