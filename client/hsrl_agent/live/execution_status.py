"""Bounded monotonic wait accounting, diagnostic only; no recovery authorization."""
import time


def wait_code(reason):
    text=reason.lower()
    if 'foreground' in text or 'human mouse' in text:return 'focus_or_operator'
    if 'history' in text or 'source log' in text:return 'log_history'
    if 'playback' in text:return 'client_playback'
    if 'ocr' in text or 'overlay' in text or 'connection' in text:return 'interruption'
    if 'row' in text or 'layout' in text or 'stability' in text or 'visual' in text or 'covered' in text:return 'visual_readiness'
    if 'receipt' in text:return 'server_receipt'
    if 'next_turn' in text or 'decision phase' in text:return 'next_phase'
    if 'unknown' in text or 'halted' in text:return 'unresolved'
    if text in ('sent','applied','branch_sent','target_sent','confirm_sent'):return 'progress'
    return 'other'


class WaitTracker:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.last_at=None;self.code=None;self.key=None;self.since=None;self.totals={}
    def update(self,reason,data,pending):
        now=self.clock();code=wait_code(reason)
        if self.last_at is not None:self.totals[self.code]=self.totals.get(self.code,0)+max(0,now-self.last_at)
        self.last_at=now;b=pending[0].binding if pending else None
        identity=((data or {}).get('game_id'),(data or {}).get('turn'),b.source if b else None,
                  b.choice_id if b else ((data or {}).get('interaction') or {}).get('request_id'))
        changed=(identity,code)!=self.key
        if changed:self.key=(identity,code);self.since=now
        self.code=code
        return dict(wait_code=code,reason=reason,waiting_seconds=max(0,now-self.since),
            elapsed_seconds_by_category=dict(self.totals),game_id=identity[0],turn=identity[1],source=identity[2],
            request_id=identity[3],pending=bool(pending),diagnostic_only=True)
