"""Projection, bounded visual evidence and semantic pre-dispatch validation."""
import time
from collections import defaultdict,deque
from contextlib import contextmanager
from .contracts import Blocked,state_stamp
from .frame_gate import FrameGate
from .interactions import branch_snapshot,target_snapshot


class Measurements:
    def __init__(self):
        self.values=defaultdict(lambda:deque(maxlen=256));self.counts=defaultdict(int)
        self.totals=defaultdict(float);self.maxima=defaultdict(float)
    @contextmanager
    def measure(self,name):
        started=time.perf_counter()
        try:yield
        finally:
            ms=(time.perf_counter()-started)*1000
            self.values[name].append(ms);self.counts[name]+=1;self.totals[name]+=ms;self.maxima[name]=max(self.maxima[name],ms)
    def report(self):
        import numpy as np
        return {k:dict(count=self.counts[k],retained=len(v),p50_ms=float(np.median(v)),p90_ms=float(np.quantile(v,.9)),
                       mean_ms=self.totals[k]/self.counts[k],total_ms=self.totals[k],max_ms=self.maxima[k])
                for k,v in self.values.items() if v}


def bind_unknown_cards(view,policy,approximation):
    if isinstance(view,dict):
        for key,value in view.items():
            if key in ('card_id','hero','power0','power1','deity','pending_source') and isinstance(value,str):
                if value not in policy.vocabulary:
                    policy.vocabulary[value]=0
                if policy.vocabulary[value]==0:approximation['fields']['oov.'+value]=dict(value=0,reason='unknown observation identity mapped to padding; actual entity retained')
            elif isinstance(value,(dict,list)):bind_unknown_cards(value,policy,approximation)
    elif isinstance(view,list):
        for value in view:
            if isinstance(value,str):bind_unknown_cards(dict(card_id=value),policy,approximation)
            else:bind_unknown_cards(value,policy,approximation)


class LivePipeline:
    def __init__(self,reader,projector,backend,profile,policy,pending,out,*,gate=None):
        self.reader=reader;self.projector=projector;self.backend=backend;self.profile=profile
        self.policy=policy;self.pending=pending;self.out=out;self.gate=gate or FrameGate(backend,profile,out=out/'visual-evidence')
        self.metrics=Measurements();self.saved_key=None
        self.admission=None
        self.overdue_evidence=deque(maxlen=2);self.overdue_saves=0
    def record_overdue(self,data):
        """Passive diagnostics once per uncertain stage, never input admission."""
        from dataclasses import asdict
        from .contracts import digest
        from .runtime import save
        pending=self.pending()
        if not pending:return
        key=digest(asdict(pending[0]))
        if key in self.overdue_evidence:return
        self.overdue_evidence.append(key);self.overdue_saves+=1
        folder=self.out/'overdue';folder.mkdir(parents=True,exist_ok=True);name=str(self.overdue_saves%2)
        evidence=dict(decision=asdict(pending[0]),client=data,diagnostic_only=True,automatic_resend=False,image=None)
        # A rotated slot must never pair a new blocked-capture JSON with an
        # older stage's photo. The files belong to this new run only.
        from .contracts import remove_file
        try:remove_file(folder/(name+'.jpg'))
        except OSError as exc:evidence['stale_image_cleanup_error']=str(exc)
        try:
            frame,window=self.backend.capture();frame.convert('RGB').save(folder/(name+'.jpg'),quality=90)
            evidence['window_token']=window.token()
            evidence['image']=str(folder/(name+'.jpg'))
        except (Blocked,OSError) as exc:evidence['capture_blocked']=str(exc)
        save(folder/(name+'.json'),evidence)
    def project(self,data):
        if self.reader.connection!='connected':raise Blocked('client connection is not confirmed connected')
        with self.metrics.measure('projection'):s=self.projector.build_public(self.reader,data)
        return s
    def finish(self,s,visual,*,pinned=None,use_current=True):
        pending=self.pending() if use_current else pinned
        s=branch_snapshot(s,pending,visual)
        s=target_snapshot(s,pending,visual,allow_public_hero=True)
        bind_unknown_cards(s['view'],self.policy,s['approximation'])
        s['visual']=visual
        return s
    def prepare(self,data):
        s=self.project(data)
        with self.metrics.measure('visual_evidence'):visual,image=self.gate.observe(s,self.pending())
        s=self.finish(s,visual);key=state_stamp(s)
        if key!=self.saved_key:
            from .runtime import save
            save(self.out/'approximation.json',s['approximation'])
            save(self.out/'coverage.json',dict(game_id=s['game_id'],revision=s['revision'],unknown=s['view'].get('unknown',[])))
            self.saved_key=key
        return s,image
    def check_action(self,old,binding=None):
        with self.metrics.measure('pre_dispatch'):
            data=self.reader.poll();new=self.project(data)
            # Synthetic branch/target stages come from the same pinned protocol
            # context. Compare before expensive capture and again afterwards.
            new=self.finish(new,old['visual'])
            if state_stamp(new)!=state_stamp(old):raise Blocked('decision input or executable authorization changed')
            with self.metrics.measure('pre_dispatch_capture'):
                visual,_=self.gate.observe(new,self.pending(),binding,revalidate=True)
            # Logs can advance while pixels are captured; verify once more.
            new=self.finish(self.project(self.reader.poll()),visual)
            if state_stamp(new)!=state_stamp(old):raise Blocked('decision input changed during visual admission')
            if visual['window_token']!=old['visual']['window_token']:raise Blocked('window changed before dispatch')
            self.admission=(state_stamp(new),self.pending(),visual['observed_at'])
            self.gate.admitted_binding=binding
            if hasattr(self.gate,'save_dispatch'):
                new['visual']['dispatch_evidence']=self.gate.save_dispatch(new,binding)
            # All semantic and visual evidence was just re-confirmed. The press
            # deadline starts here; earlier frame/check time is not charged twice.
            self.gate.evidence_at=self.gate.clock()
            return new
    def __call__(self,old):return self.check_action(old)
    def pre_press(self,old):
        if not self.admission or self.admission[0]!=state_stamp(old):raise Blocked('no matching visual admission for mouse down')
        with self.metrics.measure('pre_press_semantic_check'):
            new=self.finish(self.project(self.reader.poll()),old['visual'],pinned=self.admission[1],use_current=False)
            if state_stamp(new)!=state_stamp(old):raise Blocked('state changed during mouse hover; no button down sent')
        # Each recheck captures a fresh frame and validates the same admitted
        # decision, so the final budget is measured from that fresh evidence.
        if old['visual'].get('phase')=='hero':self.gate.recheck_hero(old,self.admission[1])
        self.gate.recheck_choice(old)
        self.gate.recheck_interaction(old,self.admission[1])
        if self.gate.evidence_at is None or self.gate.clock()-self.gate.evidence_at>self.gate.max_age:
            raise Blocked('visual admission expired during pre-press checks')
