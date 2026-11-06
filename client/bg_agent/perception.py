"""Timestamped, read-only log + desktop acquisition, stability and diagnostics."""
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageChops

from .runtime import AuditLog


class Stability:
    def __init__(self,quiet_seconds=.3,threshold=.025):
        self.quiet_seconds,self.threshold=quiet_seconds,threshold
        self.previous=None;self.since=None;self.token=None
        self.was_stable=False;self.waiting_since=None

    def feed(self,image,window_token,timestamp):
        # Downsample the whole client; stability is only necessary, never proof
        # that a target-selection modal or clickability is known.
        small=np.asarray(image.convert("L").resize((256,160)),dtype=np.float32)
        delta=None
        if self.previous is not None and self.token==window_token:
            delta=float(np.mean(np.abs(small-self.previous))/255)
        if delta is None or delta>self.threshold:self.since=timestamp
        self.previous=small;self.token=window_token
        wait=timestamp-self.since
        stable=delta is not None and wait>=self.quiet_seconds
        if self.waiting_since is None or (self.was_stable and not stable):self.waiting_since=timestamp
        waited=(timestamp-self.waiting_since)*1000 if stable and not self.was_stable else None
        self.was_stable=stable
        return dict(stable=stable,delta=delta,stabilization_wait_ms=waited,
                    stable_for_seconds=wait,threshold=self.threshold,method="downsampled full-client mean luma change; necessary not sufficient")


class Metrics:
    def __init__(self):self.values={};self.counts={}
    def sample(self,name,value):
        if value is not None and value>=0:self.values.setdefault(name,[]).append(value)
    def count(self,name,n=1):self.counts[name]=self.counts.get(name,0)+n
    def report(self):
        return dict(counts=self.counts,latency_ms={name:dict(n=len(xs),p50=float(np.percentile(xs,50)),p95=float(np.percentile(xs,95)))
                                                  for name,xs in self.values.items() if xs})


class SynchronizedSampler:
    def __init__(self,logs,windows,directory: Path,*,bound_hwnd=None,local_readiness=False,
                 audit_max_bytes=None,audit_backup_count=0,compact_audit=False,share_snapshot=False):
        self.logs,self.windows,self.directory=logs,windows,directory
        directory.mkdir(parents=True,exist_ok=True)
        self.audit=AuditLog(directory/"samples.jsonl",max_bytes=audit_max_bytes,backup_count=audit_backup_count)
        self.compact_audit=compact_audit;self.share_snapshot=share_snapshot;self.latest_snapshot=None
        self.stability=Stability();self.metrics=Metrics();self.index=0
        self.lock=threading.RLock();self.latest=None
        self.bound_hwnd=bound_hwnd;self.local_readiness=local_readiness
        from .capture_broker import CaptureBroker
        self.broker=CaptureBroker(windows)
        self.region_selector=None

    def sample(self,*,persist=True):
        with self.lock:
            read=getattr(self.logs,'synchronize',self.logs.observe)
            start=time.perf_counter();obs=read();session=str(self.logs.session)
            def client_proofs():
                state=getattr(self.logs,'state',None)
                if not getattr(state,'client_objects',None):return None
                with self.logs.lock:return self.logs.state.client_objects.proofs(obs)
            objects_before=client_proofs()
            self.metrics.sample("log_read_to_observation",(time.perf_counter()-start)*1000)
            now=time.time()
            if obs.log_event_at is not None:self.metrics.sample("latest_log_event_age_at_observation",(now-obs.log_event_at)*1000)
            self.index+=1
            stem=f"frame-{self.index:06d}"
            window=self.windows.snapshot(self.bound_hwnd) if self.bound_hwnd is not None else self.windows.unique()
            region=self.region_selector(obs,window) if not persist and self.region_selector else None
            stamp=(session,obs.game_id,obs.revision,obs.client_task_list,obs.client_submission_position,obs.client_playback_position)
            image,capture=self.broker.capture(window,self.directory/(stem+".png") if persist else None,stamp=stamp,region=region)
            after=read()
            from .client_objects import bind_capture
            capture={**capture,'client_objects':bind_capture(objects_before,client_proofs())}
            sync=(obs.revision==after.revision and obs.game_id==after.game_id and session==str(self.logs.session)
                  and obs.client_task_list==after.client_task_list
                  and obs.client_playback_position==after.client_playback_position
                  and obs.client_submission_position==after.client_submission_position)
            if getattr(self,'green_signal',False):
                from .readiness_evidence import layout_context,readiness_binding
                def context(o):return (o.mode,layout_context(o,()),readiness_binding(o),o.options_pending,
                    o.ready,o.warnings,o.actions,o.client_submission_position)
                sync=session==str(self.logs.session) and context(obs)==context(after)
            if getattr(self,'green_signal',False):
                stability=dict(method='current_frame_action_signal')
            elif region is not None:
                stability=dict(stable=False,delta=None,stabilization_wait_ms=None,stable_for_seconds=0,
                    method='local_capture_uses_independent_layout_readiness')
            else:stability=self.stability.feed(image,window.token,capture["end"])
            reasons=list(window.blockers)
            if not capture["geometry_unchanged"]:reasons.append("window_changed_during_capture")
            if not sync:reasons.append("log_changed_during_capture")
            # Live recruit/choice gestures have a stricter action-region gate.
            # A second full-client stability wait only measures unrelated art.
            local=getattr(self,'green_signal',False) or self.local_readiness and str(obs.phase) in ('recruit','choice')
            if not stability.get("stable",False) and not local:reasons.append("ui_animation_or_stability_unknown")
            # A recent unrelated log line need not restart a global settle
            # timer in live recruit/choice. These paths independently require
            # synchronized logs, a visible phase and stable action regions.
            if not local and obs.log_event_at and capture["end"]-obs.log_event_at<.15:reasons.append("log_visual_settle_guard")
            if self.share_snapshot:
                from .recording import compact_observation
                diagnostic=compact_observation(obs)
            else:diagnostic=obs
            row=dict(id=stem,observation=diagnostic.to_dict(),window=window.to_dict(),capture=capture,
                     session=session,log_revision_after=after.revision,log_snapshot_unchanged=sync,ui=stability,
                     blockers=reasons,created_at=time.time())
            if getattr(self,'green_signal',False):row['capture_sync_protocol']='local_action_state_and_options'
            self.metrics.sample("screenshot_capture",(capture["end"]-capture["start"])*1000)
            self.metrics.sample("ui_stability_wait",stability.get("stabilization_wait_ms"))
            self.metrics.count("samples")
            if reasons:self.metrics.count("blocked_samples")
            if persist:(self.directory/(stem+".json")).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
            audit_row=row
            if self.compact_audit:
                # Repeated receipts and their snapshots dominate long live
                # sessions. Only the diagnostic copy is bounded; verification
                # and the live observation keep the complete causal state.
                from .recording import compact_observation_dict
                audit_row={**row,'observation':compact_observation_dict(row['observation']),
                           'receipt_post_states_omitted':True}
            self.audit.write("synchronized_sample",**audit_row)
            self.latest=row;self.latest_snapshot=obs
            return row,image
