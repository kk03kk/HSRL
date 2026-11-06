"""Bounded operation-chain telemetry, independent of input authorization."""
from collections import Counter,defaultdict,deque
from contextlib import contextmanager
import math
import threading
import time
from functools import wraps


def measured_stage(name):
    def decorate(function):
        @wraps(function)
        def call(self,*args,**kwargs):
            metrics=(getattr(self,'chain_metrics',None) or getattr(self,'metrics',None) or
                     getattr(getattr(self,'provider',None),'chain_metrics',None) or
                     getattr(getattr(self,'observations',None),'chain_metrics',None))
            if not isinstance(metrics,ChainMetrics):return function(self,*args,**kwargs)
            with metrics.measure(name):return function(self,*args,**kwargs)
        return call
    return decorate


class ChainMetrics:
    def __init__(self,limit=250000):
        self.limit=limit;self.values=defaultdict(lambda:deque(maxlen=limit))
        self.counts=Counter();self.lock=threading.RLock()
        self.sample_counts=Counter();self.totals=Counter()
        self._active=threading.local();self.exclusive=Counter()

    def count(self,name,n=1):
        with self.lock:self.counts[name]+=n
        for scope in getattr(self._active,'scopes',()):scope['counts'][name]+=n

    def sample(self,name,milliseconds):
        if not math.isfinite(milliseconds) or milliseconds<0:return
        with self.lock:
            self.values[name].append(milliseconds);self.sample_counts[name]+=1;self.totals[name]+=milliseconds
        for scope in getattr(self._active,'scopes',()):
            scope['stage_ms'][name]+=milliseconds;scope['stage_samples'][name]+=1

    @contextmanager
    def action_scope(self):
        """Per-thread totals: parent includes child work, background I/O excluded."""
        scopes=getattr(self._active,'scopes',None)
        if scopes is None:scopes=[];self._active.scopes=scopes
        scope={key:Counter() for key in ('counts','stage_ms','stage_samples','exclusive_ms')}
        scopes.append(scope)
        try:yield scope
        finally:scopes.pop()

    @contextmanager
    def measure(self,name):
        stack=getattr(self._active,'stack',None)
        if stack is None:stack=[];self._active.stack=stack
        frame=[name,time.perf_counter(),0.0];stack.append(frame)
        try:yield
        finally:
            elapsed=(time.perf_counter()-frame[1])*1000;stack.pop()
            if stack:stack[-1][2]+=elapsed
            self.sample(name,elapsed)
            exclusive=max(0,elapsed-frame[2])
            with self.lock:self.exclusive[name]+=exclusive
            for scope in getattr(self._active,'scopes',()):scope['exclusive_ms'][name]+=exclusive

    def report(self):
        with self.lock:
            result={}
            for key,values in self.values.items():
                xs=sorted(values)
                if xs:result[key]=dict(n=self.sample_counts[key],retained=len(xs),total_ms=self.totals[key],
                    percentile_scope='all_samples' if len(xs)==self.sample_counts[key] else 'retained_tail',
                    p50=xs[(len(xs)-1)//2],p95=xs[min(len(xs)-1,math.ceil(len(xs)*.95)-1)])
            return dict(counts=dict(self.counts),latency_ms=result,exclusive_total_ms=dict(self.exclusive),
                timing_note='latency_ms is inclusive; sum exclusive_total_ms to avoid nested double counting',
                retained_samples_per_stage=self.limit)
