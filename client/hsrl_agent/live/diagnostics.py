"""Bounded passive observations. These records never authorize input."""
import copy
import json
from pathlib import Path
import time
from .contracts import Blocked
from .encoding import encode
from .runtime import save


def archive_events(source,target,started_at,ended_at,*,start_offset=0):
    """每局保存自己的模型/输入事件，避免只能去共享大日志里追查。"""
    source=Path(source);target=Path(target)
    if not source.exists():return dict(status='absent',events=0)
    count=0;malformed=0
    with source.open(encoding='utf8') as src,target.open('x',encoding='utf8') as dst:
        if not 0<=start_offset<=source.stat().st_size:raise ValueError('event archive cursor no longer exists')
        src.seek(start_offset)
        for line in src:
            try:row=json.loads(line)
            except json.JSONDecodeError:malformed+=1;continue
            if started_at<=row.get('at',-1)<=ended_at:dst.write(line);count+=1
    return dict(status='saved' if not malformed else 'saved_with_malformed_source_rows',events=count,
                malformed_source_rows=malformed,path=str(target),started_at=started_at,ended_at=ended_at,source_start_offset=start_offset)


def status_payload(reason,data=None,snapshot=None,observation=None):
    matching=bool(data and observation and observation.get('game_id')==data['game_id'] and
                  observation.get('revision')==data['revision'])
    unknown=(snapshot or {}).get('view',{}).get('unknown')
    if unknown is None and matching:unknown=observation.get('unknown')
    return dict(reason=reason,client=data,unknown=unknown,unknown_count=len(unknown) if unknown is not None else None,
        observation_available=unknown is not None,last_observation=observation,
        interpretation='null unknown means unmeasured, not a complete model observation')


class ObservationRecorder:
    def __init__(self,out,vocabulary,*,clock=time.monotonic,interval=1.,max_samples=24):
        self.out=Path(out);self.vocabulary=vocabulary;self.clock=clock;self.interval=interval
        self.max_samples=max_samples;self.last=None;self.next_at=0.;self.sampled=set()

    def probe(self,data,produce):
        now=self.clock()
        if now<self.next_at:return self.last
        self.next_at=now+self.interval
        row=dict(at=time.time(),game_id=data['game_id'],revision=data['revision'],turn=data.get('turn'),
            phase=data.get('phase'),unknown=None,model_input_ready=False,projection_available=False,
            authority='diagnostic_only_no_input');snapshot=None
        try:
            snapshot=produce();row['projection_available']=True
            row['unknown']=snapshot['view'].get('unknown',[])
            encode(snapshot['view'],self.vocabulary);row['model_input_ready']=True;row['reason']='encodable'
        except Blocked as exc:
            row['reason']='unknown public fields' if str(exc).startswith('unknown public fields:') else str(exc)
        row['unknown_count']=len(row['unknown']) if row['unknown'] is not None else None
        key=(row['game_id'],row['turn'],row['phase'],row['reason'])
        if key not in self.sampled and len(self.sampled)<self.max_samples:
            self.sampled.add(key)
            # Receipts contain large prior post-states. Raw logs preserve them;
            # diagnostic snapshots retain current facts without nested history.
            compact={k:v for k,v in data.items() if k not in ('resolutions','actions')}
            payload=dict(diagnostic=row,client=compact,receipts_omitted=True)
            if snapshot:payload.update(view=snapshot['view'],options=snapshot['options'])
            name='observation-%03d.json'%len(self.sampled);save(self.out/'samples'/name,payload)
            row['sample']='samples/'+name
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/'observations.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        self.last=copy.deepcopy(row);save(self.out/'observation-status.json',row)
        return row
