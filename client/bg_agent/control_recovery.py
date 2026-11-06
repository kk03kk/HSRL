"""Recover omitted control entities at historical capture time from raw Power.

Strictly supplementary: original screenshots/Observations remain immutable.
Only exact semantic projection matches may contribute controls. Timestamp cutoff
is the observation time, never future lines or post-capture resource state.
"""
from datetime import datetime
import hashlib,json,re
from pathlib import Path
from .power import PowerState


def signature(obs):
    d=obs.to_dict() if hasattr(obs,'to_dict') else obs
    return dict(mode=d['mode'],phase=str(d['phase']),local_player_id=d['local_player_id'],turn=d['turn'],gold=d['gold'],
                tavern_tier=d['tavern_tier'],zones={z:[(c['entity_id'],c['card_id'],c['position']) for c in d[z]] for z in ('shop','hand','board','choices')},
                actions=[(str(a['kind']),a['source'],a['target'],a['option_id'],a.get('choice_id'),a.get('sub_option_id')) for a in d['actions']])


def recover(source,log,output):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(Path(source).glob('frame-*.json'))]
    stamp=datetime.fromtimestamp(rows[0]['observation']['captured_at']).date()
    projection=PowerState();iterator=iter(Path(log).open(encoding='utf-8',errors='replace'));pending=next(iterator,None)
    records={};last_time=None;lines=0
    def timestamp(line):
        m=re.match(r'^[DWE] (\d\d:\d\d:\d\d\.\d+)',line)
        if not m:return None
        return datetime.fromisoformat(str(stamp)+'T'+m[1][:15]).timestamp()
    for row in rows:
        cutoff=row['observation']['captured_at']
        while pending is not None:
            at=timestamp(pending)
            if at is not None and at>cutoff:break
            projection.feed(pending);last_time=at or last_time;lines+=1;pending=next(iterator,None)
        obs=projection.observe();matched=signature(obs)==signature(row['observation'])
        records[row['id']]=dict(matched=matched,metadata_sha256=hashlib.sha256((Path(source)/(row['id']+'.json')).read_bytes()).hexdigest(),
             observation_revision=row['observation']['revision'],projection_revision=obs.revision,last_event_at=last_time,cutoff=cutoff,
             controls=[__import__('dataclasses').asdict(c) for c in obs.controls] if matched else [],
             action_parameters={a.key:dict(resource_cost=a.resource_cost,authorization_source=a.authorization_source) for a in obs.actions} if matched else {},
             mismatch_fields=[] if matched else [k for k in signature(obs) if signature(obs)[k]!=signature(row['observation'])[k]])
    result=dict(schema=1,source_log=str(Path(log).resolve()),sha256=hashlib.sha256(Path(log).read_bytes()).hexdigest(),lines=lines,
                matched=sum(r['matched'] for r in records.values()),total=len(records),records=records)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result
