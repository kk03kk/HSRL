"""Archive bounded file prefixes; never starts or controls the game."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]


def copy_prefix(source,target,*,start_offset=0):
    source=Path(source);target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src,target.open('xb') as dst:
        before=os.fstat(src.fileno())
        if type(start_offset) is not int or not 0<=start_offset<=before.st_size:raise ValueError('invalid archive byte offset')
        src.seek(start_offset);length=before.st_size-start_offset;remaining=length;digest=hashlib.sha256()
        while remaining:
            chunk=src.read(min(1024*1024,remaining))
            if not chunk:raise RuntimeError('source truncated during archive: '+str(source))
            dst.write(chunk);digest.update(chunk);remaining-=len(chunk)
        dst.flush();os.fsync(dst.fileno())
        # A growing log is allowed. Its already-copied prefix must be unchanged.
        src.seek(start_offset);remaining=length;verify=hashlib.sha256()
        while remaining:
            chunk=src.read(min(1024*1024,remaining))
            if not chunk:raise RuntimeError('source truncated during verification: '+str(source))
            verify.update(chunk);remaining-=len(chunk)
        after=os.fstat(src.fileno())
    if digest.digest()!=verify.digest():raise RuntimeError('source prefix changed: '+str(source))
    check=hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):check.update(chunk)
    archived=check.hexdigest()
    if archived!=digest.hexdigest():raise RuntimeError('archive checksum mismatch')
    return dict(source=str(source.resolve()),target=str(target.resolve()),bytes=length,source_start_offset=start_offset,
        sha256=archived,source_mtime_ns=before.st_mtime_ns,source_bytes_after=after.st_size,
        source_identity=[before.st_dev,before.st_ino],capture_kind='verified_file_range' if start_offset else 'verified_file_prefix',
        complete_file_at_capture=start_offset==0 and before.st_size==after.st_size)


def archive(game_dir,out,*,session=None,evidence=None,include_shared_events=True,power_offset=0):
    out=Path(out).resolve();game_dir=Path(game_dir).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('archive must be inside main project')
    sessions=list((game_dir/'Logs').glob('Hearthstone_*'))
    selected=game_dir/'Logs'/session if session else max(sessions,key=lambda p:p.name) if sessions else None
    if selected is None or not selected.is_dir() or selected.parent!=game_dir/'Logs':raise ValueError('invalid log session')
    sources=[p for p in selected.iterdir() if p.is_file() and p.suffix=='.log' and
             p.name.startswith(('Power','LoadingScreen','GameNetLogger','Gameplay'))]
    if not any(p.name.startswith('Power') for p in sources):raise ValueError('Power log absent')
    evidence=Path(evidence).resolve() if evidence else None
    if evidence and (not evidence.is_relative_to(ROOT) or out.is_relative_to(evidence)):
        raise ValueError('evidence must be a separate main-project directory')
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for p in sorted(sources):rows.append(copy_prefix(p,out/'raw'/selected.name/p.name,start_offset=power_offset if p.name=='Power.log' else 0))
    if evidence:
        for p in sorted(evidence.rglob('*')):
            if p.is_file():rows.append(copy_prefix(p,out/'evidence'/p.relative_to(evidence)))
    events=ROOT/'artifacts/runs/live-state/events.jsonl'
    if include_shared_events and events.exists():rows.append(copy_prefix(events,out/'events.jsonl'))
    result=dict(format='hsrl.live-session-archive.v1',captured_at=time.time(),session=selected.name,
        files=rows,game_started=False,native_input_acquired=False,
        interpretation='Per-file verified byte ranges; see source_start_offset. Not an atomic cross-log snapshot or proof of a complete match.')
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--game-dir',required=True)
    p.add_argument('--out',required=True);p.add_argument('--session');p.add_argument('--evidence')
    a=p.parse_args();r=archive(a.game_dir,a.out,session=a.session,evidence=a.evidence)
    print(json.dumps(dict(session=r['session'],files=len(r['files']),out=str(Path(a.out).resolve()))))

if __name__=='__main__':main()
