"""Opt-in bounded observation recorder. Never launches a game or sends input.

Only the explicit CLI entry instantiates NativeWindows. F9/F10 are passive key
state reads; the recorder never consumes keys, focuses a window or moves a
cursor. Images are saved only from a foreground, unobscured verified client.
"""
from dataclasses import dataclass, replace, field
from io import BytesIO
import argparse
import hashlib
import json
from pathlib import Path
import time
import threading


@dataclass
class Budget:
    limits: dict
    used: dict
    lock: object = field(default_factory=threading.RLock,repr=False)

    @classmethod
    def create(cls, *, logs=1024**3, images=256*1024**2, metadata=64*1024**2):
        if min(logs,images,metadata)<=0:raise ValueError('budgets must be positive')
        return cls(dict(logs=logs,images=images,metadata=metadata),dict(logs=0,images=0,metadata=0))

    def remaining(self,kind):
        with self.lock:return self.limits[kind]-self.used[kind]

    def permits(self,**sizes):
        with self.lock:return all(0<=size<=self.remaining(kind) for kind,size in sizes.items())

    def consume(self,**sizes):
        with self.lock:
            if not self.permits(**sizes):raise RuntimeError('capture_budget_exceeded')
            for kind,size in sizes.items():self.used[kind]+=size

    def reserve(self,**sizes):
        """Atomically reserve a complete frame for concurrent live writers."""
        with self.lock:
            if not self.permits(**sizes):return False
            for kind,size in sizes.items():self.used[kind]+=size
            return True


class Schedule:
    """Stable semantic edges trigger a short PNG burst, not every log revision."""
    def __init__(self):
        self.signature=None;self.burst_until=-1;self.next_frame=0

    def due(self,obs,now,*,mark=False):
        ctx=obs.interaction
        sig=(obs.game_id,obs.turn,obs.phase,
             (ctx.request_id,ctx.display_state,tuple(c.entity_id for c in obs.choices)) if ctx else None,
             tuple(c.entity_id for c in obs.branches))
        edge=sig!=self.signature
        if edge:self.signature=sig;self.burst_until=now+3
        if mark:self.burst_until=now+3
        if not mark and now<self.next_frame:return None
        png=mark or now<self.burst_until
        self.next_frame=now+(.75 if png else 3)
        return 'png' if png else 'jpg'


class LiveEvidenceSchedule:
    """Archive phase/request edges and sparse heartbeats, not every capture."""
    def __init__(self,heartbeat=30):
        self.heartbeat=heartbeat;self.signature=None;self.ready_saved=False;self.last_saved=float('-inf')

    def due(self,obs,now):
        ctx=obs.interaction
        signature=(obs.game_id,obs.phase,obs.turn,
            (ctx.request_id,ctx.source_entity,ctx.task_list,tuple(c.entity_id for c in obs.choices)) if ctx else None)
        reason=None
        if signature!=self.signature:
            self.signature=signature;self.ready_saved=obs.ready;reason='phase_or_request_changed'
        elif obs.ready and not self.ready_saved:
            self.ready_saved=True;reason='phase_first_ready'
        elif now-self.last_saved>=self.heartbeat:reason='heartbeat'
        if reason:self.last_saved=now
        return reason


def encoded_frame(image,extension):
    from .capture_broker import ClientImage
    if isinstance(image,ClientImage):image=image.image
    output=BytesIO()
    if extension=='jpg':
        image=image.copy();image.thumbnail((1600,1000))
        image.convert('RGB').save(output,format='JPEG',quality=80,optimize=True)
    else:image.save(output,format='PNG',compress_level=3)
    return output.getvalue(),image.size


def save_frame(folder,index,image,extension,metadata,budget):
    """Reserve both files before writing; full resolution only for event PNGs."""
    payload,size=encoded_frame(image,extension)
    record={**metadata,'image':dict(format=extension,size=size,bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),lossless=extension=='png')}
    raw=json.dumps(record,ensure_ascii=False,indent=2).encode('utf-8')
    if not budget.reserve(images=len(payload),metadata=len(raw)):return False
    stem=folder/(f'frame-{index:06d}' if isinstance(index,int) else index)
    stem.with_suffix('.'+extension).write_bytes(payload)
    stem.with_suffix('.json').write_bytes(raw)
    return True


def collect(game_dir,directory,*,seconds=3600,budget=None,hotkeys=True):
    from .connection import latest_session
    from .power import PowerState
    from .recording import RawMirror
    from .windows import NativeWindows
    from .interaction_types import describe
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    frames=directory/'frames';frames.mkdir()
    budget=budget or Budget.create()
    # Reserve a bounded summary area before any disk writes.
    budget.consume(metadata=65536)
    windows=NativeWindows(Path(game_dir)/'Hearthstone.exe')
    keys=None
    if hotkeys:
        import ctypes
        from ctypes import wintypes
        keys=windows.u.GetAsyncKeyState
        keys.argtypes=[wintypes.INT];keys.restype=wintypes.SHORT
    started=time.monotonic();schedule=Schedule();session=None;mirrors={};raw_rows=[]
    state=PowerState();pending=b'';index=0;last_digest=None;marks=0;held=False
    errors=[];stop='duration';blocked=0
    print('只读采集：不启动游戏、不操作鼠标键盘。F9 标记，F10 或 Ctrl+C 停止。',flush=True)
    try:
        while time.monotonic()-started<seconds:
            down=bool(keys and keys(0x78)&0x8000)
            mark=down and not held;held=down
            if mark:marks+=1
            if keys and keys(0x79)&0x8000:stop='human_stop';break
            current=latest_session(Path(game_dir))
            if current!=session:
                session=current;mirrors={};state=PowerState();pending=b''
            if session:
                for name in ('Power.log','LoadingScreen.log','GameNetLogger.log'):
                    source=session/name
                    if source.exists() and name not in mirrors:
                        mirror=RawMirror(source,directory/'raw'/session.name/name)
                        mirrors[name]=mirror;raw_rows.append(mirror)
                backlog=False
                for name,mirror in mirrors.items():
                    remaining=budget.remaining('logs')
                    if remaining<=0:stop='log_budget';break
                    chunk=mirror.poll(max_bytes=min(4*1024**2,remaining))
                    budget.consume(logs=len(chunk))
                    if name=='Power.log':
                        rows=(pending+chunk).split(b'\n');pending=rows.pop()
                        for row in rows:state.feed(row.decode('utf-8',errors='replace'))
                    # Scene logs are preserved for offline synchronization.
                    backlog|=mirror.offset<mirror.source.stat().st_size
                if stop=='log_budget':break
                if backlog:continue  # Never attach current pixels to a historical log cursor.
            obs=state.observe();now=time.monotonic()
            extension=schedule.due(obs,now,mark=mark)
            if extension:
                candidates=windows.find()
                if len(candidates)==1 and not candidates[0].blockers:
                    w=candidates[0];image,capture=windows.capture(w,None)
                    after=capture.get('window_after',{})
                    if not capture.get('geometry_unchanged') or after.get('blockers'):
                        blocked+=1;continue
                    digest=capture['sha256']
                    if digest!=last_digest or mark:
                        meta=dict(schema=1,mark=mark,unix_time=time.time(),
                            log_cursors=[m.manifest() for m in mirrors.values()],
                            interaction=describe(obs),
                            sample=dict(observation=replace(obs,resolutions=()).to_dict(),
                                        window=w.to_dict(),capture=capture))
                        if not save_frame(frames,index,image,extension,meta,budget):
                            stop='image_or_metadata_budget';break
                        index+=1;last_digest=digest
                else:blocked+=1
            time.sleep(.2)
    except KeyboardInterrupt:stop='human_interrupt'
    except (OSError,RuntimeError,ValueError) as exc:
        stop='capture_error';errors.append(type(exc).__name__+': '+str(exc)[:500])
    finally:
        result=dict(schema=1,stop=stop,frames=index,marks=marks,blocked_captures=blocked,
            elapsed_seconds=time.monotonic()-started,budget=dict(limits=budget.limits,used=budget.used),
            logs=[m.manifest() for m in raw_rows],errors=errors,game_launched=False,input_sent=False,
            live_execution_validated=False,limits=[
                'Per-invocation cap; older capture folders are never deleted automatically.',
                'Low-rate JPEG context is lossy; event PNGs preserve original pixels.',
                'Sampling visibility checks cannot prove every screen pixel was unobscured.',
                'Raw logs end at their recorded cursors; partial sessions are not full-match acceptance.'])
        raw=json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8')
        # Session count is bounded by duration; explicitly truncate if unusually large.
        if len(raw)>65536:
            result['logs']=[];result['errors'].append('summary_log_manifest_too_large; see raw folder')
            raw=json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8')
        (directory/'capture.json').write_bytes(raw)
        print(str(directory/'capture.json'),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-dir',type=Path,default=Path(__import__('os').environ.get('HEARTHSTONE_DIR', 'Hearthstone')))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--seconds',type=float,default=3600)
    parser.add_argument('--no-hotkeys',action='store_true')
    args=parser.parse_args()
    if args.seconds<=0:parser.error('--seconds must be positive')
    out=args.out or Path('runtime')/('interaction-capture-'+time.strftime('%Y%m%d-%H%M%S'))
    return collect(args.game_dir,out,seconds=args.seconds,hotkeys=not args.no_hotkeys)


if __name__=='__main__':main()
