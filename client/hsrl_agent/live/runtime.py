"""Explicitly invoked client runtime. Importing this module performs no OS I/O."""
from contextlib import contextmanager,nullcontext
import copy
import hashlib
import json
from pathlib import Path
import time

from .contracts import Blocked,state_stamp,replace_file


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');replace_file(tmp,path)


def catalog(root):
    return {c['id']:c for c in load(Path(root)/'engine/crates/hsbg-engine/data/cards.json')['cards']}


@contextmanager
def exclusive(path):
    """Kernel-released Windows byte lock, including after interpreter crashes."""
    import msvcrt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if path.stat().st_size==0:f.write(b'0');f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise Blocked('another agent owns this session') from exc
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def visual_state(backend,profile,data,previous=None,*,pending=None):
    from PIL import Image
    from .vision import stable,crop_box,template_score
    first,window=backend.capture()
    time.sleep(profile.get('settle_seconds',.25))
    second,after=backend.capture()
    if window.token()!=after.token():raise Blocked('window changed between stability frames')
    if list(second.size)!=profile.get('client_size') or window.dpi!=profile.get('dpi'):
        raise Blocked('client size/DPI differs from reviewed calibration')
    region=profile.get('stability_region',[0,.2,1,.8])
    if not stable(first,second,crop_box(region,second.size),profile.get('stability_threshold',3.)):
        raise Blocked('client animation has not settled')
    interaction=data.get('interaction') or {}
    phase='hero' if data['choices'] and all(c['card_type']=='HERO' for c in data['choices']) else 'discover' if data['choices'] else data['phase']
    guard=profile.get('scene_guards',{}).get(phase)
    if not guard:raise Blocked('no reviewed visual scene guard:'+phase)
    def matches(spec):
        path=Path(profile['_directory'])/spec['template']
        # Templates are workspace files, not executable producers.
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise Blocked('visual template changed')
        with Image.open(path) as template:
            return template_score(second.crop(crop_box(spec['rect'],second.size)),template)<=spec.get('max_error',.035)
    if not matches(guard):raise Blocked('client scene disagrees with reviewed visual template')
    status=dict(window_token=after.token(),phase=phase)
    for key in ('target_open','confirm_enabled','available_races_shown'):
        spec=profile.get('signals',{}).get(key)
        status[key]=bool(spec and matches(spec))
    from .perception import selection,branch
    status['selection']=selection(second,profile,data)
    status.update(branch(second,profile,data,pending))
    return status,second


def project(reader,data,projector,info,profile,facts_path=None,lobby_races=None):
    from .facts import context_facts
    facts=context_facts(reader,data,info,lobby_races=lobby_races,
                        supplied=load(facts_path) if facts_path else None)
    return projector.build(reader,data,facts=facts)


def run(root,args):
    """Preserve per-file log prefixes after this run, including failure exits."""
    import uuid
    token=uuid.uuid4().hex;archived=True;result=None;failure=None
    try:
        result=_run(root,args,token)
    except BaseException as exc:
        import traceback
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        try:archived=_finalize_run(root,args,token,result,failure)
        except Exception as exc:
            import sys
            archived=False
            print('Run finalization failed: '+str(exc),file=sys.stderr,flush=True)
    return result if archived else 2


def _finalize_run(root,args,token,result,failure):
    archived=True
    out=Path(args.out).resolve();manifest=out/'session.json'
    if getattr(args,'live_trace',None) is not None:
        try:save(out/'trace-status.json',args.live_trace.close())
        except Exception as exc:save(out/'trace-status.json',dict(error=type(exc).__name__+': '+str(exc)))
    # Never modify/append to a pre-existing run rejected by _run().
    if manifest.exists() and load(manifest).get('run_token')==token:
        metadata=load(manifest);archives=[]
        save(out/'exit.json',dict(exit_code=result,completed_normally=failure is None,error=failure))
        # Copy, never clear or reconcile, the exact uncertain ledger at exit.
        ledger=Path(root)/'artifacts/runs/live-state/inflight.json'
        if getattr(args,'enable_input',False) and ledger.exists():
            from .session_archive import copy_prefix
            try:copy_prefix(ledger,out/'unresolved-input.json')
            except Exception as exc:save(out/'ledger-archive-error.json',dict(reason=str(exc)))
        from .diagnostics import archive_events
        events=Path(root)/'artifacts/runs/live-state/events.jsonl' if getattr(args,'enable_input',False) else out/'shadow-state/events.jsonl'
        try:trace=archive_events(events,out/'agent-events.jsonl',metadata['started_at'],time.time(),start_offset=metadata.get('event_start_offset',0)) if 'started_at' in metadata else dict(status='missing_run_start')
        except Exception as exc:trace=dict(status='failed',error=str(exc))
        save(out/'event-archive-status.json',trace)
        if getattr(args,'pipeline_stats',None):save(out/'latency.json',args.pipeline_stats.report())
        from .session_archive import archive
        for session in metadata.get('log_sessions',[]):
            target=out/'session-archives'/session
            try:
                r=archive(args.game_dir,target,session=session,include_shared_events=False,
                          power_offset=metadata.get('power_offsets',{}).get(session,0))
                archives.append(dict(session=session,status='saved',manifest=str(target/'manifest.json')))
            except Exception as exc:
                archived=False;archives.append(dict(session=session,status='failed',error=str(exc)))
        save(out/'archive-status.json',dict(archives=archives,complete=archived and bool(archives)))
        if not archives:archived=False
        save(out/'exit.json',dict(exit_code=(result if archived else 2) if failure is None else None,
            runtime_exit_code=result,completed_normally=failure is None,archive_complete=archived,error=failure))
        print('Log archive saved.' if archived else 'Log archive incomplete; inspect archive-status.json.',flush=True)
    return archived



def settle_placement(reader,data,*,settle=2.5,interval=.1,stable_for=.15,sleep=time.sleep):
    """Read the *final* leaderboard standing, not the provisional death one.

    The client writes a provisional ``PLAYER_LEADERBOARD_PLACE`` at the moment
    of death and the final value 0.1-0.5 s later (measured in the 02:27, 02:54
    and 04:06 archived games). Reading the tag once at ``game_over`` therefore
    produced an off-by-one placement in the 02:54 game. Two equal reads
    separated by ``stable_for`` are required; the last observed value is
    returned with its provenance so the reader knows whether it settled.
    """
    last=None;seen_at=None;best=None;deadline=time.monotonic()+settle
    while True:
        tag=(data.get('hero') or {}).get('tags',{}).get('PLAYER_LEADERBOARD_PLACE')
        if tag not in (None,'0',''):
            if tag==last and seen_at is not None and time.monotonic()-seen_at>=stable_for:
                return tag,'stable'
            if tag!=last:last=tag;seen_at=time.monotonic();best=tag
        if time.monotonic()>=deadline:break
        sleep(interval)
        try:fresh=reader.poll()
        except Exception:fresh=None
        if isinstance(fresh,dict) and fresh.get('game_id')==data.get('game_id'):data=fresh
    return best,'timeout'


def _run(root,args,run_token):
    from .logs import LogReader
    from .projection import Projector
    from .encoding import encode
    from .policy import ModelPolicy
    from .controller import Controller
    from .vision import Geometry
    from .windows import NativeBackend
    profile=load(args.profile);profile['_directory']=str(Path(args.profile).resolve().parent)
    if args.enable_input and args.facts:raise Blocked('manual context sidecars are for shadow/offline diagnosis only')
    if profile.get('review_state')!='reviewed':raise Blocked('profile is reference-only; capture and review calibration first')
    game_dir=Path(args.game_dir).resolve();exe=game_dir/'Hearthstone.exe'
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=profile.get('executable_sha256'):
        raise Blocked('Hearthstone executable differs from reviewed calibration')
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not (getattr(args,'queue_prepared',False) and {p.name for p in out.iterdir()}=={'startup'}):
        raise Blocked('run output must be a new/empty folder; shared inflight ledger remains separate')
    out.mkdir(parents=True,exist_ok=True)
    session_meta=dict(format='hsrl.live-run.v1',run_token=run_token,started_at=time.time(),
        mode='input' if args.enable_input else 'shadow',log_sessions=[],checkpoint=str(args.checkpoint))
    events=Path(root)/'artifacts/runs/live-state/events.jsonl' if args.enable_input else out/'shadow-state/events.jsonl'
    session_meta['event_start_offset']=events.stat().st_size if events.exists() else 0
    save(out/'session.json',session_meta)
    backend=getattr(args,'preloaded_backend',None) or NativeBackend(exe,getattr(args,'stop_file',out/'STOP'),enable_input=args.enable_input,timing=getattr(args,'timing','balanced'))
    practical=bool(getattr(args,'practical',False))
    if practical:
        from .adaptive import AdaptiveProjector
        from .practical import PracticalGeometry
        Projector=AdaptiveProjector;Geometry=PracticalGeometry
    reader=LogReader(game_dir,args.player_id);projector=Projector(profile,catalog(root))
    reader.start_latest=practical
    from .policy import configure_threads
    configure_threads(getattr(args,'inference_threads',None))
    policy=getattr(args,'preloaded_policy',None) or ModelPolicy(args.checkpoint,args.device);info=catalog(root)
    args.preloaded_policy=policy
    session_meta['checkpoint_sha256']=policy.sha
    session_meta['policy_id']=getattr(policy,'policy_id','unspecified-test-policy')
    session_meta['inference_omitted_auxiliary']=getattr(policy,'meta',{}).get('inference_omitted_auxiliary',[])
    save(out/'session.json',session_meta)
    from .diagnostics import ObservationRecorder,status_payload
    diagnostics=ObservationRecorder(out,policy.vocabulary)
    # One writer thread per match. The controller enqueues compact/full
    # decision rows and gesture receipts; it never formats JSON or fsyncs on
    # the input hot path.
    from .trace import GameTrace
    trace=GameTrace(out,side='live',store_vectors=True)
    args.live_trace=trace
    last_report=None;last_decision=None;started=time.monotonic();tribe_memory={};ocr=None
    stale_ledger_checked=False
    status_saved_at=0.;last_status_key=None;closed_unresolved=False;game_completed=False;operator_stopped=False
    data=None;snapshot=None;last_blocker=None
    # Unexpected executor faults must pause input and keep the ledger, never
    # end the process while a match is running. Bounded, then non-zero exit.
    layout_evidence=set()
    def capture_layout_evidence(reason,data):
        """未标定的选择行会让输入暂停；留一张帧，下一次才能真的标定它。

        ``uncalibrated layout:discover/choices/5``（时空扭曲一类的五选一）以前只留
        一行状态文本，事后无法复现那一屏。这里抓一帧 + 选择实体清单，仍然不发任何
        输入；同一 (reason, turn) 只留一次，避免刷盘。
        """
        if not (reason.startswith('uncalibrated layout:')
                or reason.startswith('layout has not been visually reviewed:')):return
        key=(reason,(data or {}).get('turn'))
        if key in layout_evidence:return
        layout_evidence.add(key)
        try:
            image,window=backend.capture()
            folder=out/'visual-evidence';folder.mkdir(parents=True,exist_ok=True)
            safe=''.join(c if c.isalnum() else '-' for c in reason)[:48]
            # Writer thread, not the input loop: the save happens while the
            # bounded no-input pause continues, never inside a press sequence.
            trace.frame(folder/(safe+'.jpg'),image,dict(reason=reason,game_id=(data or {}).get('game_id'),
                turn=(data or {}).get('turn'),phase=(data or {}).get('phase'),
                choices=[{k:c.get(k) for k in ('entity_id','card_id','card_type','position')}
                         for c in (data or {}).get('choices') or []],
                window_token=window.token(),
                note='no input sent; this choice row has no calibrated geometry'),quality=90)
        except Exception as exc:
            layout_evidence.discard(key)
            try:save(out/'visual-evidence'/'layout-uncalibrated-error.json',
                     dict(reason=reason,error=type(exc).__name__+': '+str(exc)))
            except Exception:pass
    internal_errors=0;last_internal=None;progress_at=time.monotonic()
    # Global lock lives at a fixed workspace path, independent of --out. Two
    # different output folders must never authorize two input controllers.
    with (nullcontext() if getattr(args,'client_lock_owned',False) else exclusive(Path(root)/'artifacts/runs/live-client.lock')):
        controller=Controller(policy,Geometry(profile),backend,Path(root)/'artifacts/runs/live-state' if args.enable_input else out/'shadow-state',enable_input=args.enable_input,practical=practical,clock=time.monotonic,trace=trace)
        fast=None
        if practical:
            from .pipeline import LivePipeline
            fast=LivePipeline(reader,projector,backend,profile,policy,lambda:controller.pending,out)
            args.pipeline_stats=fast.metrics
            backend.pre_press=fast.pre_press
            from .interruptions import InterruptionGuard
            fast.gate.interruptions=InterruptionGuard(getattr(args,'lobby_ocr',None))
            fast.gate.frame_writer=trace.frame
        from .execution_status import WaitTracker
        waits=WaitTracker()
        last_health=0.
        def report(reason,data=None,snapshot=None):
            nonlocal last_report,status_saved_at,last_status_key,last_health
            execution=waits.update(reason,data,controller.pending)
            from .lifecycle import protocol_state
            execution.update(protocol_state(controller,data or {}))
            if time.monotonic()-last_health>=5.:
                last_health=time.monotonic()
                try:
                    from .supervisor import health_payload
                    save(out/'health.json',health_payload(reason=reason,data=data,controller=controller,waits=waits))
                except Exception:pass
            if reason!=last_report:
                text=('unknown public fields (see observation-status.json and coverage.json)' if reason.startswith('unknown public fields:') else reason)
                if reason.startswith('uncalibrated layout:'):
                    text=(reason+' — this choice row has no calibrated click geometry; input is paused. '
                          'Fill it in by hand; a frame was saved under visual-evidence/.')
                print(text,flush=True);controller.event('status',reason=reason);last_report=reason
            key=(reason,(data or {}).get('game_id'),(data or {}).get('turn'),(data or {}).get('phase'))
            if key!=last_status_key or time.monotonic()-status_saved_at>=1.:
                status_data=data
                if practical and data:
                    status_data={**data,'resolutions':[{k:v for k,v in r.items() if k!='post_state'} for r in data.get('resolutions',[])]}
                payload=status_payload(reason,status_data,snapshot,diagnostics.last)
                payload['last_blocker']=last_blocker
                payload['execution']=execution
                save(out/'status.json',payload)
                status_saved_at=time.monotonic();last_status_key=key
        def prepare(data):
            nonlocal ocr
            if fast:return fast.prepare(data)
            if reader.connection!='connected':raise Blocked('client connection is not confirmed connected')
            visual,image=visual_state(backend,profile,data,pending=controller.pending)
            if visual.get('target_open') and not (controller.pending and controller.pending[0].binding.targets):
                raise Blocked('target UI has no owned, pinned source transaction')
            gid=data['game_id'];races=tribe_memory.get(gid)
            observed=profile.get('observed_lobby_context',{})
            if races is None and observed.get('game_id')==gid:races=observed['races']
            if races is None and profile.get('available_races_roi') and visual['phase']=='hero' and visual['available_races_shown']:
                from .vision import OCR,races_from_text,crop_box
                ocr=ocr or OCR();rows=ocr.read(image.crop(crop_box(profile['available_races_roi'],image.size)))
                races=races_from_text(rows);tribe_memory.clear();tribe_memory[gid]=races
                controller.event('observed_lobby_races',game_id=gid,rows=rows,races=races)
            s=project(reader,data,projector,info,profile,args.facts,races)
            from .interactions import target_snapshot,branch_snapshot
            s=branch_snapshot(s,controller.pending,visual)
            s=target_snapshot(s,controller.pending,visual)
            save(out/'coverage.json',dict(game_id=gid,revision=data['revision'],unknown=s['view'].get('unknown',[])))
            try:encode(s['view'],policy.vocabulary)
            except Blocked:
                if not (out/'blocked-first.png').exists():
                    image.save(out/'blocked-first.png');save(out/'blocked-first.json',s)
                raise
            s['visual']=visual
            return s,image
        def fresh(old):
            data=reader.poll();new,_=prepare(data)
            if state_stamp(new)!=state_stamp(old):raise Blocked('state changed during decision/animation')
            if new['visual']['window_token']!=old['visual']['window_token']:raise Blocked('window changed before dispatch')
            return new
        guard=fast or fresh
        while args.seconds<=0 or time.monotonic()-started<args.seconds:
            if backend.stopped():operator_stopped=True;report('STOP/F8 requested');break
            data=None;snapshot=None
            try:
                data=reader.poll()
                if reader.session and reader.session.name not in session_meta['log_sessions']:
                    session_meta['log_sessions'].append(reader.session.name)
                    session_meta.setdefault('power_offsets',{})[reader.session.name]=getattr(reader,'game_offset',0)
                    save(out/'session.json',session_meta)
                if not reader.caught_up:
                    report('reconstructing complete log history')
                    if controller.pending and controller.pending_timeout()=='halted':report(controller.halted,data);break
                    # An incomplete trailing line is common while Unity writes.
                    # Do not spin at 100% CPU waiting for its newline.
                    time.sleep(.01);continue
                if controller.pending and not stale_ledger_checked:
                    # A ledger restored from a previous run belongs to that run's
                    # turn. Retire it once at start instead of inheriting a dead
                    # 120 s wait (and a halt) into this run.
                    stale_ledger_checked=True
                    if controller.retire_stale_inflight(dict(game_id=data['game_id'],revision=data['revision'],client=data)):
                        report('retired stale inflight ledger from an earlier turn')
                if practical and data['phase']=='game_over':
                    player=reader.state.entities.get(reader.state.players.get(data['local_player_id']))
                    game=reader.state.entities.get(reader.state.game_entity)
                    from .lifecycle import terminal_proof
                    terminal=terminal_proof(reader.state,data,history_complete=reader.history_complete)
                    proof=terminal['proof'] if terminal else None
                    while controller.pending:
                        outcome=controller.poll_pending(dict(game_id=data['game_id'],revision=data['revision'],client=data))
                        if outcome not in ('applied','expired_unsent_input','externally_resolved_choice','externally_resolved_hero','externally_resolved_play'):break
                    unresolved=bool(controller.pending)
                    closed_unresolved=unresolved
                    if unresolved:controller.close_game(data,out,proof)
                    placement,placement_source=settle_placement(reader,data)
                    save(out/'result.json',dict(game_id=data['game_id'],result=player.tags.get('PLAYSTATE') if player else None,
                        placement=placement,placement_source=placement_source,
                        turn=data['turn'],model_sha256=policy.sha,adapter='public-approx-v1',unresolved_input_archived=unresolved,
                        # This file is only written from the game_over branch, so its
                        # presence is the terminal proof. Record it explicitly (and the
                        # lifecycle proof string) so an overnight report can count real
                        # finished games instead of inferring from a placement field.
                        terminal=True,terminal_proof=proof,log_position=data.get('log_position')))
                    if unresolved:report('game completed; unresolved input archived without retry',data)
                    else:report('game completed',data)
                    game_completed=True
                    break
                # A model EndTurn is a durable wait, not a request to re-capture
                # and re-encode the same quiet board every 150ms.
                from .lifecycle import owned_selection
                owned_choice=owned_selection(data)
                if not controller.pending and controller.ended==(data['game_id'],data.get('turn')) and not owned_choice:
                    report('waiting_next_turn',data);time.sleep(.08);continue
                if not args.enable_input:
                    # Diagnose log observability independently of UI guards.
                    # This never bypasses prepare(), encode() or input checks.
                    diagnostics.probe(data,lambda:projector.build_public(reader,data) if practical else project(reader,data,projector,info,profile,
                        args.facts,tribe_memory.get(data['game_id'])))
                if controller.pending:
                    # Receipts are read even while the policy projection is
                    # unavailable during playback/combat. Never require NN
                    # features to acknowledge a past input.
                    raw=dict(game_id=data['game_id'],revision=data['revision'],client=data)
                    outcome=controller.poll_pending(raw)
                    if outcome=='waiting_receipt_overdue_no_resend' and fast:fast.record_overdue(data)
                    if outcome=='retry_unsent':
                        report(outcome,data);last_decision=None;time.sleep(.05);continue
                    if outcome=='duplicate_gesture_suppressed':
                        report(outcome,data);time.sleep(.25);continue
                    if outcome in ('applied','context_changed','expired_unsent_input','externally_resolved_hero','externally_resolved_choice','externally_resolved_play'):report(outcome,data);continue
                    if outcome=='halted':report(controller.halted,data);break
                    if practical and not controller.needs_ui(data):
                        report(outcome,data);time.sleep(.05);continue
                    try:snapshot,image=prepare(data)
                    except Blocked as exc:
                        last_blocker=dict(reason=str(exc),game_id=data['game_id'],turn=data.get('turn'))
                        report(str(exc),data);time.sleep(.15);continue
                    outcome=controller.poll_pending(snapshot,before_send=guard)
                else:
                    snapshot,image=prepare(data)
                    stamp=state_stamp(snapshot)
                    if stamp==last_decision:time.sleep(.15);continue
                    outcome=controller.tick(snapshot,before_send=guard)
                    if outcome!='dry_run' or not args.enable_input:last_decision=stamp
                if outcome in ('sent','applied','branch_sent','target_sent','confirm_sent','waiting_next_turn'):
                    last_blocker=None;internal_errors=0;last_internal=None;progress_at=time.monotonic()
                if outcome=='retry_unsent':last_decision=None
                report(controller.halted if outcome=='halted' and controller.halted else outcome,data,snapshot)
                if controller.halted and controller.halted!='input interrupted; delivery unknown':break
            except Blocked as exc:
                last_blocker=dict(reason=str(exc),game_id=(data or {}).get('game_id'),turn=(data or {}).get('turn'))
                capture_layout_evidence(str(exc),data)
                report(str(exc),data,snapshot)
                if controller.pending and controller.pending_timeout()=='halted':
                    report(controller.halted,data);break
            except Exception as exc:
                # A code fault in projection, geometry or verification is an
                # unavailable-evidence pause, not permission to exit mid-game.
                # Nothing is sent, the durable intent is untouched, and a
                # repeated identical fault still terminates with a non-zero
                # code after a bounded number of attempts.
                import traceback
                internal_errors+=1
                same=bool(last_internal and last_internal.get('type')==type(exc).__name__
                          and last_internal.get('reason')==str(exc))
                signature=dict(type=type(exc).__name__,reason=str(exc),
                    traceback=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))[-4000:],
                    count=internal_errors,same_repeats=(last_internal.get('same_repeats',0)+1 if same else 1))
                last_internal=signature
                last_blocker=dict(reason='internal '+signature['type']+': '+signature['reason'],
                                  game_id=(data or {}).get('game_id'),turn=(data or {}).get('turn'),internal=True)
                try:
                    controller.event('internal_error',game_id=(data or {}).get('game_id'),
                        turn=(data or {}).get('turn'),pending=bool(controller.pending),
                        input_paused=True,automatic_resend=False,**signature)
                    save(out/'internal-error-last.json',dict(**signature,at=time.time(),
                        game_id=(data or {}).get('game_id'),turn=(data or {}).get('turn'),
                        input_paused=True,automatic_resend=False,
                        retained_pending=bool(controller.pending)))
                except Exception:pass
                if controller.halted:
                    report(controller.halted,data);break
                if (internal_errors>=20
                        or (signature['same_repeats']>=6 and time.monotonic()-progress_at>20)):
                    controller.halted=('internal '+signature['type']+' repeated; input paused, '
                                       'durable ledger retained for review')
                    report(controller.halted,data);break
                report('internal error while revalidating; input paused, no retry of uncertain input',data)
                time.sleep(.2);continue
            except KeyboardInterrupt:operator_stopped=True;report('user interrupted');break
            time.sleep(.06 if practical else .15)
    if fast:save(out/'latency.json',fast.metrics.report())
    save(out/'wait-summary.json',dict(seconds_by_category=waits.totals,diagnostic_only=True))
    # Closed in _finalize_run, after the event log and session archives are
    # already on disk; closing here would delay the archive finalization.
    if practical and args.enable_input and not game_completed and not operator_stopped and not controller.halted:
        report('run time limit reached before game completion; no successful game result',data,snapshot)
        return 2
    return 2 if controller.halted or closed_unresolved else 0


def calibrate(root,args):
    """Read-only capture, invoked later by the human. Never approves itself."""
    from PIL import ImageDraw
    from .logs import LogReader
    from .windows import NativeBackend
    from .vision import Geometry,crop_box
    profile=load(args.profile);out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise Blocked('calibration output must be a new/empty folder')
    out.mkdir(parents=True,exist_ok=True)
    reader=LogReader(args.game_dir,args.player_id)
    for _ in range(1000):
        data=reader.poll()
        if reader.caught_up:break
    else:raise Blocked('log history exceeds calibration limit')
    if not reader.history_complete:raise Blocked('missing CREATE_GAME')
    backend=NativeBackend(Path(args.game_dir)/'Hearthstone.exe',out/'STOP')
    image,window=backend.capture();image.save(out/'capture.png')
    pending=1 if args.scene=='hero' else 2 if args.scene=='discover' else 0
    overlay=image.copy();draw=ImageDraw.Draw(overlay)
    candidate=copy.deepcopy(profile);candidate['review_state']='needs_review'
    points,_=Geometry(candidate).locations(dict(client=data,view=dict(pending=pending)))
    for label,(x,y) in points.items():
        x*=image.width;y*=image.height;draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2);draw.text((x+7,y),label,fill='yellow')
    overlay.save(out/'overlay.png');save(out/'client.json',data)
    # Reuse only on a matching aspect ratio; never stretch 16:10 into 16:9.
    if abs(image.width/image.height-profile['aspect'])>.005:raise Blocked('reference aspect differs; add a new layout')
    exe_sha=hashlib.sha256(Path(window.exe).read_bytes()).hexdigest()
    same=(profile.get('executable_sha256')==exe_sha and profile.get('allowed_builds')==[data['build']]
          and profile.get('client_size')==list(image.size) and profile.get('dpi')==window.dpi)
    if not same:
        profile['scene_guards']={};profile['signals']={};profile['reviewed_layouts']=[]
        profile['selection_guards']={};profile['branch_guards']={};profile.pop('magnetize_drop',None)
    profile.update(review_state='needs_review',allowed_builds=[data['build']],client_size=list(image.size),dpi=window.dpi,
                   executable_sha256=exe_sha)
    # Existing templates are copied by hash, never resolved through a stale
    # relative folder after another calibration.
    specs=list(profile.get('scene_guards',{}).values())+list(profile.get('signals',{}).values())
    specs += [c for group in ('selection_guards','branch_guards') for g in profile.get(group,{}).values() for c in g.get('classes',[])]
    for spec in specs:
        source=(Path(args.profile).resolve().parent/spec['template']).resolve()
        target=(out/spec['template']).resolve()
        if not source.is_relative_to(Path(args.profile).resolve().parent) or not target.is_relative_to(out):
            raise Blocked('calibration template path escapes its profile directory')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['sha256']:raise Blocked('prior calibration template changed')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    keys=set(profile.get('reviewed_layouts',[]));family=args.scene
    for z in ('board','shop','hand','choices'):
        if (family=='recruit' and z=='choices') or (family!='recruit' and z!='choices'):continue
        zf=family if z=='choices' else 'recruit';count=len(data[z])
        if any(x['family']==zf and x['zone']==z and x['count']==count for x in profile['layouts']):
            keys.add(zf+'/'+z+'/'+str(count))
    for layout in profile['layouts']:
        if layout['family']==family and layout['zone']=='hud':keys.add(family+'/hud/'+str(layout['count']))
    profile['reviewed_layouts']=sorted(keys)
    profile.setdefault('scene_guards',{})
    rect=profile['calibration_guard_rects'][args.scene]
    name='guard-'+args.scene+'.png';image.crop(crop_box(rect,image.size)).save(out/name)
    profile['scene_guards'][args.scene]=dict(rect=rect,template=name,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),max_error=.035)
    save(out/'profile.json',profile)
    print('Read-only capture saved. Inspect overlay and guard; review is still required:',out/'profile.json')
    return 0
