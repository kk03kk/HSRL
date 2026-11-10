"""Explicit operator-started activation and bounded matchmaking recovery.

Import is inert. Tests inject windows/OCR/logs and never acquire the real client.
"""
import hashlib
import json
from pathlib import Path
import time
import re
from .contracts import Blocked,InputNotSent,FocusUnavailable
from .lobby import LobbyFlow
from .runtime import load,save


def no_protocol_bytes(power):
    """255790 在客户端启动时就会创建 Power.log；0 字节文件没有任何协议历史。"""
    if not power.exists():return True
    return power.stat().st_size==0


def prior_session_input(root,game_dir,current):
    """只允许新客户端、首局前大厅隔离旧会话账本；不猜旧动作结果。"""
    path=Path(root)/'artifacts/runs/live-state/inflight.json'
    if not path.exists() or current.get('power_status')!='awaiting_first_game' or current['active'] or current['rebuilding']:return None
    raw=path.read_bytes();record=json.loads(raw.decode('utf8'));gid=record['decision']['game_id']
    match=re.match(r'^(Hearthstone_[0-9_]+):game@',gid)
    if not match:return None
    session=Path(current['signature'][0]).parent;power=session/'Power.log'
    sessions=[p for p in (Path(game_dir)/'Logs').glob('Hearthstone_*') if p.is_dir()]
    if (not sessions or session!=max(sessions,key=lambda p:p.name) or session.name<=match[1]
            or str(power)!=current['signature'][0] or not no_protocol_bytes(power)):return None
    loading=session/'LoadingScreen.log'
    if not loading.is_file():return None
    scenes=re.findall(r'LoadingScreen.OnSceneLoaded\(\) - prevMode=\w+ currMode=(\w+)',loading.read_text(encoding='utf-8-sig'))
    if not scenes or scenes[-1]!='BACON':return None
    return dict(sha256=hashlib.sha256(raw).hexdigest(),old_game_id=gid,old_session=match[1],new_session=session.name,
        new_power='absent' if not power.exists() else 'empty')


def retire_prior_session_input(root,game_dir,current,expected,out):
    proof=prior_session_input(root,game_dir,current)
    if proof!=expected:raise Blocked('prior-session input evidence changed; no matchmaking input sent')
    path=Path(root)/'artifacts/runs/live-state/inflight.json'
    target=path.parent/'retired-session-inputs'/(proof['sha256']+'.json')
    target.parent.mkdir(exist_ok=True)
    if target.exists():raise Blocked('prior-session archive already exists; refusing overwrite')
    # 同磁盘原子改名保留完整根动作、父子栈与阶段标记，不重放或记成功。
    path.rename(target)
    save(Path(out)/'startup/retired-input.json',dict(**proof,archive=str(target),
        outcome='abandoned_prior_client_session_unknown_not_applied',basis='new client session, absent/zero-byte Power.log, logged BACON scene and two fresh lobby frames'))
    return target


def completed_session_input(root,current):
    """Exact same-game termination permits archival, never success attribution."""
    path=Path(root)/'artifacts/runs/live-state/inflight.json';closed=current.get('closed_game')
    if not path.exists() or not closed or not current.get('completed') or current.get('active') or current.get('rebuilding'):return None
    raw=path.read_bytes();record=json.loads(raw.decode('utf8'))
    for row in [record]+record.get('parents',[]):
        if (row['decision']['game_id']!=closed['game_id']
                or row['before']['client'].get('local_player_id')!=closed['local_player_id']):return None
    if closed.get('proof') not in ('WON','LOST','TIED','CONCEDED','4','5','6','8','COMPLETE','3'):return None
    return dict(sha256=hashlib.sha256(raw).hexdigest(),closed_game=dict(closed))


def retire_completed_session_input(root,current,expected,out):
    if completed_session_input(root,current)!=expected:raise Blocked('completed-game input evidence changed; no lobby input sent')
    path=Path(root)/'artifacts/runs/live-state/inflight.json'
    target=path.parent/'retired-completed-inputs'/(expected['sha256']+'.json');target.parent.mkdir(exist_ok=True)
    if target.exists():raise Blocked('completed-game archive already exists; refusing overwrite')
    path.rename(target)
    save(Path(out)/'startup/retired-completed-input.json',dict(**expected,archive=str(target),
         outcome='unresolved_at_authoritative_game_end_no_retry_not_applied'))
    return target


def prepare_input(root,args,out):
    from .windows import NativeBackend
    # Reject malformed launch settings before doing any foreground activation.
    LobbyFlow(max_restarts=getattr(args,'queue_retries',3))
    timeout=getattr(args,'queue_timeout',240.)
    if not isinstance(timeout,(float,int)) or not 10<=timeout<=1800:raise Blocked('queue timeout must be 10..1800 seconds')
    profile=load(args.profile);exe=Path(args.game_dir)/'Hearthstone.exe'
    if profile.get('review_state')!='reviewed':raise Blocked('autoplay profile has not been reviewed')
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=profile.get('executable_sha256'):
        raise Blocked('Hearthstone executable differs from calibrated build')
    startup=out/'startup';startup.mkdir(parents=True,exist_ok=True);args.queue_prepared=True
    backend=NativeBackend(exe,getattr(args,'stop_file',out/'STOP'),enable_input=True,timing=getattr(args,'timing','balanced'))
    activation=dict(status='blocked',reason='not attempted')
    try:
        started=time.monotonic()
        if not getattr(args,'no_auto_focus',False):
            window=backend.activate()
            activation=dict(status='activated',hwnd=window.hwnd,pid=window.pid,
                process_created=window.created,executable=window.exe,rect=window.rect,dpi=window.dpi,
                activation_seconds=round(time.monotonic()-started,3))
        else:activation=dict(status='operator_requested_manual_focus',activation_seconds=0.)
        save(startup/'activation.json',activation)
    except Blocked as exc:
        save(startup/'activation.json',dict(activation,status='blocked',reason=str(exc)));raise
    args.preloaded_backend=backend
    # The OCR engine (~1.0 s) and torch/checkpoint (~1.5 s) are independent:
    # build the engine in a background thread so both finish while the operator
    # only waits once. A failure surfaces at first use, exactly as before.
    from .vision import AsyncOCR
    args.prewarmed_ocr=AsyncOCR()
    # Torch and the checkpoint load only after the verified window is foreground.
    # The operator sees the switch at once rather than a silent import pause.
    from .policy import ModelPolicy,configure_threads
    configure_threads(getattr(args,'inference_threads',None))
    model_started=time.monotonic()
    if getattr(args,'preloaded_policy',None) is None:
        args.preloaded_policy=ModelPolicy(args.checkpoint,device=args.device)
    timings=dict(activation_seconds=activation.get('activation_seconds'),
                 model_load_seconds=round(time.monotonic()-model_started,3),
                 preparation_seconds=round(time.monotonic()-started,3))
    save(startup/'timings.json',timings)
    return backend,profile


def lobby_ocr_rows(ocr,image):
    """One calibrated lobby pass at half width; full width only if unclassified.

    The measured cost on this machine is 1.55 s at 1280 px versus 1.10 s at
    1024 px for identical lobby rows. An unrecognisable screen (dialog text is
    small) still gets the full-width pass before any conclusion is drawn.
    """
    from .lobby import screen_hint
    previous=getattr(ocr,'max_width',None)
    try:
        try:ocr.max_width=min(previous or 1024,1024)
        except Exception:pass
        rows=ocr.locate(image)
        if screen_hint(rows):return rows
        try:ocr.max_width=1280
        except Exception:return rows
        return ocr.locate(image)
    finally:
        try:
            if previous is None:del ocr.max_width
            else:ocr.max_width=previous
        except Exception:pass


def closed_game_proof(reader,data):
    from .lifecycle import terminal_proof
    return terminal_proof(reader.state,data,history_complete=reader.history_complete)


def lobby_context(reader,game_dir):
    from .logs import latest_game_info
    sessions=[p for p in (Path(game_dir)/'Logs').glob('Hearthstone_*') if p.is_dir()]
    if not sessions:raise Blocked('no Hearthstone log session; enable client logs before testing')
    session=max(sessions,key=lambda p:p.name);power=session/'Power.log'
    if not power.exists():
        if reader.session==session and 'Power.log' in reader.tails:
            raise Blocked('Power.log disappeared from the current session; matchmaking forbidden')
        # A freshly launched client creates Power.log lazily, at its first
        # match. This is permission to inspect the lobby only: LobbyFlow still
        # requires its two-frame button proof, and gameplay cannot start here.
        # Never borrow a prior process/session's Power.log or create an empty
        # replacement to satisfy the strict gameplay reader.
        return dict(active=False,rebuilding=False,completed=False,power_status='awaiting_first_game',
                    signature=(str(power),'absent'))
    info=latest_game_info(power)
    stat=power.stat();signature=(str(power),stat.st_dev,stat.st_ino,info['offset'],info['complete'])
    data=reader.poll()
    if not reader.caught_up or reader.session!=session:
        return dict(active=False,rebuilding=True,completed=False,power_status='present',signature=signature)
    if reader.spectating:raise Blocked('spectator session cannot be taken over')
    empty=power.stat()
    if empty.st_size==0 and not reader.history_complete:
        # 255790 启动即创建 Power.log，但首局开始前不写入任何协议字节。0 字节
        # 与文件不存在同属“尚未有对局”：只允许大厅检查，LobbyFlow 仍需双帧
        # 大厅开始按钮；任何字节写入都会改变签名并撤销此状态。
        return dict(active=False,rebuilding=False,completed=False,power_status='awaiting_first_game',
                    signature=(str(power),'empty',empty.st_dev,empty.st_ino))
    active=reader.history_complete and reader.connection=='connected' and data['mode']=='GT_BATTLEGROUNDS' and data['phase'] in ('choice','recruit','combat')
    if data['mode'] not in ('GT_BATTLEGROUNDS','unknown') and data['phase'] not in ('menu','unknown'):
        raise Blocked('another game mode is active; matchmaking forbidden')
    closed=closed_game_proof(reader,data)
    return dict(active=active and not closed,rebuilding=False,completed=bool(closed),closed_game=closed,power_status='present',signature=signature)


def control_texts(kind):
    """OCR texts that identify the control a committed lobby action targets."""
    from .lobby import STARTS
    return {'start':STARTS,
            'enter_lobby':frozenset(('酒馆战棋',)),
            'dismiss_error':frozenset(('确定','确认')),
            'dismiss_result':frozenset(('点击继续','点击以继续','点击屏幕继续','点击任意位置继续','点击任意位置以继续'))}[kind]


def reverify_control(image,ocr,action,tolerance=.01):
    """Re-recognise the authorised control on a frame captured just before the click.

    The OCR pass that produced ``action`` is 0.3-2.8 s stale by the time the
    button may be pressed (OCR + JPEG evidence + commit). Without this check a
    message that closed, a queue that started, or a dialog that appeared in that
    window would still receive a blind click. Returns the fresh point, or None
    when the control is gone or has moved outside ``tolerance``.
    """
    from .lobby import normalized,trusted_rows,RESULT_BANNER
    wanted=control_texts(action.kind)
    for row in trusted_rows(lobby_ocr_rows(ocr,image)):
        text=normalized(row.get('text',''))
        # The end-of-game banner has no 点击继续 label of its own; the banner row
        # is the dismiss anchor, so it is accepted as its own identity.
        if text not in wanted and not (action.kind=='dismiss_result' and RESULT_BANNER.search(text)):
            continue
        if max(abs(a-b) for a,b in zip(row['point'],action.point))<=tolerance:
            return tuple(row['point'])
    return None


def wait_for_match(root,args,backend,profile,out,*,clock=time.monotonic,sleep=time.sleep,context=None,ocr=None):
    from .logs import LogReader
    from .vision import OCR
    reader=LogReader(args.game_dir,args.player_id);reader.start_latest=True
    context=context or (lambda:lobby_context(reader,args.game_dir))
    flow=LobbyFlow(max_restarts=getattr(args,'queue_retries',3))
    timeout=getattr(args,'queue_timeout',240.)
    if not isinstance(timeout,(float,int)) or not 10<=timeout<=1800:raise Blocked('queue timeout must be 10..1800 seconds')
    started=clock();startup=out/'startup';startup.mkdir(parents=True,exist_ok=True)
    image=None;window=None;rows=[];last_status=None;last_power_status=None;current={}
    focus_paused=False;inputs_sent=0;next_progress=5.
    clicked_at=None
    overlay_grace=float(getattr(args,'lobby_overlay_grace',20.))
    idle_since=None;overlay_reported=False
    retirement_flow=LobbyFlow()
    def event(kind,**fields):
        with (startup/'events.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(at=time.time(),event=kind,state=flow.state,restarts=flow.restarts,**fields),ensure_ascii=False)+'\n')
    def pause_focus(exc):
        nonlocal focus_paused
        flow.proof=None
        retirement_flow.proof=None
        if not focus_paused:
            event('focus_paused',reason=str(exc),details=getattr(exc,'details',{}))
            print('Lobby: foreground unavailable; input paused. Waiting for Hearthstone foreground without re-sending queue input.',flush=True)
        focus_paused=True
    try:
        while clock()-started<timeout:
            if backend.stopped():raise Blocked('autoplay stopped before queue')
            current=context()
            power_status=current.get('power_status','unknown')
            if power_status!=last_power_status:
                event('power_status',status=power_status)
                if power_status=='awaiting_first_game':
                    print('Lobby: no gameplay data yet (Power.log absent or empty); verifying lobby before matchmaking.',flush=True)
                last_power_status=power_status
            if current['rebuilding']:sleep(.05);continue
            if current['active']:
                elapsed=round(clock()-started,1)
                save(startup/'queue-report.json',dict(status='game_ready',restarts=flow.restarts,actions=flow.actions,
                    inputs_sent=inputs_sent,match_started_seconds=elapsed,
                    start_clicked_seconds=None if clicked_at is None else round(clicked_at-started,1)))
                print('Lobby: match running after %.1f s (%d lobby input(s)).'%(elapsed,inputs_sent),flush=True)
                event('game_ready',seconds=elapsed)
                args.lobby_ocr=ocr;return
            try:image,window=backend.capture()
            except FocusUnavailable as exc:
                pause_focus(exc);sleep(.25);continue
            if focus_paused:
                focus_paused=False;flow.proof=None
                event('focus_resumed');print('Lobby: foreground restored; rebuilding fresh UI proof.',flush=True)
            if list(image.size)!=profile['client_size'] or window.dpi!=profile['dpi']:raise Blocked('lobby size/DPI changed')
            if ocr is None:
                prewarmed=getattr(args,'prewarmed_ocr',None)
                ocr=prewarmed.get() if prewarmed is not None else OCR()
            scan_started=clock();rows=lobby_ocr_rows(ocr,image)
            event('lobby_ocr',seconds=round(clock()-scan_started,3),rows=len(rows))
            pending=(Path(root)/'artifacts/runs/live-state/inflight.json').exists()
            if pending:
                proof=prior_session_input(root,args.game_dir,current)
                closed=completed_session_input(root,current)
                if proof or closed:
                    proposal=retirement_flow.inspect(rows,window.token(),clock(),pending_gameplay=False,completed_gameplay=bool(closed))
                    if proposal and proposal.kind not in (('start','dismiss_result') if closed else ('start',)):
                        raise Blocked('unresolved prior input requires verified lobby or same-game result; popup input forbidden')
                    if proposal:
                        fresh=context()
                        if fresh!=current:retirement_flow.proof=None;sleep(.05);continue
                        target=retire_completed_session_input(root,current,closed,out) if closed else retire_prior_session_input(root,args.game_dir,current,proof,out)
                        event('completed_game_input_archived' if closed else 'prior_session_input_archived',archive=str(target),applied=False)
                        print('Lobby: ended/prior-session input archived as unresolved; proceeding without replay.',flush=True)
                        pending=False
                    else:sleep(.25);continue
            action=flow.inspect(rows,window.token(),clock(),pending_gameplay=pending,completed_gameplay=current.get('completed',False))
            # No calibrated lobby control, no popup, no match: the client is on
            # an overlay we cannot name (shop/quest/collection). A live probe on
            # 2026-11-10 showed Escape in this lobby opens 游戏菜单 and a second
            # press opened 商店 — an uncalibrated key is NOT a safe dismissal.
            # Do not guess: record the screen, then fail this match fast as an
            # environment (S3) incident instead of burning the whole timeout.
            if action is None and not pending and not current.get('active') and not current.get('completed'):
                if flow.state in ('idle','waiting_lobby'):
                    if idle_since is None:idle_since=clock()
                    elif clock()-idle_since>overlay_grace and not overlay_reported:
                        overlay_reported=True
                        from .lobby import normalized,trusted_rows
                        screen=[normalized(r['text']) for r in trusted_rows(rows)][:12]
                        event('overlay_suspected',seconds=round(clock()-started,1),state=flow.state,screen=screen)
                        print('Lobby: unclassified screen for %.0f s (no queue control, no popup); '
                              'recording evidence and failing this match as an environment incident.'
                              %(clock()-idle_since),flush=True)
                        raise Blocked('lobby overlay suspected: no calibrated queue control on screen; '
                                      'no uncalibrated key or click is sent')
                else:idle_since=None
            else:idle_since=None
            status=(flow.state,flow.restarts)
            if status!=last_status:
                event('state');print('Lobby:',flow.state,'restarts='+str(flow.restarts),flush=True);last_status=status
            if action:
                # Recognition may take time. A newly created game must win over
                # a stale lobby/error frame, even when no button was sent yet.
                fresh=context()
                if fresh['active'] or fresh['rebuilding'] or fresh['signature']!=current['signature']:
                    flow.proof=None;sleep(.05);continue
                # The authorising OCR pass is 0.3-2.8 s stale (OCR + JPEG evidence +
                # commit). Re-recognise the same control on a frame captured now;
                # if it moved or vanished, drop the proof and re-inspect instead of
                # sending a blind click into whatever replaced it.
                try:check_image,check_window=backend.capture()
                except FocusUnavailable as exc:
                    pause_focus(exc);sleep(.25);continue
                if check_window.token()!=window.token():
                    flow.proof=None;sleep(.05);continue
                if reverify_control(check_image,ocr,action) is None:
                    event('control_not_reverified',action=action.kind,point=list(action.point))
                    print('Lobby: the lobby control moved or disappeared before the click; re-inspecting without input.',flush=True)
                    flow.proof=None;sleep(.1);continue
                index=flow.actions+1;name=f'{index:03d}-{action.kind}'
                # The fresh OCR pass above is the authorization. Never encode
                # an image on the press path: a trace writer may take the JPEG
                # after the click has been sent.
                evidence=dict(rows=rows,action=dict(kind=action.kind,point=action.point),
                    window_token=window.token(),context=current,screenshot=name+'.jpg')
                writer=getattr(args,'live_trace',None)
                if writer is not None:
                    writer.frame(startup/(name+'.jpg'),image,evidence,quality=90)
                else:
                    image.convert('RGB').save(startup/(name+'.jpg'),quality=90)
                    save(startup/(name+'.json'),evidence)
                rollback={k:getattr(flow,k) for k in ('state','restarts','sent_at','needs_restart')}
                flow.commit(action,clock());event('dispatch_intent',action=action.kind,point=action.point)
                previous_guard=getattr(backend,'pre_press',None)
                def before_press(snapshot):
                    latest=context()
                    if latest['active'] or latest['rebuilding'] or latest['signature']!=current['signature']:
                        raise Blocked('game context changed before lobby mouse down')
                    if action.kind=='dismiss_result' and (not latest.get('completed') or latest.get('closed_game')!=current.get('closed_game')):
                        raise Blocked('completed-game context changed before result dismissal')
                backend.pre_press=before_press
                try:
                    backend.execute([('click',list(action.point)),('move_pointer',[.96,.91])],dict(visual=dict(window_token=window.token())))
                except InputNotSent as exc:
                    event('cancelled_before_press',reason=str(exc),action=action.kind)
                    cause=exc.__cause__
                    if isinstance(cause,FocusUnavailable):
                        # Known unsent: restore logical state but retain the
                        # monotonic attempt index and its evidence files.
                        for key,value in rollback.items():setattr(flow,key,value)
                        pause_focus(cause);sleep(.25);continue
                    latest=context()
                    if not (latest['active'] or latest['rebuilding'] or latest['signature']!=current['signature']):raise
                    flow.proof=None;sleep(.05);continue
                except FocusUnavailable as exc:
                    # execute() only exposes this directly after a button down.
                    # Its outcome is uncertain, unlike a failed capture.
                    raise Blocked('lobby focus lost after possible button down; outcome unknown, no automatic resend') from exc
                finally:backend.pre_press=previous_guard
                inputs_sent+=1
                if action.kind=='start':
                    clicked_at=clock()
                    print('Lobby: start clicked after %.1f s; the client now owns matchmaking.'
                          %(clicked_at-started),flush=True)
                event('input_sent',action=action.kind,seconds=round(clock()-started,1))
            elapsed=clock()-started
            if elapsed>=next_progress:
                next_progress=elapsed+5.
                if clicked_at is None:
                    print('Lobby: waiting for a usable lobby button (%.0f s, state=%s)'%(elapsed,flow.state),flush=True)
                else:
                    print('Lobby: client matchmaking in progress (%.0f s since start click; no script stall)'
                          %(elapsed-(clicked_at-started)),flush=True)
                event('queue_wait',seconds=round(elapsed,1),inputs_sent=inputs_sent)
            sleep(.25 if flow.state in ('idle','error_proof','dismiss_pending','waiting_lobby') else .75)
        if focus_paused:raise Blocked('queue timeout while foreground unavailable; input remained paused, no focus stealing or resend')
        if current.get('power_status')=='awaiting_first_game':
            if len(current.get('signature',()))>1 and current['signature'][1]=='empty':
                raise Blocked('Power.log stayed empty within bounded queue timeout; gameplay was not started; check the queue click and client logging')
            raise Blocked('Power.log was not created within bounded queue timeout; gameplay was not started; check client logging if a match is visible')
        raise Blocked('match did not start within bounded queue timeout')
    except BaseException as exc:
        if image is not None:
            image.convert('RGB').save(startup/'blocked-last.jpg',quality=90)
            save(startup/'blocked-last.json',dict(rows=rows,reason=str(exc),state=flow.state))
        save(startup/'queue-report.json',dict(status='stopped',reason=str(exc),restarts=flow.restarts,actions=flow.actions,
                                            power_status=current.get('power_status','unknown'),inputs_sent=inputs_sent,focus_paused=focus_paused,
                                            start_clicked_seconds=None if clicked_at is None else round(clicked_at-started,1),
                                            elapsed_seconds=round(clock()-started,1)))
        event('stopped',reason=str(exc))
        try:
            from .session_archive import archive
            archive(args.game_dir,startup/'log-archive',include_shared_events=False)
        except Exception as archive_error:save(startup/'archive-error.json',dict(reason=str(archive_error)))
        raise
