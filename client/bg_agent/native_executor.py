"""Windows input bound to an exact fixture or explicitly enabled live provider."""
from dataclasses import asdict
from contextlib import nullcontext
import threading,time,sys
from pathlib import Path
from .execution import Dispatch,Delivery
from .native_bridge import NativeInputPlanner
from .native_input import WindowsInput
from .interaction_timing import SAFE_TIMING

class NativeExecutor:
    def __init__(self,provider,audit,*,stopped=lambda:False,guard=lambda:None,before_input=None,during_drag=None,gesture_timeout=.6,planner=None,allow_live_client=False,timing=None,gesture_clock='precise'):
        if gesture_clock not in ('event','precise'):raise ValueError('unknown_gesture_clock')
        self.provider,self.audit,self.stopped,self.guard=provider,audit,stopped,guard
        b=provider.binding
        self.live=bool(allow_live_client and getattr(provider,'live_client',False))
        if self.live and Path(b.executable).name.casefold()!='hearthstone.exe':raise ValueError('live_target_must_be_hearthstone')
        if (Path(b.executable).name.casefold()=='hearthstone.exe' and not self.live) or not b.nonce or b.pid<=0:
            raise ValueError('native_executor_fixture_only; Hearthstone is prohibited')
        self.before_input,self.during_drag=before_input,during_drag
        self.timing=timing or SAFE_TIMING
        self.planner=planner or NativeInputPlanner();self.transport=WindowsInput(audit);self.cancel=threading.Event()
        from .gesture_pacing import GesturePacer
        self.pacer=GesturePacer(self.cancel,enabled=self.live and gesture_clock=='precise')
        self.gesture_timeout=gesture_timeout;self.lock=threading.Lock();self.plan_ms=[];self.dispatch_ms=[]

    def wait(self,seconds):
        metrics=getattr(self.provider,'chain_metrics',None)
        with metrics.measure('gesture_pacing') if metrics else nullcontext():
            return getattr(self,'pacer',self.cancel).wait(seconds)

    def close(self):
        self.pacer.close()

    def _prepare_press(self,action,obs,plan):
        """尚未按下：在同一意图内刷新过期证据，不让主循环重新选策略。"""
        from dataclasses import replace
        from .decision_ahead import decision_key
        from .native_bridge import NativePlan
        deadline=time.monotonic()+8;attempt=0
        original_window=plan.window_token
        expected=decision_key(obs)
        while True:
            self.guard()
            if attempt and time.monotonic()>=deadline:
                raise RuntimeError('input_preparation_timeout:prepress_refresh_budget')
            try:
                self.check(plan)
                park=getattr(self.provider,'park_neutral',None)
                if park and not park():raise RuntimeError('prepress_neutral_position_unavailable')
                prepress=getattr(self.provider,'before_mouse_down',None)
                checked=prepress(plan) if prepress else None
                if isinstance(checked,NativePlan):plan=checked
                self.check(plan)
                if attempt and time.monotonic()>=deadline:
                    raise RuntimeError('input_preparation_timeout:prepress_refresh_budget')
                return plan
            except (ValueError,TimeoutError) as exc:
                reason=str(exc)
                if reason not in ('prepress_capture_expiring','prepress_plan_expired',
                        'native_plan_expired','native_observation_changed'):raise
                # 此函数没有任何按下操作；发生按下后的异常绝不能走这里。
                if self.transport.mouse_down:raise RuntimeError('prepress_refresh_with_button_held') from exc
                self.guard()
                if self.stopped() or self.cancel.is_set() or self.transport.pressed(0x77):
                    raise InterruptedError('human_stop') from exc
                if time.monotonic()>=deadline:raise RuntimeError('input_preparation_timeout:'+reason) from exc
                self.provider.invalidate_readiness('prepress_refresh_same_intent')
                self.provider.cache_until=0
                capture=(getattr(self.provider,'last_row',None) or {}).get('capture',{})
                stamp=capture.get('end')
                if (reason=='prepress_capture_expiring' and isinstance(stamp,(int,float))
                        and not capture.get('reused_capture') and not capture.get('stage_context_changed')
                        and 0<=time.time()-stamp<.75):
                    # 刚才的新帧已通过身份/坐标/对象检查，仅手势余量不够。
                    # 只重做独立按下前取证，不再重复完整观测、规划和策略周期。
                    # 临时计划仍绑定这张真实捕获的原750ms上限；未按下，下一轮
                    # 必须再次取新帧并通过完整prepress，不能直接用它发送输入。
                    attempt+=1
                    plan=replace(plan,created_at=time.time(),expires_at=stamp+.75)
                    self.audit.write('prepress_refresh_same_intent',action_key=action.key,reason=reason,
                        attempt=attempt,input_sent=False,selection_preserved=True,scope='prepress_only',
                        remaining_seconds=max(0,deadline-time.monotonic()))
                    continue
                if reason=='native_observation_changed':
                    # 局部视觉未就绪时限速重采，避免快速拒绝形成忙循环。
                    self.cancel.wait(min(.04,max(0,deadline-time.monotonic())))
                current=self.provider.observe()
                # 不把ready的暂时变化当作策略变化；未就绪时仍由input_plan取证等待。
                if decision_key(replace(current,ready=obs.ready))!=expected or action not in current.actions:
                    raise ValueError('native_observation_changed') from exc
                if self.provider.last_window.token!=original_window:
                    raise RuntimeError('live_window_layout_or_focus_changed') from exc
                if time.monotonic()>=deadline:raise RuntimeError('input_preparation_timeout:'+reason) from exc
                attempt+=1
                self.audit.write('prepress_refresh_same_intent',action_key=action.key,reason=reason,
                    attempt=attempt,input_sent=False,selection_preserved=True,
                    scope='full_preparation',
                    remaining_seconds=max(0,deadline-time.monotonic()))
                plan=self.provider.input_plan(action,current)
                if plan.action_key!=action.key or plan.window_token!=original_window:
                    raise RuntimeError('native_plan_action_binding_mismatch')
                if time.monotonic()>=deadline:raise RuntimeError('input_preparation_timeout:'+reason)
                self.audit.write('native_refreshed_input_plan',action_key=action.key,plan=asdict(plan),
                    input_sent=False,selection_preserved=True)
    def recover_matchmaking(self,before,after):
        if not self.live:return False
        if not self.lock.acquire(False):return False
        try:
            from .queue_recovery import recover
            return recover(self.provider,self,before,after)
        finally:
            original_error=sys.exc_info()[1]
            try:
                # Recovery runs outside execute(); always lift a held button
                # if focus changes or F8 interrupts this separate UI gesture.
                if self.transport.mouse_down:self.transport.release()
            except Exception as cleanup_error:
                if original_error is None:raise
                original_error.cleanup_failure=str(cleanup_error)
            finally:self.lock.release()
    def check(self,plan,deadline=None,*,motion=False):
        if self.stopped() or self.cancel.is_set():raise InterruptedError('human_stop')
        if self.transport.pressed(0x77):raise InterruptedError('emergency_stop_F8')
        if time.time()>plan.expires_at:raise TimeoutError('native_plan_expired')
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('native_gesture_timeout')
        if self.live:
            if motion and callable(getattr(self.provider,'check_motion',None)):return self.provider.check_motion(plan)
            return self.provider.check_input(plan)
        b=self.provider.binding;w=self.provider.windows.snapshot(b.hwnd)
        if w.pid!=b.pid or w.process_created!=b.process_created or w.title!='BG Native Fixture '+b.nonce:raise RuntimeError('native_window_identity_changed')
        if w.blockers:raise RuntimeError('native_window_blocked:'+','.join(w.blockers))
        if w.token!=plan.window_token:raise RuntimeError('native_window_token_changed')
        for point in (plan.source,plan.destination):
            if point:
                if not w.client.x<=point[0]<w.client.x+w.client.width or not w.client.y<=point[1]<w.client.y+w.client.height:raise RuntimeError('planned_point_outside_client')
                if not self.provider.windows.owns_point(b.hwnd,*point):raise RuntimeError('planned_point_occluded')
        p=self.provider.packet()
        if p['layout_version']!=plan.layout_version or p['observation']['revision']!=plan.revision or not p['ready']:
            raise RuntimeError('native_ui_or_revision_changed')
        if time.time()-p['published_at']>.5:raise RuntimeError('native_snapshot_stale')
        return p
    def execute(self,action,revision):
        if not self.lock.acquire(False):return Dispatch(Delivery.NOT_SENT,'native_executor_busy')
        dispatched=False;start=time.perf_counter();result=None
        try:
            self.guard()
            if self.live:self.provider.transport=self.transport
            revalidate=getattr(self.provider,'revalidate_observation',None)
            bound=self.provider.last_obs if self.live else None
            obs=revalidate(bound) if self.live and revalidate and bound is not None else self.provider.observe()
            if obs.revision!=revision:
                from .decision_ahead import decision_key
                if not (self.live and bound is not None and bound.revision==revision
                        and decision_key(bound)==decision_key(obs)):
                    return Dispatch(Delivery.NOT_SENT,'native_observation_changed')
                revision=obs.revision
            if self.live and action not in obs.actions:return Dispatch(Delivery.NOT_SENT,'live_action_not_advertised')
            from .model import Kind
            if self.live and action.kind==Kind.END:
                from .client_protocol import passive_end
                if not passive_end(action,obs):return Dispatch(Delivery.NOT_SENT,'phase_wait_context_invalid')
                metrics=getattr(self.provider,'chain_metrics',None)
                if metrics:metrics.count('passive_end_wait')
                suppressed=[dict(kind=k[2],source=k[3],reason=v) for k,v in
                    getattr(self.provider,'unavailable_actions',{}).items() if k[:2]==(obs.game_id,obs.turn)]
                self.audit.write('client_phase_wait',revision=revision,turn=obs.turn,physical_input=False,
                    unexecuted_actions=suppressed,waiting_after_execution_refusals=bool(suppressed))
                return Dispatch(Delivery.ACCEPTED,'No end-turn control is calibrated; waiting for logged phase transition')
            t=time.perf_counter();plan=self.provider.input_plan(action,obs) if self.live else self.planner.build(action,obs,self.provider.last_packet,self.provider.last_window)
            if self.live and plan.revision!=revision:
                from .decision_ahead import decision_key
                latest=self.provider.last_obs
                if latest is None or latest.revision!=plan.revision or decision_key(latest)!=decision_key(obs):
                    raise RuntimeError('native_plan_action_binding_mismatch')
                revision=plan.revision
            if plan.action_key!=action.key or plan.revision!=revision:raise RuntimeError('native_plan_action_binding_mismatch')
            elapsed=(time.perf_counter()-t)*1000;self.plan_ms.append(elapsed)
            self.audit.write('native_input_plan',plan=asdict(plan),action=asdict(action),planning_ms=elapsed)
            if self.before_input:self.before_input(self,action,plan)
            self.guard()
            if not self.live:self.check(plan)
            button=getattr(plan,'mouse_button','left')
            if button not in ('left','right'):raise ValueError('unknown_mouse_button')
            send_button=self.transport.right_button if button=='right' else self.transport.button
            if self.transport.pressed(1) or button=='right' and self.transport.pressed(2):return Dispatch(Delivery.NOT_SENT,'mouse_button_already_held')
            input_started=time.perf_counter()
            timing=getattr(self,'timing',SAFE_TIMING)
            if self.live:
                from .live_interactions import record_stage
                record_stage(self.provider,'initial','prepared')
            metrics=getattr(self.provider,'chain_metrics',None)
            with metrics.measure('initial_mouse_gesture') if metrics else nullcontext():
                if plan.key:
                    self.check(plan)
                    invalidate=getattr(self.provider,'invalidate_readiness',None)
                    if self.live and invalidate:invalidate('input_started')
                    dispatched=True;self.transport.key(plan.key,True);self.transport.key(plan.key,False)
                else:
                    if self.live:
                        # 耗时的像素复核在中性位置做，避免脚本自己制造悬停遮挡。
                        plan=self._prepare_press(action,obs,plan)
                    self.transport.move(*plan.source);self.check(plan)
                    if self.live:self.wait(timing.hover);self.check(plan)
                    invalidate=getattr(self.provider,'invalidate_readiness',None)
                    if self.live and invalidate:invalidate('input_started')
                    dispatched=True;send_button(True)
                    if plan.destination:
                        if self.live:
                            # 拖拽也需要客户端先采到源位置上的按下；不能同帧移入邻牌。
                            self.wait(timing.click_hold);self.check(plan)
                        deadline=time.monotonic()+self.gesture_timeout
                        for i in range(1,timing.drag_steps+1):
                            if self.during_drag:self.during_drag(self,action,i)
                            if callable(getattr(self.provider,'check_motion',None)):self.check(plan,deadline,motion=True)
                            else:self.check(plan,deadline)
                            p=tuple(a+(b-a)*i/timing.drag_steps for a,b in zip(plan.source,plan.destination))
                            if not self.provider.windows.owns_point(self.provider.binding.hwnd,*p):raise RuntimeError('drag_point_occluded')
                            self.transport.move(*p)
                            self.wait(timing.drag_step) # Motion pacing; state synchronization uses receipts/readiness.
                    elif self.live:
                        # Unity may sample button transitions per rendered frame.
                        # A zero-duration down/up can be invisible to the client.
                        self.wait(timing.click_hold)
                    self.check(plan);send_button(False)
                    if self.live:
                        self.provider.last_input_released_at=time.time()
                        # Unity polls cursor position and button state per frame.
                        # Keep the release location through the following frame;
                        # immediately parking the cursor can cancel a valid click.
                        self.wait(timing.release_hold)
            if self.live:
                record_stage(self.provider,'initial','sent')
                self.provider.finish_input(self,action,plan)
                self.provider.park_cursor(self)
            dispatch_ms=(time.perf_counter()-input_started)*1000;self.dispatch_ms.append(dispatch_ms)
            self.audit.write('native_input_timing',elapsed_ms=dispatch_ms,action=action.key,timing_profile=timing.name,
                pacing_mode=getattr(getattr(self,'pacer',None),'mode','event_wait'),
                pacing_fallback=getattr(getattr(self,'pacer',None),'failure',None))
            if self.live:
                # OS dispatch is not a game acknowledgement. AgentController
                # now waits for the independent semantic Verifier.
                result=Dispatch(Delivery.ACCEPTED,'Windows input sent; game effect pending log verification')
                self.provider.input_sent(action)
            end=time.monotonic()+.7
            while result is None and time.monotonic()<end:
                if self.stopped():raise InterruptedError('human_stop')
                p=self.provider.packet();receipt=p['last_delivery']
                if p['receipt_serial']>plan.receipt_serial:
                    if receipt['action_key']!=action.key or receipt['revision']!=revision:raise RuntimeError('native_receipt_mismatch')
                    result=Dispatch(Delivery(receipt['delivery']),receipt['reason']);break
                self.cancel.wait(.01)
            if result is None:result=Dispatch(Delivery.AMBIGUOUS,'native_input_receipt_timeout')
        except InterruptedError as exc:
            exc.potentially_dispatched=dispatched
            self.audit.write('native_cancelled',reason=str(exc),potentially_dispatched=dispatched)
            raise
        except Exception as exc:
            import traceback
            try:self.audit.write('native_dispatch_error',action_key=action.key,potentially_dispatched=dispatched,
                error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
            except OSError:pass
            result=Dispatch(Delivery.AMBIGUOUS if dispatched else Delivery.NOT_SENT,str(exc))
        finally:
            original_error=sys.exc_info()[1]
            try:
                foreground=int(self.provider.windows.u.GetForegroundWindow() or 0)==self.provider.binding.hwnd
                if self.live:self.transport.release(cancel=self.transport.mouse_down and foreground,right_cancel=True)
                else:self.transport.release(cancel=self.transport.mouse_down and foreground)
                # 拒绝/异常也会发生在移入源对象之后；安全释放后回位，避免留下悬停预览。
                # 人工停止和失焦时不主动移动，成功路径已经完成自己的回位。
                if (self.live and foreground and original_error is None and result is not None
                        and result.delivery in (Delivery.NOT_SENT,Delivery.AMBIGUOUS)
                        and not self.stopped() and not self.cancel.is_set() and not self.transport.pressed(0x77)):
                    park=getattr(self.provider,'park_neutral',None)
                    if park:park()
            except Exception as cleanup_error:
                try:self.audit.write('native_cleanup_failed',reason=str(cleanup_error),potentially_dispatched=dispatched,
                    original_error=str(original_error) if original_error else None)
                except OSError:pass
                if original_error is not None:
                    # Human stop must remain human stop even when release or
                    # its audit fails. The durable stage remains uncertain.
                    original_error.cleanup_failure=str(cleanup_error)
                else:
                    cleanup_error.potentially_dispatched=dispatched
                    raise
            finally:self.lock.release()
        self.audit.write('native_dispatch_result',delivery=str(result.delivery),reason=result.reason,
                         outcome='UNKNOWN' if result.delivery==Delivery.AMBIGUOUS else 'PENDING' if result.delivery==Delivery.ACCEPTED else 'REJECTED',elapsed_ms=(time.perf_counter()-start)*1000)
        return result

class NativeEmergencyStop:
    """F8/STOP monitoring also while waiting, paused or outside execute()."""
    def __init__(self,transport,controller,stop_file):
        self.transport,self.controller,self.stop_file=transport,controller,stop_file
        self.closed=threading.Event();self.thread=None
    def start(self):
        def watch():
            while not self.closed.wait(.01):
                if self.transport.pressed(0x77) or self.stop_file.exists():
                    # The stop flag must win even if disk/audit writes fail.
                    self.controller._stop.set()
                    try:self.controller.audit.write('native_emergency_stop',source='F8' if self.transport.pressed(0x77) else 'STOP file')
                    except OSError:pass
                    return
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
    def close(self):
        self.closed.set()
        if self.thread and self.thread.ident is not None:self.thread.join(timeout=1)
