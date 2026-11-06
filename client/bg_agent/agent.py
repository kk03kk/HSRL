"""Continuous, bounded agent; no UI details, game rules or fixed-delay macros."""
from dataclasses import asdict, dataclass, replace
from contextlib import nullcontext
from enum import StrEnum
import json
from pathlib import Path
import threading
import time
import uuid

from .execution import Delivery, Executor
from .model import Kind, Phase, ObservationProvider, Policy
from .decision_ahead import PolicyPreview,decision_key,snapshot_key,policy_input
from .verification import verify, Status, accepts_processed,action_progress
from .chain_metrics import measured_stage


class Lifecycle(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class AgentDeadline(TimeoutError):
    """Session deadline, distinct from a temporarily unavailable observation."""


@dataclass(frozen=True)
class AgentConfig:
    require_return_to_menu: bool = False
    poll_interval: float = .02
    quiet_period: float = .08
    observation_timeout: float = 3
    action_timeout: float = 3
    phase_timeout: float = 5
    max_observation_age: float = 1
    max_retries: int = 1
    max_actions: int = 200
    max_seconds: float = 120
    require_effect_confirmation: bool = False
    decision_ahead: bool = False
    max_abandoned_actions: int = 12
    #: 一次 WAIT 超时后允许再重采一帧/重映射一次，避免慢动画直接终止整局。
    max_stall_recoveries: int = 1


class AgentController:
    def __init__(self, observations: ObservationProvider, policy: Policy, executor: Executor, audit, *, config=None, stop_file: Path | None = None,
                 verifier=verify, resolver=None,checkpoint=None,resume_pending=None):
        self.observations, self.policy, self.executor = observations, policy, executor
        self.audit, self.config, self.stop_file, self.verifier = audit, config or AgentConfig(), stop_file, verifier
        self.state = Lifecycle.IDLE
        self.resolver = resolver
        self.checkpoint,self.resume_pending=checkpoint,resume_pending
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._run_lock = threading.Lock()
        self._audit_sig = None
        self._started = None
        self.last_action=None
        # 动作间隔/阶段耗时埋点：不参与决策，只用于报告与 timeline。
        self._last_finished_at=None
        self._cycle_started_at=None
        self._decision_at=None
        self._dispatch_at=None
        self._timing=dict(actions=0,cancelled=0,gaps=[],decisions=[],dispatches=[],by_kind={})
        self.report = dict(actions=0, successes=0, failures=0, dispatch_attempts=0, retries=0,
                           rejected_attempts=0, uncertain_dispatches=0, action_counts={}, verification_counts={}, unsupported=[],
                           phases=[], turns=[], reason=None, status=self.state, processed=0, effect_confirmed=0,verified_action_counts={},
                           abandoned_actions=0, abandoned_action_counts={})

    def _state(self, state):
        self.state = state
        self.audit.write("lifecycle", state=state)

    def pause(self):
        if self.state == Lifecycle.RUNNING:
            self._resume.clear()
            self._state(Lifecycle.PAUSED)

    def resume(self):
        if self.state == Lifecycle.PAUSED:
            self._state(Lifecycle.RUNNING)
            self._resume.set()

    def stop(self):
        self._stop.set()
        self._resume.set()
        self.audit.write("stop_requested")

    def _decide(self,observation,now=None):
        resolve_started=time.perf_counter()
        resolution=self.resolver.resolve(observation,now=now) if self.resolver else None
        filtered=resolution.for_policy(observation) if resolution else observation
        resolve_ms=(time.perf_counter()-resolve_started)*1000
        policy_started=time.perf_counter()
        decision=self.policy.decide(policy_input(filtered))
        elapsed=(time.perf_counter()-policy_started)*1000
        metrics=getattr(self.observations,'chain_metrics',None)
        if metrics:
            metrics.sample('decision',elapsed);metrics.sample('legal_resolution',resolve_ms)
        self.audit.write('policy_timing',revision=observation.revision,elapsed_ms=elapsed,resolution_ms=resolve_ms)
        if resolution:
            self.audit.write('legal_resolution',revision=observation.revision,resolution=resolution.to_dict())
            if decision.action.kind!=Kind.WAIT and decision.action not in filtered.actions:
                raise ValueError('policy_returned_action_not_confirmed_legal')
        return decision,resolution

    @measured_stage('cycle_preparation')
    def _prepare_cycle(self,timeout=None):
        """Overlap pure policy inference with read-only client preparation."""
        self._cycle_started_at=time.monotonic()
        cycle_started=time.perf_counter()
        self._guard()
        end=time.monotonic()+min(self.config.observation_timeout,timeout if timeout is not None else float('inf'))
        job=None
        peek=getattr(self.observations,'policy_observation',None)
        preview=getattr(self.policy,'preview',None)
        enabled=bool(self.config.decision_ahead and peek and preview)
        if getattr(self.observations,'execution_retry_pending',lambda:False)():enabled=False
        def start_preview(_=None):
            nonlocal job
            if not enabled or job is not None:return
            semantic=peek()
            if semantic is not None and semantic.ready and semantic.actions:
                resolution=self.resolver.resolve(semantic) if self.resolver else None
                filtered=resolution.for_policy(semantic) if resolution else semantic
                if filtered.actions:
                    job=PolicyPreview(preview,filtered)
                    self.audit.write('policy_preview_started',revision=semantic.revision,
                                     snapshot_key=job.key,candidates=len(filtered.actions))
        start_preview()
        before=self._stable(timeout=max(0,end-time.monotonic()),on_observation=start_preview if enabled else None)
        first_ready=time.monotonic()
        if job and not job.done.is_set():
            while not job.done.is_set():
                if time.monotonic()>=end:raise TimeoutError('policy_preview_deadline')
                self._sleep()
            # A slow model may finish after the earlier UI snapshot expired.
            before=self._stable(timeout=max(0,end-time.monotonic()))
        if job:
            self.audit.write('decision_cycle_ready',revision=before.revision,
                semantic_to_scene_ready_ms=(first_ready-job.started)*1000,
                semantic_to_final_scene_ready_ms=(time.monotonic()-job.started)*1000,
                inference_ms=job.elapsed_ms,action_region_gate_still_required=True)
        self.audit.write('cycle_prepared',revision=before.revision,
            elapsed_ms=(time.perf_counter()-cycle_started)*1000)
        return before,job

    def _cycle_decision(self,before,job):
        retry=getattr(self.observations,'execution_retry',None)
        if retry and job is None:
            decision=retry(before)
            if decision is not None:
                resolution=self.resolver.resolve(before) if self.resolver else None
                filtered=resolution.for_policy(before) if resolution else before
                if decision.revision!=before.revision or decision.action not in filtered.actions:
                    self.audit.write('execution_retry_discarded',reason='not_currently_authorized',
                        revision=before.revision,physical_input=False)
                    return self._decide(before)[0]
                self.audit.write('execution_retry_selected',revision=before.revision,
                    action=asdict(decision.action),new_model_decision=False,new_input_plan_required=True)
                return decision
        if job is None:return self._decide(before)[0]
        # One preview at a time; do not race mutable policy callbacks or normal
        # decide() against it. Stop/session deadlines remain responsive.
        while not job.done.is_set():self._sleep()
        self._guard()
        resolution=self.resolver.resolve(before) if self.resolver else None
        filtered=resolution.for_policy(before) if resolution else before
        decision=job.decision
        reason=job.error
        if not reason and decision_key(filtered)!=job.key:reason='semantic_state_or_candidates_changed'
        if not reason and (decision is None or decision.revision!=job.observation.revision or
                           (decision.action.kind!=Kind.WAIT and decision.action not in filtered.actions)):
            reason='preview_action_invalid'
        # Inference can outlast the age of the UI sample. The controller and
        # executor independently revalidate again before any input.
        self.audit.write('policy_preview_finished',revision=before.revision,
            preview_revision=job.observation.revision,elapsed_ms=job.elapsed_ms,
            elapsed_since_start_ms=(time.monotonic()-job.started)*1000,
            adopted=not bool(reason),discard_reason=reason)
        if reason:return self._decide(before)[0]
        decision=replace(decision,revision=before.revision)
        adopted=getattr(self.policy,'on_decision_adopted',None)
        if adopted:adopted(policy_input(filtered),decision)
        if resolution:self.audit.write('legal_resolution',revision=before.revision,resolution=resolution.to_dict())
        return decision

    def shadow_step(self, planner, observation=None):
        """One read-only planning cycle. Never dispatches or fabricates an ack.

        Shadow supervisors repeat this against human-driven observations, rather
        than feeding unexecuted recommendations into the normal run verifier.
        """
        if self._stop.is_set() or (self.stop_file and self.stop_file.exists()):
            return None
        before=observation if observation is not None else self.observations.observe()
        started=time.perf_counter()
        now=planner.row['capture']['end'] if planner.replay else time.time()
        decision,resolution=self._decide(before,now=now)
        if decision.revision!=before.revision or (decision.action.kind!=Kind.WAIT and decision.action not in before.actions):
            raise ValueError("invalid shadow decision")
        decision_ms=(time.perf_counter()-started)*1000
        decision_at=time.time()
        self.audit.write("shadow_decision",observation=before.to_dict(),decision=asdict(decision),decision_ms=decision_ms)
        plan=planner.build(before,decision)
        if resolution:
            plan['legal_resolution']=resolution.to_dict()
            selected=next((a for a in resolution.actions if a.action==decision.action),None)
            plan['semantic_legality']=str(selected.certainty) if selected else 'unknown'
            plan['ui_interactability']='unknown'  # A log option and a region are not an input probe.
            plan['target_visibility']='inferred_from_frame_components; not an input hit test'
        plan["decision_ms"]=decision_ms
        plan["decision_at"]=decision_at
        self.audit.write("shadow_plan",plan=plan)
        return plan

    def _guard(self):
        while True:
            if self._stop.is_set() or (self.stop_file and self.stop_file.exists()):
                raise InterruptedError("human_stop")
            if time.monotonic()-self._started > self.config.max_seconds:
                raise AgentDeadline("agent_deadline")
            if self._resume.is_set():
                return
            # Pausing prevents new dispatch. Deadlines remain bounded while paused.
            self._stop.wait(self.config.poll_interval)

    def _sleep(self):
        waiter=getattr(self.observations,'wait_for_observation',None)
        if waiter and not self._stop.is_set():waiter(self.config.poll_interval)
        else:self._stop.wait(self.config.poll_interval)
        self._guard()

    @staticmethod
    def _fingerprint(obs):
        from .state_stamp import state_stamp
        return state_stamp(obs)

    def _observe(self,verification_action=None):
        self._guard()
        result_reader=getattr(self.observations,'observe_result',None)
        obs = result_reader(verification_action) if verification_action is not None and result_reader else self.observations.observe()
        sig = self._fingerprint(obs)
        if sig != self._audit_sig:
            from .recording import compact_observation
            self.audit.write("observation", observation=compact_observation(obs).to_dict())
            self._audit_sig = sig
            phase = str(obs.phase)
            if not self.report["phases"] or self.report["phases"][-1] != phase:
                self.report["phases"].append(phase)
            if obs.turn is not None and obs.turn not in self.report["turns"]:
                self.report["turns"].append(obs.turn)
        return obs, sig

    def _stable(self, timeout=None, on_observation=None):
        end = time.monotonic()+min(self.config.observation_timeout, timeout if timeout is not None else float("inf"))
        last, since = None, time.monotonic()
        started=time.monotonic();notified=0.0
        while time.monotonic() < end:
            obs, sig = self._observe()
            if on_observation:on_observation(obs)
            now = time.monotonic()
            if sig != last:
                last, since = sig, now
            fresh = 0 <= time.time()-obs.captured_at <= self.config.max_observation_age
            finished_menu=bool(self.report['verified_action_counts'].get('dismiss') and
                getattr(self.observations,'completion_ready',lambda o:False)(obs))
            if fresh and finished_menu:return obs
            if obs.ready and fresh and (getattr(self.observations,'observation_is_gated',False) or now-since >= self.config.quiet_period):
                return obs
            if now-started>=5 and now-notified>=5:
                # A silent long wait is indistinguishable from a frozen script.
                notify=getattr(self.observations,'stability_stall',None)
                if notify:
                    notified=now
                    try:notify(obs)
                    except Exception:pass
            self._sleep()
        raise TimeoutError("observation_never_stabilized_or_stale")

    def _execute(self, decision, before,depth=0):
        if depth>5:raise RuntimeError('interaction_depth_exceeded')
        aid = uuid.uuid4().hex
        parent_id=self.checkpoint.state['parents'][-1]['action_id'] if self.checkpoint and self.checkpoint.state.get('parents') else None
        metrics=getattr(self.observations,'chain_metrics',None)
        binding=getattr(self.audit,'context',None)
        start=time.perf_counter();completed=False;error=None
        with binding(action_id=aid,parent_action_id=parent_id) if binding else nullcontext():
            with metrics.action_scope() if metrics else nullcontext(None) as profile:
                try:
                    with metrics.measure('action_total') if metrics else nullcontext():
                        completed=self._execute_bound(decision,before,depth,aid)
                    return completed
                except BaseException as exc:
                    error=type(exc).__name__+': '+str(exc)
                    raise
                finally:
                    if metrics:
                        # Best-effort telemetry must not hide an input failure
                        # or turn an already committed action into a failure.
                        try:self.audit.write('action_profile',action_kind=str(decision.action.kind),
                            completed=completed,error=error,elapsed_ms=(time.perf_counter()-start)*1000,
                            **{key:dict(value) for key,value in profile.items()},
                            scope='synchronous action including child actions; excludes pre-decision observation and background log/archive threads')
                        except OSError:pass

    def _execute_bound(self,decision,before,depth,aid):
        action = decision.action
        self.last_action=action
        self.report['last_decision']=dict(action_id=aid,action=asdict(action),revision=before.revision,
            game_id=before.game_id,turn=before.turn,reason=decision.reason)
        parent_id=self.checkpoint.state['parents'][-1]['action_id'] if self.checkpoint and self.checkpoint.state.get('parents') else None
        start = time.monotonic()
        # 决策耗时（周期开始 -> 决策）与“距上一个动作完成”的间隔，写入审计供 timeline 使用。
        self._decision_at=time.monotonic()
        decision_ms=((self._decision_at-self._cycle_started_at)*1000
                     if self._cycle_started_at else None)
        gap_ms=((self._cycle_started_at-self._last_finished_at)*1000
                if self._cycle_started_at and self._last_finished_at else None)
        self.audit.write("decision", action_id=aid, observation_revision=before.revision, decision=asdict(decision),
            log_position=before.log_position,client_submission_position=before.client_submission_position,
            client_playback_position=before.client_playback_position,
            decision_ms=decision_ms,decision_cycle_ms=decision_ms,gap_since_previous_action_ms=gap_ms)
        from .interaction_types import action_types
        self.audit.write('interaction_contract',action_id=aid,types=action_types(action,before),
            request=asdict(before.interaction) if before.interaction else None)
        for attempt in range(self.config.max_retries+1):
            self._guard()
            revalidate=getattr(self.observations,'revalidate_observation',None)
            current=revalidate(before) if revalidate else self._observe()[0]
            # 只比较“策略输入”：日志游标/回执/客户端回放推进不等于决策失效。
            # 输入前的独立性检查（就绪、绿光、坐标、上下文 token）仍在执行器内逐项执行。
            if (decision_key(current) != decision_key(before) or not current.ready or
                time.time()-current.captured_at > self.config.max_observation_age):
                self.audit.write("cancelled_before_dispatch", action_id=aid, reason="decision_input_changed",
                    decision_ms=decision_ms,gap_since_previous_action_ms=gap_ms,
                    since_decision_ms=(time.monotonic()-self._decision_at)*1000)
                self._timing['cancelled']+=1
                return False  # Re-plan; nothing was sent.
            self.report["dispatch_attempts"] += 1
            # 同一策略输入的新日志版本整体重绑定，避免执行器再次拒绝旧 revision。
            before=current
            decision=replace(decision,revision=current.revision)
            if self.checkpoint:
                if attempt and depth and parent_id and (self.checkpoint.state.get('pending') or {}).get('action_id')==parent_id:
                    self.checkpoint.suspend_for_child()
                self.checkpoint.begin(action,before,aid)
            try:dispatch = self.executor.execute(action, before.revision)
            except Exception as exc:
                possible=getattr(exc,'potentially_dispatched',None)
                if possible:
                    self.report['actions']+=1;self.report['uncertain_dispatches']+=1
                    counts=self.report['action_counts'];counts[str(action.kind)]=counts.get(str(action.kind),0)+1
                    if self.checkpoint:self.checkpoint.delivery(Delivery.AMBIGUOUS)
                elif possible is False and self.checkpoint:self.checkpoint.delivery(Delivery.NOT_SENT)
                if possible is not None:
                    self.report['last_dispatch']=dict(action_id=aid,delivery=str(Delivery.AMBIGUOUS if possible else Delivery.NOT_SENT),
                        reason=type(exc).__name__+': '+str(exc))
                raise
            if self.checkpoint:self.checkpoint.delivery(dispatch.delivery)
            self.report['last_dispatch']=dict(action_id=aid,delivery=str(dispatch.delivery),reason=dispatch.reason)
            self._dispatch_at=time.monotonic()
            self.audit.write("dispatch", action_id=aid, attempt=attempt, dispatch=asdict(dispatch),
                dispatch_delay_ms=(self._dispatch_at-self._decision_at)*1000 if self._decision_at else None)
            if dispatch.delivery==Delivery.NOT_SENT and dispatch.reason in ('native_observation_changed','live_observation_changed_while_waiting','live_log_revision_changed'):
                self.audit.write('cancelled_before_dispatch',action_id=aid,reason=dispatch.reason)
                return False
            if dispatch.delivery==Delivery.NOT_SENT and hasattr(self.observations,'defer_action'):
                if self.observations.defer_action(action,dispatch.reason):
                    self.audit.write('cancelled_before_dispatch',action_id=aid,reason=dispatch.reason)
                    return False
            if dispatch.delivery in (Delivery.ACCEPTED, Delivery.AMBIGUOUS):
                self.report["actions"] += 1
                counts = self.report["action_counts"]
                counts[str(action.kind)] = counts.get(str(action.kind), 0)+1
                if dispatch.delivery == Delivery.AMBIGUOUS:
                    self.report["uncertain_dispatches"] += 1
                    # A client state change during release can mean the input
                    # already succeeded. Reconcile through the same independent
                    # verifier; ambiguity never authorizes a second input.
                    self.audit.write('uncertain_delivery_reconcile',action_id=aid,reason=dispatch.reason,resend=False)
                break
            self.report["rejected_attempts"] += 1
            if attempt == self.config.max_retries:
                raise RuntimeError("dispatch_retry_exhausted: "+dispatch.reason)
            refreshed = self._stable()
            replacement = self._decide(refreshed)[0]
            if self._fingerprint(refreshed) != self._fingerprint(before) or replacement != decision:
                self.audit.write("cancelled_retry", action_id=aid, reason="context_or_policy_changed")
                return False
            self.report["retries"] += 1
            self.audit.write("retry", action_id=aid, reason=dispatch.reason, proof=dispatch.delivery)
        return self._verify_pending(aid,action,before,start,depth)

    def _verify_pending(self,aid,action,before,start,depth=0):
        budget=getattr(self.observations,'verification_timeout',lambda a:self.config.action_timeout)(action)
        deadline = time.monotonic()+budget
        last_result=None;connection_deadline=None;last_after=None;last_verification_stamp=None
        while time.monotonic() < deadline:
            try:
                # Semantic acknowledgements must not wait for animated pixels.
                # UI readiness is checked before the NEXT action separately.
                after = self._observe(verification_action=action)[0] if hasattr(self.observations,'observe_result') else self._stable(timeout=deadline-time.monotonic())
            except AgentDeadline:raise
            except TimeoutError:
                if time.monotonic()<deadline:
                    self.audit.write('verification_observation_wait',action_id=aid)
                    continue
                if last_after is not None and self._reconcile_no_effect(aid,action,before,last_after,last_result):
                    return False
                raise TimeoutError("action_unverified_no_retry")
            last_after=after
            recover_queue=getattr(self.executor,'recover_matchmaking',None)
            if action.kind==Kind.ENTER and recover_queue and recover_queue(before,after):
                self.report['matchmaking_restarts']=self.report.get('matchmaking_restarts',0)+1
                self.audit.write('matchmaking_restarted',action_id=aid,restarts=self.report['matchmaking_restarts'],
                    reason='explicit_cancelled_queue_and_confirmed_error_modal')
                last_verification_stamp=None
                continue
            from .client_protocol import interrupt_phase_wait
            wait_view=(self.resolver.resolve(after).for_policy(after)
                       if action.kind==Kind.END and self.resolver else after)
            if interrupt_phase_wait(action,before,wait_view):
                if self.checkpoint:self.checkpoint.interrupt_phase_wait()
                self.report['phase_wait_interruptions']=self.report.get('phase_wait_interruptions',0)+1
                self.audit.write('phase_wait_interrupted',action_id=aid,revision=after.revision,physical_input=False,
                    reason='local_state_or_available_actions_changed')
                return False
            # A first connection during matchmaking is not an interrupted
            # in-game action. Its wait uses the session/ENTER budget instead
            # of the 60-second reconnect grace period; never resend Start.
            if action.kind!=Kind.ENTER and getattr(self.observations,'connection_wait',lambda:False)():
                if connection_deadline is None:
                    connection_deadline=time.monotonic()+60
                    self.audit.write('pending_action_network_wait',action_id=aid,resend=False)
                if time.monotonic()>=connection_deadline:raise TimeoutError('reconnect_timeout_pending_action_not_replayed')
                deadline=max(deadline,min(connection_deadline,time.monotonic()+budget))
                self._sleep();continue
            if connection_deadline is not None:
                # Each independently observed outage has its own grace period;
                # a later outage must not inherit an expired previous timer.
                self.audit.write('pending_action_network_restored',action_id=aid,resend=False)
                connection_deadline=None
                deadline=max(deadline,time.monotonic()+budget)
            stamp=(self._fingerprint(after),after.log_position)
            if self.verifier is not verify or stamp!=last_verification_stamp or last_result is None:
                verify_started=time.perf_counter()
                result = self.verifier(action, before, after)
                metrics=getattr(self.observations,'chain_metrics',None)
                if metrics:metrics.sample('result_verification',(time.perf_counter()-verify_started)*1000)
                self.report['last_verification']=dict(action_id=aid,revision=after.revision,**asdict(result))
                last_result=result;last_verification_stamp=stamp
                self.audit.write("verification", action_id=aid, result=asdict(result), revision=after.revision,
                    progress=action_progress(action,before,after,result),
                    elapsed_ms=(time.monotonic()-start)*1000,log_position=after.log_position,
                    client_submission_position=after.client_submission_position,client_task_list=after.client_task_list,
                    client_playback_position=after.client_playback_position,options_pending=after.options_pending)
            else:result=last_result
            accepted_result=result.status==Status.CONFIRMED or (result.status==Status.RESOLVED and
                (not self.config.require_effect_confirmation or accepts_processed(action,before,after,result)))
            if not accepted_result and result.status in (Status.PENDING,Status.RESOLVED) and time.monotonic()-start>=3:
                self._pending_diagnostic(action,before,after,result,time.monotonic()-start)
            continuation=getattr(self.observations,'can_continue',None)
            if result.status in (Status.PENDING,Status.RESOLVED) and continuation and continuation(action,before,after):
                child,preview=self._prepare_cycle(timeout=deadline-time.monotonic())
                if child.phase==Phase.CHOICE and child.interaction and after.interaction and child.interaction.request_id==after.interaction.request_id:
                    selected=self._cycle_decision(child,preview)
                    if selected.action.kind in (Kind.CHOOSE,Kind.SELECT,Kind.CONFIRM):
                        self.audit.write('interaction_continuation',parent_action_id=aid,request_id=child.interaction.request_id)
                        if self.checkpoint:self.checkpoint.suspend_for_child()
                        child_started=time.monotonic()
                        completed=self._execute(selected,child,depth+1)
                        if self.checkpoint:self.checkpoint.restore_parent_if_idle()
                        if completed:
                            elapsed=time.monotonic()-child_started
                            deadline+=elapsed
                            self.audit.write('parent_verification_resumed',action_id=aid,child_elapsed_seconds=elapsed,
                                session_deadline_unchanged=True)
                        continue
            processed_ok=result.status==Status.RESOLVED and not self.config.require_effect_confirmation
            if result.status==Status.RESOLVED and self.config.require_effect_confirmation:
                processed_ok=accepts_processed(action,before,after,result)
            if result.status==Status.CONFIRMED or processed_ok:
                if 'placement_deviated' in result.proofs:
                    deviation=dict(action_id=aid,action=asdict(action),reason=result.reason,retry=False)
                    self.report.setdefault('placement_deviations',[]).append(deviation)
                    self.audit.write('placement_deviation',**deviation)
                self.report["successes"] += 1
                self.report['processed']+=1
                self.report['effect_confirmed']+=int(result.status==Status.CONFIRMED)
                verified_counts=self.report['verified_action_counts']
                verified_counts[str(action.kind)]=verified_counts.get(str(action.kind),0)+1
                counts = self.report["verification_counts"]
                counts[str(result.status)] = counts.get(str(result.status), 0)+1
                self.audit.write("action_finished", action_id=aid, verification_status=str(result.status),
                    elapsed_ms=(time.monotonic()-start)*1000, **self._action_timings(action,start))
                committed=getattr(self.policy,'on_verified',None)
                if committed:committed(action,before,after,result)
                observed_commit=getattr(self.observations,'on_verified',None)
                if observed_commit:observed_commit(action,before,after,result)
                if self.checkpoint:self.checkpoint.commit()
                return True
            if result.status in (Status.CONTEXT_CHANGED, Status.UNSUPPORTED):
                self.report['pending_action']=dict(action_id=aid,action=asdict(action),before_revision=before.revision,
                    last_result=asdict(result),timeout_seconds=budget)
                # A turn/phase boundary can arrive before the client writes the
                # receipt. When the action provably changed nothing, continuing
                # is safer than ending the session: nothing was consumed and
                # the next decision starts from a fresh observation.
                if self._reconcile_no_effect(aid,action,before,after,result):return False
                self._pending_diagnostic(action,before,after,result,time.monotonic()-start)
                raise RuntimeError("verification_"+str(result.status)+": "+result.reason)
            self._sleep()
        self.report['pending_action']=dict(action_id=aid,action=asdict(action),before_revision=before.revision,
            last_result=asdict(last_result) if last_result else None,timeout_seconds=budget)
        self.audit.write('action_unverified',**self.report['pending_action'])
        if last_after is not None and self._reconcile_no_effect(aid,action,before,last_after,last_result):
            return False
        raise TimeoutError("action_unverified_no_retry")

    def _reconcile_no_effect(self,aid,action,before,after,result):
        """Abandon a dispatch that provably changed nothing instead of failing.

        原尝试不计成功，不在这里重发。Provider默认本回合屏蔽它；购买可在
        独立无提交及新对象证明通过后，状态未变时保留原意图最多重试一次。
        A bounded budget converts a systemic
        mapping/policy fault back into an ordinary fail-closed stop.
        """
        from .verification import no_effect_proven
        try:
            proven,why=no_effect_proven(action,before,after)
        except Exception as exc:
            self.audit.write('no_effect_reconciliation_error',action_id=aid,
                reason=type(exc).__name__+': '+str(exc))
            return False
        if not proven:
            self.audit.write('no_effect_reconciliation_refused',action_id=aid,reason=why,
                verification_status=str(result.status) if result is not None else None)
            return False
        abandoned=self.report.get('abandoned_actions',0)+1
        if abandoned>self.config.max_abandoned_actions:
            self.audit.write('no_effect_reconciliation_exhausted',action_id=aid,abandoned=abandoned)
            return False
        self.report['abandoned_actions']=abandoned
        counts=self.report.setdefault('abandoned_action_counts',{})
        counts[str(action.kind)]=counts.get(str(action.kind),0)+1
        detail=dict(action_id=aid,action=asdict(action),proof=why,
            verification_status=str(result.status) if result is not None else None,
            verification_reason=result.reason if result is not None else None,
            before_revision=before.revision,after_revision=after.revision,
            before_turn=before.turn,after_turn=after.turn,resend=False)
        history=self.report.setdefault('abandoned',[])
        if len(history)<20:history.append(detail)
        self.audit.write('action_abandoned_no_effect',**detail)
        # The action is resolved (no effect): it must not keep the report or a
        # reader believing an input is still awaiting verification.
        self.report['pending_action']=None
        if self.checkpoint:self.checkpoint.abandon()
        hook=getattr(self.observations,'abandon_action',None)
        if hook:
            try:hook(action,before,after,why)
            except Exception as exc:
                self.audit.write('abandon_hook_failed',action_id=aid,reason=type(exc).__name__+': '+str(exc))
        teacher=getattr(self.policy,'on_abandoned',None)
        if teacher:teacher(action,before,after,why)
        return True

    def _action_timings(self,action,start):
        """单动作时间线：间隔 / 决策 / 投递延迟 / 校验，并累计到报告汇总。"""
        now=time.monotonic()
        gap=((self._cycle_started_at-self._last_finished_at)*1000
             if self._cycle_started_at and self._last_finished_at else None)
        decision=((self._decision_at-self._cycle_started_at)*1000
                  if self._decision_at and self._cycle_started_at else None)
        dispatch_delay=((self._dispatch_at-self._decision_at)*1000
                        if self._dispatch_at and self._decision_at else None)
        verify=(now-self._dispatch_at)*1000 if self._dispatch_at else None
        elapsed=(now-start)*1000
        self._last_finished_at=now;self._decision_at=self._dispatch_at=None
        self._timing['actions']+=1
        for name,value in (('gaps',gap),('decisions',decision),('dispatches',dispatch_delay)):
            if value is not None:self._timing[name].append(value)
        bucket=self._timing['by_kind'].setdefault(str(action.kind),
            dict(n=0,total_ms=0.0,total_gap_ms=0.0,gaps=0))
        bucket['n']+=1;bucket['total_ms']+=elapsed
        if gap is not None:bucket['total_gap_ms']+=gap;bucket['gaps']+=1
        return dict(gap_since_previous_action_ms=gap,decision_ms=decision,
                    dispatch_delay_ms=dispatch_delay,verify_ms=verify,action_elapsed_ms=elapsed)

    @staticmethod
    def _percentiles(values,quantiles=(.5,.9,.95,1.0)):
        if not values:return dict(n=0)
        ordered=sorted(values)
        out=dict(n=len(ordered),mean=round(sum(ordered)/len(ordered),1))
        for q in quantiles:
            out['p%d'%int(q*100)]=round(ordered[min(len(ordered)-1,int(len(ordered)*q))],1)
        return out

    def _timing_summary(self):
        kinds={kind:dict(n=value['n'],mean_ms=round(value['total_ms']/value['n'],1),
                         mean_gap_ms=(round(value['total_gap_ms']/value['gaps'],1)
                                      if value['gaps'] else None))
               for kind,value in sorted(self._timing['by_kind'].items())}
        return dict(actions=self._timing['actions'],cancelled_plans=self._timing['cancelled'],
                    gap_between_actions_ms=self._percentiles(self._timing['gaps']),
                    decision_ms=self._percentiles(self._timing['decisions']),
                    decision_cycle_ms=self._percentiles(self._timing['decisions']),
                    dispatch_delay_ms=self._percentiles(self._timing['dispatches']),
                    by_kind=kinds,
                    note='gap = 上一个动作完成 -> 本动作决策周期开始；decision_ms为兼容字段，等于decision_cycle_ms（含界面等待和推理），不是搜索耗时；'
                         '实际推理看policy_timing/policy_preview_finished及timeline.policy_compute_ms；'
                         'dispatch = 决策 -> 输入计划完成/投递；verify = 投递 -> 结果确认')

    def _pending_diagnostic(self,action,before,after,result,elapsed):
        """Best-effort read-only evidence; never an authorization to replay."""
        capture=getattr(self.observations,'pending_diagnostic',None)
        if capture:
            try:capture(action,before,after,result,elapsed)
            except InterruptedError:raise
            except Exception as exc:
                try:self.audit.write('pending_diagnostic_failed',reason=str(exc))
                except OSError:pass

    def _failure_diagnostic(self):
        """Retain the failed contract even for a proven-unsent action."""
        context=getattr(self.observations,'failure_context',None)
        if context:
            try:
                details=context(self.last_action)
                self.report['failure_context']=details
                self.audit.write('failure_context',**details)
                archive=getattr(self.observations,'archive_failure',None)
                if archive:archive(self.last_action,details)
            except Exception as exc:
                self.report['failure_diagnostic_error']=type(exc).__name__+': '+str(exc)

    def run(self):
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("agent already running")
        if self.state != Lifecycle.IDLE:
            self._run_lock.release()
            raise RuntimeError("create a new controller for a new run")
        self._started = time.monotonic()
        seen_game_over = False
        wait_sig, waiting_since = None, time.monotonic()
        try:
            self._state(Lifecycle.RUNNING)
            while self.resume_pending:
                action,before,aid=self.resume_pending
                seen_game_over |= before.phase==Phase.GAME_OVER
                self.report['actions']+=1
                self.audit.write('reconcile_pending_dispatch',action_id=aid,action=asdict(action),resend=False)
                recover=getattr(self.observations,'recover_pending',None)
                if recover:recover(action,before)
                self._verify_pending(aid,action,before,time.monotonic())
                self.resume_pending=self.checkpoint.pending() if self.checkpoint else None
            while True:
                before,preview = self._prepare_cycle()
                seen_game_over |= before.phase == Phase.GAME_OVER
                menu_ready=getattr(self.observations,'menu_ready',lambda o:True)(before)
                menu_ready |= bool(self.report['verified_action_counts'].get('dismiss') and getattr(self.observations,'completion_ready',lambda o:False)(before))
                if ((seen_game_over and before.phase == Phase.MENU and menu_ready) or
                    (before.phase == Phase.GAME_OVER and not before.actions and not self.config.require_return_to_menu)):
                    self.report["reason"] = "game_complete_returned_to_menu" if before.phase == Phase.MENU else "game_complete"
                    self._state(Lifecycle.COMPLETED)
                    break
                if self.report["actions"] >= self.config.max_actions:
                    raise RuntimeError("action_budget_exhausted")
                decision = self._cycle_decision(before,preview)
                if decision.revision != before.revision:
                    raise ValueError("policy returned stale revision")
                if decision.action.kind == Kind.WAIT:
                    sig = (before.game_id, before.phase, before.turn,before.interaction.request_id if before.interaction else None)
                    if wait_sig != sig:
                        wait_sig, waiting_since = sig, time.monotonic()
                        self.audit.write("waiting", phase=before.phase,game_id=before.game_id,turn=before.turn,
                            revision=before.revision,reason=decision.reason,
                            advertised_actions=[a.key for a in before.actions])
                    wait_budget=getattr(self.observations,'waiting_timeout',lambda o:self.config.phase_timeout)(before)
                    if time.monotonic()-waiting_since > wait_budget:
                        # One bounded retry: a slow animation or a stale frame can
                        # outlive the wait budget without being a dead interaction.
                        # Recovery only re-observes; it never sends input, and the
                        # ordinary dispatch gates still authorize any later action.
                        recover=getattr(self.observations,'recover_stall',None)
                        tried=getattr(self,'_stall_recoveries',{})
                        if recover and tried.get(sig,0)<self.config.max_stall_recoveries:
                            tried[sig]=tried.get(sig,0)+1
                            self._stall_recoveries=tried
                            try:recovered=bool(recover(before,decision))
                            except InterruptedError:raise
                            except Exception as exc:
                                recovered=False
                                self.audit.write('stall_recovery_error',reason=type(exc).__name__+': '+str(exc))
                            self.audit.write('stall_recovery',signature=list(sig),attempt=tried[sig],
                                budget_seconds=wait_budget,recovered=recovered,physical_input=False)
                            if recovered:
                                wait_sig=None;waiting_since=time.monotonic()
                                self._sleep();continue
                        if before.phase != Phase.COMBAT:
                            self.report["unsupported"].append(dict(phase=before.phase, reason=decision.reason,
                                                                  revision=before.revision))
                        raise TimeoutError("combat_timeout" if before.phase == Phase.COMBAT else "unsupported_or_stalled_interaction")
                    self._sleep()
                    continue
                if decision.action not in before.actions:
                    raise ValueError("policy returned unadvertised action")
                completed=self._execute(decision, before)
                wait_sig = None
                # A verified action already provides a new-state boundary.
                # Next cycle starts log/inference work immediately; failed or
                # cancelled attempts retain backoff and all UI gates remain.
                if not completed:self._sleep()
        except KeyboardInterrupt:
            self._stop.set()
            self.report["reason"] = "human_stop_keyboard_interrupt"
            try:self._state(Lifecycle.STOPPED)
            except OSError:pass
            raise
        except InterruptedError as exc:
            self.report["reason"] = str(exc)
            try:self._state(Lifecycle.STOPPED)
            except OSError:pass
        except Exception as exc:
            import traceback
            self.report['exception_traceback']=traceback.format_exc()
            if isinstance(exc,RuntimeError) and str(exc)=='window_destroyed':
                explicit_stop=self._stop.is_set() or bool(self.stop_file and self.stop_file.exists())
                self.report.update(reason='human_stop_window_closed' if explicit_stop else 'client_window_closed',
                    stop_category='human_stop' if explicit_stop else 'external_window_closed',
                    original_error='window_destroyed')
                self.state=Lifecycle.STOPPED
                try:
                    self.audit.write('client_window_closed',explicit_human_stop=explicit_stop,
                        pending_action_preserved=True,requeue=False)
                    self._state(Lifecycle.STOPPED)
                except OSError:pass
            else:
                self.report["failures"] += 1
                self.report["reason"] = str(exc)
                self.state=Lifecycle.FAILED
                try:
                    self.audit.write("failure", error_type=type(exc).__name__, reason=str(exc),traceback=self.report['exception_traceback'])
                    self._state(Lifecycle.FAILED)
                except OSError:pass  # Preserve the original error and pending ledger.
            self._failure_diagnostic()
        finally:
            self.report.update(status=str(self.state), elapsed_seconds=time.monotonic()-self._started,
                               unverified_actions=max(0,self.report["actions"]-self.report["successes"]-
                                   self.report.get('abandoned_actions',0)-self.report.get('phase_wait_interruptions',0)),
                               unresolved_checkpoint=bool(self.checkpoint and self.checkpoint.pending()),
                               timing=self._timing_summary(),
                               stall_recoveries=dict(count=sum(getattr(self,'_stall_recoveries',{}).values()),
                                                    attempts=[[list(k),v] for k,v in getattr(self,'_stall_recoveries',{}).items()]))
            try:self.audit.write("agent_finished", report=self.report)
            except OSError:pass
            finally:self._run_lock.release()
        return self.report
