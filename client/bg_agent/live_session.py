"""Live resource ownership. Importing this module never acquires input APIs."""
from contextlib import ExitStack
import hashlib,json,os,time
from pathlib import Path


def validate_start_state(initial,*,resuming=False):
    if not resuming and (str(initial.phase)!='menu' or initial.scene!='BACON'):
        raise ValueError('fresh_run_requires_solo_battlegrounds_start_menu; active games require explicit checkpoint reconciliation')


def run_session(args,root,audit,*,factories=None):
    from .live_client import LiveGameProvider,LiveActionResolver
    from .native_executor import NativeExecutor,NativeEmergencyStop
    from .live_panel import start_panel
    from .agent import AgentController,AgentConfig
    from .checkpoint import ActionCheckpoint
    from .demonstrations import create_policy
    from .model import Phase
    from .recording import write_json
    from .interaction_timing import timing_profile
    from dataclasses import asdict
    deps=dict(provider=LiveGameProvider,panel=start_panel,executor=NativeExecutor,emergency=NativeEmergencyStop)
    deps.update(factories or {})
    result=dict(status='failed',reason='startup_incomplete',full_game_verified=False)
    agent=None
    try:
        with ExitStack() as resources:
            timing=timing_profile(getattr(args,'interaction_speed','safe'))
            configuration={name:getattr(args,name) for name in ('game_dir','ui_profile','hand_profile') if getattr(args,name,None) is not None}
            for name in ('geometry_mode','layout_catalog','local_layout'):
                if getattr(args,name,None) is not None:configuration[name]=getattr(args,name)
            provider=deps['provider'](root,audit,**configuration);resources.callback(provider.close)
            provider.uses_green_signal=True
            # 准入器由Provider拥有；会话装配不能覆盖客户端对象依赖证明。
            from .dependency_readiness import DependencyReadiness
            readiness=dict(input_readiness='action_dependencies' if isinstance(provider.gate,DependencyReadiness) else 'custom',
                input_readiness_gate=type(provider.gate).__name__)
            result.update(readiness)
            provider.run_seconds=args.seconds
            panel=deps['panel'](provider)
            resources.callback(panel.server_close);resources.callback(panel.shutdown)
            stopped=lambda:bool(agent and agent._stop.is_set()) or (root/'STOP').exists()
            executor=deps['executor'](provider,audit,allow_live_client=True,stopped=stopped,
                guard=lambda:agent._guard() if agent else None,timing=timing,
                gesture_clock=getattr(args,'gesture_clock','precise'))
            if callable(getattr(executor,'close',None)):resources.callback(executor.close)
            resources.callback(executor.transport.release)
            # 本地扩展点：只注入策略，复用完整输入、核验和恢复链。
            policy=deps.get('policy',create_policy)(args.policy,dark_gifts_per_turn=args.dark_gifts_per_turn,
                demonstration_path=root/'demonstrations.jsonl' if args.policy=='heuristic' else None)
            if callable(getattr(policy,'bind_stop',None)):policy.bind_stop(stopped)
            provider.decision_ahead=bool(getattr(args,'decision_ahead',False) and callable(getattr(policy,'preview',None)))
            checkpoint=ActionCheckpoint(root/'checkpoint.json',policy,provider.logs.session)
            provider.checkpoint=checkpoint
            agent=AgentController(provider,policy,executor,audit,resolver=LiveActionResolver(),stop_file=root/'STOP',checkpoint=checkpoint,
                config=AgentConfig(require_return_to_menu=True,quiet_period=.05,observation_timeout=60,action_timeout=240,phase_timeout=240,
                    max_seconds=args.seconds,max_actions=1200,max_retries=0,max_observation_age=3,require_effect_confirmation=True,
                    decision_ahead=provider.decision_ahead))
            emergency=deps['emergency'](executor.transport,agent,root/'STOP')
            resources.callback(emergency.close);emergency.start()
            # Register every resource and emergency stop before any UI action.
            if stopped():raise InterruptedError('human_stop_before_activation')
            provider.activate();provider.park_cursor(executor)
            initial=provider.observe()
            validate_start_state(initial,resuming=bool(args.resume))
            checkpoint.state['session']=str(provider.logs.session)
            agent.resume_pending=checkpoint.load(args.resume,initial,provider.logs.session) if args.resume else None
            if args.resume and initial.phase==Phase.MENU and not agent.resume_pending:
                raise ValueError('checkpoint_match_already_at_menu_no_requeue')
            checkpoint.save()
            provider.input_guard=agent._guard
            provider.recover_pending=lambda action,before:LiveGameProvider.recover_pending(provider,executor,action,before)
            write_json(root/'run.json',dict(pid=os.getpid(),started_at=time.time(),initial_phase=str(initial.phase),policy=args.policy,
                policy_metadata=policy.metadata() if hasattr(policy,'metadata') else None,
                dark_gifts_per_turn=args.dark_gifts_per_turn,resume=str(args.resume) if args.resume else None,
                configuration={k:str(v) for k,v in configuration.items()},
                decision_ahead=agent.config.decision_ahead,interaction_timing=asdict(timing),
                gesture_clock=getattr(args,'gesture_clock','precise'),
                **readiness,
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                objective='one real solo game from start to start; no human game inputs'))
            print('真实输入已启用。F8 / STOP 文件停止。面板：http://127.0.0.1:8796\n证据：'+str(root.resolve()),flush=True)
            result=agent.run();counts=result['verified_action_counts']
            result.update(pid=os.getpid(),initial_phase=str(initial.phase),policy_state=policy.checkpoint_state(),
                game_ids=sorted(provider.game_ids),hero_choice_verified=provider.hero_choice_verified,
                demonstration_exports_truncated=sorted(getattr(policy,'truncated_paths',())),
                audit_retention={name:dict(segment_bytes=log.max_bytes,backup_count=log.backup_count,
                    rotations=log.rotations,omitted_records=log.omitted_records)
                    for name,log in [('agent',audit),('samples',getattr(getattr(provider,'sampler',None),'audit',None))] if log is not None},
                skipped_optional_actions=getattr(provider,'skipped_optional_actions',[]),
                unsafe_skipped_actions=getattr(provider,'unsafe_skipped_actions',[]),
                returned_start_ready=provider.menu_ready(provider.last_obs),returned_start_observed=provider.returned_start_observed,
                full_game_verified=bool(result['status']=='completed' and initial.phase==Phase.MENU and counts.get('enter')==1 and
                    provider.hero_choice_verified and len(provider.game_ids)==1 and counts.get('buy',0)>0 and counts.get('play',0)>0 and
                    len(result['turns'])>=2 and result['unverified_actions']==0 and not result.get('unresolved_checkpoint') and provider.completion_ready(provider.last_obs)))
        result.update(evidence_truncated=getattr(provider,'evidence_truncated',None),
            evidence_skipped=getattr(provider,'evidence_skipped',0))
        if getattr(provider,'storage_error',None):
            result.update(status='failed',reason=provider.storage_error,full_game_verified=False)
    except (Exception,KeyboardInterrupt) as exc:
        import traceback
        if agent and result.get('reason')=='startup_incomplete':result.update(agent.report)
        result.update(status='stopped' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',
            reason=type(exc).__name__+': '+str(exc),exception_traceback=traceback.format_exc(),full_game_verified=False)
        if isinstance(exc,RuntimeError) and str(exc)=='window_destroyed':
            explicit_stop=bool(agent and agent._stop.is_set()) or (root/'STOP').exists()
            result.update(status='stopped',reason='human_stop_window_closed' if explicit_stop else 'client_window_closed',
                stop_category='human_stop' if explicit_stop else 'external_window_closed',original_error='window_destroyed')
        try:audit.write('session_failure',**result)
        except OSError:pass  # A full audit must not suppress the final report.
    finally:
        provider=locals().get('provider')
        if 'readiness' in locals():result.update(readiness)
        pending=getattr(provider,'control_deferrals',{})
        control_counts=dict(getattr(provider,'control_readiness_counts',{}))
        result['control_readiness']=dict(events=control_counts,
            selection_policy='hold_selected_until_ready_or_context_change',
            pending=[dict(kind=k[0],source=k[1],reason=v['reason'],eligibility_state=v['eligibility_state'])
                     for k,v in pending.items()],
            review_required=bool(pending or control_counts.get('expired',0) or control_counts.get('selected_timeout',0)))
        result['buy_recovery']=dict(events=dict(getattr(provider,'buy_recovery_counts',{})),
            retry_selection='retain_original_buy_if_policy_state_unchanged',
            ready_retry=asdict(provider.buy_retry_ready['action']) if getattr(provider,'buy_retry_ready',None) else None,
            pending=[dict(action=asdict(ticket['action']),before_revision=ticket['before'].revision)
                for ticket in getattr(provider,'buy_recoveries',{}).values()],
            max_reoffers_per_entity_per_turn=1,observation_budget_seconds=8)
        from .evidence_window import recording_status
        result['evidence_recording']=recording_status(provider,audit)
        metrics=getattr(provider,'chain_metrics',None)
        if metrics:
            result['operation_chain']=metrics.report()
            counts=result.get('action_counts',{});verified=result.get('verified_action_counts',{})
            result['execution_acceptance']=dict(
                gesture_actions=sum(n for k,n in counts.items() if k not in ('end','wait')),
                passive_end_waits=counts.get('end',0),unverified_actions=result.get('unverified_actions'),
                coordinate_skips=len(result.get('skipped_optional_actions',())),
                unsafe_source_skips=len(result.get('unsafe_skipped_actions',())),
                full_localizer_calls=metrics.counts.get('full_localization',0),
                captures_full=metrics.counts.get('capture_full',0),captures_roi=metrics.counts.get('capture_roi',0),
                captures_reused=metrics.counts.get('capture_reused',0),
                contour_recoveries=metrics.counts.get('layout_motion_mismatch_recovery',0),
                local_layout_recoveries=metrics.counts.get('layout_local_recovery',0),
                verified_gesture_actions=sum(n for k,n in verified.items() if k not in ('end','wait')),
                misclicks='requires review of receipts and evidence; not inferred from accepted input',
                late_game_and_unseen_special_interactions='require live evidence')
        mapper=getattr(provider,'mapper',None)
        if getattr(mapper,'mode',None):result['geometry_mode']=mapper.mode
        write_json(root/'report.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['status']=='completed' else 1
