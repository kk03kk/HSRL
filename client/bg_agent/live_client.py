"""Explicitly enabled, identity-bound real client adapter; no injected receipts."""
from dataclasses import replace,asdict
from pathlib import Path
import time,uuid,queue,threading
import ctypes as C
from ctypes import wintypes as W
import cv2
import numpy as np
from .model import Observation,Action,Decision,Kind,Phase,Resolution
from .mapping import Mapping,StabilityBasis
from .windows import Box,NativeWindows,WindowSnapshot
from .connection import HearthstoneLogs
from .perception import SynchronizedSampler
from .client_session import ClientLifecycle
from .adaptive_spatial import AdaptiveSpatialMapper
from .client_readiness import ClientReadiness
from .planning import ShadowPlanner
from .native_bridge import FixtureBinding,NativePlan
from .recording import write_json,RawMirror
from .legal import LegalActionResolver,Certainty
from .action_context import input_context_token
from .magnetic import magnetic,mechanical,magnetic_context
from .network_state import blocks_game_input
from .geometry_cache import pixel_detector
from .geometry import client_to_screen
from .chain_metrics import measured_stage
from .action_budgets import result_timeout,prepare_timeout,PREPARE_SECONDS,CHOICE_SECONDS,CONFIRM_SECONDS


class LiveActionResolver(LegalActionResolver):
    """Execution capability mask, separate from the unchanged RulePolicy."""
    def resolve(self,obs,**kwargs):
        result=super().resolve(obs,**kwargs);rows=[]
        for entry in result.actions:
            a=entry.action;reason=None
            if a.target in {c.entity_id for c in obs.hand} and a.evidence!='verified_hand_target':
                reason='live_hand_target_adapter_unverified'
            if a.kind in (Kind.CHOOSE,Kind.SELECT) and any(c.tags.get('QUEST')=='1' for c in obs.choices):
                if not obs.interaction or obs.interaction.protocol!='verified_quest_pair':
                    reason='live_paired_quest_layout_unverified'
            if a.sub_option_id is not None:
                from .interaction_protocol import branch_entities
                try:branch_entities(a)
                except ValueError:reason='live_branch_entity_missing'
            if a.kind==Kind.PLAY and a.interaction and a.interaction.placement=='magnetize':
                c=next((c for c in obs.hand if c.entity_id==a.source),None)
                target=next((c for c in obs.board if c.entity_id==a.target),None)
                known=magnetic_context(obs,a)
                if not known:reason='live_magnetic_semantics_unknown'
                elif entry.eligible:entry=replace(entry,certainty=Certainty.CONFIRMED)
            rows.append(replace(entry,eligible=False,reasons=entry.reasons+(reason,)) if reason else entry)
        return replace(result,actions=tuple(rows))


@pixel_detector
def menu_button(image):
    """Measured blue circular start button, only used with BACON menu evidence."""
    width,height=image.size
    hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([88,110,125]),np.array([121,255,255]))
    mask[:int(height*.55)]=0;mask[:,:int(width*.6)]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((13,13),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if .07*width<w<.2*width and .10*height<h<.27*height and .7<w/h<1.4:
            fill=cv2.contourArea(c)/(w*h)
            if fill>.3:candidates.append(Box(x+round(w*.3),y+round(h*.3),round(w*.4),round(h*.4)))
    return candidates[0] if len(candidates)==1 else None


def phase_view_blockers(obs,maps):
    """Server can finish combat before the client finishes showing it."""
    visible={m.key for m in maps if m.status=='matched' and m.rect}
    if obs.phase==Phase.CHOICE:
        if 'scene:choice' not in visible or any(f'entity:{c.entity_id}' not in visible for c in obs.choices):
            return ['log_choice_ahead_of_client_scene']
        return []
    if obs.phase!=Phase.RECRUIT:return []
    if 'scene:recruit' not in visible:return ['log_recruit_ahead_of_client_scene']
    if obs.shop and not any(f'entity:{c.entity_id}' in visible for c in obs.shop):
        return ['log_shop_ahead_of_client_frames']
    return []


@pixel_detector
def result_panel(image):
    width,height=image.size;hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    if np.mean(hsv[:,:,1]<65)<.5:return None
    mask=cv2.inRange(hsv,np.array([125,60,45]),np.array([174,255,255]))
    mask[:int(height*.18)]=0;mask[int(height*.85):]=0
    mask[:,:int(width*.30)]=0;mask[:,int(width*.70):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((19,19),np.uint8))
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in cs:
        x,y,w,h=cv2.boundingRect(c)
        if .13*width<w<.38*width and .28*height<h<.72*height:
            candidates.append(Box(round(x+w*.45),round(y+h*.4),round(w*.1),round(h*.1)))
    return candidates[0] if len(candidates)==1 else None


@pixel_detector
def hero_regions(image,count):
    width,height=image.size;hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([130,65,55]),np.array([174,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if .09*width<w<.24*width and .2*height<h<.55*height and .2*height<y<.5*height:
            boxes.append(Box(x+round(w*.3),y+round(h*.2),round(w*.4),round(h*.25)))
    boxes.sort(key=lambda b:b.x)
    return boxes if len(boxes)==count else []


@pixel_detector
def confirm_button(image):
    width,height=image.size;hsv=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([88,65,100]),np.array([125,255,255]))
    mask[:int(height*.70)]=0;mask[int(height*.90):]=0
    mask[:,:int(width*.40)]=0;mask[:,int(width*.60):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if .045*width<w<.17*width and .02*height<h<.10*height:
            boxes.append(Box(x,y,w,h))
    return boxes[0] if len(boxes)==1 else None


def power_choice_regions(image,count):
    from .live_choices import power_card_row
    return power_card_row(image,count)


from .hand_mapping import hand_markers


class LiveGameProvider:
    live_client=True
    observation_is_gated=True
    def __init__(self,directory,audit,game_dir=Path('D:/Hearthstone'),*,ui_profile=None,hand_profile=None,geometry_mode='geometry',layout_catalog=None,local_layout=None):
        self.directory=directory;self.audit=audit;self.phase_wait_enabled=True
        self.windows=NativeWindows(game_dir/'Hearthstone.exe');self.windows.defer_unsaved_digest=True
        window=self.windows.unique()
        self.binding=FixtureBinding(window.hwnd,window.pid,window.process_created,window.executable,uuid.uuid4().hex,'')
        # This provider has explicit network-event and fresh UI gates. A long
        # recruit timer with no new log lines does not revoke server Options.
        self.logs=HearthstoneLogs(game_dir,keep_options_during_silence=True);self.sampler=SynchronizedSampler(self.logs,self.windows,directory/'samples',bound_hwnd=window.hwnd,local_readiness=True,share_snapshot=True,
            audit_max_bytes=128*1024**2,audit_backup_count=2,compact_audit=True)
        from .local_layout import DEFAULT_PATH,LocalLayoutProfile,display_settings,executable_hash
        self.local_profile=None
        selected=Path(local_layout) if local_layout else DEFAULT_PATH
        if geometry_mode in ('geometry','compare'):
            self.local_profile=LocalLayoutProfile.load(selected)
            self.local_profile.require_ready(window,self.logs.observe().build,display_settings(),executable_hash(window.executable))
        from .interaction_capture import Budget,LiveEvidenceSchedule
        self.storage=Budget.create(images=384*1024**2,metadata=96*1024**2)
        # Diagnostics must survive routine evidence. In the 21:33 run the
        # metadata quota filled 36 s before the final stall, so the one frame
        # that would explain it was never written.
        self.diagnostic_storage=Budget.create(images=64*1024**2,metadata=16*1024**2)
        self.evidence_schedule=LiveEvidenceSchedule()
        self.storage_error=None
        from .resources import calibration_root
        calibration=calibration_root()
        self.mapper=AdaptiveSpatialMapper(ui_profile or calibration/'v05_ui/profile.json',hand_profile or calibration/'v05_hand/profile.json')
        from .geometry_mapper import GeometryMapper
        from .layout_resolver import LayoutCatalog
        from .dependency_readiness import DependencyReadiness
        from .chain_metrics import ChainMetrics
        self.chain_metrics=ChainMetrics();self.windows.metrics=self.chain_metrics;self.logs.metrics=self.chain_metrics
        self.sampler.broker.metrics=self.chain_metrics
        catalog=LayoutCatalog.load(layout_catalog or Path(__file__).parent/'data/layout_catalog.json')
        if self.local_profile:catalog=self.local_profile.catalog(catalog)
        self.mapper=GeometryMapper(self.mapper,catalog,
            mode=geometry_mode,directory=directory,metrics=self.chain_metrics)
        self.sampler.region_selector=self._capture_region
        # 旧兼容标志表示采用当前帧独立准入、禁用全局动画等待，并非要求共享绿光。
        self.uses_green_signal=True;self.mapper.uses_green_signal=True;self.sampler.green_signal=True
        self.lifecycle=ClientLifecycle();self.gate=DependencyReadiness(self.mapper,self.client_object_snapshot)
        from .recruit_start import RecruitStartGate
        self.recruit_start=RecruitStartGate()
        self._force_full_capture=True;self.verified_log_position=None
        from .state_health import ChannelHealth
        self.health=ChannelHealth()
        self.last_row=None;self.last_image=None;self.last_obs=None;self.last_window=window;self.mappings=[]
        self.cache_until=0;self.enter_sent=False;self.matched=False;self.last_evidence=0
        self.returned_start_ready=False;self.mirrors={}
        self.returned_start_observed=False
        self.ui_waits={}
        self.selection_snapshot=None;self.selection_revision=0
        self.game_ids=set();self.hero_choice_sent=None;self.hero_choice_verified=False
        self.archive_queue=queue.Queue(maxsize=3);self.closed=threading.Event()
        self.archive_thread=threading.Thread(target=self._archive,daemon=True);self.archive_thread.start()
        self.logs.start_reader()

    def _archive(self):
        while not self.closed.is_set() or not self.archive_queue.empty():
            try:stem,image,metadata=self.archive_queue.get(timeout=.1)
            except queue.Empty:continue
            try:
                folder=self.directory/'evidence';folder.mkdir(exist_ok=True)
                self.persist_evidence(stem,image,metadata)
            except Exception as exc:
                self.storage_error='evidence_archive_failed:'+str(exc)
                try:self.audit.write('archive_error',reason=str(exc))
                except OSError:pass
            finally:self.archive_queue.task_done()

    def close(self):
        try:
            for mirror in self.mirrors.values():
                if not self.storage_error:self._mirror(mirror)
        finally:
            if hasattr(self.logs,'close'):self.logs.close()
            try:self._queue_diagnostic_window('closing')
            except OSError as exc:self.storage_error=self.storage_error or 'closing_evidence_failed:'+str(exc)
            self.closed.set();self.archive_thread.join(timeout=5)
            self.archive_drained=not self.archive_thread.is_alive() and self.archive_queue.unfinished_tasks==0
            write_json(self.directory/'raw-manifest.json',dict(files=[m.manifest() for m in self.mirrors.values()],
                storage_error=self.storage_error,budget=dict(limits=self.storage.limits,used=self.storage.used),
                diagnostic_budget=dict(limits=self.diagnostic_storage.limits,used=self.diagnostic_storage.used),
                evidence_truncated=getattr(self,'evidence_truncated',None),
                evidence_skipped=getattr(self,'evidence_skipped',0),
                archive_drained=self.archive_drained,archive_pending=self.archive_queue.unfinished_tasks,
                context_frames_not_queued=getattr(self,'context_frames_not_queued',0),
                perception_metrics=self.sampler.metrics.report(),geometry_cache=self.mapper.cache_metrics,
                operation_chain=self.chain_metrics.report(),capture_broker=self.sampler.broker.metrics.report(),
                geometry_comparisons=self.mapper.comparisons))

    def _mirror(self,mirror):
        while True:
            remaining=self.storage.remaining('logs')
            chunk=mirror.poll(max_bytes=min(8*1024**2,remaining))
            self.storage.consume(logs=len(chunk))
            if mirror.offset>=mirror.source.stat().st_size:return
            if self.storage.remaining('logs')==0:
                self.storage_error='raw_log_storage_limit'
                raise OSError(self.storage_error)

    def ensure_mirror(self,source):
        """Version raw evidence by identity, not just a reused log filename."""
        stat=source.stat();identity=(stat.st_dev,stat.st_ino)
        for mirror in self.mirrors.values():
            if mirror.identity==identity and mirror.source.parent==source.parent:return mirror
        folder=self.directory/'raw'/self.logs.session.name
        destination=folder/source.name;version=1
        while destination.exists():
            version+=1;destination=folder/(source.stem+f'.part-{version}'+source.suffix)
        mirror=RawMirror(source,destination)
        self.mirrors[str(source)+'#'+str(identity)]=mirror
        return mirror

    # Frames that explain a failure are never starved by routine evidence.
    DIAGNOSTIC_PREFIXES=('failure','pending-action','ui-','choice-','target-','branch-','confirm-','context-')

    @measured_stage('evidence_persistence')
    def persist_evidence(self,stem,image,metadata):
        from .interaction_capture import save_frame
        critical=stem.startswith(self.DIAGNOSTIC_PREFIXES)
        if getattr(self,'evidence_truncated',None) and not critical:
            self.evidence_skipped=getattr(self,'evidence_skipped',0)+1;return False
        folder=self.directory/'evidence';folder.mkdir(exist_ok=True)
        metadata=dict(metadata);extension=metadata.pop('archive_format','png')
        budgets=(self.diagnostic_storage,self.storage) if critical else (self.storage,)
        saved=False
        try:
            for budget in budgets:
                if save_frame(folder,stem,image,extension,metadata,budget):saved=True;break
        except (RuntimeError,OSError) as exc:
            self.storage_error='evidence_write_failed:'+str(exc);raise OSError(self.storage_error) from exc
        if saved:return True
        if critical:
            # A quota is not a game/input failure; keep the action running and
            # say that this one diagnostic frame was dropped.
            self.evidence_skipped=getattr(self,'evidence_skipped',0)+1
            self.audit.write('diagnostic_evidence_skipped',reason='diagnostic_storage_limit',stem=stem)
            return False
        if not getattr(self,'evidence_truncated',None):
            # A configured diagnostic quota is not a game/input failure.
            # Actual encoding or disk I/O failures still stop through the
            # error path above; never silently claim complete evidence.
            self.evidence_truncated='image_or_metadata_storage_limit'
            self.audit.write('evidence_archive_truncated',reason=self.evidence_truncated,
                input_state_affected=False,budget=dict(limits=self.storage.limits,used=dict(self.storage.used)))
        self.evidence_skipped=getattr(self,'evidence_skipped',0)+1
        return False

    def stability_stall(self,obs):
        """Visible, bounded progress while the client UI is not ready.

        A silent 40-60 s wait looks like a frozen script. This reports the
        current blockers every few seconds and keeps one diagnostic frame per
        request from the reserved budget.
        """
        now=time.monotonic()
        request=obs.interaction.request_id if obs.interaction else None
        key=(obs.game_id,obs.turn,str(obs.phase),request)
        if getattr(self,'stall_key',None)!=key:
            self.stall_key=key;self.stall_since=now;self.stall_at=0
        if now-getattr(self,'stall_at',0)<5:return
        self.stall_at=now
        blockers=list(getattr(self,'last_row',{}).get('blockers') or ())
        self.audit.write('stability_stall',game_id=obs.game_id,turn=obs.turn,phase=str(obs.phase),
            request_id=request,ready=obs.ready,blockers=blockers,warnings=list(obs.warnings),
            display_state=obs.interaction.display_state if obs.interaction else None,
            stalled_seconds=round(now-getattr(self,'stall_since',now),1))
        print('等待客户端界面就绪：'+str(obs.phase)+' 回合='+str(obs.turn)+
              ' 阻塞='+(','.join(blockers) or '无')+' 警告='+(','.join(obs.warnings) or '无'),flush=True)
        # A hover tooltip can persist for as long as the cursor rests on a
        # card; move off the cards before waiting any longer.
        if getattr(self,'transport',None) is not None:self.park_neutral()
        archived=getattr(self,'stall_archived',set())
        if key in archived or self.last_image is None:return
        archived.add(key);self.stall_archived=archived
        from .live_interactions import archive_interaction
        from .model import Action
        try:
            archive_interaction(self,'ui-stall',Action(Kind.WAIT),self.last_obs,self.last_window,
                self.last_image,self.last_row.get('capture',{}),blockers=blockers,
                warnings=list(obs.warnings),display_state=obs.interaction.display_state if obs.interaction else None)
        except Exception as exc:
            try:self.audit.write('stall_frame_failed',reason=type(exc).__name__+': '+str(exc))
            except OSError:pass

    def menu_ready(self,obs):return self.returned_start_ready and obs.phase==Phase.MENU

    def completion_ready(self,obs):
        return bool(obs.phase==Phase.MENU and obs.scene=='BACON' and obs.mode=='GT_BATTLEGROUNDS' and self.returned_start_observed)

    def on_verified(self,action,before,after,outcome):
        if (before.phase==Phase.RECRUIT and before.turn==after.turn and
                action.kind not in (Kind.END,Kind.WAIT,Kind.CANCEL)):
            self.buy_handoff_key=(str(self.logs.session),after.game_id,after.turn)
        reoffered=getattr(self,'buy_reoffered_entities',set())
        recovered_key=(before.game_id,before.turn,str(action.kind),action.source)
        if action.kind==Kind.BUY and recovered_key in reoffered:
            reoffered.remove(recovered_key)
            self._buy_recovery_event('verified_after_reoffer',action=asdict(action),
                receipt_revision=outcome.receipt_revision)
        self.verified_log_position=max((r.log_position or 0 for r in after.resolutions if r.kind==action.kind and r.source==action.source and r.at_revision==outcome.receipt_revision),default=None)
        if (action.kind==Kind.REFRESH and str(outcome.status)=='confirmed'
                and (before.game_id,before.turn)==(after.game_id,after.turn) and after.shop
                and {c.entity_id for c in before.shop}!={c.entity_id for c in after.shop}):
            # 这里只确认服务器新商店；下一次BUY仍需独立捕获的客户端布局、
            # 本体与等级旗帜到位，不借共享绿光或全局任务列表判断。
            self.shop_refresh_pending=None
            audit=getattr(self,'audit',None)
            if audit:audit.write('refresh_stock_confirmed',revision=after.revision,
                shop_entities=[c.entity_id for c in after.shop],wait_for_all_animations=False,
                next_action_requires_fresh_objects=True)
        if action.kind==Kind.REORDER and 'placement_deviated' in outcome.proofs:
            # A known completed permutation is safe to continue from, but is
            # not evidence that another corrective drag would hit its slot.
            self.reorder_deviation_turn=(before.game_id,before.turn)
            self.cache_until=0
            self.audit.write('reorder_quarantined',game_id=before.game_id,turn=before.turn,
                requested=list(action.order),actual=[c.entity_id for c in after.board],until='next_turn',resend=False)
        if (action.kind==Kind.CHOOSE and str(outcome.status)=='confirmed' and
            any(c.entity_id==action.source and c.card_type=='HERO' for c in before.choices) and
            after.hero and after.hero.entity_id==action.source):
            self.hero_choice_sent=action.source;self.hero_choice_verified=True

    @measured_stage('semantic_projection')
    def semantic_observation(self,obs):
        from .state_health import projection_errors
        errors=projection_errors(obs) if obs.phase in (Phase.RECRUIT,Phase.CHOICE) else ()
        if errors:obs=replace(obs,warnings=obs.warnings+tuple('state_health:'+e for e in errors))
        network=getattr(getattr(self,'lifecycle',None),'network',None)
        if network and network.match and obs.mode=='GT_BATTLEGROUNDS':
            obs=replace(obs,game_id='match-'+network.match)
        snapshot=getattr(self,'selection_snapshot',None)
        if snapshot and obs.interaction and (obs.game_id,obs.interaction.request_id)==snapshot[:2]:
            obs=replace(obs,interaction=replace(obs.interaction,selected=snapshot[2],protocol='verified_multi_select'),ui_revision=self.selection_revision)
        return obs

    def policy_observation(self):
        """Log-only speculative input, NEVER a mouse authorization or last_obs.

        Client WAITING may be ignored for computation, not for execution.
        Partial root processing, I/O/network faults and unknown multi-selection
        fall back to the normal observed-state path.
        """
        if getattr(self,'storage_error',None):raise OSError(self.storage_error)
        with self.logs.lock:
            raw=self.logs.observe()
            if self.logs.session:self.lifecycle.network.inspect(str(self.logs.session))
            obs=self.semantic_observation(raw)
            if obs.phase not in (Phase.RECRUIT,Phase.CHOICE):return None
            if self.logs.error or blocks_game_input(self.lifecycle.network.connection,obs.phase):return None
            state=self.logs.state
            if obs.phase==Phase.RECRUIT and state.block_stack:return None
            if self.logs.tail and self.logs.tail.pending:return None
            if obs.interaction and (obs.interaction.display_state=='hidden' or obs.interaction.max_select!=1):return None
            if obs.phase==Phase.CHOICE and (not obs.choices or any(not c.card_id or not c.card_type for c in obs.choices)):return None
            warnings=tuple(w for w in obs.warnings if w!='choice_client_waiting' and not (
                w=='gold_unknown' and obs.choices and all(c.card_type=='HERO' for c in obs.choices)))
            if warnings:return None
            if not obs.ready and not (obs.interaction and obs.interaction.display_state=='waiting'):return None
            health=self.health.inspect(obs,session=self.logs.session,network=self.lifecycle.network.connection)
            if not health['healthy']:return None
            # Preserve the adapter's real source identity, just as observe()
            # does. Rebranding only the preview causes every reuse check to
            # fail even when all game state and candidates are identical.
            obs=replace(obs,ready=True,warnings=warnings,captured_at=time.time())
            previous=getattr(self,'last_obs',None)
            if previous is not None and (previous.game_id,previous.revision)!=(obs.game_id,obs.revision):self.cache_until=0
            return replace(obs,actions=self._capabilities(obs,[],semantic=True))

    def verification_timeout(self,action):
        # Saved ChoiceCardMgr traces include almost 30 seconds of task-list
        # animation before BEGIN. The parent must survive that forced wait.
        # Queuing is bounded by the whole session, not a short input timeout.
        # A long queue must never cause a second click on Start.
        return result_timeout(action,getattr(self,'run_seconds',2400))

    def waiting_timeout(self,obs):
        # A game-server reconnect is not an unsupported interaction: extend the
        # wait instead of ending the session while the client is recovering.
        if self.connection_wait():return 240
        if obs.phase==Phase.CHOICE:
            # 发现/饰品/任务选择入场动画可长于 8 秒；仍为显式超时，不无限等。
            return 20
        if obs.phase==Phase.RECRUIT and (obs.gold or 0)>0 and not obs.actions:
            # An intentional skip is an explained stall: wait for the turn to
            # end instead of reporting an unsupported interaction.
            if any(key[:2]==(obs.game_id,obs.turn) for key in getattr(self,'unavailable_actions',{})):return 240
            return 30
        return 240

    def recover_stall(self,obs,decision):
        """一次有界 STA/WAIT 恢复：只重采一帧并重映射，绝不发送输入。

        慢动画、旧帧或一次性映射失败会让 WAIT 超过预算；这里强制一次全屏
        采集，若新快照本身已可决策就让控制器重新决策一次。恢复失败仍按原
        逻辑停止，因为继续在不可信状态上点击更危险。
        """
        key=(obs.game_id,obs.turn,obs.interaction.request_id if obs.interaction else None,
             decision.action.key)
        tried=getattr(self,'stall_recoveries_done',set())
        if key in tried:return False
        tried.add(key);self.stall_recoveries_done=tried
        blockers=list((getattr(self,'last_row',None) or {}).get('blockers') or ())
        self.audit.write('stall_recovery_started',revision=obs.revision,phase=str(obs.phase),
            turn=obs.turn,request_id=obs.interaction.request_id if obs.interaction else None,
            blockers=blockers,warnings=list(obs.warnings),action=asdict(decision.action))
        self.cache_until=0;self._force_full_capture=True
        try:fresh=self.observe()
        except Exception as exc:
            self.audit.write('stall_recovery_failed',reason=type(exc).__name__+': '+str(exc))
            return False
        # 客户端层只判断“这一帧是否已可执行”；选择请求证据仍由策略层闸门复核。
        blocked=list((getattr(self,'last_row',None) or {}).get('blockers') or ())
        recovered=bool(fresh.actions) and fresh.ready and not blocked and not fresh.warnings
        self.audit.write('stall_recovery_result',recovered=recovered,revision=fresh.revision,
            actions=[a.key for a in fresh.actions],blockers=blocked,warnings=list(fresh.warnings),
            phase=str(fresh.phase),physical_input=False)
        return recovered

    def _note_skip(self,action,reason):
        """Count an intentional per-turn skip; a systemic fault still stops.

        A few unlocatable actions are normal (animation, occlusion, a framing
        gap). Dozens in one session mean the mapper/policy contract is broken,
        and silently idling until the session deadline would hide it.
        """
        total=getattr(self,'skipped_total',0)+1
        self.skipped_total=total
        if total>self.SKIP_LIMIT:
            self.audit.write('live_skip_limit_reached',action=asdict(action),reason=reason,total=total)
            raise RuntimeError('live_action_skip_limit_reached:'+str(total))

    SKIP_LIMIT=30

    def _skip_unsafe(self,action,reason):
        """Defer a never-sent action; new stable geometry may re-offer it twice.

        Only used when the action's own source slot or zone could not be
        independently proven. No input was sent and no game state changed, so
        continuing with other verified actions is safe. The action stays
        unavailable until new evidence or the next turn; the policy re-plans. Forced
        interactions and ambiguous deliveries keep their stop behaviour.
        """
        self._note_skip(action,reason)
        key=(self.last_obs.game_id,self.last_obs.turn,str(action.kind),action.source)
        unavailable=getattr(self,'unavailable_actions',{})
        unavailable[key]=reason;self.unavailable_actions=unavailable
        from .geometry_recovery import remember
        remember(self,action,reason)
        history=getattr(self,'unsafe_skipped_actions',[])
        if len(history)<50:history.append(dict(action=asdict(action),reason=reason,turn=self.last_obs.turn,game_id=self.last_obs.game_id))
        self.unsafe_skipped_actions=history
        self.audit.write('live_unsafe_action_skipped',action=asdict(action),reason=reason,
            game_id=self.last_obs.game_id,turn=self.last_obs.turn,input_sent=False,until='new_stable_geometry_or_next_turn')
        self.cache_until=0
        print('该动作源槽位无法独立证明，暂缓并继续其他操作：'+str(action.kind)+' / '+str(action.source),flush=True)
        raise ValueError('live_unsafe_action_no_input:'+reason)

    def abandon_action(self,action,before,after,reason):
        """A dispatch independently proven to have had no effect was abandoned.

        不计成功。默认本回合屏蔽；购买仅在重新证明没有客户端提交、
        没有匹配回执且当前对象重新就绪后，同状态保留原购买意图最多重试一次。
        """
        key=(before.game_id,before.turn,str(action.kind),action.source)
        unavailable=getattr(self,'unavailable_actions',{})
        unavailable[key]=reason;self.unavailable_actions=unavailable
        from .late_response import LateResponseWatch
        if not hasattr(self,'late_responses'):self.late_responses=LateResponseWatch()
        self.late_responses.abandon(action,before,after)
        uses=getattr(self,'buy_recovery_uses',{})
        uses={k:v for k,v in uses.items() if k[:2]==(after.game_id,after.turn)}
        self.buy_recovery_uses=uses
        if (action.kind==Kind.BUY and reason=='shop_source_still_offered'
                and before.turn==after.turn and after.phase==Phase.RECRUIT):
            count_key=(after.game_id,after.turn,action.source)
            if uses.get(count_key,0)==0:
                uses[count_key]=1
                self.buy_recoveries={key:dict(action=action,session=str(self.logs.session),
                    before=replace(before,resolutions=(),actions=(action,)),
                    started=time.monotonic(),after_time=time.time(),sample=None,
                    selection_key=self._buy_retry_state(before))}
                self._buy_recovery_event('scheduled',action=asdict(action),input_sent=False,
                    reason=reason,max_reoffers=1,observation_budget_seconds=8)
            else:
                self._buy_recovery_event('exhausted',action=asdict(action),input_sent=False,
                    reason='single_reoffer_already_used')
        # 普通几何恢复不能直接重发该尝试；购买重开必须经过独立无提交取证。
        getattr(self,'geometry_deferrals',{}).pop(key,None)
        self.cache_until=0
        self.audit.write('live_action_abandoned',action=asdict(action),reason=reason,
            game_id=before.game_id,turn=before.turn,after_revision=after.revision,resend=False)

    def _ui_wait_expired(self,action,reason):
        key=(self.last_obs.game_id,self.last_obs.turn,self.last_obs.interaction.request_id if self.last_obs.interaction else None,action.key)
        waits=getattr(self,'ui_waits',None)
        if waits is None:self.ui_waits={};waits=self.ui_waits
        first=waits.setdefault(key,time.monotonic())
        elapsed=time.monotonic()-first
        if elapsed>=4 and key not in getattr(self,'slow_archived',set()):
            # Half the budget: keep one frame of the state that cannot be
            # located, so a slow mapping has evidence without extra captures
            # during normal play.
            if all(hasattr(self,name) for name in ('last_image','last_row','last_window','mapper')):
                from .live_interactions import archive_interaction
                archive_interaction(self,'ui-slow',action,self.last_obs,self.last_window,
                    self.last_image,self.last_row.get('capture',{}),reason=reason,elapsed_seconds=elapsed,
                    mappings=[asdict(m) for m in self.mappings],ui=self.mapper.last_ui_state,
                    diagnostics=self.mapper.last_diagnostics)
            archived=getattr(self,'slow_archived',set());archived.add(key);self.slow_archived=archived
        budget=45 if self.last_obs.phase==Phase.CHOICE and action.kind in (Kind.CHOOSE,Kind.SELECT,Kind.CONFIRM) else 8
        if elapsed>=budget:
            if all(hasattr(self,name) for name in ('last_image','last_row','last_window','mapper')):
                from .live_interactions import archive_interaction
                archive_interaction(self,'ui-unavailable',action,self.last_obs,self.last_window,
                    self.last_image,self.last_row.get('capture',{}),reason=reason,
                    mappings=[asdict(m) for m in self.mappings],ui=self.mapper.last_ui_state,
                    diagnostics=self.mapper.last_diagnostics)
            if getattr(self,'uses_green_signal',False) and getattr(getattr(self,'mapper',None),'mode',None)=='geometry':
                raise RuntimeError('input_admission_timeout:'+reason)
            parts=reason.removeprefix('live_plan_blocked:').split(',')
            local_geometry=bool(parts) and all(r in ('target_pixels_not_stable','target_quiet_window_pending','target_regions_unknown','action_green_outline_absent','recruit_green_signal_absent','action_source_geometry_unconfirmed') or r.startswith('unknown_mapping:') for r in parts)
            # Any ordinary recruited action may be skipped: a temporary mapping
            # or animation gap must not end the session. Forced interactions
            # keep their stop behaviour - the phase and interaction guards
            # below already exclude choice/target/confirm/cancel.
            skippable=action.kind in (Kind.BUY,Kind.SELL,Kind.PLAY,Kind.UPGRADE,Kind.REFRESH,
                Kind.FREEZE,Kind.HERO_POWER,Kind.ACTIVATE,Kind.REORDER)
            if skippable and local_geometry and self.last_obs.phase==Phase.RECRUIT and self.last_obs.interaction is None:
                # Called only during planning or after a proven NOT_SENT.
                # Forced choices, pending targets and ambiguous deliveries
                # must retain their stop/reconcile behavior.
                self._note_skip(action,reason)
                unavailable=getattr(self,'unavailable_actions',{})
                unavailable[(self.last_obs.game_id,self.last_obs.turn,str(action.kind),action.source)]=reason
                self.unavailable_actions=unavailable;self.cache_until=0
                from .geometry_recovery import remember
                remember(self,action,reason)
                history=getattr(self,'skipped_optional_actions',[])
                history.append(dict(action=asdict(action),reason=reason,turn=self.last_obs.turn,game_id=self.last_obs.game_id))
                self.skipped_optional_actions=history
                self.audit.write('live_optional_action_skipped',action=asdict(action),reason=reason,
                    game_id=self.last_obs.game_id,turn=self.last_obs.turn,input_sent=False,until='new_stable_geometry_or_next_turn')
                print('该动作暂时无法安全定位，暂缓并继续其他操作：'+str(action.kind)+' / '+str(action.source),flush=True)
                raise ValueError('live_optional_action_unavailable_no_input:'+reason)
            self.audit.write('live_ui_unavailable',action=asdict(action),reason=reason,waited_seconds=time.monotonic()-first)
            raise RuntimeError('live_ui_unavailable_no_input:'+reason)

    @measured_stage('result_poll')
    def observe_result(self,action):
        if (action.kind==Kind.END and (getattr(self,'control_deferrals',None) or getattr(self,'buy_recoveries',None))
                and time.monotonic()>=getattr(self,'next_control_capture',0)):
            # 被动END不能把未知控件饿死到回合结束；只重观测，不发送输入。
            self.next_control_capture=time.monotonic()+.5;self.cache_until=0
            return self.observe()
        if action.kind==Kind.ENTER and getattr(self,'enter_sent',False) and not getattr(self,'matched',False):
            # Poll logs on every verification cycle. Re-check queue pixels only
            # every two seconds, or immediately when an actionable phase appears.
            with self.logs.lock:
                raw=self.logs.observe()
                if self.logs.session:self.lifecycle.network.inspect(str(self.logs.session))
                obs=self.semantic_observation(raw)
                for mirror in getattr(self,'mirrors',{}).values():
                    if mirror.source.stat().st_size!=mirror.offset:self._mirror(mirror)
            now=time.monotonic()
            if obs.phase in (Phase.CHOICE,Phase.RECRUIT,Phase.COMBAT) or now>=getattr(self,'next_queue_capture',0):
                self.next_queue_capture=now+2;self.cache_until=0
                return self.observe()
            return replace(obs,phase=Phase.UNKNOWN,actions=(),ready=False,source='hearthstone_live',
                turn=None,gold=None,hero=None,hand=(),board=(),shop=(),choices=(),controls=(),
                branches=(),choice_rewards=(),interaction=None)
        if action.kind in (Kind.ENTER,Kind.DISMISS,Kind.SELECT,Kind.CANCEL):
            self.cache_until=0;return self.observe()
        with self.logs.lock:
            raw=self.logs.observe()
            if self.logs.session:self.lifecycle.network.inspect(str(self.logs.session))
            obs=self.semantic_observation(raw)
            # Receipt polling need not capture images, but must preserve the
            # raw action block even if the console is closed before close().
            for mirror in getattr(self,'mirrors',{}).values():
                if mirror.source.stat().st_size!=mirror.offset:self._mirror(mirror)
            if self.hero_choice_sent and obs.hero and obs.hero.entity_id==self.hero_choice_sent:
                self.hero_choice_verified=True
            offset=getattr(self.logs,'revision_offset',0)
            pending=[Resolution(at_revision=obs.revision,complete=False,log_position=obs.log_position,**{**frame['action'],'started_at_revision':frame['action']['started_at_revision']+offset})
                for frame in self.logs.state.block_stack if frame['action']]
            if action.kind==Kind.END:
                # Compare the same supported action vocabulary used by policy,
                # without capturing pixels merely to interrupt an input-free wait.
                obs=replace(obs,actions=self._capabilities(obs,(),semantic=True))
            return replace(obs,resolutions=obs.resolutions+tuple(pending))

    def _remember_evidence(self,*,pin=False):
        row=getattr(self,'last_row',None);image=getattr(self,'last_image',None)
        if not row or image is None or row.get('blockers'):return
        from .evidence_window import EvidenceWindow
        if not hasattr(self,'evidence_window'):self.evidence_window=EvidenceWindow()
        sample={key:row.get(key) for key in ('id','session','observation','window','capture','log_snapshot_unchanged')}
        key=row.get('id',row['capture'].get('frame_sequence'))
        self.evidence_window.remember(key,image,sample)
        if pin:self.evidence_window.pin(key)

    def _queue_diagnostic_window(self,reason):
        ring=getattr(self,'evidence_window',None);work=getattr(self,'archive_queue',None)
        if ring is None or work is None:return
        for key,image,sample in ring.diagnostic():
            stem=f'context-{reason}-{key}-{time.time_ns()}'
            metadata=dict(stage='preceding_context',reason=reason,frame_id=key,
                input_frame=key==ring.pinned,sample=sample)
            try:work.put_nowait((stem,image,metadata))
            except queue.Full:
                self.context_frames_not_queued=getattr(self,'context_frames_not_queued',0)+1
                self.audit.write('evidence_window_backpressure',frame_id=key,reason=reason,
                    input_frame=key==ring.pinned,capture=sample['capture'],image_saved=False)

    def pending_diagnostic(self,action,before,after,result,elapsed):
        if action.kind in (Kind.ENTER,Kind.END):return  # Passive waits are not stalled gestures.
        key=(before.game_id,before.turn,before.revision,action.key)
        if getattr(self,'pending_diagnostic_key',None)==key:return
        self.pending_diagnostic_key=key
        self.audit.write('pending_action_diagnostic',action=asdict(action),before_revision=before.revision,
            after_revision=after.revision,elapsed_seconds=elapsed,result=asdict(result),resend=False)
        # Capture the failure while the original hand/target may still be on
        # screen. This does not submit or cancel anything in the client.
        self.cache_until=0;self.observe()
        if self.last_window.blockers:return
        self._queue_diagnostic_window('pending')
        from .live_interactions import archive_interaction
        archive_interaction(self,'pending-action',action,self.last_obs,self.last_window,self.last_image,
            self.last_row.get('capture',{}),result=asdict(result),elapsed_seconds=elapsed,
            mappings=[asdict(m) for m in self.mappings],diagnostics=self.mapper.last_diagnostics)

    def failure_context(self,action):
        obs=getattr(self,'last_obs',None);row=getattr(self,'last_row',None) or {}
        relevant={e for e in (action.source,action.target) if e is not None} if action else set()
        return dict(frame_id=row.get('id'),revision=obs.revision if obs else None,
            game_id=obs.game_id if obs else None,turn=obs.turn if obs else None,
            phase=str(obs.phase) if obs else None,logical_ready=obs.ready if obs else None,
            blockers=row.get('blockers',[]),warnings=list(obs.warnings) if obs else [],
            action=asdict(action) if action else None,readiness=getattr(self,'last_readiness',None),
            readiness_invalidation=getattr(self,'last_readiness_invalidation',None),
            hand_source_binding=getattr(self,'last_hand_source_binding',None),
            regions=[asdict(m) for m in getattr(self,'mappings',()) if m.entity_id in relevant or m.key.startswith(('position:board:','region:board'))],
            ui=getattr(getattr(self,'mapper',None),'last_ui_state',{}),
            root_processing=bool(getattr(getattr(self.logs,'state',None),'block_stack',())))

    def archive_failure(self,action,context):
        # Reuse the last acquired client image. Never take an extra desktop
        # screenshot during failure cleanup or save an occluded window.
        if not action or not getattr(self,'last_image',None) or self.last_window.blockers:return
        self._queue_diagnostic_window('failure')
        from .live_interactions import archive_interaction
        archive_interaction(self,'failure-last-frame',action,self.last_obs,self.last_window,self.last_image,
            self.last_row.get('capture',{}),context=context)

    def connection_wait(self):return self.lifecycle.network.connection in ('disconnected','reconnecting','restoring','connecting')

    def can_continue(self,action,before,after):
        return bool(action.kind in (Kind.PLAY,Kind.HERO_POWER,Kind.ACTIVATE,Kind.CHOOSE) and after.phase==Phase.CHOICE and after.interaction and
            before.game_id==after.game_id and before.turn==after.turn and
            (before.interaction is None or before.interaction.request_id!=after.interaction.request_id) and
            any(r.kind==action.kind and r.source==action.source and r.target==action.target and r.started_at_revision is not None and r.started_at_revision>before.revision for r in after.resolutions))

    def defer_action(self,action,reason):
        if reason.startswith('input_preparation_timeout:'):
            # 同一未输入意图已用尽执行器内预算，不能被外层重试重置。
            raise RuntimeError(reason)
        if reason in ('prepress_capture_expiring','prepress_plan_expired','native_plan_expired'):
            # 只有NOT_SENT才会调用此恢复：重新观测与决策，绝不重发已按下的动作。
            # 此预算不随input_plan成功而清空，避免昂贵预检反复过期形成无限循环。
            obs=self.last_obs
            from .action_context import input_context_token
            key=(str(self.logs.session),obs.game_id,obs.turn,action.key,
                getattr(getattr(self,'last_window',None),'token',None),
                input_context_token(obs,action,ignore_playback=True))
            wait=getattr(self,'prepress_expiry_wait',None)
            if wait is None or wait[0]!=key:self.prepress_expiry_wait=(key,time.monotonic())
            elapsed=time.monotonic()-self.prepress_expiry_wait[1]
            self.invalidate_readiness('prepress_evidence_expired');self.cache_until=0
            self.audit.write('live_prepress_expiry_deferred',action=asdict(action),reason=reason,
                elapsed_seconds=elapsed,budget_seconds=8,input_sent=False,resend=False)
            if elapsed>=8:raise RuntimeError('input_preparation_timeout:'+reason)
            return True
        if getattr(self,'uses_green_signal',False) and getattr(getattr(self,'mapper',None),'mode',None)=='geometry' and 'unknown_mapping:' in reason:
            raise RuntimeError('unsupported_coordinate_layout:'+reason)
        if reason.startswith(('live_optional_action_unavailable_no_input:','live_unsafe_action_no_input:')):
            self.cache_until=0
            return True
        transient=reason.startswith('live_plan_blocked:') or reason in (
            'native_plan_expired','live_window_layout_or_focus_changed','start_button_not_ready',
            'live_action_not_advertised','left_button_already_held','live_action_context_changed',
            'native_observation_changed','live_observation_changed_while_waiting')
        if transient:
            try:self._ui_wait_expired(action,reason)
            except ValueError as exc:
                if not str(exc).startswith('live_optional_action_unavailable_no_input:'):raise
                return True
            self.cache_until=0
            self.audit.write('live_action_deferred',action=asdict(action),reason=reason)
        return transient

    def _buy_recovery_event(self,event,**data):
        counts=getattr(self,'buy_recovery_counts',{})
        counts[event]=counts.get(event,0)+1;self.buy_recovery_counts=counts
        self.audit.write('buy_recovery_'+event,**data)

    def _recover_buy_candidates(self,obs,maps,*,semantic):
        """漏买后短暂只取证；重开的是当前候选，不是旧输入计划。"""
        recoveries=getattr(self,'buy_recoveries',{})
        if not recoveries:return False
        if semantic:return self._buy_recovery_waiting(obs)
        row=getattr(self,'last_row',None) or {};capture=row.get('capture',{})
        from .verification import no_effect_proven
        for key,ticket in list(recoveries.items()):
            action=ticket['action'];before=ticket['before']
            from .action_budgets import BUY_RECOVERY_SECONDS
            expired=(obs.phase!=Phase.RECRUIT or obs.interaction is not None or
                (str(self.logs.session),obs.game_id,obs.turn,obs.local_player_id,obs.mode)!=
                (ticket['session'],before.game_id,before.turn,before.local_player_id,before.mode)
                or time.monotonic()-ticket['started']>=BUY_RECOVERY_SECONDS)
            if expired:
                recoveries.pop(key,None)
                self._buy_recovery_event('expired',action=asdict(action),input_sent=False,
                    reason='context_changed_or_observation_budget_expired')
                continue
            if self.poll_late_responses(obs):
                recoveries.pop(key,None)
                self._buy_recovery_event('late_response',action=asdict(action),input_sent=False,resend=False)
                continue
            current=next((a for a in obs.actions if a.kind==Kind.BUY and a.source==action.source),None)
            prior=next((c for c in before.shop if c.entity_id==action.source),None)
            card=next((c for c in obs.shop if c.entity_id==action.source),None)
            if not current or not prior or not card:
                recoveries.pop(key,None)
                self._buy_recovery_event('expired',action=asdict(action),input_sent=False,
                    reason='source_no_longer_offered')
                continue
            stamp=capture.get('end')
            if (not obs.ready or obs.options_pending or obs.client_task_list is not None
                    or row.get('blockers') or row.get('recruit_root_processing')
                    or not row.get('log_snapshot_unchanged') or capture.get('reused_capture')
                    or getattr(self.logs,'error',None) or not isinstance(stamp,(int,float))
                    or not 0<=time.time()-stamp<=.75 or stamp<=ticket['after_time']):continue
            if (card.card_id,card.controller,card.zone)!=(prior.card_id,prior.controller,prior.zone):continue
            if self._buy_retry_state(obs)!=ticket.get('selection_key',self._buy_retry_state(before)):
                recoveries.pop(key,None)
                self._buy_recovery_event('selection_changed',action=asdict(action),input_sent=False,
                    reason='policy_state_changed_no_automatic_retry')
                continue
            proven,reason=no_effect_proven(action,before,obs)
            if not proven:
                recoveries.pop(key,None)
                self._buy_recovery_event('refused',action=asdict(action),input_sent=False,reason=reason)
                continue
            gate=getattr(self,'gate',None)
            if not callable(getattr(gate,'object_evidence',None)):continue
            objects=gate.object_evidence(obs,current,self.last_image,maps,capture,row['window']['token'],str(self.logs.session))
            if not objects['ready']:continue
            if not self._buy_input_handoff(obs,maps)['ready']:continue
            identity=capture.get('frame_sequence',row.get('id'))
            sample=ticket['sample']
            if sample is None:
                ticket['sample']=(identity,stamp);continue
            if identity==sample[0] or not .033<=stamp-sample[1]<=2:continue
            # 两次当前独立捕获都已核验对象，原失败仍无提交/回执。
            # 撤销的仅是该失败尝试的迟到监测，其他票据继续保留。
            watcher=getattr(self,'late_responses',None)
            retired=watcher.retire_no_submission(action,before) if watcher else 0
            self.unavailable_actions.pop(key,None);recoveries.pop(key,None)
            reoffered=getattr(self,'buy_reoffered_entities',set());reoffered.add(key)
            self.buy_reoffered_entities=reoffered
            self.buy_retry_ready=dict(action=current,before=obs,session=str(self.logs.session),
                original_action=action,original_before=before,selection_key=self._buy_retry_state(obs))
            self.invalidate_readiness('buy_candidate_reoffered')
            self._buy_recovery_event('reoffered',action=asdict(current),input_sent=False,
                reason='fresh_no_submission_and_object_proofs',retired_old_watch=retired,
                original_revision=before.revision,current_revision=obs.revision,
                original_log_position=before.log_position,current_log_position=obs.log_position,
                model_must_choose_again=False,retained_selection=True)
        return bool(recoveries)

    @staticmethod
    def _buy_retry_state(obs):
        from .decision_ahead import decision_key
        # 授权编号可以更新；不能因一次无效果输入再次推进模型决策。
        return decision_key(replace(obs,actions=(),ready=True))

    def _buy_recovery_waiting(self,obs):
        """纯读：只阻塞同一招募上下文中的短暂取证，不阻塞战斗或强制选择。"""
        from .action_budgets import BUY_RECOVERY_SECONDS
        if obs.phase!=Phase.RECRUIT or obs.interaction is not None:return False
        return any((str(self.logs.session),obs.game_id,obs.turn,obs.local_player_id,obs.mode)==
            (t['session'],t['before'].game_id,t['before'].turn,t['before'].local_player_id,t['before'].mode)
            and time.monotonic()-t['started']<BUY_RECOVERY_SECONDS
            for t in getattr(self,'buy_recoveries',{}).values())

    def execution_retry_pending(self):
        return bool(getattr(self,'buy_recoveries',{}) or getattr(self,'buy_retry_ready',None))

    def execution_retry(self,obs):
        ticket=getattr(self,'buy_retry_ready',None)
        if not ticket:return None
        self.buy_retry_ready=None
        action=ticket['action']
        current=next((a for a in obs.actions if a==replace(action,option_id=a.option_id,
            authorization_source=a.authorization_source)),None)
        from .verification import no_effect_proven
        proven,reason=no_effect_proven(ticket['original_action'],ticket['original_before'],obs)
        processing=any(f.get('action') for f in getattr(getattr(self.logs,'state',None),'block_stack',()))
        if not proven or processing:
            key=(obs.game_id,obs.turn,str(action.kind),action.source)
            self.unavailable_actions[key]='retry_proof_invalidated'
            self.invalidate_readiness('buy_retry_proof_invalidated');self.cache_until=0
            self._buy_recovery_event('refused',action=asdict(action),input_sent=False,
                reason=reason if not proven else 'processing_started_before_retry')
            return None
        if (ticket['session']!=str(self.logs.session) or current is None or
                obs.phase!=Phase.RECRUIT or obs.interaction is not None or
                self._buy_retry_state(obs)!=ticket['selection_key']):
            self._buy_recovery_event('selection_changed',action=asdict(action),input_sent=False)
            return None
        action=current
        self._buy_recovery_event('selected_retry',action=asdict(action),input_sent=False,
            retained_selection=True,current_revision=obs.revision)
        return Decision(action,obs.revision,'execution_retry: confirmed no submission; retained BUY')

    def _buy_input_handoff(self,obs,maps):
        """每回合只取一次输入交接证据；不把卡面存在当作客户端已开放输入。"""
        key=(str(self.logs.session),obs.game_id,obs.turn)
        if getattr(self,'buy_handoff_key',None)==key:return dict(ready=True,already_confirmed=True)
        from .dependency_readiness import CONTROLS
        from .green_readiness import action_signal
        candidates=list(obs.actions);offered={a.source for a in candidates}
        # 模型候选不一定列出所有HUD按钮。当前自有、槽位已知的特殊控件
        # 也能提供纯视觉交接见证；不把此探针加入合法动作，更不点击它。
        candidates.extend(Action(Kind.ACTIVATE,c.entity_id) for c in obs.controls
            if c.entity_id not in offered and c.controller==obs.local_player_id
            and c.card_type=='GAME_MODE_BUTTON' and c.tags.get('GAME_MODE_BUTTON_SLOT') is not None)
        for candidate in candidates:
            control=next((c for c in obs.controls if c.entity_id==candidate.source),None)
            if not control or not (candidate.kind in CONTROLS or candidate.kind==Kind.ACTIVATE):continue
            if control.tags.get('LITERALLY_UNPLAYABLE') in ('1',1):continue
            # 只查看当前绑定HUD控件自身的正面信号；不用手牌或随从的绿图案。
            current_maps=maps
            if hasattr(self.mapper,'ensure'):
                from .capture_broker import FullCaptureRequired
                try:current_maps=self.mapper.ensure(obs,candidate,self.last_image,self.last_row['window'],maps)
                except FullCaptureRequired:
                    self._force_full_capture=True
                    continue
                except ValueError as exc:
                    if str(exc).startswith('unknown_mapping:'):continue
                    raise
            signal=action_signal(self.mapper,obs,candidate,self.last_image,current_maps)
            if signal['ready']:
                self.buy_handoff_key=key
                self.audit.write('buy_round_input_handoff',game_id=obs.game_id,turn=obs.turn,
                    witness_kind=str(candidate.kind),witness_entity=candidate.source,signal=signal)
                return dict(ready=True,signal=signal)
        return dict(ready=False,reason='recruit_input_handoff_pending')

    def _control_event(self,event,**data):
        counts=getattr(self,'control_readiness_counts',{})
        counts[event]=counts.get(event,0)+1;self.control_readiness_counts=counts
        self.audit.write('control_readiness_'+event,**data)

    def _captured_capabilities(self,obs,maps):
        """调用前已发布本次row/image；只有此真实捕获入口推进购买恢复。"""
        self._recover_buy_candidates(obs,maps,semantic=False)
        return self._capabilities(obs,maps,semantic=getattr(self,'decision_ahead',False))

    def _capabilities(self,obs,maps,*,semantic=False):
        # Observe publishes the fresh row/image before calling this helper;
        # speculative semantic-only calls have no maps and cannot recover.
        from .geometry_recovery import recover
        if not (getattr(self,'uses_green_signal',False) and getattr(getattr(self,'mapper',None),'mode',None)=='geometry'):recover(self,obs,maps)
        visible={m.key:m for m in maps if m.status=='matched' and m.rect}
        result=[];rejected=[]
        unavailable={key:reason for key,reason in getattr(self,'unavailable_actions',{}).items() if key[:2]==(obs.game_id,obs.turn)}
        self.unavailable_actions=unavailable
        if self._buy_recovery_waiting(obs):
            return ()  # 纯投影不推进恢复；强制选择和过期上下文永远不受此锁影响。
        unavailable=self.unavailable_actions
        processing=any(f.get('action') for f in getattr(getattr(getattr(self,'logs',None),'state',None),'block_stack',()))
        # 视觉等待不修改合法候选，保留策略已经选定的操作。
        for a in obs.actions:
            # 合法候选不因播放动画被隐藏；准入统一在派发时处理。
            if (obs.game_id,obs.turn,str(a.kind),a.source) in unavailable:continue
            card=next((c for c in obs.hand if c.entity_id==a.source),None)
            if a.target in {c.entity_id for c in obs.hand}:
                box=getattr(getattr(self,'mapper',None),'target_box',lambda *a:None)(obs,a.target)
                if box is not None:a=replace(a,evidence='verified_hand_target')
            if a.kind==Kind.PLAY and card and card.tags.get('LITERALLY_UNPLAYABLE') in ('1',1):continue
            if a.kind==Kind.PLAY and magnetic(card) and not (a.interaction and a.interaction.placement=='magnetize') and (a.target is not None or a.sub_option_id is not None):
                # Battlecry/Choose-One does not remove Magnetic placement
                # semantics. Keep its target/branch, normalize only an
                # unspecified ordinary summon to the safe append position.
                if len(obs.board)>=7:continue
                if obs.board and a.position is None:a=replace(a,position=len(obs.board)+1,evidence='verified_insertion')
            if a.kind==Kind.PLAY and magnetic(card) and a.target is None and a.sub_option_id is None:
                # A gap to the LEFT of a Mech is a fusion zone, even when the
                # gap is empty wood. Ordinary magnetic summons use the far
                # right gap; fusion is represented as a distinct action.
                if len(obs.board)<7:
                    result.append(replace(a,position=len(obs.board)+1,evidence='verified_insertion') if obs.board else a)
                if card.tags.get('BATTLECRY')!='1':
                    from .model import InteractionSpec
                    for target in obs.board:
                        if mechanical(target) and target.controller==obs.local_player_id and (semantic or all(k in visible for k in (f'entity:{target.entity_id}',f'position:board:{target.position}'))):
                            result.append(replace(a,target=target.entity_id,position=None,evidence='verified_magnetic',
                                interaction=InteractionSpec(placement='magnetize',mandatory=True)))
                continue
            from .action_geometry import route
            geometry=route(a,obs,visible)
            if geometry.error:continue
            if a.kind==Kind.REORDER:
                if getattr(self,'reorder_deviation_turn',None)==(obs.game_id,obs.turn):continue
                if not semantic and any(k not in visible for k in geometry.required):continue
                a=replace(a,evidence='verified_board_move')
            keys=geometry.required
            missing=[k for k in keys if k not in visible]
            # Supported semantic actions survive temporary animation/mapping
            # gaps. The selected action waits for current geometry at dispatch.
            # Hiding PLAY here lets a policy spend its budget on REFRESH instead.
            if a.kind==Kind.ACTIVATE:
                c=next((c for c in obs.controls+obs.board if c.entity_id==a.source),None)
                if not c or not (c.tags.get('INTERACTABLE_OBJECT')=='1' or c.card_type=='GAME_MODE_BUTTON'):continue
            result.append(a)
            card=next((c for c in obs.hand if c.entity_id==a.source),None)
            if a.kind==Kind.PLAY and card and card.card_type=='MINION' and a.target is None and a.sub_option_id is None and 0<len(obs.board)<7:
                for position in range(1,len(obs.board)+2):
                    if semantic or f'position:board:{position}' in visible:result.append(replace(a,position=position,evidence='verified_insertion'))
            if missing:rejected.append(dict(action=asdict(a),missing=missing))
        from .client_protocol import cancellable_target
        if cancellable_target(obs):result.append(Action(Kind.CANCEL,choice_id=obs.interaction.request_id,evidence='verified_target_cancel'))
        # END仅表示可中断的本地等待，不发送输入；动画不能剥夺保留场面的候选。
        if getattr(self,'phase_wait_enabled',False) and obs.phase==Phase.RECRUIT and obs.ready and not obs.options_pending and not obs.interaction and obs.turn is not None and not obs.warnings:
            result.append(Action(Kind.END,evidence='verified_phase_wait'))
        signature=(obs.game_id,obs.revision,tuple(a.key for a in result))
        if hasattr(self,'audit') and signature!=getattr(self,'capability_signature',None):
            self.audit.write('live_capability_resolution',revision=obs.revision,server_actions=[asdict(a) for a in obs.actions],supported=[a.key for a in result],awaiting_mapping=rejected)
            self.capability_signature=signature
        # Prefer a currently measured hand slot among equivalent PLAY choices.
        # Keep every semantic PLAY: if none is mapped, the policy still waits
        # on PLAY instead of falling through to BUY/REFRESH.
        if semantic:return tuple({a.key:a for a in result}.values())
        plays=iter(sorted((a for a in result if a.kind==Kind.PLAY),key=lambda a:f'entity:{a.source}' not in visible))
        return tuple({a.key:a for a in (next(plays) if a.kind==Kind.PLAY else a for a in result)}.values())

    def activate(self):
        w=self.windows.snapshot(self.binding.hwnd)
        if w.pid!=self.binding.pid or w.process_created!=self.binding.process_created:raise RuntimeError('live_binding_changed_before_activation')
        u=self.windows.u;u.ShowWindow.argtypes=[W.HWND,C.c_int];u.SetForegroundWindow.argtypes=[W.HWND]
        u.ShowWindow(w.hwnd,9);u.SetForegroundWindow(w.hwnd)

    def _capture_region(self,obs,window):
        if getattr(self,'uses_green_signal',False):return None
        if getattr(self,'_force_full_capture',False):return None
        if not getattr(self.mapper,'used_geometry',False):return None
        if self.last_obs is None or (obs.game_id,obs.phase,obs.turn)!=(self.last_obs.game_id,self.last_obs.phase,self.last_obs.turn):return None
        return self.mapper.capture_region(obs,(window.client.width,window.client.height),window.to_dict())

    def _reuse_geometry_observation(self):
        if getattr(self,'uses_green_signal',False):return None
        mapper=self.mapper;evidence=getattr(self.gate,'evidence',None)
        if getattr(self,'_force_full_capture',False) or not getattr(mapper,'used_geometry',False) or evidence is None:return None
        raw=self.semantic_observation(self.logs.observe())
        if raw.phase!=Phase.RECRUIT or raw.client_task_list is not None:return None
        if any(f.get('action') for f in getattr(getattr(self.logs,'state',None),'block_stack',())):return None
        if getattr(self.logs,'error',None):return None
        network=getattr(self.lifecycle,'network',None)
        if network and self.logs.session:network.inspect(str(self.logs.session))
        if network and blocks_game_input(network.connection,raw.phase):return None
        window=self.windows.snapshot(self.binding.hwnd)
        projection=mapper.resolver.resolve(raw,mapper.viewport,session=mapper.session)
        if not evidence.accepts(raw,window,projection.layouts,time.time(),verified_log_position=getattr(self,'verified_log_position',None)):return None
        maps=mapper.reuse_projection(raw,window.to_dict())
        obs=replace(raw,captured_at=time.time())
        row={**self.last_row,'capture_observation':self.last_row.get('capture_observation',self.last_row['observation']),
            'observation':replace(obs,resolutions=()).to_dict(),'window':window.to_dict(),'log_snapshot_unchanged':False,
            'readiness_reused':True,'readiness_verified_log_position':getattr(self,'verified_log_position',None)}
        self.last_row=row;self.last_obs=obs;self.last_window=window;self.mappings=maps
        obs=replace(obs,actions=self._capabilities(obs,maps,semantic=getattr(self,'decision_ahead',False)))
        from .recording import compact_observation
        row['observation']=compact_observation(obs).to_dict();self.last_obs=obs
        self.chain_metrics.count('action_frame_lease_reused')
        return obs

    @measured_stage('observation_cycle')
    def observe(self):
        if getattr(self,'storage_error',None):raise OSError(self.storage_error)
        recovering=bool(getattr(self,'buy_recoveries',{}))
        if recovering:self.cache_until=0;self._force_full_capture=True
        if self.last_obs is not None and time.monotonic()<self.cache_until:
            version=getattr(self.logs,'publication_version',None)
            cached=getattr(self,'_capture_log_version',None)
            if version is None or cached is None or version()==cached:return self.last_obs
            # A new Options/choice/playback publication must not sit behind
            # the image TTL. Old pixels still cannot authorize a new action.
            self.cache_until=0
            if hasattr(self,'chain_metrics'):self.chain_metrics.count('observation_cache_event_invalidated')
        if hasattr(self,'chain_metrics') and not recovering:
            reused=self._reuse_geometry_observation()
            if reused is not None:return reused
        # Log playback and combat can advance without another screenshot.
        # Never refresh last_row/last_image or advertise inputs on this path.
        # Choices and playback completion immediately resume normal capture.
        if not recovering and self.last_obs is not None and getattr(self,'last_row',None):
            from .capture_cadence import passive_wait
            prior=self.last_obs
            if callable(getattr(self.logs,'observe',None)) and prior.phase in (Phase.COMBAT,Phase.RECRUIT):
                latest=self.semantic_observation(self.logs.observe())
                processing=any(f.get('action') for f in getattr(getattr(self.logs,'state',None),'block_stack',()))
                opening=getattr(self,'recruit_start',None)
                admitted=bool(opening and callable(getattr(opening,'allows',None))
                              and opening.allows(latest,getattr(self.logs,'session',None)))
                # 招募入口未确认时，动画中的诊断采样最多延迟一个节拍；
                # 这只限制采图频率，不以播放清空作为入口条件。当前截图一旦
                # 确认招募场景，后续逐帧检查实际动作对象，可在无关动画中出手。
                inspect_signal=bool(getattr(self,'uses_green_signal',False) and latest.phase==Phase.RECRUIT
                    and not latest.options_pending
                    and (admitted or (latest.client_task_list is None and not processing)))
                # 战斗/回合切换期间不可能发输入，画面只用于“迟到的客户端动画”确认：
                # 1s 一帧足够；招募动画仍保持 0.35s 的既有诊断节拍（回归测试锁定）。
                # 之前无论战斗还是动画都按 80ms 缓存节拍采图，30s 战斗会白采数十张全屏图。
                interval=1.0 if latest.phase==Phase.COMBAT else .35
                if not inspect_signal and passive_wait(prior,latest,self.last_row['capture']['end'],time.time(),
                                                       interval=interval,processing=processing):
                    self.invalidate_readiness('passive_client_playback')
                    if hasattr(self.sampler,'metrics'):self.sampler.metrics.count('passive_captures_avoided')
                    return replace(latest,ready=False,actions=(),captured_at=time.time())
        started=time.perf_counter()
        row,image=self.sampler.sample(persist=False);row['lifecycle']=self.lifecycle.inspect(row)
        row['input_released_at']=getattr(self,'last_input_released_at',0.)
        sampled=time.perf_counter()
        if self.logs.session:
            network_tail=self.lifecycle.network.tail
            for source in (self.logs.tail.path if self.logs.tail else None,self.logs.scene_tail.path if self.logs.scene_tail else None,network_tail.path if network_tail else None):
                if source:
                    self._mirror(self.ensure_mirror(source))
        shared=getattr(self.sampler,'latest_snapshot',None) if getattr(self.sampler,'latest',None) is row else None
        raw=shared if shared is not None else Observation.from_dict(row['observation'])
        from .connection import observation_version
        self._capture_log_version=observation_version(raw)
        obs=self.semantic_observation(raw);window=WindowSnapshot.from_dict(row['capture']['window_after'])
        if self.poll_late_responses(obs):row['blockers'].append('late_response_resync_pending')
        previous=getattr(self,'last_obs',None)
        dirty=getattr(self,'changed_layout_zones',{})
        if previous is not None and (previous.game_id,previous.turn)==(obs.game_id,obs.turn):
            for zone in ('shop','board','hand'):
                layout=lambda o:tuple((c.entity_id,c.position) for c in getattr(o,zone))
                if layout(previous)!=layout(obs):dirty[zone]=layout(obs)
        else:dirty={}
        self.changed_layout_zones=dirty
        row['changed_layout_zones']=dict(dirty)
        from .live_choices import ui_build_error
        build_error=ui_build_error(obs,getattr(self.mapper,'profile',{}))
        if build_error:raise RuntimeError(build_error)
        if window.identity!=row['window']['identity'] or window.pid!=self.binding.pid or window.process_created!=self.binding.process_created:
            raise RuntimeError('live_window_identity_changed')
        scene=row['lifecycle']['scene'] or obs.scene
        # GameNetLogger describes the game-server socket, not menu/BNet
        # readiness. A canceled previous queue can leave it disconnected while
        # the actual BACON Start menu is fully usable. Scope that stale socket
        # state only after proving the current, unobscured blue Start button.
        start_button=None
        if obs.phase==Phase.MENU and scene=='BACON' and not window.blockers:
            if hasattr(self.mapper,'control'):start_button=self.mapper.control(obs,image,row['window'],'enter')
            if start_button is None:start_button=menu_button(image)
        start_menu=bool(start_button and not self.enter_sent and row['log_snapshot_unchanged'])
        network=row['lifecycle']['connection']
        health_network='unknown' if start_menu else network
        base_ready=obs.ready and not row['blockers'] and row['log_snapshot_unchanged']
        if obs.phase==Phase.RECRUIT and obs.interaction is None and obs.options_pending:
            row['blockers'].append('server_options_pending');base_ready=False
        if not getattr(self,'uses_green_signal',False) and obs.phase==Phase.RECRUIT and obs.interaction is None and obs.client_task_list is not None:
            row['blockers'].append('client_task_list_processing');base_ready=False
        state=getattr(self.logs,'state',None)
        row['recruit_root_processing']=bool(obs.phase==Phase.RECRUIT and any(f.get('action') for f in getattr(state,'block_stack',())))
        if not getattr(self,'uses_green_signal',False) and obs.phase==Phase.RECRUIT and any(f.get('action') for f in getattr(state,'block_stack',())):
            # A matching receipt may prove the purchase itself before its
            # triggered effects finish. Do not dispatch the NEXT ordinary
            # action from a half-processed root. Forced choices remain usable.
            row['blockers'].append('recruit_action_processing');base_ready=False
        if not obs.ready:row['blockers'].append('logical_state_not_ready')
        health=self.health.inspect(obs,session=self.logs.session,error=self.logs.error if self.logs.tail else None,network=health_network)
        health.update(game_server_connection=network,connection_scope='verified_start_menu' if start_menu else 'game')
        row['channel_health']=health
        if not health['healthy']:
            row['blockers'].extend(health['reasons']);base_ready=False
        if health_network in ('disconnected','reconnecting','restoring','connecting'):
            row['blockers'].append('network_'+health_network);base_ready=False
        if hasattr(self.mapper,'begin_capture'):self.mapper.begin_capture(row)
        try:maps=self.mapper.map(obs,image,row['window'],blocked=bool(window.blockers))
        except ValueError as exc:
            from .capture_broker import FullCaptureRequired
            if not isinstance(exc,FullCaptureRequired):raise
            self._force_full_capture=True;self.cache_until=0
            return self.observe()
        self._force_full_capture=False
        pixels=getattr(self.mapper,'last_frame_pixels',None)
        detection_image=pixels if getattr(pixels,'image',None) is image else image
        if self.mapper.last_ui_state.get('mode')=='pause_menu':
            row['blockers'].append('pause_menu_visible');base_ready=False
        if obs.phase==Phase.RECRUIT and self.mapper.last_ui_state.get('mode')=='recruit_transition':
            row['blockers'].append('recruit_transition_overlay');base_ready=False
        if getattr(self.mapper,'used_geometry',False) or getattr(self.mapper,'used_layout_recovery',False):
            obs,maps=self.mapper.project_interaction(obs,image,maps)
            if not obs.ready:
                row['blockers'].append('geometry_interaction_state_unconfirmed');base_ready=False
            if obs.interaction and obs.interaction.protocol=='verified_multi_select':
                snapshot=(obs.game_id,obs.interaction.request_id,obs.interaction.selected)
                if snapshot!=getattr(self,'selection_snapshot',None):
                    self.selection_revision=getattr(self,'selection_revision',0)+1;self.selection_snapshot=snapshot
                obs=replace(obs,ui_revision=self.selection_revision)
        elif not (getattr(self,'uses_green_signal',False) and getattr(getattr(self,'mapper',None),'mode',None)=='geometry'):
            from .live_interactions import discover_regions
            if obs.phase==Phase.RECRUIT and any(m.key=='scene:recruit' and m.status=='matched' for m in maps):
                if not hasattr(self,'hand_tracking'):self.hand_tracking={}
                pixels=getattr(self.mapper,'last_frame_pixels',None)
                hand_image=pixels if getattr(pixels,'image',None) is image else image
                measured=hand_markers(hand_image,obs.hand,tracker=self.hand_tracking,fan_hint=self.mapper.last_diagnostics.get('hand_fan'),window_token=(obs.game_id,row['window']['token']))
                for card,box in measured:
                    key=f'entity:{card.entity_id}';maps=[m for m in maps if m.key!=key]
                    reason='log_stable_hand_and_reobserved_badge_portrait' if self.hand_tracking.get('used_tracking') else 'complete_log_slot_and_observed_cost_tier_marker_sequence'
                    maps.append(Mapping(key,'matched',box,.8,reason,entity_id=card.entity_id,stability_basis=StabilityBasis.GEOMETRY))
            if obs.phase==Phase.CHOICE and obs.choices and all(c.card_type=='HERO' for c in obs.choices):
                # Gold is not needed for free hero selection. Locked heroes remain
                # present in layout but are independently removed by the resolver.
                obs=replace(obs,warnings=tuple(w for w in obs.warnings if w!='gold_unknown'))
                boxes=hero_regions(detection_image,len(obs.choices))
                ordered=sorted(obs.choices,key=lambda c:c.position)
                if boxes and [c.position for c in ordered]==list(range(1,len(ordered)+1)):
                    maps=[m for m in maps if m.key not in {f'entity:{c.entity_id}' for c in ordered}]
                    maps.extend(Mapping(f'entity:{c.entity_id}','matched',box,.9,'log_hero_slot_and_purple_portrait_frame',entity_id=c.entity_id) for c,box in zip(ordered,boxes))
                    maps.append(Mapping('scene:choice','matched',boxes[0],.9,'hero_portrait_group_with_server_hero_choices'))
            if obs.phase==Phase.CHOICE and obs.interaction and obs.choices and all(c.card_type=='BATTLEGROUND_TRINKET' for c in obs.choices):
                visible={m.entity_id for m in maps if m.status=='matched' and m.entity_id is not None}
                if all(c.entity_id in visible for c in obs.choices):
                    obs=replace(obs,interaction=replace(obs.interaction,protocol='verified_trinket_select'))
            if obs.phase==Phase.CHOICE and obs.choices and all(c.card_type=='HERO_POWER' for c in obs.choices):
                boxes=power_choice_regions(detection_image,len(obs.choices))
                if boxes:
                    keys={f'entity:{c.entity_id}' for c in obs.choices}|{'scene:choice'}
                    maps=[m for m in maps if m.key not in keys]
                    maps.append(Mapping('scene:choice','matched',boxes[0],.9,'observed_green_power_card_group'))
                    maps.extend(Mapping(f'entity:{c.entity_id}','matched',b,.9,'log_choice_order_and_power_frame',entity_id=c.entity_id) for c,b in zip(obs.choices,boxes))
            quest_choice=any(c.tags.get('QUEST')=='1' for c in obs.choices)
            if obs.phase==Phase.CHOICE and quest_choice:
                from .live_choices import quest_regions,quest_links
                boxes=quest_regions(detection_image,len(obs.choices)) if quest_links(obs) else []
                if boxes and obs.interaction and not window.blockers:
                    keys={f'entity:{c.entity_id}' for c in obs.choices}|{'scene:choice'}
                    maps=[m for m in maps if m.key not in keys]
                    maps.append(Mapping('scene:choice','matched',boxes[0],.9,'complete_paired_quest_panel_row'))
                    maps.extend(Mapping(f'entity:{c.entity_id}','matched',b,.9,'server_quest_reward_link_and_measured_quest_panel',entity_id=c.entity_id)
                        for c,b in zip(obs.choices,boxes))
                    obs=replace(obs,interaction=replace(obs.interaction,protocol='verified_quest_pair'))
            if obs.phase==Phase.CHOICE and not quest_choice and obs.choices and all(c.card_type not in ('HERO','HERO_POWER','BATTLEGROUND_TRINKET') for c in obs.choices):
                boxes=discover_regions(detection_image,len(obs.choices))
                if boxes:
                    keys={f'entity:{c.entity_id}' for c in obs.choices}|{'scene:choice'}
                    maps=[m for m in maps if m.key not in keys]
                    maps.append(Mapping('scene:choice','matched',boxes[0],.9,'complete_aligned_discover_frames'))
                    maps.extend(Mapping(f'entity:{c.entity_id}','matched',b,.9,'server_choice_order_and_complete_frame',entity_id=c.entity_id) for c,b in zip(obs.choices,boxes))
            if obs.phase==Phase.CHOICE and obs.interaction and obs.interaction.max_select>1 and not window.blockers:
                from .live_interactions import multi_choice_state
                selection=multi_choice_state(detection_image,len(obs.choices))
                if selection:
                    frames,indices=selection;chosen=tuple(obs.choices[i].entity_id for i in indices)
                    snapshot=(obs.game_id,obs.interaction.request_id,chosen)
                    if snapshot!=self.selection_snapshot:self.selection_revision+=1;self.selection_snapshot=snapshot
                    obs=replace(obs,interaction=replace(obs.interaction,selected=chosen,protocol='verified_multi_select'),
                        ui_revision=self.selection_revision,warnings=tuple(w for w in obs.warnings if w!='multi_select_choice_requires_adapter'))
                    keys={f'entity:{c.entity_id}' for c in obs.choices}|{'scene:choice','control:confirm'}
                    maps=[m for m in maps if m.key not in keys]
                    maps.append(Mapping('scene:choice','matched',frames[0],.9,'server_multi_choice_and_unambiguous_frame_states'))
                    for c,b in zip(obs.choices,frames):
                        maps.append(Mapping(f'entity:{c.entity_id}','matched',Box(round(b.x+b.width*.3),round(b.y+b.height*.25),round(b.width*.4),round(b.height*.3)),.9,'log_choice_order_and_selection_rails',entity_id=c.entity_id))
                    actions=[];ctx=obs.interaction
                    if len(chosen)<ctx.max_select:
                        actions=[Action(Kind.SELECT,c.entity_id,choice_id=ctx.request_id,selected=chosen+(c.entity_id,),evidence='server_choices') for c in obs.choices if c.entity_id not in chosen]
                    confirm=confirm_button(detection_image)
                    if confirm and ctx.min_select<=len(chosen)<=ctx.max_select:
                        maps.append(Mapping('control:confirm','matched',confirm,.9,'observed_enabled_multi_choice_confirm'))
                        actions.append(Action(Kind.CONFIRM,choice_id=ctx.request_id,selected=chosen,evidence='server_choices'))
                    obs=replace(obs,actions=tuple(actions))
                else:
                    row['blockers'].append('multi_choice_selection_state_unknown');base_ready=False
        pending=getattr(self,'cancel_pending',None)
        if pending and (obs.game_id,obs.turn)==pending[:2] and obs.interaction is None and obs.phase==Phase.RECRUIT and row['capture']['end']>pending[3] and row['log_snapshot_unchanged'] and not row['blockers'] and self.mapper.last_ui_state.get('mode')=='unknown' and any(m.key=='scene:recruit' and m.status=='matched' for m in maps):
            obs=replace(obs,ui_revision=max(obs.ui_revision,pending[4]+1));self.cancel_pending=None
        view_blockers=phase_view_blockers(obs,maps)
        if view_blockers:
            row['blockers'].extend(view_blockers);base_ready=False
        opening=getattr(self,'recruit_start',None)
        if opening and obs.phase==Phase.RECRUIT and obs.interaction is None:
            was_admitted=opening.allows(obs,self.logs.session)
            row['recruit_start']=opening.inspect(row,obs,image,maps,self.mapper.last_ui_state)
            if not row['recruit_start']['ready']:
                row['blockers'].append('recruit_start_pending');base_ready=False
            elif not was_admitted:
                self.audit.write('recruit_start_admitted',game_id=obs.game_id,turn=obs.turn,evidence=row['recruit_start'])
        if not base_ready or row['blockers']:
            self.invalidate_readiness('observed_frame_blocked',frame_id=row.get('id'))
        if obs.phase==Phase.MENU and scene=='BACON':
            obs=replace(obs,turn=None,gold=None,hero=None,board=(),hand=(),shop=(),choices=(),controls=(),interaction=None)
            button=start_button
            self.returned_start_ready=bool(button and base_ready)
            if self.matched and button and not any('occluded' in b or 'offscreen' in b for b in window.blockers) and row['capture']['geometry_unchanged']:
                self.returned_start_observed=True
            if button:
                maps.append(Mapping('control:enter','matched',button,.9,'measured_blue_start_button_with_BACON_scene'))
                obs=replace(obs,source='hearthstone_live',mode='GT_BATTLEGROUNDS',scene=scene,
                    warnings=(),actions=(() if self.enter_sent else (Action(Kind.ENTER,evidence='verified_client_ui'),)))
            elif not (self.enter_sent and not self.matched):base_ready=False
        else:self.returned_start_ready=False
        if obs.phase==Phase.GAME_OVER:
            panel=getattr(self.mapper,'control',lambda *a:None)(obs,image,row['window'],'dismiss') or result_panel(detection_image)
            if panel:
                maps.append(Mapping('control:dismiss','matched',panel,.9,'observed_purple_result_panel_and_desaturated_background'))
                obs=replace(obs,source='hearthstone_live',warnings=(),actions=(Action(Kind.DISMISS,evidence='verified_client_ui'),))
        if self.enter_sent and obs.phase==Phase.MENU and not self.matched:
            # Do not enqueue twice. Matching begins without another click.
            obs=replace(obs,phase=Phase.UNKNOWN,actions=(),warnings=(),scene=scene,source='hearthstone_live')
        if obs.phase in (Phase.CHOICE,Phase.RECRUIT,Phase.COMBAT):self.matched=True
        if obs.phase in (Phase.CHOICE,Phase.RECRUIT,Phase.COMBAT):self.game_ids.add(obs.game_id)
        if self.last_row is not None:self.previous_sample=(self.last_row,self.last_image,self.mappings)
        # Geometry recovery must inspect this capture, never the preceding one.
        self.last_row,self.last_image=row,image
        obs=replace(obs,ready=base_ready,captured_at=time.time())
        # 恢复只由真实捕获推进，与是否开启策略预判无关。先清理再投影候选。
        obs=replace(obs,actions=self._captured_capabilities(obs,maps))
        from .recording import compact_observation
        row['observation']=compact_observation(obs).to_dict();self.last_row,self.last_image,self.last_obs,self.last_window,self.mappings=row,image,obs,window,maps
        self._remember_evidence()
        if hasattr(self.mapper,'record_visual'):self.mapper.record_visual(obs,image,maps,row)
        self.audit.write('live_observation_timing',frame_id=row['id'],revision=obs.revision,sampling_ms=(sampled-started)*1000,mapping_and_projection_ms=(time.perf_counter()-sampled)*1000,total_ms=(time.perf_counter()-started)*1000,geometry_cache=getattr(self.mapper,'cache_metrics',{}))
        self.cache_until=time.monotonic()+.08
        archive_reason=self.evidence_schedule.due(obs,time.monotonic()) if hasattr(self,'evidence_schedule') else None
        if archive_reason and not window.blockers:
            self.last_evidence=time.monotonic();folder=self.directory/'evidence';folder.mkdir(exist_ok=True)
            from .recording import compact_observation_dict
            sample={**row,'observation':compact_observation_dict(row['observation'])}
            try:self.archive_queue.put_nowait((row['id'],image,dict(archive_format='jpg',archive_reason=archive_reason,sample=sample,mapping=[asdict(m) for m in maps])))
            except queue.Full:self.audit.write('archive_backpressure',frame_id=row['id'])
        return obs

    @measured_stage('observation_revalidation')
    def revalidate_observation(self,before):
        """Cheap semantic recheck; fresh pixels are still required by input_plan."""
        from .decision_ahead import decision_key
        if self.last_obs is not before and (self.last_obs is None or decision_key(self.last_obs)!=decision_key(before)):
            self.invalidate_readiness('observation_context_changed')
            self.cache_until=0
            return self.observe()
        # 普通手势还需约0.26秒；复用帧必须给输入和最终检查留足0.4秒，
        # 不能省掉截图后让计划在按住鼠标期间越过0.75秒证据期限。
        max_age=.35 if before.phase==Phase.RECRUIT and before.interaction is None else .75
        if not 0<=time.time()-self.last_row['capture']['end']<=max_age:
            self.cache_until=0
            return self.observe()
        network=getattr(getattr(self,'lifecycle',None),'network',None)
        raw=getattr(self.logs,'synchronize',self.logs.observe)()
        if network and self.logs.session:network.inspect(str(self.logs.session))
        obs=self.semantic_observation(raw)
        unhealthy=bool(getattr(self.logs,'error',None) or any(w.startswith('state_health:') for w in obs.warnings) or
            network and blocks_game_input(network.connection,obs.phase))
        if obs.phase==Phase.RECRUIT and obs.interaction is None and before.phase==Phase.RECRUIT and before.interaction is None:
            # observe()发布的是客户端能力投影（插入/换位编码、受支持动作、被动END），
            # 原始日志候选不能直接与它比较，否则每次都假报变化并重截整屏。
            # 复用同一投影，不更新像素时间；最终绿光与输入上下文仍独立复核。
            obs=replace(obs,actions=self._capabilities(obs,getattr(self,'mappings',()),semantic=getattr(self,'decision_ahead',False)))
        from .decision_ahead import decision_key
        from .readiness_evidence import observation_context
        # 只有“策略输入”变化才需要重新采集/重新规划；revision/回执/客户端回放游标
        # 推进不算。输入前的独立检查（上下文 token、绿光、坐标）仍在执行器内执行。
        if unhealthy or decision_key(obs)!=decision_key(before):
            if unhealthy or observation_context(obs)!=observation_context(before):
                self.invalidate_readiness('observation_context_changed')
            self.cache_until=0
            return self.observe()
        windows=getattr(self,'windows',None)
        if windows is not None:
            w=windows.snapshot(self.binding.hwnd)
            if w.token!=self.last_window.token or w.blockers:
                self.cache_until=0;self.invalidate_readiness('window_context_changed');return self.observe()
        audit=getattr(self,'audit',None)
        if audit is not None:
            audit.write('observation_revalidated_without_capture',revision=before.revision,
                revision_only=True,decision_input_unchanged=True)
        # 返回最新的语义投影（revision/游标已更新）但不重采像素；策略输入与
        # before 完全一致，执行器仍会用最新的日志与画面证据独立复核后才发输入。
        self.last_obs=obs
        self.last_row={**self.last_row,'observation':replace(obs,resolutions=()).to_dict()}
        return obs

    @measured_stage('input_preparation')
    def input_plan(self,action,obs):
        # Long entry animations must not consume the ordinary four-second
        # preparation budget. This extends waiting only; no input gate weakens.
        from .dependency_readiness import CONTROLS
        selected_control=bool(obs.interaction is None and (action.kind in CONTROLS or
            action.kind==Kind.ACTIVATE and any(c.entity_id==action.source for c in obs.controls)))
        from .action_budgets import CONTROL_POLL_SECONDS,CONTROL_WAIT_SECONDS
        selected_wait=selected_control or action.kind==Kind.BUY
        wait_event=self._control_event if selected_control else self._buy_recovery_event
        budget=CONTROL_WAIT_SECONDS if selected_wait else prepare_timeout(action,obs.phase)
        deadline=time.monotonic()+budget
        capture_started=time.monotonic()
        started=time.perf_counter();samples=0;cursor_parked=False;held=False;diagnosed=False;last_notice=0.
        while True:
            guard=getattr(self,'input_guard',None)
            if guard:guard()
            try:
                plan=self._input_plan(action,obs)
                if held:wait_event('selected_ready',action=asdict(action),input_sent=False,
                    wait_ms=(time.perf_counter()-started)*1000,selection_preserved=True)
                self.audit.write('action_region_ready',action_key=action.key,revision=obs.revision,
                    preparation_ms=(time.perf_counter()-started)*1000,additional_samples=samples,
                    frame_id=self.last_row.get('id'),input_interactability='unknown_until_game_ack')
                return plan
            except ValueError as exc:
                reasons=str(exc).removeprefix('live_plan_blocked:').split(',')
                transient={'recruit_input_handoff_pending','control_readiness_unknown','control_explicitly_disabled','recruit_start_pending','shop_refresh_pending','action_green_outline_absent','recruit_green_signal_absent','recruit_green_source_unavailable','recruit_signal_context_unavailable','action_source_geometry_unconfirmed','target_pixels_not_stable','target_quiet_window_pending','ui_animation_or_stability_unknown','log_visual_settle_guard','log_capture_not_synchronized','log_changed_during_capture','logical_state_not_ready','capture_expired','readiness_evidence_expired','local_object_picture_pending','local_action_processing','local_capture_reused','capture_precedes_input_release','choice_panel_picture_pending','choice_panel_pending','target_stage_pending'}
                waiting=lambda r:r in transient or r.startswith(('unknown_mapping:','client_object_pending:','client_object_capture_pending:','local_object_absent:','local_object_position_pending:')) or r in ('board_hover_panel_visible','board_point_covered_by_panel','choice_object_pending','client_object_channel_missing','local_capture_scope_pending','target_regions_unknown','visual_choice_scene_unconfirmed','visual_scene_unconfirmed','visual_recruit_scene_unconfirmed')
                if not reasons or any(not waiting(r) for r in reasons):raise
                if selected_wait:
                    # 等待仍属于同一个未投递动作；不使用可选动作的跳过/隔离路径。
                    elapsed=time.perf_counter()-started;held=True
                    if time.monotonic()>deadline:
                        wait_event('selected_timeout',action=asdict(action),reason=str(exc),
                            input_sent=False,selection_preserved=True)
                        raise RuntimeError(('selected_control_readiness_timeout:' if selected_control else 'selected_buy_readiness_timeout:')+str(exc)) from exc
                    if not last_notice or elapsed-last_notice>=5:
                        wait_event('selected_wait',action=asdict(action),reason=str(exc),
                            input_sent=False,selection_preserved=True,wait_ms=elapsed*1000,
                            poll_min_interval_ms=CONTROL_POLL_SECONDS*1000,
                            signal=getattr(self,'last_readiness',{}).get('signal',{}))
                        last_notice=elapsed or .000001
                    if elapsed>=8 and not diagnosed:
                        from .live_interactions import archive_interaction
                        archive_interaction(self,'selected-control-wait',action,obs,self.last_window,
                            self.last_image,self.last_row.get('capture',{}),reason=str(exc),
                            readiness=getattr(self,'last_readiness',{}))
                        diagnosed=True
                else:self._ui_wait_expired(action,str(exc))
                if not cursor_parked and any(r in ('board_hover_panel_visible','board_point_covered_by_panel') or r.startswith(('local_object_absent:board:','local_object_position_pending:board')) for r in reasons):
                    cursor_parked=self.park_neutral()
                if selected_wait:
                    # 处理耗时计入间隔；慢帧不追赶、不额外睡眠，快帧补足120ms。
                    self._wait_control_capture(capture_started+CONTROL_POLL_SECONDS)
                    now=time.monotonic()
                    metrics=getattr(self,'chain_metrics',None)
                    if metrics:metrics.sample('selected_control_capture_start_gap',(now-capture_started)*1000)
                    capture_started=now
                self.cache_until=0;updated=self.observe()
                samples+=1
                from .decision_ahead import decision_key
                comparison=replace(updated,ready=obs.ready) if selected_wait else updated
                if decision_key(comparison)!=decision_key(obs):
                    if held:wait_event('selected_context_changed',action=asdict(action),
                        input_sent=False,reason='decision_state_changed',old_revision=obs.revision,
                        new_revision=updated.revision)
                    raise ValueError('live_observation_changed_while_waiting')
                obs=updated

    def _wait_control_capture(self,next_capture):
        """仅等待新捕获节拍；每至多20ms检查停止，不访问输入API。"""
        while True:
            guard=getattr(self,'input_guard',None)
            if guard:guard()
            remaining=next_capture-time.monotonic()
            if remaining<=0:return
            time.sleep(min(.02,remaining))

    @measured_stage('input_planning_and_readiness')
    def _input_plan(self,action,obs):
        from .capture_broker import FullCaptureRequired
        self.last_hand_source_binding=None
        opening=getattr(self,'recruit_start',None)
        if opening and not opening.allows(obs,self.logs.session):raise ValueError('live_plan_blocked:recruit_start_pending')
        from .recruit_start import shop_update_pending

        if action not in obs.actions:raise ValueError('live_action_not_advertised')
        row=self.last_row
        if hasattr(self.mapper,'ensure') and action.kind not in (Kind.ENTER,Kind.DISMISS,Kind.CANCEL):
            try:self.mappings=self.mapper.ensure(obs,action,self.last_image,row['window'],self.mappings)
            except ValueError as exc:
                if not isinstance(exc,FullCaptureRequired):raise
                self._force_full_capture=True
                raise ValueError('live_plan_blocked:unknown_mapping:action_layout') from exc
        geometric=bool(getattr(self.mapper,'authorize_geometry',lambda *a:False)(obs,action))
        if action.kind==Kind.CANCEL:
            from .client_protocol import cancellable_target
            if not cancellable_target(obs,action) or row['blockers'] or self.mapper.last_ui_state.get('mode')!='targeting_suspected':raise ValueError('cancel_protocol_unconfirmed')
            if time.time()-row['capture']['end']>.5:raise ValueError('cancel_capture_expired')
            w=self.last_window;source=w.client.point(.97,.5);destination=None
            steps=[dict(kind='right_click',point_screen=source,region='owned_neutral_client')]
        elif action.kind in (Kind.ENTER,Kind.DISMISS):
            m=next((m for m in self.mappings if m.key=='control:'+str(action.kind) and m.rect),None)
            if not m or row['blockers']:raise ValueError('start_button_not_ready')
            x,y=m.rect.point();w=self.last_window
            source=tuple(client_to_screen((x,y),w.client));destination=None
            steps=[dict(kind='click',point_screen=source,region=m.key)]
        else:
            # Resolve optional HUD controls only for the selected action. A
            # buy/play loop need not run circle searches for power/freeze/gift.
            if action.kind in (Kind.HERO_POWER,Kind.FREEZE,Kind.ACTIVATE) and not geometric:
                from .live_interactions import hud_mappings
                pixels=getattr(self.mapper,'last_frame_pixels',None)
                hud_image=pixels if getattr(pixels,'image',None) is self.last_image else self.last_image
                measured=hud_mappings(replace(obs,actions=(action,)),hud_image,self.mappings,
                    cache=getattr(self.mapper,'hud_locations',None))
                keys={m.key for m in measured}
                self.mappings=[m for m in self.mappings if m.key not in keys]+measured
            plan=ShadowPlanner(row,self.mappings,ttl=2.5,compact=True).build(obs,Decision(action,obs.revision,'live executor geometry'))
            if not getattr(self.gate,'uses_green_signal',False) and not getattr(self.gate,'uses_layout_evidence',False) and getattr(self,'previous_sample',None) and (not self.gate.previous or self.gate.previous[0][3]!=action.key):
                old,image,maps=self.previous_sample
                if old['observation']['game_id']==obs.game_id and action in Observation.from_dict(old['observation']).actions:
                    if action.kind in (Kind.HERO_POWER,Kind.FREEZE,Kind.ACTIVATE):
                        maps=maps+hud_mappings(replace(Observation.from_dict(old['observation']),actions=(action,)),image,maps)
                    self.gate.inspect(old,image,maps,action,{})
            try:readiness=self.gate.inspect(row,self.last_image,self.mappings,action,self.mapper.last_ui_state,proposal=plan)
            except FullCaptureRequired:
                self._force_full_capture=True
                raise ValueError('live_plan_blocked:local_capture_scope_pending')
            self.last_readiness=dict(action_key=action.key,revision=obs.revision,**readiness)
            reasons=sorted(set(plan['rejection_reasons']+readiness['reasons']))
            if action.kind==Kind.BUY and not reasons:
                handoff=self._buy_input_handoff(obs,self.mappings)
                self.last_readiness['round_handoff']=handoff
                if not handoff['ready']:
                    reasons.append('recruit_input_handoff_pending')
                    readiness={**readiness,'ready':False,'state':'WAITING',
                        'reasons':reasons,'round_handoff':handoff}
                    self.last_readiness.update(readiness)
            if reasons:
                if 'layout_contour_mismatch' in reasons:self._force_full_capture=True
                self.audit.write('live_readiness_wait',action_key=action.key,readiness=readiness)
                raise ValueError('live_plan_blocked:'+','.join(reasons))
            steps=[s for s in plan['steps'] if s['kind'] not in ('await_ui_state','deferred_target','deferred_branch')]
            if len(steps)==1 and steps[0]['kind'] in ('click','click_target'):
                source=tuple(steps[0]['point_screen']);destination=None
            elif len(steps)==2 and [s['kind'] for s in steps]==['drag_start','drag_release']:
                source=tuple(steps[0]['point_screen']);destination=tuple(steps[1]['point_screen'])
            else:raise ValueError('multi_stage_requires_new_observation')
        # Independent final boundary: a permissive or faulty mapper cannot
        # authorize a hand drag from the resource HUD or the wrong board row.
        from .interaction_regions import allowed
        w=self.last_window
        client_source=(source[0]-w.client.x,source[1]-w.client.y)
        from .action_geometry import route
        zone=route(action,obs).source_zone
        if not allowed(client_source,self.last_image.size,zone):
            if zone=='hand':self._skip_unsafe(action,'live_source_outside_semantic_region:'+str(zone))
            raise ValueError('live_source_outside_semantic_region:'+str(zone))
        if zone=='hand':
            card=next((c for c in obs.hand if c.entity_id==action.source),None)
            if card and card.tags.get('LITERALLY_UNPLAYABLE') in ('1',1):
                self._skip_unsafe(action,'live_source_explicitly_unplayable')
            if geometric:
                if not self.mapper.authorize_geometry(obs,action,client_source):self._skip_unsafe(action,'live_hand_source_slot_mismatch')
                self.last_hand_source_binding=dict(frame_id=row['id'],source=action.source,
                    binding='verified_layout_and_current_log_slot',proposed_point=list(client_source),
                    layouts=[s.id for s in self.mapper.projection.layouts])
            else:
                # Independently bind the proposed source to the complete measured
                # card-frame row. A permissive mapper cannot swap a white-cost
                # spell with a locked card's coin or artwork in a neighboring
                # slot - for ANY hand size, not only small all-spell rows.
                from .hand_mapping import hand_frame_binding,hand_card_frames
                pixels=getattr(self.mapper,'last_frame_pixels',None)
                image=pixels if getattr(pixels,'image',None) is self.last_image else self.last_image
                frames=hand_card_frames(image)
                binding=hand_frame_binding(image,obs.hand,frames=frames)
                locked=any(c.tags.get('LITERALLY_UNPLAYABLE') in ('1',1) for c in obs.hand)
                self.last_hand_source_binding=dict(frame_id=row.get('id'),revision=obs.revision,
                    source=action.source,proposed_point=list(client_source),
                    hand_slots=[dict(entity=c.entity_id,position=c.position,
                        explicitly_unplayable=c.tags.get('LITERALLY_UNPLAYABLE') in ('1',1)) for c in obs.hand],
                    observed_frames=[asdict(b) for b in frames],complete_row=bool(binding),
                    bound_points=[dict(entity=c.entity_id,point=list(b.point())) for c,_,b in binding])
                if binding:
                    index=next((i for i,(c,_,_) in enumerate(binding) if c.entity_id==action.source),None)
                    if index is None:self._skip_unsafe(action,'live_hand_slot_binding_unconfirmed')
                    frame=binding[index][1];x=client_source[0]
                    others=[b for i,(_,b,_) in enumerate(binding) if i!=index]
                    # The horizontal slot is the decisive evidence. Vertical
                    # jitter inside the measured card body is tolerated; a badge
                    # or a neighbouring card's slot is not.
                    if not (frame.x-2<=x<=frame.x+frame.width+2) or any(b.x-2<=x<=b.x+b.width+2 for b in others):
                        self._skip_unsafe(action,'live_hand_source_slot_mismatch')
                elif locked:
                    from .mixed_hand import locked_prefix_binding
                    mixed=locked_prefix_binding(image,obs.hand)
                    own=next((hit for c,_,hit in mixed if c.entity_id==action.source),None)
                    self.last_hand_source_binding['mixed_badge_row']=bool(mixed)
                    self.last_hand_source_binding['mixed_source_confirmed']=own is not None
                    # A locked card keeps its physical slot, and its gold artwork
                    # can impersonate another card's cost badge. Without a
                    # complete measured frame or lock/tier row, refuse the drag.
                    if own is None:self._skip_unsafe(action,'live_hand_slot_binding_unconfirmed')
                    x,y=client_source
                    if not (own.x-2<=x<=own.x+own.width+2 and own.y-2<=y<=own.y+own.height+2):
                        self._skip_unsafe(action,'live_hand_source_slot_mismatch')
                # The mapper may supply a point in the hand band but on a tier
                # shield, which the client does not treat as the card body.
                # Independently reject that point before any button is pressed.
                from .hand_fan import tier_markers
                badges=[Box(**b) for b in getattr(self.mapper,'last_diagnostics',{}).get('badges',())]
                nearby=[b for b in badges if b.x<=client_source[0]<=b.x+b.width and b.y<=client_source[1]<=b.y+b.height]
                if nearby and tier_markers(self.last_image,nearby):
                    self._skip_unsafe(action,'live_source_on_tier_badge')
        if destination and not allowed((destination[0]-w.client.x,destination[1]-w.client.y),self.last_image.size):
            raise ValueError('live_destination_in_forbidden_region')
        now=time.time();expires=min(now+1.5,row['capture']['end']+.75)
        if expires<=now:raise ValueError('live_plan_blocked:capture_expired')
        from .green_readiness import needs_green_signal
        shared=getattr(self,'last_readiness',{})
        green_input=bool(getattr(self,'uses_green_signal',False) and (needs_green_signal(obs,action) or
            shared.get('action_key')==action.key and shared.get('ready') and shared.get('protocol') in ('shared_recruit_green_signal','action_dependencies')))
        p=NativePlan(obs.revision,action.key,self.last_window.token,now,expires,0,source,destination,None,obs.revision,
            obs.game_id,str(self.logs.session),obs.interaction.request_id if obs.interaction else None,
            input_context_token(obs,action,ignore_playback=green_input) if action.kind not in (Kind.ENTER,Kind.DISMISS) else '',
            green_signal=green_input)
        if action.kind==Kind.CANCEL:p=replace(p,mouse_button='right')
        self.pending_action=action
        # 准入已完成；只允许同一捕获、同一请求和同一操作点在短时限内复用。
        # 不缓存下一动作的授权，不延长截图寿命，输入开始或观测失效即清除。
        self.last_input_admission=dict(action_key=p.action_key,window_token=p.window_token,
            state_token=p.state_token,session=str(self.logs.session),capture_end=row['capture']['end'],
            frame_sequence=row['capture'].get('frame_sequence'),image_identity=id(self.last_image),
            source=p.source,destination=p.destination,
            object_ready=bool(shared.get('action_key')==p.action_key
                and shared.get('checked_at')==row['capture']['end']
                and shared.get('protocol')=='action_dependencies' and shared.get('ready')
                and shared.get('objects',{}).get('ready')))
        self._remember_evidence(pin=True)
        getattr(self,'ui_waits',{}).clear()
        from .decision_ahead import policy_input
        self.audit.write('real_input_plan',action=asdict(action),plan=asdict(p),steps=steps,
            readiness=shared if shared.get('action_key')==action.key else None,
            frame_id=row.get('id'),capture_age_ms=max(0,now-row['capture']['end'])*1000,
            capture_region=row['capture'].get('region_client'),input_released_at=getattr(self,'last_input_released_at',None),
            observation=policy_input(obs).to_dict(),receipt_post_states_omitted=True,
            hand_source_binding=self.last_hand_source_binding)
        return p

    def check_motion(self,plan):
        return self.check_input(plan,motion=True)

    def client_object_snapshot(self):
        if not getattr(getattr(self.logs,'state',None),'client_objects',None):return None
        with self.logs.lock:
            tracker=getattr(self.logs.state,'client_objects',None)
            return tracker.snapshot() if tracker else None

    @measured_stage('log_event_wait')
    def wait_for_observation(self,timeout):
        if hasattr(self.logs,'wait_for_update'):
            # Sampler and verifier share the publisher. A consumed event must
            # not wake every poll while an image cache is still alive.
            previous=getattr(self,'_wait_sequence',0)
            sequence=max(previous,getattr(self.logs,'last_observed_sequence',0))
            current=self.logs.wait_for_update(sequence,timeout)
            if current is not None:self._wait_sequence=current
        else:time.sleep(timeout)

    @measured_stage('board_prepress')
    def before_mouse_down(self,plan,*,target_stage=False,required_remaining=.4):
        from .board_input_guard import before_mouse_down
        return before_mouse_down(self,plan,target_stage=target_stage,required_remaining=required_remaining)

    @measured_stage('input_context_check')
    def check_input(self,plan,*,motion=False):
        if getattr(self,'storage_error',None):raise OSError(self.storage_error)
        lightweight=getattr(self.windows,'motion_snapshot',None)
        w=lightweight(self.binding.hwnd,self.last_window) if motion and lightweight else self.windows.snapshot(self.binding.hwnd)
        if w.pid!=self.binding.pid or w.process_created!=self.binding.process_created or w.executable!=self.binding.executable:raise RuntimeError('live_identity_changed')
        if w.token!=plan.window_token or w.blockers:raise RuntimeError('live_window_layout_or_focus_changed')
        network=getattr(getattr(self,'lifecycle',None),'network',None)
        raw=(getattr(self.logs,'synchronize',self.logs.observe)() if not motion else self.logs.observe())
        if not motion and getattr(self,'local_profile',None):
            from .local_layout import display_settings,executable_hash
            self.local_profile.require_ready(w,raw.build,display_settings(),executable_hash(w.executable))
        if not motion and network and self.logs.session:network.inspect(str(self.logs.session))
        obs=self.semantic_observation(raw)
        opening=getattr(self,'recruit_start',None)
        if getattr(getattr(self,'gate',None),'objects',None) and self.pending_action is not None:
            from .dependency_readiness import dependencies
            tracker=self.client_object_snapshot()
            for zone in dependencies(obs,self.pending_action):
                if tracker is None or not tracker.inspect(obs,zone)['ready']:
                    raise RuntimeError('live_client_object_changed:'+zone)
        if self.poll_late_responses(obs):raise RuntimeError('late_response_resync_pending')
        if opening and not opening.allows(obs,self.logs.session):raise RuntimeError('live_recruit_start_pending')
        from .recruit_start import shop_update_pending

        if plan.state_token and (not obs.ready or getattr(self.logs,'error',None) or any(w.startswith('state_health:') for w in obs.warnings) or
            network and blocks_game_input(network.connection,obs.phase)):
            raise RuntimeError('live_log_channel_unhealthy')
        if obs.game_id!=plan.game_id or str(self.logs.session)!=plan.session:raise RuntimeError('live_game_session_changed')
        if plan.state_token and input_context_token(obs,self.pending_action,ignore_playback=plan.green_signal)!=plan.state_token:
            raise RuntimeError('live_action_context_changed')
        if obs.revision!=plan.revision:
            if not plan.state_token:raise RuntimeError('live_action_context_changed')
            self.audit.write('irrelevant_log_update_during_gesture',planned_revision=plan.revision,current_revision=obs.revision)
        for point in (plan.source,plan.destination):
            if point and (not w.client.x<=point[0]<w.client.x+w.client.width or not w.client.y<=point[1]<w.client.y+w.client.height or not self.windows.owns_point(w.hwnd,*point)):
                raise RuntimeError('live_input_point_not_owned')
        return None

    def invalidate_readiness(self,reason,*,frame_id=None):
        self.last_input_admission=None
        self.last_readiness_invalidation=dict(reason=reason,at=time.time(),
            frame_id=frame_id or (getattr(self,'last_row',None) or {}).get('id'))
        gate=getattr(self,'gate',None)
        if callable(getattr(gate,'invalidate',None)):gate.invalidate()
        broker=getattr(getattr(self,'sampler',None),'broker',None)
        if callable(getattr(broker,'invalidate',None)):broker.invalidate()

    def input_sent(self,action):
        self.control_wait_samples={}
        self.prepress_expiry_wait=None
        if action.kind==Kind.ENTER:self.enter_sent=True
        if action.kind==Kind.REFRESH:self.shop_refresh_pending=(self.last_obs.game_id,self.last_obs.turn)
        if action.kind==Kind.CANCEL:self.cancel_pending=(self.last_obs.game_id,self.last_obs.turn,action.choice_id,self.last_row['capture']['end'],self.last_obs.ui_revision)
        self.cache_until=0
        # Successful OS dispatch consumes pre-input evidence, not the action's
        # semantic result. The next stage/action must acquire current pixels.
        self.invalidate_readiness('input_sent')

    def poll_late_responses(self,obs):
        watcher=getattr(self,'late_responses',None)
        if watcher is None or not watcher.pending:return False
        state=getattr(self.logs,'state',None);offset=getattr(self.logs,'revision_offset',0)
        pending=[Resolution(at_revision=obs.revision,complete=False,log_position=obs.log_position,
            **{**f['action'],'started_at_revision':f['action']['started_at_revision']+offset})
            for f in getattr(state,'block_stack',()) if f.get('action')]
        late=watcher.inspect(obs,pending)
        if late:
            self.cache_until=0
            self.invalidate_readiness('late_abandoned_action_response')
            self.audit.write('late_abandoned_action_response',receipts=late,resend=False)
        return bool(late)

    def park_cursor(self,executor):
        self.transport=getattr(executor,'transport',None)
        return self.park_neutral()

    def park_neutral(self):
        """Move the cursor off card art; hovering a trinket covers the board.

        A hover tooltip left open by a resting cursor is a persistent overlay,
        not an animation: it can hide the scene the agent is waiting for.
        Moving the cursor is the only input used here, and only inside an
        unobscured, foreground, owned client.
        """
        transport=getattr(self,'transport',None);windows=getattr(self,'windows',None)
        if transport is None or windows is None:return False
        try:
            guard=getattr(self,'input_guard',None)
            if guard:guard()
            if transport.mouse_down or transport.pressed(1) or transport.pressed(2):return False
            w=windows.snapshot(self.binding.hwnd)
            if w.blockers or w.pid!=self.binding.pid:return False
            point=w.client.point(.97,.5)
            if not windows.owns_point(w.hwnd,*point):return False
            transport.move(*point)
            return True
        except InterruptedError:
            raise
        except Exception:
            return False

    @measured_stage('followup_interaction')
    def finish_input(self,executor,action,plan):
        self.transport=getattr(executor,'transport',None)
        if action.kind==Kind.DISMISS:
            from .result_flow import finish_rewards
            finish_rewards(self,executor,action,plan)
            return
        if action.sub_option_id is not None:
            from .live_interactions import finish_branch
            finish_branch(self,executor,action,plan)
        if action.target is not None:
            card=next((c for c in self.last_obs.hand if c.entity_id==action.source),None)
            hand_target=any(c.entity_id==action.target for c in self.last_obs.hand)
            if action.kind in (Kind.HERO_POWER,Kind.ACTIVATE) or (action.kind==Kind.PLAY and card and (card.card_type=='MINION' or action.sub_option_id is not None or hand_target)):
                if not (action.interaction and action.interaction.placement=='magnetize'):
                    from .live_interactions import finish_target
                    finish_target(self,executor,action,plan)
        if action.kind!=Kind.CHOOSE or not self.last_obs.choices:return
        types={c.card_type for c in self.last_obs.choices}
        if types not in ({'HERO'},{'BATTLEGROUND_TRINKET'}):return
        # Hovering the selected card can cover its neighbor with tooltips,
        # destroying the complete-row evidence needed for confirmation.
        self.park_cursor(executor)
        from .live_interactions import ensure_stage_unsent
        ensure_stage_unsent(self,'confirm')
        selected_position=[c.entity_id for c in self.last_obs.choices].index(action.source)+1
        references=[next((m.rect for m in self.mappings if m.key==f'entity:{c.entity_id}' and m.status=='matched'),None)
                    for c in self.last_obs.choices]
        from .readiness_window import stage_context
        from .green_readiness import SignalReadiness
        from .confirmation import request_identity,confirmation_reasons
        readiness=SignalReadiness();before=self.last_obs;stage_prepared=False
        deadline=time.monotonic()+CONFIRM_SECONDS;first=True;last_frame=None
        unselected_since=None
        saved=getattr(self,'checkpoint',None)
        reselected=bool(saved and saved.stage_state('hero_reselect') in ('prepared','sent'))
        while time.monotonic()<deadline:
            if executor.stopped() or executor.transport.pressed(0x77):raise InterruptedError('human_stop')
            w=self.windows.snapshot(self.binding.hwnd)
            if w.blockers or w.pid!=self.binding.pid or w.token!=plan.window_token:raise RuntimeError('hero_confirm_focus_changed')
            obs=self.semantic_observation(getattr(self.logs,'synchronize',self.logs.observe)())
            if request_identity(before)!=request_identity(obs) or tuple(c.entity_id for c in before.choices)!=tuple(c.entity_id for c in obs.choices):
                raise RuntimeError('choice_confirmation_refused:choice_request_context_changed')
            from .live_interactions import capture_stage
            image,capture=capture_stage(self,w,obs,kind='confirm',action=action);box=None;selected=()
            if capture and not capture.get('geometry_unchanged',True):raise RuntimeError('choice_capture_geometry_changed')
            from .live_interactions import archive_interaction
            last_frame=(obs,w,image,capture)
            if first:
                archive_interaction(self,'choice-first',action,*last_frame);first=False
            local=getattr(getattr(self,'mapper',None),'confirmation',lambda *a:None)(obs,image,w.to_dict())
            if local is not None:box,selected=local
            elif types=={'BATTLEGROUND_TRINKET'}:
                _,_,evidence=self.mapper.trinkets.detect(image,len(obs.choices))
                if evidence.get('confirm_appearance')=='cyan_selected_like' and evidence.get('selected_position_inferred')==selected_position and evidence.get('confirm_region'):
                    box=Box(**evidence['confirm_region'])
                    selected=(action.source,)
            else:
                box=confirm_button(image)
                from .live_choices import selected_hero
                index=selected_hero(image,references) if all(references) else None
                if index is not None:selected=(obs.choices[index].entity_id,)
            # 英雄点击只改变本地选择，确认才提交。完整未选画面连续出现时，
            # 只允许同一未锁定英雄重选一次；不用于饰品、打牌、购买或未知结果。
            if types=={'HERO'} and not reselected and not box and not selected:
                from .confirmation import unselected_hero_evidence
                same=not confirmation_reasons(before,obs,action,selected=(action.source,),enabled=True)
                ledger=getattr(self,'checkpoint',None)
                unsent=not ledger or ledger.stage_state('confirm') not in ('prepared','sent')
                valid=same and unsent and not capture.get('stage_context_changed') and unselected_hero_evidence(self.mapper,obs,image,w.to_dict(),action)
                if valid:
                    if unselected_since is None:unselected_since=capture['end']
                    elif capture['end']-unselected_since>=.6:
                        from .live_interactions import stage_click,record_stage,StageEvidenceExpired
                        own=next((m.rect for m in self.mappings if m.entity_id==action.source and m.status=='matched'),None)
                        if own is not None:
                            if not ledger or ledger.stage_state('hero_reselect') is None:
                                record_stage(self,'hero_reselect','prepared',entity=action.source)
                            try:stage_click(self,executor,action,plan,obs,w,own,stage='hero_reselect',capture=capture)
                            except StageEvidenceExpired:unselected_since=None;continue
                            reselected=True
                            record_stage(self,'hero_reselect','sent',entity=action.source)
                            self.audit.write('hero_unselected_reselect',source=action.source,attempts=1,
                                request_id=action.choice_id,consuming_action_resend=False)
                            readiness.invalidate();continue
                else:unselected_since=None
            if box and selected:
                reasons=confirmation_reasons(before,obs,action,selected=selected,enabled=True)
                if reasons==('choice_client_not_shown',):
                    readiness.invalidate();executor.cancel.wait(.05);continue
                if reasons:
                    from .live_interactions import archive_interaction
                    archive_interaction(self,'confirm-refused',action,obs,w,image,capture,reasons=reasons)
                    raise RuntimeError('choice_confirmation_refused:'+','.join(reasons))
            evidence=readiness.inspect(stage_context(obs,action,'confirm'),capture,
                (box.x,box.y,box.width,box.height) if box else (),valid=bool(box and selected==(action.source,)),window=w.token)
            if evidence['ready']:
                current=self.semantic_observation(getattr(self.logs,'synchronize',self.logs.observe)())
                reasons=confirmation_reasons(before,current,action,selected=selected,enabled=True)
                if reasons:raise RuntimeError('choice_confirmation_refused:'+','.join(reasons))
                from .live_interactions import record_stage,stage_click,StageEvidenceExpired
                if not stage_prepared:
                    record_stage(self,'confirm','prepared',entity=action.source);stage_prepared=True
                try:stage_click(self,executor,action,plan,obs,w,box,stage='confirm',capture=capture)
                except StageEvidenceExpired:
                    self.audit.write('stage_evidence_recapture',stage='confirm',input_sent=False)
                    readiness.invalidate();continue
                record_stage(self,'confirm','sent',entity=action.source)
                if types=={'HERO'}:self.hero_choice_sent=action.source
                self.audit.write('choice_confirm_sent',entity_id=action.source,card_types=sorted(types),region=asdict(box));return
            time.sleep(.08)
        if last_frame:archive_interaction(self,'choice-failed',action,*last_frame,reason='selection_or_confirm_unproven')
        raise TimeoutError('hero_confirmation_control_not_confirmed')

    def resume_selected_choice(self,executor,action,before):
        """Finish only an independently proven selected trinket, never reselect."""
        if action.kind!=Kind.CHOOSE:return
        obs=self.semantic_observation(self.logs.observe())
        if (obs.game_id,obs.turn)!=(before.game_id,before.turn):return
        if obs.phase!=Phase.CHOICE or not obs.interaction or obs.interaction.request_id!=action.choice_id:return
        if not obs.choices or any(c.card_type!='BATTLEGROUND_TRINKET' for c in obs.choices):return
        ids=[c.entity_id for c in obs.choices]
        if action.source not in ids:return
        self.park_cursor(executor)
        w=self.windows.snapshot(self.binding.hwnd)
        if w.blockers:return
        from .live_interactions import capture_stage
        image,_=capture_stage(self,w,obs,kind='confirm')
        local=getattr(self.mapper,'confirmation',lambda *a:None)(obs,image,w.to_dict())
        if local is not None:
            box,selected=local
            if box is None or selected!=(action.source,):return
        else:
            _,_,evidence=self.mapper.trinkets.detect(image,len(ids))
            if evidence.get('confirm_appearance')!='cyan_selected_like' or evidence.get('selected_position_inferred')!=ids.index(action.source)+1:return
            if not evidence.get('confirm_region'):return
            box=Box(**evidence['confirm_region'])
        from .confirmation import confirmation_reasons
        current=self.semantic_observation(self.logs.observe())
        if confirmation_reasons(before,current,action,selected=(action.source,),enabled=True):return
        ledger=getattr(self,'checkpoint',None)
        if ledger and ledger.stage_state('confirm') in ('prepared','sent'):return
        x,y=box.point();now=time.time()
        p=NativePlan(obs.revision,action.key,w.token,now,now+1,0,(w.client.x+x,w.client.y+y),None,None,obs.revision,
            obs.game_id,str(self.logs.session),obs.interaction.request_id,input_context_token(obs,action))
        self.pending_action=action
        self.audit.write('resume_confirmation_only',source=action.source,choice_id=action.choice_id,reselect=False)
        # Recovery uses the same selected-identity and enabled-control gate.
        # finish_input(CHOOSE) only confirms; it never clicks a candidate.
        self.last_obs=before
        self.finish_input(executor,action,p)
        self.cache_until=0

    def recover_pending(self,executor,action,before):
        """Resume a never-prepared stage, never replay an ambiguous stage."""
        from .verification import verify,Status
        from .live_interactions import finish_branch,finish_target
        obs=self.semantic_observation(self.logs.observe())
        outcome=verify(action,before,obs)
        if outcome.status==Status.CONFIRMED:return
        if outcome.status==Status.RESOLVED and any(r.complete and r.kind==action.kind and r.source==action.source and r.target==action.target for r in obs.resolutions):return
        if (obs.game_id,obs.turn)!=(before.game_id,before.turn):return
        if action.kind==Kind.CHOOSE:
            self.resume_selected_choice(executor,action,before);return
        ledger=getattr(self,'checkpoint',None)
        if not ledger or ledger.stage_state('initial')!='sent':return
        w=self.windows.snapshot(self.binding.hwnd)
        if w.blockers:return
        now=time.time()
        plan=NativePlan(obs.revision,action.key,w.token,now,now+1,0,None,None,None,obs.revision,
            obs.game_id,str(self.logs.session),obs.interaction.request_id if obs.interaction else None,input_context_token(obs,action))
        self.pending_action=action;self.last_obs=before;self.cache_until=0
        if action.sub_option_id is not None:
            state=ledger.stage_state('branch')
            if state is None:finish_branch(self,executor,action,plan)
            elif state!='sent':return
        if action.target is not None:
            if ledger.stage_state('target') is not None:return
            finish_target(self,executor,action,plan)
        self.cache_until=0


def main():
    import argparse
    from .runtime import AuditLog
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--enable-input',action='store_true');parser.add_argument('--seconds',type=int,default=2400)
    parser.add_argument('--policy',choices=('heuristic','coverage','baseline'),default='heuristic')
    parser.add_argument('--geometry-mode',choices=('geometry','legacy','compare'),default='geometry')
    parser.add_argument('--layout-catalog',type=Path)
    parser.add_argument('--local-layout',type=Path,help='Reviewed local fullscreen configuration; default requires completed local calibration')
    parser.add_argument('--decision-ahead',action=argparse.BooleanOptionalAction,default=False,help='Opt in to pure preview during client animation; current heuristic uses the simpler default path')
    parser.add_argument('--interaction-speed',choices=('safe','experimental'),default='safe',help='Experimental gesture/region timings require live validation; safe preserves existing timings')
    parser.add_argument('--gesture-clock',choices=('precise','event'),default='precise',help='Private high-resolution timer or legacy Event.wait; same configured gesture minimum delays')
    parser.add_argument('--dark-gifts-per-turn',type=int,choices=(1,3),default=1)
    parser.add_argument('--resume',type=Path,help='Reconcile an existing checkpoint without replaying pending inputs')
    parser.add_argument('--game-dir',type=Path,default=Path(__import__('os').environ.get('HEARTHSTONE_DIR', 'Hearthstone')))
    parser.add_argument('--ui-profile',type=Path,help='Generic UI anchors; defaults to the project calibration')
    parser.add_argument('--hand-profile',type=Path,help='Hand marker calibration; defaults to the project calibration')
    args=parser.parse_args()
    if not args.enable_input:parser.error('Explicit --enable-input required')
    root=args.out or Path(__file__).resolve().parents[2]/'runtime'/('live-play-'+time.strftime('%Y%m%d-%H%M%S'));root.mkdir(parents=True,exist_ok=False)
    audit=AuditLog(root/'agent.jsonl',max_bytes=128*1024**2,backup_count=2)
    from .live_session import run_session
    from .live_lease import LiveLease
    try:
        with LiveLease(args.game_dir):return run_session(args,root,audit)
    except RuntimeError as exc:
        write_json(root/'report.json',dict(status='failed',reason=str(exc),full_game_verified=False))
        print(str(exc),flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
