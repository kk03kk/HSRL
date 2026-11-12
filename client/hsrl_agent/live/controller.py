"""Single-flight model -> input -> causal receipt. No retry of uncertain input."""
from dataclasses import asdict,replace
import copy
import os
import json
from pathlib import Path
import time
from .actions import bind_actions
from .contracts import Blocked,InputNotSent,Binding,Decision,state_stamp,execution_stamp,replace_file,remove_file,compact_snapshot,digest
from .verification import verify

class Controller:
    def __init__(self,policy,geometry,backend,out,*,enable_input=False,clock=time.monotonic,timeout=12.,practical=False,recovery_timeout=120.,trace=None):
        self.policy=policy;self.geometry=geometry;self.backend=backend;self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=True);self.enable_input=enable_input
        self.clock=clock;self.timeout=timeout;self.pending=None;self.ended=None;self.halted=None
        self.practical=practical;self.turn_counts={}
        self.recovery_timeout=max(timeout,recovery_timeout) if practical else timeout
        self.overdue=False
        self.target_sent=False
        self.confirm_sent=False
        self.branch_sent=False
        self.hero_reselected=False;self.hero_unselected_since=None;self.hero_unselected_last=None;self.hero_unselected_key=None
        self.parents=[]
        self.unsent_retries=set()
        self.candidate=None
        self.trace=trace
        self.current_decision_id=None
        # LockAll is an animated toggle.  In the archived 255790 game the
        # second press 1.39 s after the first reached the server, while a third
        # press only 0.94 s after the second was swallowed by the button
        # animation (no SendOption/Power root).  Keep a short local cooldown
        # after a Freeze dispatch instead of shortening every wait or blindly
        # resending uncertain input.
        self.freeze_cooldown_until=0.
        self.freeze_cooldown_key=None
        self.freeze_cooldown_noted=False
        self.freeze_cooldown_seconds=1.35
        # A gesture the client never submitted needs no 120 s of silence: the
        # client's own SendOption cursor proves it. Grace is deliberately far
        # above one frame/animation tick so a late submit can never be mistaken
        # for a swallowed click.
        self.unsent_grace=2.5
        # A submitted-but-rejected option leaves the same option authorized and
        # produces no root. Six seconds is ~60x the observed client/server update
        # latency; the 120 s hard stop stays as the last resort.
        self.rejected_grace=6.0
        # Identical gesture repeats: only a *confirmed applied* repeat is
        # suppressed (see _gesture_allowed), so a swallowed first click can
        # always be retried.
        self.repeat_window=3.0
        self.repeat_sensitive=frozenset(('move','choose','branch','target','end','freeze'))
        self.gesture_state={}
        self.pending_gesture_key=None
        # Match the frozen training/eval decode contract exactly:
        # max 80 actions per turn, at most 5 moves, repeat limit 2 and no
        # immediate inverse move.  Only applied actions are counted; a
        # cancelled pre-press gesture never consumes a slot.
        self.move_history={}
        state_file=self.out/'controller-state.json'
        if enable_input and practical and state_file.exists():
            state=json.loads(state_file.read_text(encoding='utf8'))
            self.ended=tuple(state['ended']) if state.get('ended') else None
            self.turn_counts={tuple(row['turn']):row['counts'] for row in state.get('turn_counts',[])}
            self.move_history={tuple(row['turn']):row for row in state.get('move_history',[])}
        if enable_input and (self.out/'inflight.json').exists():
            raw=json.loads((self.out/'inflight.json').read_text(encoding='utf-8'))
            decision=raw['decision'];b=decision['binding'];b['targets']=tuple(b.get('targets',()))
            b['branches']=tuple((r[0],r[1],r[2],tuple(r[3])) for r in b.get('branches',()))
            decision['binding']=Binding(**b)
            self.pending=(Decision(**decision),raw['before'],clock())
            self.target_sent=raw.get('target_sent',False)
            self.confirm_sent=raw.get('confirm_sent',False)
            self.branch_sent=raw.get('branch_sent',False)
            self.hero_reselected=raw.get('hero_reselected',False)
            if raw.get('delivery_unknown'):self.halted='input interrupted; delivery unknown'
            self.parents=raw.get('parents',[])
            if hasattr(policy,'sha') and any(row.get('checkpoint_sha256')!=policy.sha
                    for row in [decision]+[parent['decision'] for parent in self.parents]):
                raise Blocked('unresolved input belongs to a different policy; resume its pinned checkpoint before continuation')
    #: 需要进对局记录（trace/events.jsonl）的桥接事件。
    #: 只收"重试/阻断/分支/回执"这类低频率、对还原因果必需的事件；
    #: ``status`` / ``model_decision`` / ``decision_reused`` / ``input_stage`` /
    #: ``native_input_trace`` 等高频诊断留在 live-state/events.jsonl，不进本局记录。
    TRACE_EVENTS=frozenset((
        'suspend_for_server_choice','confirm_model_choice','model_target','model_branch',
        'model_prospective_target','cancelled_before_press','retried_unsent_input',
        'expired_unsent_input','retired_externally_resolved_play',
        'retired_externally_resolved_choice','reselect_unsubmitted_hero',
        'duplicate_gesture_suppressed','waiting_control_settle','halted'))
    def event(self,kind,**data):
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(at=time.time(),event=kind,**data),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        if self.trace is not None and kind in self.TRACE_EVENTS:
            try:self.trace.note(kind,**data)
            except Exception:self.trace=None
    def persist(self):
        if not self.enable_input:return
        d,before,_=self.pending
        path=self.out/'inflight.json';tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(dict(decision=asdict(d),before=before,target_sent=self.target_sent,confirm_sent=self.confirm_sent,branch_sent=self.branch_sent,
                           hero_reselected=self.hero_reselected,parents=self.parents,
                           delivery_unknown=self.halted=='input interrupted; delivery unknown'),f,ensure_ascii=False)
            f.flush();os.fsync(f.fileno())
        replace_file(tmp,path)
    def clear(self):
        self.overdue=False
        self.pending_gesture_key=None
        self.pending=None;self.target_sent=False;self.confirm_sent=False;self.branch_sent=False
        self.hero_reselected=False;self.hero_unselected_since=None;self.hero_unselected_last=None;self.hero_unselected_key=None
        if self.parents:
            raw=self.parents.pop();decision=raw['decision'];b=decision['binding']
            b['targets']=tuple(b.get('targets',()))
            b['branches']=tuple((r[0],r[1],r[2],tuple(r[3])) for r in b.get('branches',()))
            decision['binding']=Binding(**b)
            self.pending=(Decision(**decision),raw['before'],self.clock())
            self.target_sent=raw['target_sent'];self.branch_sent=raw['branch_sent'];self.confirm_sent=raw['confirm_sent']
            self.hero_reselected=raw.get('hero_reselected',False)
            self.persist();return
        if self.enable_input:remove_file(self.out/'inflight.json')
    def _save_state(self):
        if not self.enable_input or not self.practical:return
        from .runtime import save
        self.turn_counts=dict(list(self.turn_counts.items())[-4:])
        self.move_history=dict(list(self.move_history.items())[-4:])
        save(self.out/'controller-state.json',dict(ended=self.ended,
            turn_counts=[dict(turn=list(k),counts=v) for k,v in self.turn_counts.items()],
            move_history=[dict(v,turn=list(k)) for k,v in self.move_history.items()]))
    def _history(self,turn):
        return self.move_history.setdefault(turn,dict(last_action=None,counts={}))
    def _note_action(self,turn,action):
        history=self._history(turn)
        history['last_action']=int(action)
        if 196<=int(action)<245:
            action=int(action)
            history['counts'][str(action)]=history['counts'].get(str(action),0)+1
        counts=self.turn_counts.setdefault(turn,dict(actions=0,moves=0))
        counts['actions']+=1
        if 196<=int(action)<245:counts['moves']+=1
    def _guarded_bindings(self,snapshot,bindings,*,turn_key=None):
        """Apply the frozen 80/5/2/inverse move rules before model sampling."""
        turn=turn_key or (snapshot['game_id'],snapshot['client'].get('turn'))
        history=self._history(turn)
        counts=self.turn_counts.get(turn,dict(actions=0,moves=0))
        end=tuple(b for b in bindings if b.kind=='end')
        if counts['actions']>=80 and end:
            return end
        last=history.get('last_action');counts_by_action=history.get('counts',{})
        cap=counts.get('moves',0)>=5
        inverse=None
        if isinstance(last,int) and 196<=last<245:
            src=(last-196)//7;dst=(last-196)%7
            if src!=dst:inverse=196+dst*7+src
        out=[]
        for binding in bindings:
            action=int(binding.action)
            if not 196<=action<245:
                out.append(binding);continue
            if cap or action==inverse or counts_by_action.get(str(action),0)>=2:
                continue
            out.append(binding)
        return tuple(out)
    def _fresh(self,snapshot,before_send,binding=None):
        if self.enable_input and self.practical and snapshot.get('comparison')=='semantic-v1' and before_send is None:
            raise Blocked('semantic runtime requires a pre-dispatch admission guard')
        if before_send:
            fresh=before_send.check_action(snapshot,binding) if hasattr(before_send,'check_action') else before_send(snapshot)
            if execution_stamp(fresh)!=execution_stamp(snapshot):
                raise Blocked('executable state changed before interaction continuation')
            if fresh.get('visual',{}).get('window_token')!=snapshot.get('visual',{}).get('window_token'):
                raise Blocked('window changed before interaction continuation')
            if (self.enable_input and self.practical and snapshot.get('comparison')=='semantic-v1'
                    and not fresh.get('visual',{}).get('admitted')):
                raise Blocked('two-frame action admission is missing')
            return fresh
        return snapshot
    def _execute(self,plan,snapshot,*,rollback=None,flags=None):
        self.event('input_stage',game_id=snapshot['game_id'],plan=plan,
                   log_position=snapshot['client'].get('log_position'),
                   visual_evidence=snapshot.get('visual',{}).get('dispatch_evidence'))
        status='sent';error=None;outcome=None
        physical=False
        try:
            self.backend.execute(plan,snapshot)
            physical=True
            if getattr(self.backend,'last_timing',None):self.event('input_timing',**self.backend.last_timing)
        except InputNotSent as exc:
            status='cancelled_before_press';error=str(exc)
            self.event('cancelled_before_press',game_id=snapshot['game_id'])
            if rollback:
                self.pending=rollback;self.target_sent,self.confirm_sent,self.branch_sent=flags;self.persist()
            else:self.clear()
            raise
        except BaseException as exc:
            status='delivery_unknown';error=type(exc).__name__+': '+str(exc)
            self.halted='input interrupted; delivery unknown';self.persist();self.event('halted',reason=self.halted);raise
        finally:
            if getattr(self.backend,'last_input_trace',None):self.event('native_input_trace',**self.backend.last_input_trace)
            if self.trace is not None and self.current_decision_id:
                try:self.trace.execution(self.current_decision_id,snapshot=snapshot,plan=plan,status=status,error=error,
                    timing=getattr(self.backend,'last_timing',None),
                    native_trace=getattr(self.backend,'last_input_trace',None),outcome=outcome,
                    physical=physical or status=='delivery_unknown')
                except Exception:self.trace=None
    def needs_ui(self,data):
        if not self.pending:return False
        if self.halted=='input interrupted; delivery unknown':return False
        from .lifecycle import continuations
        return bool(continuations(self.pending[0].binding,data,confirm_sent=self.confirm_sent,
            target_sent=self.target_sent,branch_sent=self.branch_sent))
    def retire_stale_inflight(self,snapshot):
        """A ledger restored from a previous run cannot outlive its turn.

        ``inflight.json`` exists so a restart inside the *same* turn never
        resubmits an uncertain gesture. When the restored intent belongs to an
        earlier turn (or another game), its effect window is over: keeping it
        would make every later attach wait 120 s and halt on the same dead
        ledger (seen at 14:36 and again on the 14:52 attach). Archive it as
        unresolved — never as success — and let this run start clean.
        """
        if not self.pending:return False
        d,before,_=self.pending;new=snapshot['client']
        if d.game_id!=snapshot.get('game_id'):reason='different game'
        elif isinstance(new.get('turn'),int) and isinstance(before['client'].get('turn'),int) \
                and new['turn']>before['client']['turn']:reason='turn advanced while unattended'
        else:return False
        from .runtime import save
        archive=self.out/'retired-stale'/(digest(asdict(d))+'.json')
        if not archive.exists():
            save(archive,dict(decision=asdict(d),before=before,parents=self.parents,reason=reason,
                outcome='restored_ledger_expired_before_any_input_no_resend'))
        self.event('retired_stale_inflight',game_id=d.game_id,action=d.action,reason=reason,archive=str(archive))
        self.parents=[];self.halted=None;self.clear();return True
    def unsent_reconciliation(self,snapshot):
        """Prove the client never submitted this gesture and retire it early.

        ``client_submission_position`` is the power-line index of the client's
        own last SendOption/SendChoices. When it did not advance past the
        dispatch point, nothing left the client for this gesture: the click was
        swallowed by an animation/cooldown or landed on a control that refused
        it. That is a *known unsent* outcome, so the durable intent can be
        archived and the model may decide again from live state. It never
        resends the old gesture, and it never shortens the wait when the client
        did submit.
        """
        if not self.practical or not self.pending or self.parents:return None
        if self.target_sent or self.confirm_sent or self.branch_sent:return None
        d,before,started=self.pending
        if self.clock()-started<self.unsent_grace:return None
        new=snapshot['client'];old=before['client']
        if (snapshot.get('game_id')!=before.get('game_id')
                or new.get('local_player_id')!=old.get('local_player_id')
                or not isinstance(old.get('log_position'),int)):return None
        submitted=new.get('client_submission_position')
        if submitted is None:return None
        if (submitted or 0)>(old.get('log_position') or 0):return None
        from .runtime import save
        archive=self.out/'expired-unsent'/(digest(asdict(d))+'.json')
        if not archive.exists():
            save(archive,dict(decision=asdict(d),before=before,parents=self.parents,
                proof=dict(kind='client_never_submitted',grace_seconds=self.unsent_grace,
                    log_position_at_dispatch=old.get('log_position'),
                    client_submission_position=new.get('client_submission_position'),
                    client_playback_position=new.get('client_playback_position'),
                    turn=new.get('turn')),
                outcome='client_never_submitted_archived_no_resend'))
        self.event('expired_unsent_input',action=d.action,action_kind=d.binding.kind,
            source=d.binding.source,proof='client_never_submitted',archive=str(archive))
        self.clear();return True
    def rejected_reconciliation(self,snapshot):
        """Retire a submitted option the server never took.

        A rejected option (board full, illegal target, out-of-gold, animation
        swallow) produces no Power.log root at all. The client cursor advanced,
        so this is *not* the never-submitted case, and silence alone must never
        be called success. The distinguishing evidence is that the **exact same
        option is still authorized**: if the server had taken it, the card would
        have left the hand, the slot would be empty or the shop would differ.
        After a grace far above any observed update latency, the durable intent
        is archived as not applied and the model decides again from live state.
        """
        if not self.practical or not self.pending or self.parents:return None
        if self.target_sent or self.confirm_sent or self.branch_sent:return None
        d,before,started=self.pending
        if self.clock()-started<self.rejected_grace:return None
        new=snapshot['client'];old=before['client']
        if (snapshot.get('game_id')!=before.get('game_id')
                or new.get('local_player_id')!=old.get('local_player_id')
                or new.get('turn')!=old.get('turn') or new.get('phase')!=old.get('phase')):return None
        submitted=new.get('client_submission_position')
        if submitted is None or (submitted or 0)<=(old.get('log_position') or 0):return None
        if any((r.get('log_position') or 0)>(old.get('log_position') or 0)
               for r in new.get('resolutions',[])):return None
        try:current=bind_actions(snapshot)
        except Blocked:return None
        if not any(binding==d.binding for binding in current):return None
        from .runtime import save
        archive=self.out/'expired-unsent'/(digest(asdict(d))+'.json')
        if not archive.exists():
            save(archive,dict(decision=asdict(d),before=before,parents=self.parents,
                proof=dict(kind='server_did_not_take_still_legal',grace_seconds=self.rejected_grace,
                    log_position_at_dispatch=old.get('log_position'),
                    client_submission_position=submitted,turn=new.get('turn'),phase=new.get('phase')),
                outcome='submitted_option_still_authorized_archived_not_applied'))
        self.event('expired_unsent_input',action=d.action,action_kind=d.binding.kind,
            source=d.binding.source,proof='server_did_not_take_still_legal',archive=str(archive))
        self.clear();return True
    def pending_timeout(self):
        """A soft deadline changes diagnostics, never retransmits an input.

        Keep reading receipts through one normal turn. A complete later turn
        may prove no submission; a late receipt may prove acceptance. Silence
        alone still cannot clear the durable intent or authorize a new action.
        """
        if not self.pending:return None
        age=self.clock()-self.pending[2]
        if age>self.recovery_timeout:
            self.halted='input outcome unknown after bounded recovery; automatic retry disabled'
            self.event('halted',reason=self.halted,action=self.pending[0].action,age_seconds=age)
            return 'halted'
        if age>self.timeout:
            if not self.overdue:
                self.overdue=True
                self.event('receipt_overdue',action=self.pending[0].action,age_seconds=age,
                           recovery_limit_seconds=self.recovery_timeout,automatic_resend=False)
            return 'waiting_receipt_overdue_no_resend'
        return 'waiting_receipt'
    def _progress(self):
        d,before,_=self.pending
        self.pending=(d,before,self.clock());self.overdue=False
    def close_game(self,data,out,proof):
        """Archive an unresolved closed-game intent; never claim it succeeded."""
        if not self.pending:return
        if (data.get('phase')!='game_over' or data.get('game_id')!=self.pending[0].game_id
                or proof not in ('WON','LOST','TIED','CONCEDED','4','5','6','8','COMPLETE','3')):
            raise Blocked('closed-game retirement lacks authoritative same-game termination')
        from .runtime import save
        d,before,_=self.pending
        save(Path(out)/'unresolved-game-end.json',dict(decision=asdict(d),before=before,parents=self.parents,
            target_sent=self.target_sent,confirm_sent=self.confirm_sent,branch_sent=self.branch_sent,proof=proof,
            outcome='unresolved_at_authoritative_game_end_no_retry'))
        self.event('retired_closed_game_input',game_id=d.game_id,action=d.action,proof=proof)
        self.parents=[];self.clear()
    def retire_resolved_hero(self,snapshot):
        """A different authoritative hero closes this local choice, not a success.

        Preserve the intent and actual result. This cannot retire purchases,
        spells, or uncertain requests belonging to another game/owner.
        """
        if not self.practical or not self.pending or self.parents:return False
        d,before,_=self.pending;b=d.binding;old=before['client'];new=snapshot['client']
        choices=old.get('choices',[]);hero=new.get('hero') or {};actual=hero.get('entity_id')
        if (b.kind!='choose' or not choices or any(c['card_type']!='HERO' for c in choices)
                or snapshot['game_id']!=before['game_id'] or new.get('local_player_id')!=old.get('local_player_id')
                or new.get('phase') not in ('recruit','combat','game_over') or not isinstance(new.get('turn'),int) or new['turn']<1
                or actual not in {c['entity_id'] for c in choices} or actual==b.source
                or (new.get('interaction') or {}).get('request_id')==b.choice_id):return False
        receipts=[r for r in new.get('resolutions',[]) if r.get('kind')=='choose' and r.get('complete')
            and r.get('source')==actual and r.get('block_type')=='CHOICE:'+str(b.choice_id)
            and r.get('choice_owner')==old.get('local_player_id') and r.get('log_position',0)>(old.get('log_position') or 0)]
        if len(receipts)!=1:return False
        from .runtime import save,load
        path=self.out/'retired-heroes'/(digest(asdict(d))+'.json')
        if path.exists():
            if load(path).get('decision')!=asdict(d):raise Blocked('hero retirement archive identity differs')
        else:save(path,dict(decision=asdict(d),before=before,actual_hero=actual,receipt=receipts[0],
            outcome='different_hero_authoritatively_resolved_input_not_attributed',hero_reselected=self.hero_reselected))
        self.event('retired_externally_resolved_hero',requested=b.source,actual=actual,archive=str(path),model_choice_applied=False)
        self.clear();return True
    def poll_pending(self,snapshot,*,before_send=None):
        if not self.pending:return None
        d,before,started=self.pending
        rollback=(d,before,started);flags=(self.target_sent,self.confirm_sent,self.branch_sent)
        check=snapshot
        if self.practical and snapshot['client'].get('choices') and (snapshot['client'].get('interaction') or {}).get('owner')==snapshot['client'].get('local_player_id'):
            check={**snapshot,'client':{**snapshot['client'],'options_pending':False,'client_task_list':None}}
        # Completed server causality and client readiness are separate. The
        # projector/visual gate still forbids the next input during playback.
        outcome=verify(d,before,check,require_playback=not self.practical,allow_options_boundary=self.practical)
        if outcome=='context_changed':
            # A log/session epoch can change during reconnection to the same
            # match. It is not proof that the uncertain transaction vanished.
            self.halted='context changed with unresolved input; retain ledger for review'
            self.event('halted',reason=self.halted,action=d.action);return 'halted'
        if outcome=='applied':
            recovered=self.halted=='input interrupted; delivery unknown'
            if recovered:self.halted=None
            if self.pending_gesture_key:
                state=self.gesture_state.get(self.pending_gesture_key)
                if state is not None:state['confirmed_at']=self.clock()
                self.pending_gesture_key=None
            key=(before['game_id'],before['client']['turn'])
            self._note_action(key,d.action)
            self._save_state()
            self.event(outcome,action=d.action,game_id=d.game_id,action_kind=d.binding.kind,source=d.binding.source,recovered_after_interruption=recovered,
                       receipt_positions=[r['log_position'] for r in snapshot['client'].get('resolutions',[])
                           if r.get('source')==d.binding.source and (r.get('complete') or r.get('closed_by')=='next_options')
                           and r.get('log_position',0)>before['client'].get('log_position',0)],
                       receipt_boundaries=[r.get('closed_by') for r in snapshot['client'].get('resolutions',[])
                           if r.get('source')==d.binding.source and r.get('log_position',0)>before['client'].get('log_position',0)])
            if self.trace is not None and self.current_decision_id:
                try:self.trace.execution(self.current_decision_id,snapshot=None,plan=[],status='receipt',
                    outcome='applied',after=snapshot.get('client'))
                except Exception:self.trace=None
            self.clear();return outcome
        if outcome=='rejected':
            self.halted='observed action outcome differs from requested action';return 'halted'
        if self.unsent_reconciliation(snapshot):return 'expired_unsent_input'
        if self.rejected_reconciliation(snapshot):return 'expired_unsent_input'
        if self.retire_resolved_hero(snapshot):return 'externally_resolved_hero'
        if self.retire_resolved_choice(snapshot):return 'externally_resolved_choice'
        if self.retire_resolved_play(snapshot):return 'externally_resolved_play'
        if (self.practical and snapshot['game_id']==before['game_id']
                and snapshot['client'].get('local_player_id')==before['client'].get('local_player_id')
                and (snapshot['client'].get('log_position') or 0)>(before['client'].get('log_position') or 0)
                and isinstance(snapshot['client'].get('turn'),int) and isinstance(before['client'].get('turn'),int)
                and snapshot['client']['turn']>before['client']['turn']
                and (snapshot['client'].get('client_submission_position') or 0)<=(before['client'].get('log_position') or 0)
                and not self.target_sent and not self.branch_sent and not self.confirm_sent):
            # A completed turn plus a complete log with no SendOption/Choices
            # after dispatch proves this gesture was never submitted. Retire
            # it; the next turn receives a new model decision, never a resend.
            from .runtime import save
            archive=self.out/'expired-unsent'/(digest(asdict(d))+'.json')
            save(archive,dict(decision=asdict(d),before=before,parents=self.parents,
                proof=dict(game_id=snapshot['game_id'],local_player_id=snapshot['client'].get('local_player_id'),
                    turn=snapshot['client']['turn'],log_position=snapshot['client'].get('log_position'),
                    client_submission_position=snapshot['client'].get('client_submission_position')),
                outcome='no_submission_through_authoritative_later_turn_not_applied'))
            self.event('expired_unsent_input',action=d.action,source=d.binding.source,archive=str(archive),
                       old_turn=before['client']['turn'],new_turn=snapshot['client']['turn'])
            if self.halted=='input interrupted; delivery unknown':self.halted=None
            self.clear();return 'expired_unsent_input'
        if self._retry_unsent(snapshot,d,before,started):
            # A play/sell/move whose exact source is still in its old zone
            # cannot have applied: a successful play leaves the hand, a sell
            # leaves the board. With no SendOption and no receipt after the
            # intent, retire it and allow one fresh decision. Buys are excluded
            # because a delayed shop update cannot prove the card is still
            # purchasable; that preserves the late-receipt safety case.
            from .runtime import save
            archive=self.out/'retried-unsent'/(digest(asdict(d))+'.json')
            save(archive,dict(decision=asdict(d),before=before,parents=self.parents,
                proof=dict(game_id=snapshot['game_id'],local_player_id=snapshot['client'].get('local_player_id'),
                    turn=snapshot['client'].get('turn'),log_position=snapshot['client'].get('log_position'),
                    client_submission_position=snapshot['client'].get('client_submission_position')),
                outcome='no_submission_same_turn_source_unchanged_retry_not_attributed'))
            self.event('retried_unsent_input',action=d.action,source=d.binding.source,archive=str(archive),
                       turn=snapshot['client'].get('turn'))
            self.clear();return 'retry_unsent'
        interaction=snapshot['client'].get('interaction') or {}
        if (self.practical and snapshot.get('view') and snapshot['client'].get('choices')
                and interaction.get('mode')=='choice' and interaction.get('owner')==snapshot['client']['local_player_id']
                and d.binding.kind!='choose'
                and (snapshot['client'].get('client_submission_position') or 0)>(before['client'].get('log_position') or 0)):
            if len(self.parents)>=4:raise Blocked('nested choice depth exceeds four')
            parent_flags=(self.target_sent,self.confirm_sent,self.branch_sent)
            self.parents.append(dict(decision=asdict(d),before=before,target_sent=self.target_sent,
                                     confirm_sent=self.confirm_sent,branch_sent=self.branch_sent,hero_reselected=self.hero_reselected))
            self.pending=None;self.target_sent=self.confirm_sent=self.branch_sent=False
            self.event('suspend_for_server_choice',source=d.binding.source,request_id=interaction['request_id'])
            try:return self.tick(snapshot,before_send=before_send)
            finally:
                # Inference/planning/freshness can fail before child intent is
                # durable. Retain the original parent, including in-memory state.
                if self.pending is None:
                    self.parents.pop();self.pending=(d,before,started)
                    self.target_sent,self.confirm_sent,self.branch_sent=parent_flags
        if d.binding.branches and not self.branch_sent and interaction.get('mode')=='branch':
            if snapshot.get('view',{}).get('pending')!=3:raise Blocked('branch lacks a complete model observation')
            if interaction.get('source_entity')!=d.binding.source or interaction.get('request_id')!='branch:'+d.stamp:
                raise Blocked('branch request identity differs')
            if snapshot['client']['turn']!=before['client']['turn']:raise Blocked('branch turn changed')
            if [c['entity_id'] for c in snapshot['client']['choices']]!=[r[0] for r in d.binding.branches]:
                raise Blocked('branch candidates differ')
            selected=self.policy.decide(snapshot,bind_actions(snapshot))
            if self.trace is not None and self.current_decision_id:
                try:self.trace.decision(snapshot,selected,policy_ms=0.,reused=False,
                    bindings=bind_actions(snapshot),parent=self.current_decision_id,
                    seat=snapshot['client'].get('local_player_id'),role='model',
                    inputs=getattr(self.policy,'last_inputs',None))
                except Exception:self.trace=None
            spec=next((r for r in d.binding.branches if r[0]==selected.binding.source and r[1]==selected.binding.sub_option_id and r[2]=='NONE'),None)
            if spec is None:raise Blocked('model branch is not authorized')
            snapshot=self._fresh(snapshot,before_send,selected.binding)
            self.branch_sent=True
            completed=replace(d,binding=replace(d.binding,sub_option_id=spec[1],targets=tuple(spec[3])))
            self.pending=(completed,before,started);self._progress();self.persist();self.event('model_branch',decision=asdict(selected))
            self._execute(self.geometry.plan(selected,snapshot),snapshot,rollback=rollback,flags=flags);return 'branch_sent'
        hero=snapshot.get('visual',{}).get('hero_ui') or {}
        old_choice=before['client'].get('interaction') or {}
        same_candidates=([(c['entity_id'],c['card_id']) for c in snapshot['client'].get('choices',[])]==
                         [(c['entity_id'],c['card_id']) for c in before['client'].get('choices',[])])
        own_unselected_hero=(d.binding.kind=='choose' and not self.confirm_sent and not self.hero_reselected
            and snapshot.get('view',{}).get('pending')==1 and hero.get('visible') and hero.get('selected_index') is None
            and hero.get('button_neutral') and not hero.get('button_enabled') and same_candidates
            and interaction.get('request_id')==d.binding.choice_id and interaction.get('owner')==snapshot['client']['local_player_id']
            and all(old_choice.get(k)==interaction.get(k) for k in ('request_id','source_entity','owner','choice_type','task_list'))
            and snapshot['client']['turn']==before['client']['turn']
            and (snapshot['client'].get('client_submission_position') or 0)<=(before['client'].get('log_position') or 0))
        if own_unselected_hero:
            now=self.clock();proof_key=(snapshot['visual'].get('window_token'),digest(interaction))
            if (self.hero_unselected_since is None or self.hero_unselected_key!=proof_key
                    or self.hero_unselected_last is None or now-self.hero_unselected_last>.75):
                self.hero_unselected_since=now
            self.hero_unselected_key=proof_key;self.hero_unselected_last=now
            if now-self.hero_unselected_since>=.6:
                snapshot=self._fresh(snapshot,before_send,d.binding)
                fresh_hero=snapshot.get('visual',{}).get('hero_ui') or {}
                if (not fresh_hero.get('visible') or fresh_hero.get('selected_index') is not None
                        or not fresh_hero.get('button_neutral') or fresh_hero.get('button_enabled')):
                    raise Blocked('hero unselected evidence changed before one bounded reselect')
                self.hero_reselected=True;self.pending=(d,before,self.clock());self.persist()
                self.event('reselect_unsubmitted_hero',source=d.binding.source,request_id=d.binding.choice_id)
                try:self._execute(self.geometry.plan(d,snapshot),snapshot,rollback=rollback,flags=flags)
                except InputNotSent:
                    self.hero_reselected=False;self.persist();raise
                return 'hero_reselected'
        elif 'visual' in snapshot:
            # The production loop first polls raw receipts, then pixels. A raw
            # receipt poll must not erase continuously observed UI evidence.
            self.hero_unselected_since=None;self.hero_unselected_last=None;self.hero_unselected_key=None
        if (d.binding.kind=='choose' and not self.confirm_sent
                and snapshot.get('visual',{}).get('confirm_enabled')
                and interaction.get('request_id')==d.binding.choice_id
                and self._selection_matches(snapshot,d.binding.source)):
            # Confirm is a protocol continuation of the model's selected card,
            # never a new strategic decision. Persist before sending it once.
            prior=before['client'].get('interaction') or {}
            keys=('request_id','source_entity','owner','choice_type','task_list')
            if any(prior.get(k)!=interaction.get(k) for k in keys):raise Blocked('choice request identity changed')
            identity=lambda c:(c['entity_id'],c['card_id'],c.get('card_type'),c.get('controller'))
            if [identity(c) for c in before['client']['choices']]!=[identity(c) for c in snapshot['client']['choices']]:
                raise Blocked('choice candidates changed before confirm')
            if snapshot['client']['turn']!=before['client']['turn']:raise Blocked('choice turn changed')
            points,_=self.geometry.locations(snapshot)
            point=points.get('control:confirm')
            if point is None: raise Blocked('confirm control is unlocated')
            snapshot=self._fresh(snapshot,before_send,d.binding)
            if not snapshot.get('visual',{}).get('confirm_enabled') or not self._selection_matches(snapshot,d.binding.source):
                raise Blocked('model-selected card lost its visual confirmation proof')
            points,_=self.geometry.locations(snapshot)
            final_point=points.get('control:confirm')
            if final_point is None:raise Blocked('confirm control is unlocated')
            self.confirm_sent=True;self._progress();self.persist();self.event('confirm_model_choice',source=d.binding.source)
            # 确认也是"续动作"：必须和其它动作一样把指针收回右下角。
            # 2026-11-11 实机：确认点击后指针停在面板上，英雄选择时压在确认按钮、
            # 饰品选择时压在神明展示框上弹出 tooltip，遮挡了后续决策的取景。
            self._execute([('click',list(final_point)),('move_pointer',[.96,.91])],
                snapshot,rollback=rollback,flags=flags)
            return 'confirm_sent'
        # 目标续动作的触发条件：视觉信号 ``target_open`` **或** 服务端权威的
        # ``interaction.mode=='target'``。2026-11-11 T7 实机：Activate 打开发动目标
        # 选择时视觉层没有给出 target_open，续动作永远不触发，局面卡在"已选中但未指认"。
        # 下面原有的授权/身份/候选集一致性检查一条都不放松。
        target_ui = bool(snapshot.get('visual',{}).get('target_open')) or (
            interaction.get('mode')=='target'
            and interaction.get('owner')==snapshot['client'].get('local_player_id')
            and snapshot.get('view',{}).get('pending')==4)
        if d.binding.targets and d.binding.target is None and not self.target_sent and target_ui:
            # Original server target authorization remains pinned. The visual
            # source context may not change while preparing this second decision.
            if snapshot['client']['turn']!=before['client']['turn']: raise Blocked('target turn changed')
            # Do not reuse pre-play gold/hand/global features for post-play
            # targeting. A complete, fresh target view is mandatory.
            if snapshot.get('view',{}).get('pending')!=4 or interaction.get('mode')!='target':
                raise Blocked('target UI lacks a complete post-action model observation')
            if set(c['entity_id'] for c in snapshot['client']['choices'])!=set(d.binding.targets):
                raise Blocked('target authorization changed')
            selected=self.policy.decide(snapshot,bind_actions(snapshot))
            if self.trace is not None and self.current_decision_id:
                try:self.trace.decision(snapshot,selected,policy_ms=0.,reused=False,
                    bindings=bind_actions(snapshot),parent=self.current_decision_id,
                    seat=snapshot['client'].get('local_player_id'),role='model',
                    inputs=getattr(self.policy,'last_inputs',None))
                except Exception:self.trace=None
            snapshot=self._fresh(snapshot,before_send,selected.binding)
            self.target_sent=True
            completed=replace(d,binding=replace(d.binding,target=selected.binding.target))
            self.pending=(completed,before,started);self._progress();self.persist()
            self.event('model_target',decision=asdict(selected))
            self._execute(self.geometry.plan(selected,snapshot),snapshot,rollback=rollback,flags=flags)
            return 'target_sent'
        return self.pending_timeout()
    def _retry_unsent(self,snapshot,d,before,started):
        """A dropped play/sell/move may be re-decided once on the same turn.

        The source zone is the proof: a successful play removes the card from
        hand, a sell/move changes the board order. If the exact entity is still
        there and the client logged no SendOption and no receipt after the
        intent, the gesture never reached the server. Buys are excluded because
        a delayed shop update cannot prove the card is still purchasable.
        """
        if not self.practical or self.halted or self.parents:return False
        b=d.binding;old=before['client'];new=snapshot['client']
        if b.kind not in ('play','sell','move'):return False
        if self.target_sent or self.branch_sent or self.confirm_sent:return False
        if (snapshot['game_id']!=before['game_id'] or new.get('turn')!=old.get('turn')
                or new.get('phase')!='recruit' or new.get('local_player_id')!=old.get('local_player_id')
                or new.get('options_pending') or new.get('client_task_list') is not None
                or not new.get('ready')):return False
        # A Move gesture can be submitted and still take longer than the
        # generic 12 s receipt budget to appear as a reorder root.  In the
        # 2026-11-10 03:02 game, decision 200 dragged source 7364 at 03:06:36:
        # the client displayed the new order, but the root arrived at 03:06:49,
        # just after the retry had already been armed.  The retry then produced
        # a *second* reorder.  Move has an exact, local identity proof (the
        # submitted source and final order), so give its receipt the full
        # recovery window before classifying it as unsent.  This only delays a
        # retry; it never shortens a wait or authorizes a blind second drag.
        move_grace=min(self.timeout+6.,self.recovery_timeout) if b.kind=='move' else self.timeout
        if self.clock()-started<move_grace:return False
        if (new.get('client_submission_position') or 0)>(old.get('log_position') or 0):return False
        if any(r.get('source')==b.source and (r.get('log_position') or 0)>(old.get('log_position') or 0)
               for r in new.get('resolutions',[])):return False
        if new.get('choices') or new.get('interaction'):return False
        board=(lambda d:[c['entity_id'] for c in d.get('board',[])])
        if board(new)!=board(old):return False
        if b.kind=='play' and b.source not in {c['entity_id'] for c in new.get('hand',[])}:return False
        key=(snapshot['game_id'],new.get('turn'),d.stamp)
        if key in self.unsent_retries:return False
        self.unsent_retries.add(key);return True
    def _selection_matches(self,snapshot,entity):
        from .interactions import choice_context
        proof=snapshot.get('visual',{}).get('selection') or {}
        return (proof.get('entity_id')==entity and proof.get('context')==choice_context(snapshot['client'])
                and (proof.get('evidence')=='qualified_template_selection' or self.practical and
                     proof.get('evidence') in ('persistent_outline_after_pointer_exit','measured_hero_halo_and_confirm',
                                               'confirm_control_changed_after_selection'))
                and bool(proof.get('image_sha256')))
    def retire_resolved_play(self,snapshot):
        """Retire an unfinished model continuation only after its real server play.

        The actor is unknown. Never fill missing model target/branch decisions
        from the receipt or count the external completion as model success.
        """
        if not self.practical or not self.pending:return False
        d,before,_=self.pending;b=d.binding;old=before['client'];new=snapshot['client']
        if (b.kind!='play' or not (b.branches and b.sub_option_id is None or b.targets and b.target is None)
                or snapshot['game_id']!=before['game_id'] or new.get('local_player_id')!=old.get('local_player_id')):return False
        from .verification import departed_play
        matches=[]
        for receipt in new.get('resolutions',[]):
            if not receipt.get('complete') and receipt.get('closed_by')!='next_options':continue
            if (receipt.get('log_position') or 0)>(new.get('log_position') or 0):continue
            sub=receipt.get('sub_option_id');target=receipt.get('target')
            if b.branches:
                specs=[r for r in b.branches if r[1]==sub and r[2]=='NONE']
                if len(specs)!=1:continue
                targets=specs[0][3]
            else:
                if sub!=b.sub_option_id:continue
                targets=b.targets
            if (targets and target not in targets) or (not targets and target is not None):continue
            actual=replace(d,binding=replace(b,sub_option_id=sub,target=target))
            if departed_play(actual,old,new,receipt):matches.append(receipt)
        if len(matches)!=1:return False
        from .runtime import save
        path=self.out/'retired-plays'/(digest(asdict(d))+'.json')
        save(path,dict(decision=asdict(d),before=before,parents=self.parents,receipt=matches[0],
            outcome='authorized_play_resolved_without_complete_model_continuation_not_attributed'))
        self.event('retired_externally_resolved_play',source=b.source,target=matches[0].get('target'),
            sub_option_id=matches[0].get('sub_option_id'),archive=str(path),model_play_applied=False)
        if self.halted=='input interrupted; delivery unknown':self.halted=None
        self.clear();return True
    def retire_resolved_choice(self,snapshot):
        """Human/client resolution of a different offered card is not our win."""
        if not self.practical or not self.pending:return False
        d,before,_=self.pending;b=d.binding;old=before['client'];new=snapshot['client']
        context=old.get('interaction') or {};choices=old.get('choices',[])
        if (b.kind!='choose' or not choices or any(c['card_type']=='HERO' for c in choices)
                or context.get('mode')!='choice' or context.get('min_select',1)!=1 or context.get('max_select',1)!=1
                or context.get('request_id')!=b.choice_id
                or snapshot['game_id']!=before['game_id'] or new.get('local_player_id')!=old.get('local_player_id')
                or context.get('owner')!=old.get('local_player_id')
                or (new.get('interaction') or {}).get('request_id')==b.choice_id):return False
        offered={c['entity_id'] for c in choices};cursor=old.get('log_position') or 0
        receipts=[r for r in new.get('resolutions',[]) if r.get('kind')=='choose' and r.get('complete')
            and r.get('block_type')=='CHOICE:'+str(b.choice_id) and r.get('choice_owner')==old['local_player_id']
            and (r.get('log_position') or 0)>cursor and (r.get('started_at_log_position') or 0)>cursor
            and r.get('source') in offered]
        if len(receipts)!=1 or receipts[0]['source']==b.source:return False
        from .runtime import save
        path=self.out/'retired-choices'/(digest(asdict(d))+'.json')
        save(path,dict(decision=asdict(d),before=before,parents=self.parents,receipt=receipts[0],
            outcome='different_choice_authoritatively_resolved_input_not_attributed'))
        self.event('retired_externally_resolved_choice',requested=b.source,actual=receipts[0]['source'],
                   request_id=b.choice_id,archive=str(path),model_choice_applied=False)
        if self.halted=='input interrupted; delivery unknown':self.halted=None
        self.clear();return True
    def _gesture_key(self,decision,plan):
        return (decision.game_id,decision.binding.kind,json.dumps(plan,sort_keys=True))
    def _gesture_allowed(self,decision,plan,snapshot):
        """Refuse to replay a gesture whose first dispatch was *confirmed applied*.

        The 03:12 game submitted the same Move seven times in thirteen seconds;
        every one of them had a receipt, so the board really did return to the
        same order and the model asked again. Suppressing a confirmed repeat is
        safe; suppressing a *swallowed* first click would starve the panel, so
        only ``confirmed_at`` (set when the receipt for this exact plan arrived)
        ever suppresses. The window releases by itself, and it never slides.
        """
        kind=decision.binding.kind
        if kind not in self.repeat_sensitive:return True
        turn=snapshot['client'].get('turn');now=self.clock()
        key=self._gesture_key(decision,plan)
        state=self.gesture_state.get(key)
        if state and state.get('turn')==turn and state.get('confirmed_at') is not None:
            age=now-state['confirmed_at']
            if age<self.repeat_window:
                state['count']=state.get('count',1)+1
                if state['count']<=2:
                    self.event('duplicate_gesture_suppressed',action=decision.action,gesture_kind=kind,
                        seconds=round(age,3),reason='identical gesture already applied')
                return False
        self.gesture_state[key]=dict(turn=turn,at=now,count=1,
            confirmed_at=(state or {}).get('confirmed_at') if state and state.get('turn')==turn else None)
        if len(self.gesture_state)>256:
            self.gesture_state={k:v for k,v in self.gesture_state.items() if v.get('turn')==turn}
        return True
    def tick(self,snapshot,*,before_send=None):
        if self.halted: return 'halted'
        if self.pending:return self.poll_pending(snapshot,before_send=before_send)
        turn=(snapshot['game_id'],snapshot['client']['turn'])
        if self.freeze_cooldown_key and self.freeze_cooldown_key[:2]==turn:
            if self.clock()<self.freeze_cooldown_until:
                if not self.freeze_cooldown_noted:
                    self.event('waiting_control_settle',control='freeze',
                               source=self.freeze_cooldown_key[2],
                               remaining_ms=round((self.freeze_cooldown_until-self.clock())*1000,1))
                    self.freeze_cooldown_noted=True
                return 'waiting_control_settle'
            self.freeze_cooldown_key=None
        from .lifecycle import owned_selection
        owned_choice=owned_selection(snapshot['client'])
        if self.ended==turn and not owned_choice:return 'waiting_next_turn'
        bindings=bind_actions(snapshot)
        if self.practical and not snapshot['client'].get('choices') and not snapshot['client'].get('interaction'):
            bindings=self._guarded_bindings(snapshot,bindings)
        cache_key=(state_stamp(snapshot),tuple(bindings))
        reused=bool(self.practical and snapshot.get('comparison')=='semantic-v1' and self.candidate and self.candidate[0]==cache_key)
        infer_start=time.perf_counter()
        decision=self.candidate[1] if reused else self.policy.decide(snapshot,bindings)
        if not reused and self.practical and decision.binding.targets and decision.binding.kind=='play':
            source=next((c for c in snapshot['client']['hand'] if c['entity_id']==decision.binding.source),None)
            if source and source['card_type'] in ('SPELL','BATTLEGROUND_SPELL'):
                from .adaptive import prospective_target
                context=prospective_target(snapshot,decision);target=self.policy.decide(context,bind_actions(context))
                decision=replace(decision,binding=replace(decision.binding,target=target.binding.target))
                self.event('model_prospective_target',decision=asdict(target),approximation=context.get('approximation'))
        policy_ms=(time.perf_counter()-infer_start)*1000
        if self.practical and snapshot.get('comparison')=='semantic-v1':
            authorized=next((b for b in bindings if b.action==decision.action),None)
            exact=authorized==decision.binding
            prospective=bool(authorized and authorized.targets and decision.binding.target in authorized.targets
                and replace(decision.binding,target=authorized.target)==authorized)
            if (decision.game_id!=snapshot['game_id'] or decision.stamp!=state_stamp(snapshot)
                    or decision.action!=decision.binding.action or not (exact or prospective)):
                raise Blocked('model decision is inconsistent with current authorized bindings')
            if hasattr(self.policy,'sha') and decision.checkpoint_sha256!=self.policy.sha:
                raise Blocked('decision checkpoint identity differs from loaded policy')
        if self.practical:self.candidate=(cache_key,decision)
        self.event('decision_reused' if reused else 'model_decision',decision=asdict(decision),snapshot=compact_snapshot(snapshot) if self.practical else snapshot,policy_ms=policy_ms)
        if self.trace is not None:
            # Every separate model decision gets its own record, including a
            # repeated argmax after a visual retry. ``reused`` only means the
            # cache avoided a second forward pass; it does not collapse the
            # decision timeline.
            did=self.trace.decision(snapshot,decision,policy_ms=policy_ms,reused=reused,
                bindings=bindings,plan=None,
                seat=snapshot['client'].get('local_player_id'),role='model',
                inputs=getattr(self.policy,'last_inputs',None))
            self.current_decision_id=did
        try:
            plan=self.geometry.plan(decision,snapshot)
        except Blocked as exc:
            if self.trace is not None and self.current_decision_id:
                self.trace.execution(self.current_decision_id,snapshot=snapshot,plan=[],status='blocked_before_gesture',
                    error=str(exc),timing=None,native_trace=None,outcome='blocked')
            raise
        if self.trace is not None and self.current_decision_id:
            self.trace.event('gesture_plan',decision_id=self.current_decision_id,plan=plan)
        if not self.enable_input: self.event('dry_run',plan=plan);return 'dry_run'
        if not self._gesture_allowed(decision,plan,snapshot):return 'duplicate_gesture_suppressed'
        snapshot=self._fresh(snapshot,before_send,decision.binding)
        if decision.binding.kind=='choose':
            # The admission frame can refine a runtime-derived 5/6-card row.
            # Recompute the click from the final measured boxes, not from the
            # speculative pre-admission frame.
            plan=self.geometry.plan(decision,snapshot)
        if self.practical and snapshot.get('comparison')=='semantic-v1':
            # Full model context can include an unrelated public opponent
            # summary update. The physical gesture is valid iff the exact
            # authorized binding still exists in the fresh executable state.
            current=next((b for b in bind_actions(snapshot) if b.action==decision.action),None)
            exact=current==decision.binding
            prospective=bool(current and current.targets and decision.binding.target in current.targets
                and replace(decision.binding,target=current.target)==current)
            if not (exact or prospective):
                raise Blocked('decision became stale before dispatch')
        elif state_stamp(snapshot)!=decision.stamp:
            raise Blocked('decision became stale before dispatch')
        self.pending=(decision,copy.deepcopy(compact_snapshot(snapshot) if self.practical else snapshot),self.clock());self.persist()
        self.pending_gesture_key=self._gesture_key(decision,plan)
        self.event('dispatch_intent',plan=plan)
        dispatched_at=self.clock()
        self._execute(plan,snapshot)
        if decision.binding.kind=='freeze':
            self.freeze_cooldown_key=(turn[0],turn[1],decision.binding.source)
            self.freeze_cooldown_until=dispatched_at+self.freeze_cooldown_seconds
            self.freeze_cooldown_noted=False
        return 'sent'
