"""Atomic action ledger. An unresolved dispatch is reconciled, never replayed."""
from dataclasses import asdict
import json,hashlib,re
from .recording import write_json,atomic_replace
from .model import Observation,Action,Kind


class ActionCheckpoint:
    def __init__(self,path,policy,session):
        self.path=path;self.policy=policy
        self._snapshots={}
        self.state=dict(schema=1,session=str(session),pending=None,parents=[],policy=None,verified=0,game_id=None)

    def _pack(self,row):
        if not row:return row
        before=row['before']
        if not before.get('resolutions'):return row
        cached=self._snapshots.get(id(before))
        if cached is None or cached[0] is not before:
            # Preserve the complete causal state once, before publishing a
            # ledger that references it. Stage/delivery updates write only
            # the small ledger, not the same receipt history four more times.
            data=json.dumps(before,ensure_ascii=False,separators=(',',':')).encode('utf8')
            digest=hashlib.sha256(data).hexdigest();name=digest+'.json'
            directory=self.path.parent/(self.path.stem+'.snapshots');directory.mkdir(exist_ok=True)
            target=directory/name
            temporary=target.with_suffix('.tmp');temporary.write_bytes(data);atomic_replace(temporary,target)
            cached=(before,name,digest);self._snapshots[id(before)]=cached
        return {**row,'before':None,'before_ref':cached[1],'before_sha256':cached[2]}

    def save(self):
        packed={**self.state,'pending':self._pack(self.state['pending']),
            'parents':[self._pack(row) for row in self.state.get('parents',())]}
        write_json(self.path,packed)
        active={id(row['before']) for row in [self.state['pending'],*self.state.get('parents',())] if row}
        referenced={v[1] for k,v in self._snapshots.items() if k in active}
        obsolete={v[1] for k,v in self._snapshots.items() if k not in active and v[1] not in referenced}
        # Only files this ledger wrote, and only AFTER the replacement ledger
        # is committed. A retained parent or identical child shares its blob.
        for name in obsolete:
            try:(self.path.parent/(self.path.stem+'.snapshots')/name).unlink(missing_ok=True)
            except OSError:pass  # A reader may retain a handle; cleanup is optional.
        self._snapshots={k:v for k,v in self._snapshots.items() if k in active}

    def begin(self,action,before,action_id):
        if self.state['pending']:raise RuntimeError('checkpoint_has_unresolved_dispatch')
        self.state['pending']=dict(action_id=action_id,action=asdict(action),before=before.to_dict(),delivery='prepared')
        self.state['game_id']=before.game_id
        self.save()

    def suspend_for_child(self):
        if not self.state['pending']:raise RuntimeError('checkpoint_parent_missing')
        self.state.setdefault('parents',[]).append(self.state['pending']);self.state['pending']=None;self.save()

    def restore_parent_if_idle(self):
        if self.state['pending'] is None and self.state['parents']:
            self.state['pending']=self.state['parents'].pop();self.save()

    def delivery(self,delivery):
        if str(delivery) in ('not_sent','rejected'):self.state['pending']=self.state['parents'].pop() if self.state['parents'] else None
        elif self.state['pending']:self.state['pending']['delivery']=str(delivery)
        self.save()

    def stage(self,name,state,**evidence):
        """Write-ahead stage ledger: prepared is ambiguous, never replayable."""
        pending=self.state['pending']
        if pending is None:raise RuntimeError('checkpoint_stage_without_action')
        stages=pending.setdefault('stages',{})
        previous=stages.get(name,{}).get('state')
        if previous in ('prepared','sent') and state=='prepared':
            raise RuntimeError('interaction_stage_must_not_be_replayed:'+name)
        if state not in ('prepared','sent'):raise ValueError('unknown_stage_state')
        if state=='sent' and previous!='prepared':raise RuntimeError('interaction_stage_sent_without_prepare:'+name)
        stages[name]=dict(state=state,**evidence);self.save()

    def stage_state(self,name):
        pending=self.state.get('pending')
        return (pending or {}).get('stages',{}).get(name,{}).get('state')

    def abandon(self):
        """Release a dispatch independently proven to have had no effect.

        Not a success: it is never counted as verified, never replayed, and it
        restores the suspended parent the same way commit() does.
        """
        if self.state['pending'] is None:raise RuntimeError('checkpoint_has_no_pending_dispatch')
        self.state['pending']=self.state['parents'].pop() if self.state['parents'] else None
        self.state['abandoned']=self.state.get('abandoned',0)+1
        self.save()

    def interrupt_phase_wait(self):
        pending=self.state['pending']
        if (not pending or pending['action']['kind']!=Kind.END or
            pending['action'].get('evidence')!='verified_phase_wait' or pending.get('stages')):
            raise RuntimeError('only_input_free_phase_wait_can_be_interrupted')
        self.state['pending']=self.state['parents'].pop() if self.state['parents'] else None
        self.state['phase_wait_interruptions']=self.state.get('phase_wait_interruptions',0)+1
        self.save()

    def commit(self):
        self.state['pending']=self.state['parents'].pop() if self.state['parents'] else None
        self.state['verified']+=1
        exporter=getattr(self.policy,'checkpoint_state',None)
        if exporter:self.state['policy']=exporter()
        self.save()

    def load(self,path,current,session):
        state=json.loads(path.read_text(encoding='utf-8'))
        if state.get('schema')!=1:raise ValueError('checkpoint_schema_unsupported')
        for row in [state.get('pending'),*state.get('parents',())]:
            if row and row.get('before_ref'):
                name=row['before_ref'];digest=row.get('before_sha256')
                if not re.fullmatch(r'[0-9a-f]{64}\.json',name) or name!=str(digest)+'.json':raise ValueError('checkpoint_snapshot_reference_invalid')
                data=(path.parent/(path.stem+'.snapshots')/name).read_bytes()
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('checkpoint_snapshot_hash_mismatch')
                row['before']=json.loads(data)
        state.setdefault('parents',[])
        pending=state.get('pending') or (state['parents'].pop() if state['parents'] else None)
        state['pending']=pending
        before=Observation.from_dict(pending['before']) if pending else None
        if not before and state.get('game_id') and state['game_id']!=current.game_id:
            raise ValueError('checkpoint_game_mismatch')
        if before and before.game_id!=current.game_id and before.phase!='menu':raise ValueError('checkpoint_game_mismatch')
        if before and before.phase!='menu' and (before.mode!=current.mode or before.local_player_id!=current.local_player_id):raise ValueError('checkpoint_player_or_mode_mismatch')
        if state['session']!=str(session) and not (before and before.game_id.startswith('match-') and before.game_id==current.game_id):
            raise ValueError('checkpoint_session_mismatch')
        self.state=state
        restorer=getattr(self.policy,'restore_checkpoint_state',None)
        if state.get('policy') and restorer:restorer(state['policy'])
        if pending:
            raw={**pending['action'],'kind':Kind(pending['action']['kind'])}
            action=Observation.from_dict({**before.to_dict(),'actions':[raw]}).actions[0]
            return action,before,pending['action_id']
        return None

    def pending(self):
        row=self.state['pending']
        if not row:return None
        before=Observation.from_dict(row['before'])
        action=Observation.from_dict({**before.to_dict(),'actions':[row['action']]}).actions[0]
        return action,before,row['action_id']
