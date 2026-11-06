"""Environment-independent, verified-only teacher transitions (JSONL schema 1).

These are heuristic demonstrations, not expert/human labels or win-rate data.
Failed, merely dispatched and unverified decisions are never exported as
successful transitions. Full candidate identities preserve legal action masks.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path

from .model import Kind


class DemonstrationPolicy:
    def __init__(self, policy, path, *, trace_path=None,max_export_bytes=None):
        self.policy=policy;self.path=Path(path);self.pending=OrderedDict()
        self.trace_path=Path(trace_path) if trace_path is not None else None;self.trace_sig=None
        self.max_export_bytes=max_export_bytes;self.truncated_paths=set()

    def _append(self,path,content):
        if str(path) in self.truncated_paths:return
        payload=(json.dumps(content,ensure_ascii=False)+'\n').encode('utf-8')
        size=path.stat().st_size if path.exists() else 0
        if self.max_export_bytes is not None and size+len(payload)>self.max_export_bytes:
            self.truncated_paths.add(str(path));return
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab') as stream:stream.write(payload)

    def decide(self, observation):
        decision=self.policy.decide(observation)
        self.on_decision_adopted(observation,decision)
        return decision

    def preview(self,observation):
        # Never write candidate traces or teacher pending records speculatively.
        preview=getattr(self.policy,'preview',None)
        if preview is None:raise ValueError('wrapped_policy_does_not_support_preview')
        return preview(observation)

    def on_decision_adopted(self,observation,decision):
        ranking=[dict(action=asdict(r.action),utility=r.utility,reason=r.reason) for r in self.policy.rank(observation)]
        if self.trace_path is not None:
            signature=(observation.game_id,observation.revision,observation.ui_revision,observation.ready,
                decision.action.key,tuple((a.key,a.option_id,a.resource_cost,a.evidence) for a in observation.actions),
                tuple((r['action']['kind'],r['action']['source'],r['utility']) for r in ranking))
            if signature!=self.trace_sig:
                self._append(self.trace_path,dict(schema=1,time=time.time(),observation=observation.to_dict(),
                        decision=asdict(decision),ranking=ranking,verified=False,
                        evidence='decision_before_dispatch_not_success_label'))
                self.trace_sig=signature
        if decision.action.kind!=Kind.WAIT:
            key=(observation.game_id,observation.revision,decision.action.key)
            self.pending[key]=dict(observation=observation.to_dict(),action=asdict(decision.action),
                reason=decision.reason,ranking=ranking)
            while len(self.pending)>128:self.pending.popitem(last=False)

    def on_verified(self,action,before,after,outcome):
        if str(outcome.status) not in ('confirmed','resolved'):return
        self.policy.on_verified(action,before,after,outcome)
        sample=self.pending.pop((before.game_id,before.revision,action.key),None)
        if sample is None:return  # Recovery has no original filtered action mask.
        # The client can safely continue from a known placement deviation,
        # but the requested action must not become a successful teacher label.
        if {'placement_deviated','placement_unobservable'} & set(outcome.proofs):return
        content=dict(schema=1,label_source='heuristic_verified',teacher=self.policy.metadata(),
            **sample,next_observation=after.to_dict(),verification=asdict(outcome),
            terminal=after.phase.value=='game_over',reward=None)
        encoded=json.dumps(content,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        content['sample_id']=hashlib.sha256(encoded.encode()).hexdigest()
        self._append(self.path,content)

    def on_abandoned(self,action,before,after,reason):
        """A proven no-effect dispatch is neither a success nor a teacher label."""
        self.pending.pop((before.game_id,before.revision,action.key),None)

    def checkpoint_state(self):return self.policy.checkpoint_state()
    def restore_checkpoint_state(self,state):return self.policy.restore_checkpoint_state(state)
    def metadata(self):return self.policy.metadata()


def create_policy(name, *, dark_gifts_per_turn=1, demonstration_path=None):
    if name=='heuristic':
        from .heuristic import HeuristicPolicy
        policy=HeuristicPolicy()
        return DemonstrationPolicy(policy,demonstration_path,trace_path=Path(demonstration_path).with_name('policy-candidates.jsonl'),max_export_bytes=128*1024**2) if demonstration_path is not None else policy
    from .policy import RulePolicy,RuleProfile
    if name=='coverage':return RulePolicy(RuleProfile(dark_gifts_per_turn=dark_gifts_per_turn))
    if name=='baseline':return RulePolicy()
    raise ValueError('unknown_policy:'+name)
