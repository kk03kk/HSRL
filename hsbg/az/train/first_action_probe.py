"""One declared legal first action, then an unchanged research continuation."""
import numpy as np


class _FirstActionEvaluator:
    def __init__(self,base):self.base=base;self.model=getattr(base,'model',None);self.action=None;self.used=False
    def reset(self):self.used=False
    def __call__(self,features,cards,mask,seat):
        if not self.used:
            if self.action is None or not mask[self.action]:raise ValueError('declared first action is not guarded-legal')
            self.used=True;p=np.zeros(len(mask),dtype=np.float32);p[self.action]=1
            return p,0.
        return self.base(features,cards,mask,seat)


class FirstActionPolicy:
    name='research-single-first-action'
    supports_pi=False
    def __init__(self,policy):
        self.policy=policy;self.evaluator=policy.evaluator
        guard=getattr(policy,'policy',policy)
        self.first=_FirstActionEvaluator(guard.evaluator);guard.evaluator=self.first
        self.first_decision=True
    def configure(self,action):
        if type(action) is not int or not 8<=action<16:raise ValueError('bridge requires a native Buy')
        self.first.action=action
    def reset(self):self.policy.reset();self.first.reset();self.first_decision=True
    @property
    def forced_actions(self):return self.policy.forced_actions
    def pending_guard_meta(self):return self.policy.pending_guard_meta()
    def decide(self,game,seat):
        first=self.first_decision;decision=self.policy.decide(game,seat)
        if first and (not self.first.used or decision.action!=self.first.action):
            raise ValueError('first-action intervention was not executed by ordinary guard')
        decision.info.update(bridge_first_action=bool(first),bridge_declared_buy=self.first.action)
        self.first_decision=False;return decision
