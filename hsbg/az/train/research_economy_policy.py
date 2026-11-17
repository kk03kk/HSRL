"""Research-only public economic option using the ordinary evaluation guards.

This is an explicit hand-coded policy intervention, not a neural model result.
The heuristic sees only the declared own public view. The shared greedy-policy
wrapper owns action counting and guard state across option entry and exit.
"""
import json
import numpy as np
from hsbg.az.train.ppo import PPOGreedyPolicy
from hsbg.az.train.research_economy import Operator,cid

def active(view):
    owned={cid(x) for x in view['me']['board']+view['me']['hand']}
    return 'PIRATE' in view['lobby_races'] and len(owned & {'BG_LOE_077','BG33_823','BG33_825'})>=2

class _Evaluator:
    def __init__(self,base,operator_mode='legacy'):
        self.base=base;self.model=getattr(base,'model',None)
        self.operator_mode=operator_mode
        self.operator=Operator(operator_mode);self.view=None;self.reason=None;self.enabled=False
        self.last_mask=None
    def __call__(self,features,cards,mask,seat):
        self.last_mask=np.asarray(mask,dtype=bool).copy()
        self.enabled=active(self.view);self.reason=None
        if not self.enabled:return self.base(features,cards,mask,seat)
        view={**self.view,'mask':np.asarray(mask,dtype=bool).tolist()}
        action,self.reason=self.operator.act(view)
        if not mask[action]:raise ValueError('economic option violates evaluation guard')
        pi=np.zeros(len(mask),dtype=np.float32);pi[action]=1
        return pi,0.

class EconomicOptionPolicy:
    name='research-public-economic-option'
    supports_pi=False
    def __init__(self,base_evaluator,*,operator_mode='legacy',**guard_args):
        self.evaluator=_Evaluator(base_evaluator,operator_mode)
        self.policy=PPOGreedyPolicy(self.evaluator,**guard_args)
        self.previous=None
    def reset(self):
        self.policy.reset();self.evaluator.operator=Operator(self.evaluator.operator_mode);self.previous=None
    @property
    def forced_actions(self):return self.policy.forced_actions
    def pending_guard_meta(self):return self.policy.pending_guard_meta()
    def decide(self,game,seat):
        full=json.loads(game.view(seat))
        view={k:full[k] for k in ('me','mask','turn','phase','lobby_races')}
        if self.previous is not None:
            before,action=self.previous
            if before['turn']==view['turn']:
                self.evaluator.operator.note(before,view,action)
            else:self.evaluator.operator=Operator(self.evaluator.operator_mode)
        self.evaluator.view=view
        decision=self.policy.decide(game,seat)
        decision.info.update(research_option_active=self.evaluator.enabled,
                             research_operator_reason=self.evaluator.reason,
                             research_operator_mode=self.evaluator.operator_mode,
                             research_policy_intervention=True)
        self.previous=(view,int(decision.action))
        return decision
