"""Query a public-view teacher without changing the executing student's policy.

Queries are restricted to the validated gold-engine context. Unsupported
recovery states are explicit abstentions, never fabricated action labels.
"""
import json
import numpy as np
from hsbg.az.train.conditional_economy import current_gate_spec
from hsbg.az.train.ppo import PPOGreedyPolicy
from hsbg.az.train.research_economy import Operator,HEALTH_RISK,cid


def gold_context(features,cards):
    spec=current_gate_spec('gold_engine')
    owned=np.concatenate([cards[slice(*spec[k])] for k in ('board_slice','hand_slice')])
    return bool(features[spec['lobby_feature']]>.5 and
        sum(any(int(c) in group for c in owned) for group in spec['embedding_ids'])>=spec['minimum_distinct_groups'])


class QueryEvaluator:
    def __init__(self,base):
        self.base=base;self.model=getattr(base,'model',None);self.operator=Operator('growth')
        self.view=None;self.last_mask=None;self.target=None;self.reason=None;self.abstention=None;self.active=False
    def __call__(self,features,cards,mask,seat):
        self.last_mask=np.asarray(mask,dtype=bool).copy()
        self.target=None;self.reason=None;self.abstention=None;self.active=gold_context(features,cards)
        if self.active:
            pending=self.view['me'].get('pending') or {}
            if cid(pending.get('source') or {}) in HEALTH_RISK:
                self.abstention='health-risk pending sequence outside teacher domain'
            else:
                view={**self.view,'mask':self.last_mask.tolist()}
                try:self.target,self.reason=self.operator.act(view)
                except ValueError as exc:
                    if str(exc)!='destructive teacher target has no safe continuation':raise
                    self.abstention=str(exc)
                if self.target is not None and not self.last_mask[self.target]:
                    raise ValueError('shadow target violates actual student guard')
        return self.base(features,cards,mask,seat)


class ShadowEconomyPolicy:
    name='student-with-readonly-public-teacher'
    supports_pi=False
    def __init__(self,base_evaluator,**guard_args):
        self.evaluator=QueryEvaluator(base_evaluator)
        self.policy=PPOGreedyPolicy(self.evaluator,**guard_args);self.previous=None
    def reset(self):
        self.policy.reset();self.evaluator.operator=Operator('growth');self.previous=None
    @property
    def forced_actions(self):return self.policy.forced_actions
    def pending_guard_meta(self):return self.policy.pending_guard_meta()
    def decide(self,game,seat):
        full=json.loads(game.view(seat));view={k:full[k] for k in ('me','mask','turn','phase','lobby_races')}
        if self.previous is not None:
            before,action=self.previous
            if before['turn']==view['turn']:self.evaluator.operator.note(before,view,action)
            else:self.evaluator.operator=Operator('growth')
        self.evaluator.view=view;decision=self.policy.decide(game,seat)
        decision.info.update(shadow_teacher_action=self.evaluator.target,shadow_teacher_reason=self.evaluator.reason,
            shadow_teacher_abstention=self.evaluator.abstention,shadow_gold_context=self.evaluator.active,
            shadow_teacher_executes=False)
        self.previous=(view,int(decision.action));return decision


class DeploymentEvaluator(QueryEvaluator):
    """Causal diagnostic: only deploy an already owned Hooktusk into a free slot."""
    def __call__(self,features,cards,mask,seat):
        prior,value=super().__call__(features,cards,mask,seat)
        self.base_action=int(np.argmax(np.where(mask,prior,-np.inf)));self.repaired=False
        if (self.active and self.reason=='restore_core' and self.target is not None
                and 46<=self.target<126
                and cid(self.view['me']['hand'][(self.target-46)//8])=='BG36_344'):
            if 46<=self.base_action<126 and cid(self.view['me']['hand'][(self.base_action-46)//8])=='BG36_344':
                return prior,value
            pi=np.zeros(len(mask),dtype=np.float32);pi[self.target]=1
            self.repaired=True
            return pi,value
        return prior,value


class PayoffDeploymentPolicy(ShadowEconomyPolicy):
    name='research-hooktusk-deployment-intervention'
    def __init__(self,base_evaluator,**guard_args):
        self.evaluator=DeploymentEvaluator(base_evaluator)
        self.policy=PPOGreedyPolicy(self.evaluator,**guard_args);self.previous=None
    def decide(self,game,seat):
        decision=super().decide(game,seat)
        decision.info.update(research_deployment_intervention=True,
            original_greedy_action=self.evaluator.base_action,
            deployment_override_executed=bool(self.evaluator.repaired and decision.action==self.evaluator.target))
        return decision
