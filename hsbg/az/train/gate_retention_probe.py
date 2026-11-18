"""Research-only causal constraint: exclude sales that directly break a gold gate.

This changes policy support, not native legality. It is never applied by default
and is not a learned-policy improvement. The network/value remain untouched.
"""
import numpy as np
from hsbg.az.train.conditional_economy import current_gate_spec
from hsbg.az.train.shadow_economy import gold_context
from hsbg.az.train.ppo import PPOGreedyPolicy


def gate_breaking_sales(features,cards,mask):
    if not gold_context(features,cards):return []
    spec=current_gate_spec('gold_engine')
    board=cards[slice(*spec['board_slice'])]
    owned=np.concatenate((board,cards[slice(*spec['hand_slice'])]))
    blocked=[]
    for i,card in enumerate(board):
        if mask[16+i] and any(card in group and np.isin(owned,group).sum()==1 for group in spec['embedding_ids']):
            blocked.append(16+i)
    return blocked


class GateRetentionEvaluator:
    def __init__(self,base):
        self.base=base;self.model=getattr(base,'model',None)
        self.original_action=None;self.blocked=[]
    def __call__(self,features,cards,mask,seat):
        prior,value=self.base(features,cards,mask,seat)
        self.original_action=int(np.argmax(np.where(mask,prior,-np.inf)))
        self.blocked=gate_breaking_sales(features,cards,mask)
        if not self.blocked:return prior,value
        pi=np.asarray(prior).copy();pi[self.blocked]=0
        total=float(pi.sum())
        if not np.isfinite(total) or total<=0:raise ValueError('no finite retained probability for gate constraint')
        return pi/total,value


class GateRetentionPolicy:
    name='research-gold-gate-sale-constraint'
    supports_pi=False
    def __init__(self,base_evaluator,**guard_args):
        self.evaluator=GateRetentionEvaluator(base_evaluator)
        self.policy=PPOGreedyPolicy(self.evaluator,**guard_args)
    def reset(self):self.policy.reset()
    @property
    def forced_actions(self):return self.policy.forced_actions
    def pending_guard_meta(self):return self.policy.pending_guard_meta()
    def decide(self,game,seat):
        decision=self.policy.decide(game,seat)
        decision.info.update(research_gate_retention_constraint=True,
            gate_breaking_sell_actions=self.evaluator.blocked,
            original_greedy_action=self.evaluator.original_action,
            gate_retention_override_executed=bool(not decision.info.get('forced')
                and self.evaluator.original_action in self.evaluator.blocked
                and decision.action!=self.evaluator.original_action))
        return decision
