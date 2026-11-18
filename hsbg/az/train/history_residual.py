"""Research composite: frozen PPO plus a small public-history logit residual.

The residual asset is deliberately NOT a standalone PPO checkpoint. The base
identity, vocabulary, input layout and public-history schema must all match.
"""
import json
import numpy as np
import torch
from torch import nn
from hsbg.env import obs_std as O
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.public_turn_history import PublicTurnHistory,SCHEMA_SHA256 as HISTORY_SHA,project_own_record
from hsbg.az.train.conditional_economy import current_gate_spec
from hsbg.az.train.shadow_economy import gold_context
from hsbg.az.train.ppo import PPOGreedyPolicy,masked_probs_np,value_from_rank_logits_np
from hsbg.runtime.artifacts import file_identity

SCHEMA='hsai.history-logit-residual/v1'


def current_spec(use_history):
    import hsbg_engine as E
    ncols,raw=E.card_static()
    return {'schema':SCHEMA,'use_history':bool(use_history),'history_schema_sha256':HISTORY_SHA,
        'mechanism_schema_sha256':M.SCHEMA_SHA256,
        'vocabulary_sha256':current_gate_spec('gold_engine')['vocabulary_sha256'],
        'vocabulary_size':len(raw)//(4*ncols),'own_prefix_stop':O.SLICE_SHOP.stop,
        'card_embedding':16,'hidden':256,'residual_bound':6.0,'history_width':7,
        'gate':current_gate_spec('gold_engine')}


class HistoryResidual(nn.Module):
    def __init__(self,spec):
        super().__init__()
        if spec!=current_spec(spec.get('use_history',False)):raise ValueError('residual schema or vocabulary differs')
        self.spec=dict(spec)
        self.embedding=nn.Embedding(spec['vocabulary_size'],16,padding_idx=0)
        width=spec['own_prefix_stop']+M.N_EXTRA+25*16+245+7
        self.hidden=nn.Linear(width,256);self.output=nn.Linear(256,245)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,features,cards,base_logits,history):
        if features.shape[1]!=M.OBS_FLOATS or cards.shape[1]!=124 or history.shape!=(len(features),7):
            raise ValueError('residual input layout differs')
        own=torch.cat((features[:,:self.spec['own_prefix_stop']],features[:,M.LEGACY_FLOATS:]),dim=-1)
        own=own.sign()*own.abs().log1p()
        tokens=torch.cat((cards[:,O.SLICE_CARD_BOARD],cards[:,O.SLICE_CARD_HAND],cards[:,O.SLICE_CARD_SHOP]),dim=-1)
        embedded=self.embedding(tokens).flatten(1)
        prior=base_logits.sign()*base_logits.abs().log1p()
        history=history.float() if self.spec['use_history'] else torch.zeros_like(history,dtype=torch.float32)
        x=torch.cat((own,embedded,prior,history),dim=-1)
        delta=self.output(torch.nn.functional.silu(self.hidden(x)))
        return 6.0*torch.tanh(delta/6.0)


def save_adapter(path,model,base,run):
    torch.save({'schema':SCHEMA,'spec':model.spec,'base':base,'run':run,'state_dict':model.state_dict()},path)


def load_adapter(path,base,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!=SCHEMA or payload.get('base')!=base:
        raise ValueError('residual asset or base identity differs')
    model=HistoryResidual(payload['spec']);model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device).eval()


def restore_bc_state(adapter_path,state_path,base,device='cpu'):
    """Restore this residual's Adam and sampling RNG; not a PPO resume API."""
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    if (state.get('schema')!='hsai.history-residual-bc-state/v1'
            or state.get('model')!=file_identity(adapter_path)
            or type(state.get('updates')) is not int or state['updates']<0):
        raise ValueError('residual BC recovery identity differs')
    model=load_adapter(adapter_path,base,device)
    optimizer=torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(state['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    return model,optimizer,rng,state['updates']


class HistoryResidualEvaluator:
    def __init__(self,base,adapter):
        if M.schema_of(base.model)!=M.SCHEMA:raise ValueError('history residual requires the declared mechanism observation base')
        self.base=base;self.model=base.model;self.adapter=adapter
        self.history=np.zeros(7,dtype=np.uint8);self.active=False
    def __call__(self,features,cards,mask,seat):
        self.active=gold_context(features,cards)
        if not self.active:return self.base(features,cards,mask,seat)
        plog,rlog=self.base.logits(features,cards,mask)
        device=next(self.adapter.parameters()).device
        with torch.no_grad():
            delta=self.adapter(torch.as_tensor(np.asarray(features)[None],dtype=torch.float32,device=device),
                torch.as_tensor(np.asarray(cards)[None],dtype=torch.long,device=device),
                torch.as_tensor(plog,dtype=torch.float32,device=device),
                torch.as_tensor(self.history[None],device=device)).cpu().numpy()
        corrected=plog[0]+delta[0]
        if not np.isfinite(corrected).all() or not np.isfinite(rlog).all():raise ValueError('nonfinite residual policy')
        prior=masked_probs_np(corrected,np.asarray(mask,bool),self.base.temperature)
        prior=prior/max(float(prior.sum()),1e-12)
        return prior.astype(np.float32),float(value_from_rank_logits_np(rlog[0])[0])


class HistoryResidualPolicy:
    name='research-learned-history-residual'
    supports_pi=False
    def __init__(self,base_evaluator,adapter,**guard_args):
        self.evaluator=HistoryResidualEvaluator(base_evaluator,adapter)
        self.policy=PPOGreedyPolicy(self.evaluator,**guard_args);self.history=PublicTurnHistory();self.awaiting=False
    def reset(self):self.policy.reset();self.history.reset();self.awaiting=False
    @property
    def forced_actions(self):return self.policy.forced_actions
    def pending_guard_meta(self):return self.policy.pending_guard_meta()
    def decide(self,game,seat):
        if self.awaiting:raise ValueError('history policy requires actual execution feedback before another decision')
        view=json.loads(game.view(seat))
        self.before_view={'turn':view['turn'],'me':{'board':[{'eid':m['eid']} for m in view['me']['board']],
            'hand':[{'type':h['type']} for h in view['me']['hand']]}}
        marks,count=self.history.before(self.before_view);self.evaluator.history=marks
        decision=self.policy.decide(game,seat);self.awaiting=True
        decision.info.update(public_play_history=marks.tolist(),public_history_actions=count,
            history_residual_uses_marks=self.evaluator.adapter.spec['use_history'],
            history_residual_active=self.evaluator.active,history_schema_sha256=HISTORY_SHA)
        return decision
    def record_actual(self,before_state,after_state,seat,action):
        before=project_own_record(before_state,seat);after=project_own_record(after_state,seat)
        if not self.awaiting or before!=self.before_view:raise ValueError('history execution feedback does not match decision')
        self.history.after(before,after,action);self.awaiting=False
