"""Explicitly pinned compatible policies; masked greedy logits, no fallback."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from hsbg.az.train.model import load_checkpoint
from hsbg.az.train.ppo import forward_model, split_model_outputs
from hsrl_agent.compat import verify
from .contracts import Blocked,Decision,state_stamp
from .encoding import encode

CHECKPOINT_SHA256='116eb3df777ac0d0b47921372c4a41cd3ad5bf5fecac36af27354dc227a92dbb'
CHAMPION_SHA256='0efb6e356300e4e1a23e0f385e048315d21e6fa2b999b77466554c4a84124329'
POLICY_SPECS={CHECKPOINT_SHA256:dict(id='original-14.8m',auxiliary=False),
              CHAMPION_SHA256:dict(id='champion-c0-1975',auxiliary=True),
              'fffc395c4a24207acd2d297f0b4ee25e7aabde66e7249dd7a71cd388b58f500c':dict(id='s14-best-offline',auxiliary=False,parameters=16436034),
              'eb91fd41fa9957bd259d6f0fba7cb5eecdddc91dd85a8f212d2dafaa79ad3f66':dict(id='nus-1080772-step-006039',auxiliary=False,parameters=16436034,required_client_build=256589),
              # V1.1 = m1095539-v9691 final (MODEL_CARD/MODEL v1.1). The checkpoint carries
              # the BC card head and the sequence head, so it must load through
              # load_checkpoint (strict, meta-driven), never load_champion_policy.
              # trained_engine_version is the simulator rules it learned (37.0.2.256589);
              # required_client_build is the real client the bridge is calibrated for
              # (255790: calibrated executable sha256 + Power.log BuildNumber), not the
              # training engine version. Conflating the two made the gate unsatisfiable.
              '68a8a58743617bef004fef556bd263a4f4939e22b5f5f70352dfa697abc0848e':dict(
                  id='hsai-v1.1',auxiliary=False,parameters=16436034,
                  required_client_build=255790,trained_engine_version='37.0.2.256589'),
              # league-latest-12311 = m1102669-v12311 (36h league step_012311, 1,175,791,607 decisions).
              # Same entity architecture as V1.1 (16,436,034 params, no auxiliary head in the
              # inference path); selected by the server for live testing, V1.1 kept as fallback.
              '5069aa62b8d20c57f2332fb284f5f57b639501dbd9721c3e2fefed41218d29cd':dict(
                  id='league-latest-12311',auxiliary=False,parameters=16436034,
                  required_client_build=255790,trained_engine_version='37.0.2.256589'),
              # league-final-14858 = m1102669-v14858 (36h league final, 1,552,463,499 decisions).
              # E1 four-model calibration champion by point estimate (1086.53, rank CI [1,2]);
              # statistically tied with 13806, so it is a point-estimate winner, not a proven
              # best.  Structurally identical to 12311 (137 tensors, same shapes/dtypes, entity
              # arch, d_model=384/8 layers/8 heads), so it is a drop-in inference checkpoint.
              'ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7':dict(
                  id='league-final-14858',auxiliary=False,parameters=16436034,
                  required_client_build=255790,trained_engine_version='37.0.2.256589')}


def _vector_meta(array):
    """输入向量摘要（形状/范数/指纹），不复制数据本体。"""
    import numpy as _np
    flat=_np.ascontiguousarray(array).reshape(-1)
    return dict(sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
        shape=[int(v) for v in _np.shape(array)],dtype=str(flat.dtype),
        nonzero=int(_np.count_nonzero(flat)),abs_sum=round(float(_np.abs(flat).sum()),6))


def configure_threads(count):
    """Bound CPU inference threads; importable after focus activation.

    Importing torch costs several seconds on this machine. The startup path
    activates the verified window before touching torch so the operator sees
    the switch immediately instead of a silent import pause.
    """
    if count is None:return
    if type(count) is not int or not 1<=count<=32:raise Blocked('inference threads must be 1..32')
    torch.set_num_threads(count)

def load_champion_policy(checkpoint,device):
    """Omit exactly the verified training-only head, strict-load every policy tensor.

    The source checkpoint stays immutable. This is inference, not optimizer or
    trainer resume, and does not change the frozen training model implementation.
    """
    from hsbg.az.train.model import build_net
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=dict(payload['meta'])
    if (meta.get('arch'),meta.get('d_model'),meta.get('n_layers'),meta.get('n_heads'),meta.get('seq_head'))!=('entity',384,8,8,True):
        raise Blocked('champion architecture metadata differs from validated contract')
    model=build_net(arch='entity',d_model=384,n_layers=8,n_heads=8)
    state=payload['state_dict'];expected=set(model.state_dict());aux={'seq_head.weight','seq_head.bias'}
    if set(state)-expected!=aux or expected-set(state):raise Blocked('champion has unvalidated missing/extra tensors')
    if tuple(state['seq_head.weight'].shape)!=(245,384) or tuple(state['seq_head.bias'].shape)!=(245,):
        raise Blocked('champion auxiliary head shape differs')
    model.load_state_dict({key:state[key] for key in expected},strict=True)
    meta['inference_omitted_auxiliary']=sorted(aux)
    for key in ('norm_mean','norm_std'):
        if key in payload:meta[key]=np.asarray(payload[key],np.float32)
    model.to(torch.device(device));model.eval()
    return model,meta

class ModelPolicy:
    def __init__(self,checkpoint,device='cpu'):
        checkpoint=Path(checkpoint)
        self.sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        spec=POLICY_SPECS.get(self.sha)
        if spec is None:raise Blocked('checkpoint is not an approved pinned policy')
        self.policy_id=spec['id']
        if not verify()['ok']: raise Blocked('observation/card/action contract failed')
        self.device=device
        self.model,self.meta=(load_champion_policy(checkpoint,device) if spec['auxiliary'] else load_checkpoint(str(checkpoint),device=device))
        if sum(p.numel() for p in self.model.parameters())!=spec.get('parameters',14808254): raise Blocked('wrong model architecture')
        self.model.eval()
        for key in ('norm_mean','norm_std'):
            if self.meta.get(key) is not None:
                value=np.asarray(self.meta[key],np.float32)
                if value.shape!=(3227,) or not np.isfinite(value).all() or key=='norm_std' and np.any(value<=0):
                    raise Blocked('invalid checkpoint normalization:'+key)
        for p in self.model.parameters(): p.requires_grad_(False)
        import hsbg_engine as t
        self.vocabulary={t.card_info(i)[0]:i for i in range(1,t.N_CARDS)}

    @torch.inference_mode()
    def decide(self,snapshot,bindings):
        if not bindings: raise Blocked('no representable authorized action')
        f,c=encode(snapshot['view'],self.vocabulary)
        mask=np.zeros(245,bool)
        by_action={}
        for b in bindings:
            if b.action in by_action and by_action[b.action]!=b: raise Blocked('ambiguous action binding')
            by_action[b.action]=b;mask[b.action]=True
        if self.meta.get('norm_mean') is not None: f=f-np.asarray(self.meta['norm_mean'],np.float32)
        if self.meta.get('norm_std') is not None: f=f/np.asarray(self.meta['norm_std'],np.float32)
        logits,_,_=split_model_outputs(forward_model(self.model,
            torch.as_tensor(f[None],device=self.device),torch.as_tensor(c[None].astype(np.int64),device=self.device),
            torch.as_tensor(mask[None],device=self.device)))
        if not torch.isfinite(logits).all(): raise Blocked('nonfinite model logits')
        raw=logits[0].float().cpu().numpy()
        action=int(np.argmax(np.where(mask,raw,-np.inf)))
        # 输入留痕：只保存引用，不复制、不影响决策。trace 层负责落 npy，
        # 让 sim×live 能逐位比较"模型真正看到的向量"，而不是事后重建。
        self.last_inputs=dict(features=f,cards=c.astype(np.int32),mask=mask,
            features_meta=_vector_meta(f),cards_meta=_vector_meta(c))
        return Decision(snapshot['game_id'],snapshot['revision'],state_stamp(snapshot),action,
            by_action[action],self.sha,hashlib.sha256(raw.tobytes()).hexdigest(),
            hashlib.sha256(f.tobytes()+c.tobytes()).hexdigest(),hashlib.sha256(mask.tobytes()).hexdigest())
