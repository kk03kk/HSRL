"""Public-state conditional specialist with an unchanged fallback policy.

The gate is a declared architectural prior, not a learned optimal strategy:
Pirates must be in the lobby and at least two distinct members of Brann,
Rogers and Privateer must be owned on board or in hand. Shop offers never
activate it. No game handle, hidden state, future draw or teacher memory enters.
"""
from __future__ import annotations
import copy,hashlib,json
from functools import lru_cache
import torch
from torch import nn
from hsbg.env import obs_std as O
from hsbg.az.train import mechanism_obs as M

ARCH='conditional_economy_v1'
CORE_GROUPS=(('BG_LOE_077','TB_BaconUps_045'),('BG33_823','BG33_823_G'),
             ('BG33_825','BG33_825_G'))


@lru_cache(maxsize=2)
def current_gate_spec(mode='two_core'):
    import hsbg_engine as E
    ncols,raw=E.card_static();n=len(raw)//(4*ncols)
    vocabulary=['']+[E.card_info(i)[0] for i in range(1,n)]
    indices={cid:i for i,cid in enumerate(vocabulary)}
    if mode not in ('two_core','gold_engine'):raise ValueError('unknown conditional gate mode')
    groups=CORE_GROUPS if mode=='two_core' else (
        ('TB_BaconUps_045','TB_BaconUps_045'),('BG33_823','BG33_823_G'),
        ('BG33_825_G','BG33_825_G'),('BG36_523','BG36_523_G'))
    spec={'schema':'hsai.conditional-economy-gate/v1',
        'groups':[list(g) for g in groups],
        'embedding_ids':[[indices[c] for c in g] for g in groups],
        'minimum_distinct_groups':2,'lobby_feature':O.GLOBAL_LOBBY_RACES_START+6,
        'board_slice':[O.SLICE_CARD_BOARD.start,O.SLICE_CARD_BOARD.stop],
        'hand_slice':[O.SLICE_CARD_HAND.start,O.SLICE_CARD_HAND.stop],
        'vocabulary_sha256':hashlib.sha256(json.dumps(vocabulary,separators=(',',':')).encode()).hexdigest()}
    if mode=='gold_engine':
        spec.update(schema='hsai.conditional-economy-gate/v2',mode=mode,minimum_distinct_groups=4)
    return spec


class ConditionalEconomy(nn.Module):
    uses_cards=True
    arch=ARCH
    input_floats=3227
    mechanism_schema=''
    def __init__(self,base,specialist,component_meta,gate_spec=None):
        super().__init__()
        if component_meta.get('arch') not in ('entity','entity18m'):
            raise ValueError('conditional specialist requires an entity component')
        schema=str(component_meta.get('mechanism_schema','') or '')
        if schema not in ('',M.SCHEMA) or (schema and component_meta.get('mechanism_schema_sha256')!=M.SCHEMA_SHA256):
            raise ValueError('conditional component observation schema identity differs')
        if any(component_meta.get(k) is not None for k in ('norm_mean','norm_std')):
            raise ValueError('conditional gate requires an unnormalized legacy prefix')
        self.mechanism_schema=schema;self.input_floats=M.OBS_FLOATS if schema else M.LEGACY_FLOATS
        if any(M.schema_of(c)!=schema or getattr(c,'input_floats',3227)!=self.input_floats for c in (base,specialist)):
            raise ValueError('conditional components disagree with declared observation schema')
        spec=copy.deepcopy(current_gate_spec() if gate_spec is None else gate_spec)
        if spec!=current_gate_spec(spec.get('mode','two_core')):raise ValueError('conditional gate or vocabulary identity differs')
        self.base=base;self.specialist=specialist
        self.component_meta=copy.deepcopy(component_meta);self.gate_spec=spec
        self.register_buffer('_gate_ids',torch.tensor(spec['embedding_ids'],dtype=torch.long),persistent=False)

    def gate(self,features,cards):
        if features.shape[-1]!=self.input_floats or cards.shape[-1]!=124:
            raise ValueError('conditional observation layout differs')
        owned=torch.cat((cards[:,O.SLICE_CARD_BOARD],cards[:,O.SLICE_CARD_HAND]),dim=1)
        distinct=(owned[:,:,None,None]==self._gate_ids[None,None,:,:]).any(1).any(-1).sum(-1)
        return (features[:,self.gate_spec['lobby_feature']]>.5)&(distinct>=self.gate_spec['minimum_distinct_groups'])

    def forward(self,features,cards,mask=None):
        active=self.gate(features,cards)
        base=self.base(features,cards,mask)
        if not bool(active.any()):return base
        # Same batch shape preserves the fallback's exact floating-point path.
        # The base remains the shared rank/value critic, including inside the
        # option. BC must not silently substitute an uncalibrated expert critic.
        expert=self.specialist(features,cards,mask)
        policy=torch.where(active[:,None],expert[0],base[0])
        return (policy,*base[1:])

    def checkpoint_metadata(self):
        return {'arch':ARCH,'component_meta':copy.deepcopy(self.component_meta),
                'conditional_gate':copy.deepcopy(self.gate_spec),
                'input_floats':self.input_floats,'mechanism_schema':self.mechanism_schema,
                'mechanism_schema_sha256':M.SCHEMA_SHA256 if self.mechanism_schema else ''}


def fork_conditional(base,meta,*,mechanisms=False,gate_mode='two_core'):
    if mechanisms:
        base=M.enable_mechanisms(copy.deepcopy(base));meta=copy.deepcopy(meta)
        meta.update(base.arch_flags())
    return ConditionalEconomy(base,copy.deepcopy(base),meta,current_gate_spec(gate_mode)).to(next(base.parameters()).device)


def load_payload(payload,device,load_component):
    meta=dict(payload['meta']);state=payload['state_dict']
    if any(payload.get(k) is not None or meta.get(k) is not None for k in ('norm_mean','norm_std')):
        raise ValueError('conditional gate does not support external normalization')
    if 'component_meta' not in meta or 'conditional_gate' not in meta:
        raise ValueError('conditional checkpoint missing explicit identities')
    if meta['component_meta'].get('arch') not in ('entity','entity18m'):
        raise ValueError('conditional component architecture differs')
    schema=str(meta['component_meta'].get('mechanism_schema','') or '')
    if str(meta.get('mechanism_schema','') or '')!=schema or (schema and meta.get('mechanism_schema_sha256')!=M.SCHEMA_SHA256):
        raise ValueError('conditional wrapper observation schema identity differs')
    width=M.OBS_FLOATS if schema else M.LEGACY_FLOATS
    if int(meta.get('input_floats',width))!=width:raise ValueError('conditional wrapper observation width differs')
    if any(not (k.startswith('base.') or k.startswith('specialist.')) for k in state):
        raise ValueError('unexpected conditional state key')
    components=[]
    for prefix in ('base.','specialist.'):
        part={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        component,_=load_component({'state_dict':part,'meta':meta['component_meta']},device)
        components.append(component)
    model=ConditionalEconomy(*components,meta['component_meta'],meta['conditional_gate'])
    model.load_state_dict(state,strict=True);model.to(device).eval()
    return model,meta
