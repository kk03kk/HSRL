"""A frozen entity policy with a learned public, card-role action adjustment."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hsbg.env import obs_std as O

ARCH = 'quilboar_choice_v1'
TERMS = ('start_supply','start_rider','complete_rider','complete_supply','play_supply','play_rider',
         'retain_supply','retain_rider','empty_slot','use_choice','buy_choice','play_recipient',
         'buy_recipient','cycle_slot','defer_choice','retain_recipient','gem_growth','seek','extra_copy')


@lru_cache(None)
def roles():
    import hsbg_engine as engine
    path = Path(__file__).resolve().parents[3]/'engine/crates/hsbg-engine/data/cards.json'
    cards = json.loads(path.read_text(encoding='utf8'))['cards']
    table = np.zeros((len(cards)+1, 6),dtype=bool)
    for i, card in enumerate(cards,1):
        if engine.card_info(i)[0] != card['id']:
            raise ValueError('card order differs from native vocabulary')
        key = card['normal'] or card['id']
        table[i] = (key=='BG31_323', key in ('BG31_326','BG36_331'),
                    'CHOOSE_ONE' in card['keywords'], 'QUILBOAR' in card['races'] or 'ALL' in card['races'],
                    key=='BG33_885', card['kind']=='minion')
    return table, hashlib.sha256(path.read_bytes()).hexdigest()


def basis(f,c):
    if f[O.GLOBAL_LOBBY_RACES_START+7]<.5:
        return None
    role=roles()[0]
    b,h,s,d=(role[c[sl]] for sl in (O.SLICE_CARD_BOARD,O.SLICE_CARD_HAND,O.SLICE_CARD_SHOP,O.SLICE_CARD_CHOICES))
    if not any(z[:,:2].any() for z in (b,h,s,d)):
        return None
    rider,supply=b[:,:2].sum(0)
    hr,hs=h[:,:2].sum(0)
    owned_r,owned_s=rider+hr,supply+hs
    boars=int(b[:,3].sum())
    turn=round(float(f[0])*20)
    if turn<3:
        return None
    hp=f[5]*40+f[6]*20
    safe=float(np.clip((hp-5)/20,.15,1))
    gems=max(0,float(f[31]+f[32])*5)
    ready=min(1,.35+.17*boars+.06*gems)
    boardlen=int(np.count_nonzero(c[O.SLICE_CARD_BOARD]))
    x=np.zeros((245,len(TERMS)),np.float32)
    def take(dst, z):
        dst[:,0]=z[:,1]*(owned_s==0)*(owned_r==0)*safe
        dst[:,1]=z[:,0]*(owned_r==0)*(owned_s==0)*safe*ready
        dst[:,2]=z[:,0]*(owned_s>0)*(owned_r<2)
        dst[:,3]=z[:,1]*(owned_r>0)*(owned_s<2)
        dst[:,18]=(z[:,0]*(owned_r==1)+z[:,1]*(owned_s==1))*ready
    take(x[8:16],s)
    if f[O.SLICE_PENDING][2]>.5:
        take(x[30:46],d)
    play=x[46:126].reshape(10,8,-1)
    play[:,:,4]=h[:,1,None]*(supply<2)
    play[:,:,5]=h[:,0,None]*(rider<2)
    sell=x[16:23]
    sell[:,6]=-b[:,1].astype(float)*(1 if supply<=1 else .45)
    sell[:,7]=-b[:,0].astype(float)*(1 if rider<=1 else .55)
    stats=f[O.SLICE_BOARD].reshape(7,O.MINION_F)
    size=np.maximum(0,stats[:,1]+stats[:,2])
    cheap=(1-size/max(.001,float(max(size))))**2
    spare=(~b[:,:2].any(1))*cheap
    if boardlen==7 and (hr or hs) and (hr and not rider or hs and not supply):
        sell[:,8]=spare*(1-.45*b[:,3])
    if rider:
        play[:,:,9]=h[:,2,None]
        x[8:16,10]=s[:,2]*(np.count_nonzero(c[O.SLICE_CARD_HAND])<8)
        if boars<5:
            play[:,:,11]=h[:,3,None]*(~h[:,:2].any(1)[:,None])
            x[8:16,12]=s[:,3]*(~s[:,:2].any(1))*(boars<4)
        if boardlen==7 and (h[:,2]&h[:,5]).any():
            sell[:,13]=spare*(1-.75*b[:,3])
        sell[:,15]=-b[:,3].astype(float)*np.minimum(1,size/max(.001,float(max(size))))
    elif hr:
        play[:,:,14]=-h[:,2,None].astype(float)
    if owned_r or owned_s:
        x[8:16,16]=s[:,4]*safe
        play[:,:,16]=h[:,4,None]
        if owned_r==0 and owned_s and f[10]>.95 and f[7]>=.4 and not s[:,0].any():
            x[1,17]=ready*safe
    return x


class QuilboarOption(nn.Module):
    uses_cards=True
    arch=ARCH
    mechanism_schema=''
    input_floats=3227

    def __init__(self,base,meta,weights):
        super().__init__()
        if meta['arch'] not in ('entity','entity18m') or meta.get('mechanism_schema') or any(
                meta.get(k) is not None for k in ('norm_mean','norm_std')):
            raise ValueError('option requires original unnormalized entity inputs')
        self.base=base
        self.component_meta=meta
        self.weights=nn.Parameter(torch.as_tensor(weights,dtype=torch.float32).reshape(len(TERMS)).clone())

    def forward(self,features,cards,mask=None):
        logits,rank,value=self.base(features,cards,mask)
        adjustments=[]
        for i,(f,c) in enumerate(zip(features.detach().cpu().numpy(),cards.detach().cpu().numpy())):
            features_i=basis(f,c)
            if features_i is not None:
                adjustments.append((i,features_i))
        if adjustments:
            ix,values=zip(*adjustments)
            delta=torch.as_tensor(np.stack(values),device=logits.device)@self.weights
            logits=logits.index_add(0,torch.tensor(ix,device=logits.device),delta.to(logits.dtype))
        return logits,rank,value

    def checkpoint_metadata(self):
        return dict(arch=ARCH,component_meta=self.component_meta,terms=list(TERMS),card_sha=roles()[1])


def load_payload(payload,device,load_component):
    meta=payload['meta']
    if meta['terms']!=list(TERMS) or meta['card_sha']!=roles()[1]:
        raise ValueError('option vocabulary or term layout mismatch')
    state=payload['state_dict']
    base,_=load_component(dict(meta=meta['component_meta'],state_dict={
        k[5:]:v for k,v in state.items() if k.startswith('base.')}),device)
    model=QuilboarOption(base,meta['component_meta'],state['weights'].detach().cpu())
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),dict(meta)
