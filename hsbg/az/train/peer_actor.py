"""HSAI training interface for PrentendYan's content-verified research checkpoint."""
from pathlib import Path
import sys

import torch
from torch import nn
from hsbg.runtime.references import reference_directory

ARCH="peer_heads_v1"
ORIGINAL_SHA="0454a99dda1d95b03922aa5015c827b1d499c8e952893284a3b3a0e87bf0352b"


class PeerActor(nn.Module):
    uses_cards=True
    arch=ARCH

    def __init__(self,actor,config,temperature):
        super().__init__()
        self.actor=actor
        self.config=config
        self.temperature=float(temperature)
        if not 0<self.temperature<=1:
            raise ValueError("invalid rollout temperature")
        # Preserve the learned representation. Allow all native action families
        # to adapt independently from the new HSAI-utility rank predictor.
        for name,p in self.actor.named_parameters():
            p.requires_grad_(name.startswith(("h_global.","h_buy.","h_sell.","h_activate.",
                "h_choose.","h_play.","q_mag.","h_move.")))
        self.rank_head=nn.Sequential(nn.Linear(69,128),nn.Tanh(),nn.Linear(128,8))

    def checkpoint_metadata(self):
        return dict(arch=ARCH,peer_config=self.config,temperature=self.temperature,origin=ORIGINAL_SHA)

    def forward(self,features,cards,mask):
        logits,v=self.actor(features,cards,mask)
        critic_input=torch.cat((features[:,:68],v.detach().float().unsqueeze(1)),dim=1)
        scaled=(logits.float().masked_fill(~mask,0.)/self.temperature).masked_fill(~mask,-1e9)
        return scaled,self.rank_head(critic_input)


def load_payload(payload,device):
    meta=dict(payload["meta"])
    if meta["origin"]!=ORIGINAL_SHA:
        raise ValueError("unexpected external origin")
    path=str(reference_directory("hsbg-ppo"))
    if path not in sys.path:
        sys.path.insert(0,path)
    import hsbg_net
    model=PeerActor(hsbg_net.build(meta["peer_config"]),meta["peer_config"],meta["temperature"])
    model.load_state_dict(payload["state_dict"],strict=True)
    return model.to(device).eval(),meta
