"""HSAI training interface for PrentendYan's content-verified research checkpoint."""
from pathlib import Path
import sys

import torch
from torch import nn
from hsbg.env.obs_std import GLOBAL_FIELD_INDEX, GLOBAL_FIELD_SCALE, MINION_F, N_BOARD, SLICE_BOARD
from hsbg.env.action_space_std import A_SELL
from hsbg.runtime.references import reference_directory

ARCH="peer_heads_v1"
ORIGINAL_SHA="0454a99dda1d95b03922aa5015c827b1d499c8e952893284a3b3a0e87bf0352b"


def retain_material_mask(features, mask):
    """Calculate the D-161 policy's eligible actions from native observations."""
    def global_value(name):
        return (features[:, GLOBAL_FIELD_INDEX[name]].float() * GLOBAL_FIELD_SCALE[name]).round()
    board = features[:, SLICE_BOARD].reshape(-1, N_BOARD, MINION_F)
    stats = (board[:, :, 1:3].float() * 10).round().to(torch.int64).sum(dim=2)
    active = ((global_value("turn") >= 7) & (global_value("health") + global_value("armor") <= 15)
              & (global_value("board_len") <= 4))
    protect = active[:, None] & (board[:, :, 0] > .5) & (stats * 100 >= 15 * stats.sum(1, keepdim=True).clamp(min=1))
    eligible = mask.clone()
    eligible[:, A_SELL:A_SELL+N_BOARD] &= ~protect
    return torch.where(eligible.any(dim=1, keepdim=True), eligible, mask)


class PeerActor(nn.Module):
    uses_cards=True
    arch=ARCH

    def __init__(self,actor,config,temperature,*,adapt_last_block=False,retain_material=False):
        super().__init__()
        self.actor=actor
        self.config=config
        self.temperature=float(temperature)
        if type(adapt_last_block) is not bool:
            raise ValueError("adapt_last_block must be boolean")
        self.adapt_last_block = adapt_last_block
        if type(retain_material) is not bool:
            raise ValueError("retain_material must be boolean")
        self.retain_material = retain_material
        if not 0<self.temperature<=1:
            raise ValueError("invalid rollout temperature")
        # Preserve the learned representation. Allow all native action families
        # to adapt independently from the new HSAI-utility rank predictor.
        for name,p in self.actor.named_parameters():
            p.requires_grad_(name.startswith(("h_global.","h_buy.","h_sell.","h_activate.",
                "h_choose.","h_play.","q_mag.","h_move.")))
        if self.adapt_last_block:
            self.actor.encoder.layers[-1].requires_grad_(True)
        self.rank_head=nn.Sequential(nn.Linear(69,128),nn.Tanh(),nn.Linear(128,8))

    def checkpoint_metadata(self):
        meta = dict(arch=ARCH,peer_config=self.config,temperature=self.temperature,origin=ORIGINAL_SHA)
        if self.adapt_last_block:
            meta["adapt_last_block"] = True
        if self.retain_material:
            meta["sale_policy"] = "retain-material-v1"
        return meta

    def forward(self,features,cards,mask):
        if self.retain_material:
            mask = retain_material_mask(features, mask)
        logits,v=self.actor(features,cards,mask)
        critic_input=torch.cat((features[:,:68],v.detach().float().unsqueeze(1)),dim=1)
        scaled=(logits.float().masked_fill(~mask,0.)/self.temperature).masked_fill(~mask,-1e9)
        return scaled,self.rank_head(critic_input)


def load_payload(payload,device):
    meta=dict(payload["meta"])
    if meta["origin"]!=ORIGINAL_SHA:
        raise ValueError("unexpected external origin")
    sale_policy = meta.get("sale_policy")
    if sale_policy not in (None, "retain-material-v1"):
        raise ValueError("unknown peer sale policy")
    if sale_policy and any(k in payload or k in meta for k in ("norm_mean", "norm_std")):
        raise ValueError("material retention requires native feature scaling")
    path=str(reference_directory("hsbg-ppo"))
    if path not in sys.path:
        sys.path.insert(0,path)
    import hsbg_net
    model=PeerActor(hsbg_net.build(meta["peer_config"]),meta["peer_config"],meta["temperature"],
                    adapt_last_block=meta.get("adapt_last_block", False), retain_material=bool(sale_policy))
    model.load_state_dict(payload["state_dict"],strict=True)
    return model.to(device).eval(),meta
