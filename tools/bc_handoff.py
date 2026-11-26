"""Verify PPO-compatible sampling from a frozen, independently evaluated BC checkpoint."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from taverngym import N_ACTIONS, OBS_CARDS, OBS_FLOATS, build_info
from taverngym._core import BcArena
from taverngym.bc_train import json_file, value_target
from taverngym.entity_policy import EntityPolicy

run=Path(sys.argv[1])
report=json.loads((run/'evaluation/report.json').read_text())
ckpt=run/'selected.pt'
with ckpt.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==report['checkpoint_sha256']
torch.set_num_threads(2)
torch.manual_seed(19377)
torch.backends.mha.set_fastpath_enabled(False)
saved=torch.load(ckpt,map_location='cpu',weights_only=False)
assert saved['meta']['source']==report['source']==build_info()['source_hash']
utility=torch.tensor(saved['meta']['placement_utility'])
assert torch.equal(value_target(torch.arange(8),utility),torch.tensor([1.,.3,0,-.05,-.1,-.15,-.2,-.25]))
model=EntityPolicy(**saved['meta']['model'])
model.load_state_dict(saved['model'])
n=sum(p.numel() for p in model.parameters())
assert n==18147729
model.eval()
arena=BcArena(list(range(9480000, 9480016)),True,2)
def inputs():
    indices,f,c,m=arena.observe()
    n=len(indices)//4
    return [torch.from_numpy(np.frombuffer(x,dtype=d).reshape(n,w).copy()) for x,d,w in
            ((f,'<f4',OBS_FLOATS),(c,'<i4',OBS_CARDS),(m,'u1',N_ACTIONS))]
samples=0
with torch.no_grad():
    for step in range(12):
        batch=inputs()
        result=model(*batch)
        distribution=torch.distributions.Categorical(logits=result.logits)
        a=distribution.sample()
        assert batch[-1][torch.arange(len(a)),a].all() and torch.isfinite(result.value).all()
        assert torch.isfinite(distribution.log_prob(a)).all()
        samples+=len(a)
        arena.step(a.tolist())
    batch=inputs()
    old=torch.distributions.Categorical(logits=model(*batch).logits)
    a=old.sample()
    old_lp=old.log_prob(a)
out=model(*batch)
new_lp=torch.distributions.Categorical(logits=out.logits).log_prob(a)
assert torch.allclose((new_lp-old_lp).exp(),torch.ones_like(old_lp))
(-new_lp.mean()+.05*out.value.square().mean()).backward()
grads=[p.grad for p in model.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
check={'checkpoint_sha256':report['checkpoint_sha256'],'parameters':n,'model':saved['meta']['model'],'valid_samples':samples,
       'finite_log_probabilities_value_and_gradients':True,'same_weights_ratio':True,'optimizer_updates':0}
json_file(run/'selected-check.json',check)
print(check)
