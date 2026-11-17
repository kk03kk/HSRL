"""Create a new mechanism-input checkpoint; exact zero-adapter equivalence gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.mechanism_obs import enable_mechanisms
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.runtime.artifacts import atomic_json, file_identity


def upgrade(source, out, device='cpu'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, meta = load_checkpoint(str(source), device=device)
    generator = torch.Generator(device=device).manual_seed(11718)
    f = torch.randn((8, M.LEGACY_FLOATS), generator=generator, device=device)
    c = torch.randint(model.n_cards, (8, 124), generator=generator, device=device)
    extra = torch.randn((8, M.N_EXTRA), generator=generator, device=device)
    with torch.inference_mode():
        before = tuple(x.clone() for x in model(f, c))
        enable_mechanisms(model)
        after = model(torch.cat([f, extra], dim=1), c)
    assert all(torch.equal(a,b) for a,b in zip(before,after)), 'zero adapters changed outputs'
    meta.update(model.arch_flags())
    meta.update(parent_checkpoint=file_identity(source), observation_schema=M.SPEC,
                upgrade='zero-adapter fork; optimizer intentionally not inherited')
    path = out/'model.pt'
    save_checkpoint(model, str(path), meta=meta)
    reloaded, loaded_meta = load_checkpoint(str(path), device=device)
    with torch.inference_mode():
        restored = reloaded(torch.cat([f,extra],dim=1),c)
    assert all(torch.equal(a,b) for a,b in zip(after,restored)), 'checkpoint roundtrip changed outputs'
    report = {'valid':True,'zero_adapter_exact':True,'roundtrip_exact':True,
              'model':file_identity(path),'parent':file_identity(source),
              'schema':M.SPEC,'schema_sha256':M.SCHEMA_SHA256}
    atomic_json(out/'validation.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    upgrade(a.checkpoint,a.out,a.device)
