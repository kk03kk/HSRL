"""Fork an entity checkpoint with an exactly neutral, independent Activate query."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.runtime.artifacts import atomic_json, file_identity


def enable_activation_head(model):
    if getattr(model, 'arch', '') not in ('entity', 'entity18m'):
        raise ValueError('entity model required')
    if model.activate_head_split:
        raise ValueError('activation head already split')
    layer = torch.nn.Linear(model.d_model, model.d_model, bias=False)
    layer.to(next(model.parameters()).device)
    torch.nn.init.zeros_(layer.weight)
    model.q_activate_delta = layer
    model.activate_head_split = True
    return model


def upgrade(source, out):
    torch.set_num_threads(1)
    torch.manual_seed(1182301)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, meta = load_checkpoint(source)
    f = torch.randn(16, model.input_floats)
    c = torch.randint(model.n_cards, (16, 124))
    with torch.no_grad():
        before = tuple(x.clone() for x in model(f, c))
        enable_activation_head(model)
        after = model(f, c)
    if not all(torch.equal(a, b) for a, b in zip(before, after)):
        raise RuntimeError('activation upgrade changed the initial policy')
    meta.update(parent_checkpoint=file_identity(source),
                upgrade='zero activation query; fresh optimizer required')
    save_checkpoint(model, str(out / 'model.pt'), meta=meta)
    restored, _ = load_checkpoint(str(out / 'model.pt'))
    with torch.no_grad():
        final = restored(f, c)
    if not all(torch.equal(a, b) for a, b in zip(after, final)):
        raise RuntimeError('activation checkpoint roundtrip differs')
    atomic_json(out / 'validation.json', {
        'valid': True, 'initial_policy_exact': True, 'roundtrip_exact': True,
        'parent': file_identity(source), 'model': file_identity(out / 'model.pt')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    upgrade(args.checkpoint, args.out)
