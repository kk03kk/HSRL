"""Verify frozen peer tensors and measure policy movement on archived legal states."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hsbg.az.train.model import load_checkpoint
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.human_turn_preflight import verified_generation
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA


@torch.inference_mode()
def audit(segment, out):
    torch.set_num_threads(4)
    if file_identity(INITIAL)["sha256"] != INITIAL_SHA:
        raise ValueError("parent identity mismatch")
    manifest = verified_generation(segment, 232700000)
    parent, _ = load_checkpoint(str(INITIAL), "cpu")
    student, _ = load_checkpoint(manifest["model"], "cpu")
    parent.eval()
    student.eval()
    trainable = {k for k, v in parent.named_parameters() if v.requires_grad}
    student_state = student.state_dict()
    frozen = []
    moved = []
    for name, value in parent.state_dict().items():
        changed = not torch.equal(value, student_state[name])
        if name in trainable:
            if changed:
                moved.append(name)
        else:
            if changed:
                raise ValueError("frozen tensor modified: " + name)
            frozen.append(name)
    if not any(k.startswith("actor.") for k in moved) or not any(k.startswith("rank_head.") for k in moved):
        raise ValueError("policy or value did not move")
    batches = []
    rng = np.random.default_rng(774161)
    for i in range(8):
        path = Path("artifacts/runs/nus-peer-distill-20261119-1") / f"rows-{i}.npz"
        with np.load(path) as d:
            eligible, = np.where(d["arm"] == 1)
            ix = np.sort(rng.choice(eligible, 256, replace=False))
            values = [torch.from_numpy(d[k][ix]) for k in ("f","c","m")]
            turn = d["turn"][ix]
        for j in range(0, len(ix), 32):
            batch = [v[j:j+32] for v in values]
            before, after = [m(*batch)[0].log_softmax(1) for m in (parent,student)]
            batches.extend(zip(turn[j:j+32], (before.argmax(1) != after.argmax(1)).tolist(),
                               (before.exp() * (before-after)).sum(1).tolist()))
    result = {}
    for label, low, high in (("all",1,99),("through5",1,5),("6to9",6,9),("10plus",10,99)):
        selected = [(diff,kl) for t,diff,kl in batches if low<=t<=high]
        if selected:
            result[label] = dict(n=len(selected), changed=sum(v[0] for v in selected),
                                 mean_kl=float(np.mean([v[1] for v in selected])))
    atomic_json(out, dict(candidate=manifest["files"]["model.pt"], frozen_tensors=len(frozen),
                         changed_parameters=moved, fixed_states=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    audit(args.segment, args.out)
