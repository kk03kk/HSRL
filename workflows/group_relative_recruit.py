"""On-policy, leave-one-out policy gradients for paired natural recruit continuations."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.az.train.ppo import masked_log_softmax_torch
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.assess_investment_timing import UTILITY
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.learn_recruit_residual import paired_endpoints
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import FIELDS, MODEL, TimingPolicy


ROOT = Path('artifacts/runs/nus-group-relative-20261118-1')
TEMPERATURE = .75
UPDATES = 4
SHARDS = 8
GROUPS = 16
MEMBERS = 4


def output(arm, iteration, root=ROOT):
    return root / f'arm-{arm}' / f'update-{iteration}'


def checkpoint(arm, iteration, root=ROOT):
    return str(output(arm, iteration-1,root) / 'model.pt') if iteration else MODEL


@lru_cache(maxsize=4)
def evaluator(field, path):
    return make_eval_policy(path, protocol=EvalProtocol(mirror_snaps=FIELDS[field]), device='cpu')['policy'].evaluator


class GroupPolicy(TimingPolicy):
    def __init__(self, field, path, start, noise=None):
        base, self.pool, _ = native(field)
        super().__init__(base, 'parent', False)
        self.actor = evaluator(field, path)
        self.start, self.noise = start, noise

    def reset(self):
        super().reset()
        self.pending_row = None
        self.trajectory = []
        self.origin = None

    def __call__(self, f, c, mask, seat):
        if self.current < self.start:
            return self.original(f, c, mask, seat)
        if self.noise is None:
            return self.actor(f, c, mask, seat)
        if self.origin is None:
            self.origin = hashlib.sha256(f.tobytes()+c.tobytes()+mask.tobytes()).hexdigest()
        logits, _ = self.actor.logits(f, c, mask)
        scores = logits[0, mask].astype(float)/TEMPERATURE
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        options = np.flatnonzero(mask)
        index = int(self.noise.choice(len(options), p=probs))
        action = int(options[index])
        out = np.zeros(len(mask), dtype=np.float32)
        out[action] = 1
        if len(options) > 1:
            self.pending_row = (f.copy(), c.astype(np.int16), np.packbits(mask), action, float(np.log(probs[index])))
        return out, 0.

    def decide(self, game, seat):
        self.pending_row = None
        dec = super().decide(game, seat)
        row = self.pending_row
        if row is not None and not dec.info.get('forced'):
            if int(dec.action) != row[3]:
                raise ValueError('action differs from recorded behavior probability')
            self.trajectory.append(row)
        return dec


def group_task(args):
    arm, iteration, i, directory = args
    seed = 87500000 + arm*100000 + iteration*10000 + i
    start = 7 + i%3
    traces, ranks, starts = [], [], []
    for replica in range(MEMBERS):
        noise = np.random.default_rng(np.random.SeedSequence([1357901, arm, iteration, i, replica]))
        w = GroupPolicy('development', checkpoint(arm, iteration), start, noise)
        result = evaluate_policy(w, EvalConfig(games=1, seed=seed, mirror_pool=w.pool))
        check(result)
        traces.append(w.trajectory)
        ranks.append(result['rank_sequence'][0])
        starts.append(w.origin)
    if len(set(starts)) != 1:
        raise ValueError('shared natural prefix mismatch')
    all_rows = [r for trace in traces for r in trace]
    target = np.array(UTILITY)[np.array(ranks)-1]
    adv = (target-target.mean())*MEMBERS/(MEMBERS-1)
    if all_rows:
        features, cards, masks, acts, lp = zip(*all_rows)
        np.savez_compressed(directory / f'group-{i}.npz', f=np.stack(features), c=np.stack(cards),
                            m=np.stack(masks), a=np.array(acts, dtype=np.int16), logp=np.array(lp, dtype=np.float32),
                            adv=np.repeat(adv, [len(t) for t in traces]).astype(np.float32))
    return dict(i=i, seed=seed, start=start, origin=starts[0], ranks=ranks, actions=[len(t) for t in traces])


def collect(arm, iteration, shard):
    directory = output(arm, iteration) / f'rollout-{shard:02}'
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'run.json', dict(**provenance(), behavior=file_identity(checkpoint(arm, iteration))))
    tasks = [(arm, iteration, i, directory) for i in range(GROUPS*shard, GROUPS*(shard+1))]
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as pool:
        rows = list(pool.map(group_task, tasks))
    atomic_json(directory/'groups.json', rows)


def update(arm, iteration, root=ROOT, trajectories_per_group=MEMBERS):
    dev = 'cuda'
    torch.manual_seed(182000 + arm*100 + iteration)
    torch.set_num_threads(2)
    model, meta = load_checkpoint(checkpoint(arm, iteration,root), dev)
    reference, _ = load_checkpoint(MODEL, dev)
    reference.eval()
    model.eval()  # evaluation-mode dropout matches the sampling distribution; gradients stay enabled
    state_dir = output(arm, iteration,root)
    if (state_dir/'model.pt').exists():
        raise FileExistsError('fixed iteration already published')
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-5, weight_decay=0.)
    if iteration:
        opt.load_state_dict(torch.load(output(arm, iteration-1,root)/'adam.pt', weights_only=True, map_location=dev))
    samples = {}
    groups = []
    for s in range(SHARDS):
        directory = state_dir/f'rollout-{s:02}'
        rows = json.loads((directory/'groups.json').read_text())
        if [r['i'] for r in rows] != list(range(GROUPS*s, GROUPS*(s+1))):
            raise ValueError('incomplete group manifest')
        for r in rows:
            groups.append(r)
            if not sum(r['actions']):
                continue
            with np.load(directory/f"group-{r['i']}.npz") as arr:
                if len(arr['a']) != sum(r['actions']):
                    raise ValueError('incomplete group rollout')
                for k in arr:
                    samples.setdefault(k, []).append(arr[k])
    batch = {k: np.concatenate(v) for k,v in samples.items()}
    if not np.isfinite(batch['f']).all() or not np.any(batch['adv']):
        raise ValueError('invalid or empty learning signal')
    rng = np.random.default_rng(181180 + arm*100 + iteration)
    diagnostics = []
    for epoch in range(2):
        order = rng.permutation(len(batch['a']))
        for ib in range(0, len(order), 256):
            idx = order[ib:ib+256]
            f = torch.as_tensor(batch['f'][idx], device=dev)
            if meta.get('norm_mean') is not None:
                mean = torch.as_tensor(meta['norm_mean'], device=dev)
                std = torch.as_tensor(meta['norm_std'], device=dev)
                f = (f-mean) / std.clamp_min(1e-6)
            c = torch.as_tensor(batch['c'][idx].astype(np.int64), device=dev)
            m = torch.as_tensor(np.unpackbits(batch['m'][idx], axis=1)[:, :245].astype(bool), device=dev)
            a = torch.as_tensor(batch['a'][idx].astype(np.int64), device=dev)
            old = torch.as_tensor(batch['logp'][idx], device=dev)
            adv = torch.as_tensor(batch['adv'][idx], device=dev)
            logits = model(f, c, m)[0]
            lp = masked_log_softmax_torch(logits/TEMPERATURE, m)
            decision_lp = lp.gather(1, a[:, None]).squeeze(1)
            if epoch == 0 and ib == 0 and (decision_lp-old).abs().max().item() > .004:
                raise ValueError('training log probabilities do not reproduce behavior')
            ratio = (decision_lp-old).exp()
            with torch.no_grad():
                anchor = masked_log_softmax_torch(reference(f, c, m)[0]/TEMPERATURE, m)
            kl = (anchor.exp() * (anchor-lp)).masked_fill(~m, 0).sum(1).mean()
            policy = -torch.minimum(ratio*adv, ratio.clamp(.85,1.15)*adv).mean()
            loss = policy + .08*kl
            if not torch.isfinite(loss):
                raise ValueError('nonfinite group policy loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            opt.step()
            diagnostics.append([float(policy.detach()),float(kl.detach())])
    save_checkpoint(model, str(state_dir/'model.pt'), meta=meta,
                    norm_mean=meta.get('norm_mean'), norm_std=meta.get('norm_std'))
    torch.save(opt.state_dict(), state_dir/'adam.pt')
    atomic_json(state_dir/'complete.json', dict(groups=len(groups), trajectories=trajectories_per_group*len(groups),
                choices=len(batch['a']), differing_ranks=sum(len(set(r['ranks']))>1 for r in groups),
                mean_policy_and_kl=np.mean(diagnostics,axis=0).tolist(), weights=file_identity(state_dir/'model.pt')))


def game(i):
    rows = []
    for path in [MODEL, checkpoint(0, UPDATES), checkpoint(1, UPDATES)]:
        w = GroupPolicy('transfer', path, 7)
        r = evaluate_policy(w, EvalConfig(games=1, seed=87900000+i, mirror_pool=w.pool))
        check(r)
        rows.append(dict(seed=87900000+i, rank=r['rank_sequence'][0], eot=w.eot, changes=0))
    return rows


if __name__ == '__main__':
    stage = sys.argv[1]
    args = list(map(int, sys.argv[2:]))
    if stage == 'collect':
        collect(*args)
    elif stage == 'update':
        update(*args)
    elif stage == 'panel':
        shard, = args
        d = ROOT/f'panel-{shard:02}'
        d.mkdir(parents=True, exist_ok=False)
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as pool:
            rows = list(pool.map(game, range(32*shard, 32*(shard+1))))
        atomic_json(d/'rows.json', rows)
    elif stage == 'assess':
        rows = [r for s in range(8) for trio in json.loads((ROOT/f'panel-{s:02}'/'rows.json').read_text()) for r in trio]
        report = paired_endpoints(rows, 87900000, 256)
        # Network policies are evaluated on their own states; no parent shadow forward was taken.
        for entry in report['report']:
            entry.pop('changes')
        atomic_json(ROOT/'assessment.json', report)
    else:
        raise ValueError('unknown stage')
