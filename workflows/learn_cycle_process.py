"""Compare bounded cycle teaching with frozen versus trainable representations."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import hsbg_engine as engine
from hsbg.az.train.cycle_tasks import achieved, own
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.model import load_checkpoint, save_checkpoint
from hsbg.az.train.ppo import PPOEvaluator, PPOGreedyPolicy
from hsbg.az.train.process_learning import process_loss, log_policy, forward_kl
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity, environment_identity
from workflows.learn_economic_process import tensors, forward, assess_rows

HEAD_PREFIXES = ('scalar_head.', 'q_buy.', 'q_board.', 'q_activate_delta.',
                 'buy_bias', 'sell_bias', 'activate_bias', 'choose_bias', 'play_bias', 'pos_play')


def configure_parameters(model, mode):
    if mode not in ('all', 'heads'):
        raise ValueError(mode)
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == 'all' or name.startswith(HEAD_PREFIXES))
        if parameter.requires_grad:
            selected.append(name)
    return selected


@torch.no_grad()
def closed_loop(model, tasks, device, path):
    evaluator = PPOEvaluator(model, device=device)
    rows = []
    with gzip.open(path, 'xt', encoding='utf-8', compresslevel=3) as stream:
        for task in tasks:
            game = engine.Game.from_json(json.dumps(task['initial']))
            policy = PPOGreedyPolicy(evaluator)
            steps = []
            stop = 'budget'
            for index in range(12):
                if achieved(game, task['goal']):
                    stop = 'completed'
                    break
                before = json.loads(game.state_json())
                action = int(policy.decide(game, 0).action)
                native_mask = bytes(game.legal_mask(0))
                if not native_mask[action]:
                    raise RuntimeError('illegal cycle-policy action')
                if action == 0:
                    stop = 'end_turn'
                    steps.append({'action': 0, 'executed': False, 'before': before})
                    break
                game.step(0, action)
                steps.append({'action': action, 'executed': True, 'before': before,
                              'after': json.loads(game.state_json())})
                if any(game.diagnostic_counts()):
                    raise RuntimeError('cycle rollout native diagnostic')
                if not set(task['goal']['protected']) <= {m['eid'] for m in own(game)['board']}:
                    stop = 'sold_original'
                    break
            if achieved(game, task['goal']):
                stop = 'completed'
            row = {'seed': task['seed'], 'kind': task['kind'], 'variant': task['variant'],
                   'stop': stop, 'success': stop == 'completed',
                   'executed_actions': sum(x['executed'] for x in steps)}
            rows.append(row)
            stream.write(json.dumps({'result': row, 'goal': task['goal'], 'initial': task['initial'],
                                     'steps': steps}, separators=(',', ':')) + '\n')
    def summary(group):
        return {'tasks': len(group), 'success': sum(r['success'] for r in group) / max(1, len(group)),
                'sold_original': sum(r['stop'] == 'sold_original' for r in group)}
    return {'all': summary(rows), 'kinds': {k: summary([r for r in rows if r['kind'] == k])
            for k in sorted({r['kind'] for r in rows})}, 'rows': rows}


@torch.no_grad()
def anchor_check(model, reference, anchors):
    kl, changed, rank_kl = [], [], []
    for offset in range(0, len(anchors['features']), 128):
        ix = np.arange(offset, min(offset + 128, len(anchors['features'])))
        p, r = forward(model, anchors, ix)
        q, s = forward(reference, anchors, ix)
        lp = log_policy(p, anchors['masks'][ix])
        lq = log_policy(q, anchors['masks'][ix])
        kl.extend((lq.exp() * (lq - lp)).sum(-1).cpu().tolist())
        changed.extend((p.argmax(-1) != q.argmax(-1)).cpu().tolist())
        rank_kl.extend((s.log_softmax(-1).exp() * (s.log_softmax(-1) - r.log_softmax(-1))).sum(-1).cpu().tolist())
    return {'rows': len(kl), 'policy_kl': float(np.mean(kl)), 'greedy_changed': float(np.mean(changed)),
            'rank_kl': float(np.mean(rank_kl)), 'policy_kl_p95': float(np.quantile(kl, .95))}


def run(args, *, expected_schema='hsai.cycle-process-dataset/v1',
        closed_loop_fn=closed_loop, learning_tag='cycle_learning'):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.dataset)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['schema'] != expected_schema:
        raise ValueError('wrong task schema')
    if manifest.get('mechanism_schema_sha256') != M.SCHEMA_SHA256:
        raise ValueError('dataset observation schema differs')
    for name, identity in manifest['files'].items():
        if file_identity(root / name) != identity:
            raise ValueError('dataset file changed: ' + name)
    model, meta = load_checkpoint(args.checkpoint, device=args.device)
    if meta.get('norm_mean') is not None or meta.get('norm_std') is not None:
        raise ValueError('internally normalized entity model required')
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainable = configure_parameters(model, args.mode)
    raw = np.load(root / 'labels.npz', allow_pickle=False)
    labels = tensors(raw, model, args.device)
    anchors = tensors(np.load(root / 'anchors.npz', allow_pickle=False), model, args.device)
    dev = np.flatnonzero(raw['dev'])
    groups = {k: np.flatnonzero((~raw['dev']) & (raw['kinds'] == k)) for k in sorted(set(raw['kinds']))}
    if not len(dev) or any(not len(v) for v in groups.values()):
        raise ValueError('missing source split')
    with gzip.open(root / 'tasks.jsonl.gz', 'rt', encoding='utf-8') as stream:
        tasks = [row for line in stream if (row := json.loads(line))['split'] == 'dev']
    identity = {'args': vars(args), 'source': source_identity(Path(__file__).resolve().parents[1]),
                'environment': environment_identity(), 'parent': file_identity(args.checkpoint),
                'dataset_manifest': file_identity(root / 'manifest.json'), 'trainable': trainable}
    if identity['source']['tracked_dirty']:
        raise RuntimeError('committed source required')
    atomic_json(out / 'run.json', identity)
    model.eval()
    baseline = {'labels': assess_rows(model, labels, dev, raw['kinds']),
                'closed_loop': closed_loop_fn(model, tasks, args.device, out / 'before.jsonl.gz')}
    atomic_json(out / 'baseline.json', baseline)
    print(json.dumps({'baseline': baseline['closed_loop']['kinds']}), flush=True)
    if args.updates == 0:
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    started = time.monotonic()
    for update in range(1, args.updates + 1):
        ix = np.concatenate([rng.choice(v, size=args.per_kind_batch, replace=True) for v in groups.values()])
        ai = rng.integers(len(anchors['features']), size=len(ix) * 2)
        logits, _ = forward(model, labels, ix)
        skill = process_loss(logits, labels['masks'][ix], labels['targets'][ix])
        current, rank = forward(model, anchors, ai)
        with torch.no_grad():
            prior, prior_rank = forward(reference, anchors, ai)
        kl = forward_kl(log_policy(prior, anchors['masks'][ai]), log_policy(current, anchors['masks'][ai]))
        rkl = forward_kl(prior_rank.log_softmax(-1), rank.log_softmax(-1))
        loss = skill + args.anchor_kl * kl + rkl
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite cycle loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == args.updates:
            check = anchor_check(model, reference, anchors)
            row = {'update': update, 'skill': float(skill.detach()), 'anchor': check,
                   'grad_norm': float(norm), 'elapsed': time.monotonic() - started,
                   'labels': assess_rows(model, labels, dev, raw['kinds'])}
            atomic_json(out / 'checks' / f'{update:06d}.json', row)
            print(json.dumps(row), flush=True)
            if check['policy_kl'] > .03:
                raise RuntimeError('whole-anchor KL budget exceeded; no candidate saved')
    if args.mode == 'heads':
        # Prove that values/representations were preserved, not merely frozen by intent.
        prior_state = reference.state_dict()
        for name, value in model.state_dict().items():
            if name not in trainable and not torch.equal(value, prior_state[name]):
                raise RuntimeError('frozen parameter/buffer changed: ' + name)
    meta[learning_tag]={'parent': identity['parent'], 'dataset': identity['dataset_manifest'],
                       'args': vars(args), 'research_only': True}
    save_checkpoint(model, str(out / 'model.pt'), meta=meta)
    torch.save({'schema': 'hsai.cycle-bc-state/v1', 'learning_tag':learning_tag,
                'optimizer': optimizer.state_dict(),
                'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state,
                'updates': args.updates, 'model': file_identity(out / 'model.pt'),
                'note': 'Not a PPO Trainer; any PPO fork must declare a fresh optimizer'}, out / 'bc_state.pt')
    restored, _ = load_checkpoint(str(out / 'model.pt'), device=args.device)
    with torch.no_grad():
        if not all(torch.equal(a, b) for a, b in zip(forward(model, labels, dev[:16]), forward(restored, labels, dev[:16]))):
            raise RuntimeError('cycle checkpoint roundtrip differs')
    result = {'labels': assess_rows(restored, labels, dev, raw['kinds']),
              'closed_loop': closed_loop_fn(restored, tasks, args.device, out / 'after.jsonl.gz'),
              'anchor': anchor_check(restored, reference, anchors),
              'model': file_identity(out / 'model.pt'), 'bc_state': file_identity(out / 'bc_state.pt'),
              'roundtrip_exact': True}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'final': result['closed_loop']['kinds'], 'anchor': result['anchor']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=('all', 'heads'), required=True)
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--anchor-kl', type=float, default=10.)
    parser.add_argument('--per-kind-batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1182801)
    run(parser.parse_args())
