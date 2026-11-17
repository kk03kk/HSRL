"""Read-only validation of a completed training run. Run on a compute node."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path


def require(condition, detail):
    if not condition:
        raise ValueError(f'invalid completed training artifact: {detail}')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root):
    import torch
    torch.set_num_threads(2)
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    latest = json.loads((root/'latest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (root/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [row for row in rows if 'iter' in row]
    model = torch.load(root/'model.pt', map_location='cpu', weights_only=False)
    trainer = torch.load(root/'trainer.pt', map_location='cpu', weights_only=False)
    updates = int(trainer['updates'])
    require(updates == int(summary['updates']) == int(model['meta']['trainer_updates']), 'generation mismatch')
    require(bool(rows) and len(rows) == int(summary['iterations']) == int(summary['history_len']), 'history coverage')
    require(rows[-1]['ppo']['trainer_updates'] == updates, 'history final generation')
    for key in ('optimizer', 'scheduler', 'rng', 'torch_rng', 'python_rng', 'anchor_model'):
        require(trainer.get(key) is not None, key)
    require(trainer['optimizer']['state'], 'empty Adam state')
    require(set(model['state_dict']) == set(trainer['model']), 'model/Trainer keys')
    for key, value in model['state_dict'].items():
        require(torch.equal(value, trainer['model'][key]), key)
    def finite(value, path='trainer'):
        if torch.is_tensor(value):
            require(bool(torch.isfinite(value).all()), path)
        elif isinstance(value, numbers.Real):
            require(math.isfinite(value), path)
        elif isinstance(value, dict):
            for key, child in value.items(): finite(child, f'{path}/{key}')
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value): finite(child, f'{path}/{i}')
    finite(trainer)
    counters = ('illegal_actions', 'invalid_games', 'truncated_games', 'truncated_combats',
                'truncated_recruit_contexts', 'engine_stopped', 'mirror_errors', 'dropped_decisions')
    for row in rows:
        for key in counters:
            value=row['collect'].get(key,0)
            require(isinstance(value,numbers.Real) and math.isfinite(value) and value>=0 and int(value)==value, key)
    totals = {key: sum(row['collect'].get(key, 0) for row in rows) for key in counters}
    require(not any(totals.values()), totals)
    return {
        'schema': 'hsai.completed-run-validation/v1', 'valid': True,
        'updates': updates, 'history_rows': len(rows),
        'decisions': sum(row['collect'].get('decisions', 0) for row in rows),
        'latest_snapshot_version': latest['version'], 'counters': totals,
        'counter_coverage': {key:sum(key in row['collect'] for row in rows) for key in counters},
        'model_meta': model['meta'], 'trainer_extra': trainer['extra'],
        'scheduler': trainer['scheduler'], 'optimizer_states': len(trainer['optimizer']['state']),
        'files': {name: {'bytes': (root/name).stat().st_size, 'sha256': sha256(root/name)}
                  for name in ('model.pt', 'trainer.pt', 'history.jsonl', 'summary.json', 'latest.json')},
        'torch': torch.__version__,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, default=str)
    print(json.dumps({key: result[key] for key in ('valid','updates','history_rows','decisions','files')}))
