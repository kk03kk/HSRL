"""Execute a predeclared evaluation plan with exact shards and complete evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import environment_identity, source_identity
from tools.merge_panels import merge
from tools.validate_eval_evidence import validate_panel


def shard_ranges(games, shards, seed):
    if type(games) is not int or type(shards) is not int or games < 1 or not 1 <= shards <= games:
        raise ValueError('require 1 <= shards <= games')
    q, remainder = divmod(games, shards)
    result = []
    for index in range(shards):
        count = q + int(index < remainder)
        result.append((seed, count))
        seed += count
    return result


def run(plan, out, *, device='cuda:0'):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    models = plan['models']
    if not models or len({m['name'] for m in models}) != len(models):
        raise ValueError('models must be nonempty and uniquely named')
    from workflows.eval_human_gap import _preflight_outputs
    _preflight_outputs([m['name'] for m in models], out/'panel.json', False)
    ranges = shard_ranges(plan['games'], plan['shards'], plan['seed'])
    inputs = {m['checkpoint']: file_identity(ROOT/m['checkpoint']) for m in models}
    for model in models:
        if model.get('sha256') and model['sha256'] != inputs[model['checkpoint']]['sha256']:
            raise ValueError(f'checkpoint differs from planned identity: {model["name"]}')
    for key in ('mirror-snaps', 'human-opponents'):
        value = plan.get('evaluation', {}).get(key)
        if value:
            source = ROOT/value
            if not source.exists(): raise FileNotFoundError(source)
            paths = sorted(source.rglob('*')) if source.is_dir() else [source]
            for path in paths:
                if path.is_file(): inputs[str(path.relative_to(ROOT))] = file_identity(path)
    identity = {'schema':'hsai.evaluation-run/v1', 'plan':plan, 'source':source_identity(ROOT),
                'environment':environment_identity(), 'inputs':inputs,
                'shards':[{'seed':seed,'games':games} for seed,games in ranges]}
    if identity['source']['tracked_dirty']:
        raise RuntimeError('evaluation requires committed, clean tracked source')
    atomic_json(out/'run.json', identity)
    commands = []
    reserved = {'checkpoint','games','seed','device','out','record-decisions','dump-rows'}
    if reserved.intersection(plan.get('evaluation', {})):
        raise ValueError('evaluation options cannot override run identity or recording')
    for index, (seed, games) in enumerate(ranges):
        directory = out/f'shard{index}'
        directory.mkdir()
        command = [sys.executable, '-u', 'workflows/eval_human_gap.py']
        for model in models:
            command += ['--checkpoint', f'{model["name"]}={model["checkpoint"]}']
        command += ['--games', str(games), '--seed', str(seed), '--device', device,
                    '--out', str(directory/'panel.json'), '--record-decisions', '1', '--dump-rows', '1']
        for key, value in plan.get('evaluation', {}).items():
            command += ['--'+key, str(value)]
        atomic_json(directory/'command.json', command)
        commands.append((directory, command))
    env = {**os.environ, 'PYTHONUTF8':'1', 'OMP_NUM_THREADS':'1',
           'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
    def worker(item):
        directory, command = item
        with (directory/'process.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(directory/'exit.json', {'returncode':result.returncode})
        return result.returncode
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        statuses = list(pool.map(worker, commands))
    if any(statuses):
        atomic_json(out/'failed.json', {'shard_returncodes':statuses})
        raise RuntimeError(f'evaluation shard failed: {statuses}')
    panel = merge([directory for directory, _ in commands], out/'panel.json',
                  expect_games=plan['games'], expect_models=len(models))
    expected = plan.get('expected_protocol_sha256')
    if expected and panel['protocol']['protocol_sha256'] != expected:
        raise ValueError('protocol differs from the predeclared baseline')
    checked = validate_panel(out/'panel.json', replay=int(plan.get('replay_steps_per_trace', 16)))
    for relative, before in inputs.items():
        if file_identity(ROOT/relative) != before:
            raise ValueError(f'input changed during evaluation: {relative}')
    if environment_identity() != identity['environment'] or source_identity(ROOT) != identity['source']:
        raise ValueError('source or package environment changed during evaluation')
    atomic_json(out/'validation.json', checked)
    atomic_json(out/'complete.json', {'valid':True,'panel':file_identity(out/'panel.json'),
                                      'validation':file_identity(out/'validation.json')})
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.plan.read_text(encoding='utf-8')), args.out, device=args.device)))
