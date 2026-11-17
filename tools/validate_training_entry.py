"""Two one-iteration real-engine continuation probes; never publish their weights.

Uses the completed run's scientific configuration. Only execution size, paths,
device placement and explicit preservation of the restored anchor differ.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.cli import build_parser
from hsbg.runtime.artifacts import atomic_json, file_identity


def validate(run, model, out):
    run, model, out = Path(run), Path(model), Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    config = json.loads((run/'summary.json').read_text(encoding='utf-8'))['config']
    before = {name:file_identity(model/name) for name in ('model.pt','trainer.pt')}
    actions = {a.dest:a for a in build_parser()._actions}
    previous = model
    generations = []
    for segment in range(2):
        target = out/f'segment{segment}'
        overrides = dict(iterations=1, minutes=10, games=2, processes=1, workers=1,
                         gpus='0', device='cuda:0', out=str(target),
                         checkpoint=str(previous/'model.pt'), trainer_state=str(previous/'trainer.pt'),
                         reset_anchor=0)
        probe = {**config, **overrides}
        command = [sys.executable, '-u', 'workflows/train_entity_sync.py']
        for key, value in probe.items():
            if value is None: continue
            action = actions[key]
            flag = next(x for x in action.option_strings if x.startswith('--'))
            command += [flag, str(value)]
        atomic_json(out/f'command{segment}.json', {'command':command, 'overrides':overrides})
        with (out/f'segment{segment}.log').open('x', encoding='utf-8') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
        latest = json.loads((target/'latest.json').read_text(encoding='utf-8'))
        summary = json.loads((target/'summary.json').read_text(encoding='utf-8'))
        assert latest['final'] is True
        assert latest['version'] == summary['updates']
        for name, key in (('model.pt','model'), ('trainer.pt','trainer')):
            assert file_identity(latest[key]) == latest['files'][name] == file_identity(target/name)
        generations.append(latest['version'])
        previous = target
    assert generations[1] == generations[0]+1
    assert before == {name:file_identity(model/name) for name in before}
    result = {'valid':True, 'generations':generations, 'source_unchanged':True,
              'scope':'real engine, pipeline collection, PPO, final publication and second-process restore; probe weights are not candidates'}
    atomic_json(out/'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.run, args.model, args.out)))
