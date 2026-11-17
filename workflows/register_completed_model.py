"""Publish an immutable model cohort from a validated completed run (compute node)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import environment_identity, source_identity
from tools.validate_training_artifacts import validate


def register(run, destination, *, model_id, recovery_source):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model_id):
        raise ValueError('invalid model id')
    partial = destination.with_name(destination.name+'.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    checked = validate(run)
    partial.mkdir(parents=True)
    for name in ('model.pt', 'trainer.pt'):
        shutil.copy2(run/name, partial/name)
        if file_identity(partial/name) != checked['files'][name]:
            raise ValueError(f'asset copy identity mismatch: {name}')
    import hsbg_engine as engine
    from hsbg.az.train.common import engine_info
    manifest = {'schema': 'hsai.model/v1.1', 'id': model_id, 'status': 'candidate',
                'version': checked['updates'], 'cum_decisions': checked['decisions'],
                'arch': checked['model_meta']['arch'],
                'source_path': str(run), 'recovery_source': str(recovery_source),
                'files': {name: checked['files'][name] for name in ('model.pt','trainer.pt')},
                'engine': engine_info(), 'publication_source': source_identity(ROOT),
                'environment': environment_identity()}
    atomic_json(partial/'manifest.json', manifest)
    atomic_json(partial/'checkpoint.json', {'model': 'model.pt', 'trainer': 'trainer.pt',
                                          'version': checked['updates'], 'files': manifest['files']})
    atomic_json(partial/'validation.json', checked)
    (partial/'MODEL_CARD.md').write_text(
        f'# {model_id}\n\nStatus: candidate; no live pin change.\n\n'
        f'Final generation: {checked["updates"]}; decisions: {checked["decisions"]}.\n'
        'Model and Trainer are validated as the same generation.\n'
        'Scientific configuration is preserved in the original run summary and run.env.\n',
        encoding='utf-8')
    os.rename(partial, destination)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--recovery-source', required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.run, args.destination, model_id=args.model_id,
                              recovery_source=args.recovery_source), indent=2))
