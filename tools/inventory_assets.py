"""Read-only metadata inventory; never infers that a historical checkpoint is compatible.

Run on local CPU or an allocated compute node. Existing outputs are not overwritten.
Large weights and datasets are counted, not deserialized or silently migrated.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json


def inventory(workspace):
    workspace = Path(workspace).resolve()
    result = {'schema': 'hsai.asset-metadata-inventory/v1',
              'scope': 'metadata only; no tensor loading, content hash audit, compatibility claim, or migration',
              'models': [], 'libraries': [], 'json_errors': [], 'unreadable': []}
    for category in ('models', 'libraries'):
        base = workspace / 'artifacts' / category
        if not base.is_dir():
            result['unreadable'].append({'path': f'artifacts/{category}', 'reason': 'missing directory'})
            continue
        for directory in sorted(base.iterdir()):
            if not directory.is_dir():
                continue
            row = {'directory': str(directory.relative_to(workspace)).replace('\\', '/'),
                   'symlink': directory.is_symlink(), 'files': 0, 'bytes': 0,
                   'checkpoint_files': [], 'metadata': {}, 'canonical_model_directories': []}
            if row['symlink']:
                row['scope'] = 'symlink not traversed; avoids external traversal and duplicate totals'
                result[category].append(row)
                continue
            def error(exc):
                result['unreadable'].append({'path': row['directory'], 'reason': type(exc).__name__})
            for current, dirs, files in os.walk(directory, followlinks=False, onerror=error):
                dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
                names = set(files)
                if category == 'models' and 'model.pt' in names:
                    row['canonical_model_directories'].append({
                        'path': str(Path(current).relative_to(workspace)).replace('\\', '/'),
                        'missing': sorted({'manifest.json', 'checkpoint.json', 'MODEL_CARD.md'} - names)})
                for name in files:
                    path = Path(current) / name
                    if path.is_symlink():
                        continue
                    try:
                        size = path.stat().st_size
                        row['files'] += 1
                        row['bytes'] += size
                        if path.suffix == '.pt':
                            row['checkpoint_files'].append({'path': str(path.relative_to(workspace)).replace('\\', '/'),
                                                            'bytes': size})
                    except OSError as exc:
                        error(exc)
            # Only parse top-level metadata; historical formats are not rewritten.
            for name in ('manifest.json', 'checkpoint.json', 'summary.json'):
                path = directory / name
                if not path.is_file() or path.is_symlink():
                    row['metadata'][name] = 'absent'
                    continue
                try:
                    if path.stat().st_size > 5 * 1024 * 1024:
                        row['metadata'][name] = 'not parsed: exceeds 5 MiB'
                        continue
                    value = json.loads(path.read_text(encoding='utf-8-sig'))
                    row['metadata'][name] = 'valid JSON object' if isinstance(value, dict) else 'valid JSON, other type'
                except (OSError, ValueError) as exc:
                    row['metadata'][name] = 'unreadable or invalid'
                    result['json_errors'].append({'path': row['directory'] + '/' + name,
                                                  'reason': type(exc).__name__})
            result[category].append(row)
    result['totals'] = {category: {'directories': len(result[category]),
                        'files': sum(r['files'] for r in result[category]),
                        'bytes': sum(r['bytes'] for r in result[category]),
                        'checkpoint_files': sum(len(r['checkpoint_files']) for r in result[category])}
                        for category in ('models', 'libraries')}
    canonical = [d for r in result['models'] for d in r['canonical_model_directories']]
    result['canonical_models'] = {'directories': len(canonical),
                                 'complete_file_sets': sum(not d['missing'] for d in canonical),
                                 'missing_file_counts': dict(Counter(name for d in canonical for name in d['missing']))}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = inventory(args.workspace)
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in ('totals', 'canonical_models', 'json_errors', 'unreadable')}))
