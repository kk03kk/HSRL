"""Content-verified optional reference assets, kept outside maintained source."""
import json
from pathlib import Path

from .artifacts import file_identity
from .paths import workspace_asset


def reference_directory(name, *, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    manifest = json.loads((root/'config/reference-assets.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'hsai.reference-assets/v1':
        raise ValueError('invalid reference asset manifest')
    item = manifest[name]
    directory = workspace_asset(root, item['directory'])
    if not directory.is_dir():
        raise FileNotFoundError(f'optional reference asset {name!r} is not installed: {item["directory"]}')
    for relative, expected in item['files'].items():
        path = directory/relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('reference file escapes registered directory')
        if file_identity(path)['sha256'] != expected:
            raise ValueError(f'reference asset content changed: {name}/{relative}')
    actual = {str(p.relative_to(directory)).replace('\\','/') for p in directory.rglob('*.py')}
    expected_sources = {p for p in item['files'] if p.endswith('.py')}
    if actual != expected_sources:
        raise ValueError('reference source inventory differs from manifest')
    return directory
