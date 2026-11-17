"""Seal every evaluation artifact after completion, including raw per-turn rows."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity


def seal(root):
    root=Path(root).resolve()
    destination=root/'evidence-inventory.json'
    if destination.exists(): raise FileExistsError(destination)
    complete=json.loads((root/'complete.json').read_text(encoding='utf-8'))
    if complete.get('valid') is not True or complete['panel'] != file_identity(root/'panel.json') or complete['validation'] != file_identity(root/'validation.json'):
        raise ValueError('evaluation does not have a valid completion record')
    paths=sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.name.endswith('.partial') for p in paths): raise ValueError('incomplete artifacts remain')
    files={str(p.relative_to(root)).replace('\\','/'):file_identity(p) for p in paths}
    record={'schema':'hsai.evidence-inventory/v1','files':files,
            'scope':'all existing run files except this inventory itself; append analysis outside this sealed directory'}
    atomic_json(destination,record)
    return {'files':len(files),'bytes':sum(f['bytes'] for f in files.values()),'inventory':file_identity(destination)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(seal(args.run)))
