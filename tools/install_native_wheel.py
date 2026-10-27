"""Install only the native module into this isolated source tree; leave shared venv untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def install(wheel, root):
    wheel,root=Path(wheel).resolve(),Path(root).resolve()
    if not (root/'Cargo.toml').is_file() or not (root/'python/taverngym').is_dir():
        raise ValueError('target must be a TavernGym source workspace')
    with zipfile.ZipFile(wheel) as z:
        names=[n for n in z.namelist() if n.startswith('taverngym/_core') and
               n.count('/')==1 and n.endswith(('.pyd','.so'))]
        if len(names)!=1:raise ValueError('wheel needs exactly one native TavernGym module')
        data=z.read(names[0]);name=Path(names[0]).name
    target=root/'python/taverngym'/name
    others=[p for p in target.parent.glob('_core*') if p.suffix in ('.so','.pyd') and p!=target]
    if others:raise ValueError('another native filename could shadow this module; inspect before installation')
    temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(data);os.replace(temporary,target)
    result={'native_file':str(target),'sha256':hashlib.sha256(data).hexdigest(),
            'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()}
    meta=root/'EXECUTION_ENV.json'
    if meta.exists():
        execution=json.loads(meta.read_text(encoding='utf-8'))
        execution.update(native_sha256=result['sha256'],native_source=str(target),native_wheel_sha256=result['wheel_sha256'])
        meta.write_text(json.dumps(execution,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('wheel');p.add_argument('--root',default='.')
    a=p.parse_args();print(json.dumps(install(a.wheel,a.root)))
