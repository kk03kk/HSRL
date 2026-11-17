"""Small dependency-free primitives for durable, inspectable artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload, *, replace=False):
    """Publish complete JSON; reject existing evidence unless explicitly replacing a pointer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(tmp, path)
    else:
        # Atomic exclusive publication: never replace a concurrently created report.
        os.link(tmp, path)
        os.unlink(tmp)
    return path


def file_identity(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
