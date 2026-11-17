"""Record execution identity without importing the training stack."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys


def environment_identity():
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions() if d.metadata['Name']}
    payload = {'python': platform.python_version(), 'platform': platform.system(),
               'packages': dict(sorted(packages.items()))}
    return {**payload, 'sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def source_identity(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'),
            'tracked_dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
