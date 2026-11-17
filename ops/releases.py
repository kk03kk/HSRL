"""Create immutable source releases with explicit shared asset/environment mounts.

Run on NUS with Python's standard library. Does not alter the checkout or submit jobs.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess


def create_release(workspace, commit, name, environment=None):
    workspace = Path(workspace).resolve()
    environment = (workspace / (environment or '.venv')).resolve()
    if not environment.is_relative_to(workspace) or not environment.is_dir():
        raise ValueError('release environment must be an existing directory inside workspace')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('release name must be a safe component')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(workspace), *args], text=True).strip()
    commit = git('rev-parse', '--verify', commit+'^{commit}')
    target = workspace/'artifacts/releases'/name
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    git('worktree', 'add', '--detach', str(target), commit)
    mounts = {}
    for name in ('.venv', 'artifacts'):
        source = environment if name == '.venv' else (workspace/name).resolve()
        if source.exists():
            dest = target/name
            if dest.exists():
                raise FileExistsError(dest)
            dest.symlink_to(source, target_is_directory=True)
            mounts[name] = str(source)
    # data/ and records/ contain a small tracked catalog as well as large local
    # assets. Keep the commit's tracked files; mount only absent descendants.
    def attach_missing(source, dest):
        if not dest.exists():
            dest.symlink_to(source, target_is_directory=source.is_dir())
            mounts[str(dest.relative_to(target))] = str(source)
        elif source.is_dir() and dest.is_dir():
            for child in source.iterdir():
                attach_missing(child, dest/child.name)
    for name in ('data', 'records'):
        source = workspace/name
        if source.exists(): attach_missing(source, target/name)
    # Optional reference code is deliberately not imported or implicitly linked.
    # Any third-party model release needs its own content-verified source manifest.
    manifest = {'schema': 'hsai.source-release/v1', 'commit': commit,
                'source': str(target), 'mounts': mounts,
                'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out = workspace/'artifacts/releases/manifests'/f'{target.name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--environment', help='workspace-relative immutable environment directory; default .venv target')
    args = parser.parse_args()
    print(json.dumps(create_release(args.workspace, args.commit, args.name, args.environment), indent=2))
