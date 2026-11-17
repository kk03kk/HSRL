from pathlib import Path
import subprocess
import pytest

from hsbg.runtime.paths import workspace_asset


def symlink(source, target):
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f'directory symlinks unavailable: {exc}')


def test_ordinary_asset_and_parent_traversal(tmp_path):
    assert workspace_asset(tmp_path, 'artifacts/models/x.pt') == tmp_path/'artifacts/models/x.pt'
    with pytest.raises(ValueError):
        workspace_asset(tmp_path, '../outside.pt')
    with pytest.raises(ValueError):
        workspace_asset(tmp_path, str(tmp_path/'absolute.pt'))


def test_only_verified_git_worktree_asset_mount_is_allowed(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c',
                    'user.email=test@example.invalid', 'commit', '--allow-empty', '-m', 'test'],
                   check=True, capture_output=True)
    release = tmp_path/'release'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(release)],
                   check=True, capture_output=True)
    (repo/'artifacts').mkdir()
    symlink(repo/'artifacts', release/'artifacts')
    assert workspace_asset(release, 'artifacts/model.pt') == release/'artifacts/model.pt'
    outside = tmp_path/'outside'
    outside.mkdir()
    symlink(outside, repo/'artifacts/escape')
    with pytest.raises(ValueError):
        workspace_asset(release, 'artifacts/escape/model.pt')
    symlink(outside, release/'data')
    with pytest.raises(ValueError):
        workspace_asset(release, 'data/model.pt')
