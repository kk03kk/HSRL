"""Failure injection for runtime pushes and integrity-checked downloads."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
import hsai_ops as ops


@pytest.mark.parametrize('script', [
    'cd /tmp && python -m pytest', 'echo ready; cargo build',
    'cat x | python workflows/train_entity_sync.py',
    "python - <<'PY'\nimport torch\nPY",
    'echo $(python -c "import torch")',
])
def test_compound_commands_cannot_hide_heavy_work(script):
    assert ops.check_heavy(script)


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'SUSPENDED', 'COMPLETING'])
def test_arena_job_is_protected(monkeypatch, state):
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=f'1156986,{state},hsai-arena\n'.encode()))
    assert ops._resume_sensitive_jobs()


@pytest.mark.parametrize('failure', [1, 'timeout', 'malformed'])
def test_queue_uncertainty_blocks_push(monkeypatch, failure):
    def run(*a, **k):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('ssh', 20)
        return SimpleNamespace(returncode=1 if failure == 1 else 0, stdout=b'bad row')
    monkeypatch.setattr(ops.subprocess, 'run', run)
    with pytest.raises(ops.OpsError):
        ops._resume_sensitive_jobs()


def test_push_refreshes_remote_and_never_pushes_during_training(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, 'require_tunnel', lambda: None)
    monkeypatch.setattr(ops, 'ensure_server_remote', lambda: None)
    def git(args):
        calls.append(args)
        if args[0] == 'rev-parse': return 'main'
        if args[0] == 'diff': return 'deploy/slurm/stage_arena.sbatch\n'
        return ''
    monkeypatch.setattr(ops, '_git', git)
    monkeypatch.setattr(ops, '_resume_sensitive_jobs', lambda: ['1156986'])
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: pytest.fail('push reached'))
    with pytest.raises(ops.OpsError):
        ops.cmd_push(argparse.Namespace(allow_dirty=False, force_digest=False))
    assert calls.index(['fetch', 'server', 'main']) < next(i for i,a in enumerate(calls) if a[0]=='diff')


@pytest.mark.parametrize('good', [False, True])
def test_download_verifies_before_replacing_old_asset(monkeypatch, tmp_path, good):
    target = tmp_path / 'model.pt'
    target.write_bytes(b'known-good-old-model')
    expected = hashlib.sha256(b'new-model').hexdigest()
    def run(*a, **kw):
        kw['stdout'].write(b'new-model' if good else b'corrupted')
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(ops.subprocess, 'run', run)
    monkeypatch.setattr(ops, 'audit', lambda _: None)
    if good:
        ops.stream_file_down('model.pt', target, expected_sha256=expected)
        assert target.read_bytes() == b'new-model'
    else:
        with pytest.raises(ops.OpsError, match='SHA256'):
            ops.stream_file_down('model.pt', target, expected_sha256=expected)
        assert target.read_bytes() == b'known-good-old-model'
        assert len(list(tmp_path.glob('*.partial'))) == 1


def test_misspelled_file_filter_never_downloads_everything(monkeypatch):
    monkeypatch.setattr(ops, 'require_tunnel', lambda: None)
    monkeypatch.setattr(ops, 'remote_run', lambda *a, **k: (0, 'model.pt\ntrainer.pt\n', ''))
    monkeypatch.setattr(ops, 'remote_sha256', lambda *a: pytest.fail('download reached'))
    with pytest.raises(ops.OpsError, match='不存在'):
        ops.cmd_model_pull(argparse.Namespace(name='test', files=['modle.pt'], with_trainer=False))


def test_model_name_cannot_escape_asset_directory(monkeypatch):
    monkeypatch.setattr(ops, 'require_tunnel', lambda: None)
    with pytest.raises(ops.OpsError):
        ops.cmd_model_pull(argparse.Namespace(name='../../config', files=[], with_trainer=False))
