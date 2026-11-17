"""Confine asset paths to a checkout or its verified Git workspace asset mount."""
from pathlib import Path
import subprocess


def workspace_asset(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('asset must be a relative path without parent traversal')
    candidate = root / relative
    resolved = candidate.resolve()
    if resolved.is_relative_to(root):
        return candidate
    if not relative.parts or relative.parts[0] != 'artifacts':
        raise ValueError('asset escapes workspace')
    try:
        common = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--git-common-dir'],
            text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        common = (root / common).resolve()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('cannot verify shared workspace assets') from exc
    shared = common.parent / 'artifacts'
    # Only the canonical repository's assets may cross a worktree boundary.
    # Resolving shared itself would incorrectly approve an arbitrary symlink.
    if (root / 'artifacts').resolve() != shared or not resolved.is_relative_to(shared):
        raise ValueError('asset escapes verified shared workspace assets')
    return candidate
