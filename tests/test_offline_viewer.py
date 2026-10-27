from pathlib import Path
import shutil
import subprocess

import pytest


def test_viewer_logic_without_network_or_browser():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node optional; viewer DOM logic test needs Node")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/viewer_smoke.cjs"),
                             str(root / "python/taverngym/offline/viewer.html")],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
