import json
import os
from pathlib import Path

import pytest
from tools.prepare_paper_release import prepare


def test_isolated_release_preserves_environment_and_checks_native(tmp_path):
    env = tmp_path / "project"
    release = env / "runs/releases/experiment/AutoTavern"
    release.mkdir(parents=True)
    (release / "SOURCE_MANIFEST.json").write_text("{}", encoding="utf-8")
    module = env / "python/taverngym" / ("_core.pyd" if os.name == "nt" else "_core.abi3.so")
    module.parent.mkdir(parents=True)
    module.write_bytes(b"native-fixture")
    executable = env / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python-fixture")
    meta = prepare(release, env)
    assert Path(meta["source_root"]) == release
    assert module.read_bytes() == b"native-fixture"
    assert prepare(release, env) == meta
    copied = release / "python/taverngym" / module.name
    copied.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="different native"):
        prepare(release, env)
    with pytest.raises(ValueError, match="runs/releases"):
        prepare(tmp_path / "elsewhere", env)
