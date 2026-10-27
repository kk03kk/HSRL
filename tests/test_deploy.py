"""Release payload completeness, source integrity and secret/binary exclusion."""
import io
import json
from pathlib import Path
import tarfile

import pytest
from tools.package_nus import build, payload, verify


def test_release_contains_personal_trainer_and_no_reference_runtime():
    files = payload()
    assert "python/taverngym/rl/train.py" in files
    assert "crates/taverngym/src/training_obs.rs" in files
    assert "deploy/nus/setup.sbatch" in files
    assert "configs/ppo_nus.json" in files
    assert "tools/overlays/36.6.3.json" in files
    assert "tools/gen_carddb.py" in files
    overlay = json.loads(files["tools/overlays/36.6.3.json"])
    assert overlay["meta"]["client_version"] == "36.6.3.253932"
    assert all(not k.endswith((".pyd", ".pyc", ".pt", ".pem", ".key")) for k in files)
    for name, value in files.items():
        if name.startswith(("python/taverngym/rl/", "deploy/")):
            assert b"HearthstoneReferences" not in value
            assert b"import hsbg" not in value


def test_release_manifest_verifies_and_detects_changed_source(tmp_path):
    path = build(tmp_path / "source.tar.gz")
    with tarfile.open(path) as archive:
        archive.extractall(tmp_path / "extracted", filter="data")
    root = tmp_path / "extracted/AutoTavern"
    verify(root)
    target = root / "configs/ppo_nus.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or missing"):
        verify(root)


def test_incomplete_release_fails_before_shipping(monkeypatch, tmp_path):
    import tools.package_nus as package
    files = payload()
    del files["tools/overlays/36.6.3.json"]
    monkeypatch.setattr(package, "payload", lambda: files)
    with pytest.raises(FileNotFoundError, match="36.6.3"):
        build(tmp_path / "incomplete.tar.gz")


def test_delta_repairs_missing_overlay_without_changing_engine(tmp_path):
    from tools.package_nus import digest
    files = payload()
    old_files = {name: value for name, value in files.items() if name != "tools/overlays/36.6.3.json"}
    old_manifest = {"files": {name: digest(value) for name, value in old_files.items()}}
    manifest_path = tmp_path / "old-manifest.json"
    manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    archive = build(tmp_path / "repair.tar.gz", manifest_path)
    root = tmp_path / "applied/AutoTavern"
    for name, value in old_files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert "AutoTavern/tools/overlays/36.6.3.json" in names
        assert not any(name.startswith("AutoTavern/crates/") for name in names)
        tar.extractall(root.parent, filter="data")
    verify(root)
