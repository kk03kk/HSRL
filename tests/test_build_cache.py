"""Backdated archive sources must not share a stale native cache."""
import os
from tools.nus_build_id import build_id


def workspace(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    src = tmp_path / "crates/engine/src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text("pub fn old() {}\n", encoding="utf-8")
    return src / "lib.rs"


def test_cache_key_detects_native_content_change_with_epoch_mtime(tmp_path):
    p = workspace(tmp_path)
    os.utime(p, (0, 0))
    before = build_id(tmp_path)
    p.write_text("pub fn last_applied() {}\n", encoding="utf-8")
    os.utime(p, (0, 0))
    assert build_id(tmp_path) != before


def test_cache_key_ignores_mtime_and_non_native_changes(tmp_path):
    p = workspace(tmp_path)
    before = build_id(tmp_path)
    os.utime(p, (0, 0))
    (tmp_path / "README.md").write_text("new documentation", encoding="utf-8")
    assert build_id(tmp_path) == before


def test_native_build_configuration_is_part_of_cache_key(tmp_path):
    workspace(tmp_path)
    before = build_id(tmp_path)
    (tmp_path / "Cargo.toml").write_text('[workspace]\nresolver="2"\n', encoding="utf-8")
    assert build_id(tmp_path) != before
