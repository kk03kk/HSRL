"""仓库卫生门禁（``tools/verify_workspace.py``）的测试。

对应审计 P1-7：``AGENTS.md §2`` 的"根目录只允许 6 个文件""单文件 ≤5 MB 进 Git"
以前**没有任何执行机制**——`verify_workspace` 只查 BOM/语法/卡表/引擎哈希/pytest，
所以三个 39–68 MB 的 ``.pt`` 长期躺在仓库根目录。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "verify_workspace", ROOT / "tools" / "verify_workspace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VW = _load()


def _tree(tmp_path: Path, files) -> Path:
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    return tmp_path


# ────────────────────────── 根目录白名单 ──────────────────────────
def test_root_whitelist_accepts_repo_root():
    """当前仓库根目录必须干净（这条是活的回归门）。"""
    assert VW.check_root_files(ROOT) == []


def test_root_whitelist_flags_weights(tmp_path):
    root = _tree(tmp_path, ["README.md", "AGENTS.md", "model.pt",
                            "ckpt_x.pt"])
    assert VW.check_root_files(root) == ["ckpt_x.pt", "model.pt"]


def test_root_whitelist_ignores_directories(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    assert VW.check_root_files(tmp_path) == []


def test_root_whitelist_allows_all_six(tmp_path):
    root = _tree(tmp_path, sorted(VW.ROOT_ALLOWED_FILES))
    assert VW.check_root_files(root) == []


# ────────────────────────── 进 Git 的单文件大小 ──────────────────────────
def test_big_file_gate_flags_oversized_tracked(tmp_path, monkeypatch):
    root = _tree(tmp_path, ["big.bin", "small.txt"])
    (root / "big.bin").write_bytes(b"0" * int(VW.BIG_FILE_LIMIT + 10))
    monkeypatch.setattr(VW, "git_tracked_files", lambda _root=ROOT: ["big.bin",
                                                                    "small.txt"])
    bad = VW.check_big_tracked(root)
    assert len(bad) == 1 and "big.bin" in bad[0]


def test_big_file_gate_passes_small_tracked(tmp_path, monkeypatch):
    root = _tree(tmp_path, ["small.txt"])
    monkeypatch.setattr(VW, "git_tracked_files", lambda _root=ROOT: ["small.txt"])
    assert VW.check_big_tracked(root) == []


def test_big_file_gate_skips_when_git_unavailable(monkeypatch):
    monkeypatch.setattr(VW, "git_tracked_files", lambda _root=ROOT: None)
    assert VW.check_big_tracked(ROOT) == []


def test_git_tracked_files_returns_paths():
    names = VW.git_tracked_files(ROOT)
    assert names is None or "AGENTS.md" in names


# ────────────────────────── 既有门禁没被破坏 ──────────────────────────
def test_source_gate_accepts_current_tree():
    bom, syntax = VW.check_sources(ROOT)
    assert bom == [] and syntax == []
