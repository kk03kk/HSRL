"""源码卫生门禁：UTF-8 BOM 既不能造成假 SyntaxError，也不能悄悄留在代码目录。

事故（2026-11-07，NUS CPU 作业 1092598）：``tools/verify_workspace.py`` 用
``path.read_text(encoding='utf-8')`` 再 ``ast.parse``，而 ``workflows/run_forced_tier_long.py``
被 PowerShell 改写时带上了 UTF-8 BOM（U+FEFF），于是作业在 15 秒内以
``SyntaxError: invalid non-printable character U+FEFF`` 失败；同一批改写还弄脏了
``tests/test_parallel_collect.py``。这两个文件当时都不在检查器的语法清单里，
所以"查了这个漏了那个"也是缺陷的一部分。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.strip_source_bom import (BOM, find_bom, iter_sources, main as strip_main,
                                    strip_bom)
from tools.verify_workspace import (CODE_DIRS, check_sources, iter_sources as verify_sources,
                                    read_source)

CLEAN = '"""ok."""\nVALUE = 1\n'


def make_tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


# ── 读取：BOM 不再变成假语法错误 ─────────────────────────────────────────
def test_read_source_tolerates_bom(tmp_path: Path):
    path = tmp_path / "workflows" / "bommed.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(BOM + CLEAN.encode("utf-8"))
    ast.parse(read_source(path), filename=str(path))          # 不抛 SyntaxError
    with path.open("rb") as fh:                                # 原始读取仍然保留 BOM
        assert fh.read(3) == BOM


def test_plain_utf8_reader_would_have_failed(tmp_path: Path):
    """回归：证明事故形态真实存在（旧检查器正是这样失败的）。"""
    path = tmp_path / "bommed.py"
    path.write_bytes(BOM + CLEAN.encode("utf-8"))
    with pytest.raises(SyntaxError):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ── 门禁：BOM / 语法 ────────────────────────────────────────────────────
def test_check_sources_flags_bom(tmp_path: Path):
    make_tree(tmp_path, {"workflows/run_forced_tier_long.py": BOM + CLEAN.encode("utf-8"),
                         "tests/ok.py": CLEAN.encode("utf-8")})
    bom_rows, syntax_rows = check_sources(tmp_path)
    assert bom_rows == ["workflows/run_forced_tier_long.py"]
    assert syntax_rows == []


def test_check_sources_flags_syntax_error(tmp_path: Path):
    make_tree(tmp_path, {"hsbg/broken.py": b"def f(:\n"})
    bom_rows, syntax_rows = check_sources(tmp_path)
    assert bom_rows == []
    assert syntax_rows and syntax_rows[0].startswith("hsbg/broken.py:")


def test_check_sources_clean_tree(tmp_path: Path):
    make_tree(tmp_path, {"hsbg/a.py": CLEAN.encode("utf-8"),
                         "tests/b.py": CLEAN.encode("utf-8")})
    assert check_sources(tmp_path) == ([], [])


def test_repo_sources_are_bom_free_and_parse():
    """仓库自身的回归：代码目录不允许有任何 BOM（本次事故的两个文件曾在此）。"""
    bom_rows, syntax_rows = check_sources(ROOT)
    assert bom_rows == [], f"代码目录出现 UTF-8 BOM: {bom_rows}"
    assert syntax_rows == [], f"源码无法解析: {syntax_rows}"


def test_checker_covers_tests_and_workflows():
    """事故文件分别在 workflows/ 与 tests/：清单必须都覆盖，不能只查 hsbg/client。"""
    assert "tests" in CODE_DIRS and "workflows" in CODE_DIRS and "tools" in CODE_DIRS
    rels = {p.relative_to(ROOT).as_posix() for p in verify_sources(ROOT)}
    assert "workflows/run_forced_tier_long.py" in rels
    assert "tests/test_parallel_collect.py" in rels


# ── 修复工具 ────────────────────────────────────────────────────────────
def test_strip_bom_is_idempotent(tmp_path: Path):
    path = tmp_path / "workflows" / "x.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(BOM + CLEAN.encode("utf-8"))
    assert strip_bom(path) is True
    assert path.read_bytes() == CLEAN.encode("utf-8")
    assert strip_bom(path) is False               # 第二次没有改动


def test_fix_cli_clears_every_bom(tmp_path: Path, capsys):
    make_tree(tmp_path, {"workflows/a.py": BOM + CLEAN.encode("utf-8"),
                         "tools/b.py": BOM + CLEAN.encode("utf-8")})
    assert find_bom(iter_sources(tmp_path), tmp_path) == ["tools/b.py", "workflows/a.py"]
    assert strip_main(["--fix", "--root", str(tmp_path)]) == 0
    assert find_bom(iter_sources(tmp_path), tmp_path) == []
    assert "已去掉 2 个文件" in capsys.readouterr().out


def test_report_cli_exits_nonzero_when_bom_present(tmp_path: Path):
    make_tree(tmp_path, {"workflows/a.py": BOM + CLEAN.encode("utf-8")})
    assert strip_main(["--root", str(tmp_path)]) == 1


def test_iter_sources_only_touches_code_dirs(tmp_path: Path):
    make_tree(tmp_path, {"workflows/a.py": CLEAN.encode("utf-8"),
                         "hsbg/a.py": CLEAN.encode("utf-8"),
                         "records/evidence.py": CLEAN.encode("utf-8"),
                         "artifacts/local/x.py": CLEAN.encode("utf-8")})
    rels = {p.relative_to(tmp_path).as_posix() for p in iter_sources(tmp_path)}
    assert rels == {"hsbg/a.py", "workflows/a.py"}
