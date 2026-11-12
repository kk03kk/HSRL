"""登录节点守卫与文档检查器的回归测试。

`ops/hsai_ops.py` 的守卫是"不要在登录节点跑重活"这条规则的**执行者**。它曾经
误伤过一次（`git add workflows/train_entity_sync.py` 被当成训练命令拦下），也曾经
漏掉过一次（heredoc 里的提交信息被当成命令执行）。这两类都会让人绕过守卫，
所以必须有测试钉住行为。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))

from hsai_ops import check_heavy  # noqa: E402


# ─────────────────── 守卫：必须放行 ───────────────────

@pytest.mark.parametrize(
    "script",
    [
        # 只是**提到**训练脚本的文件名，不是执行
        "git add workflows/train_entity_sync.py",
        # 反斜杠续行拼接后仍然只是 git add
        "git add \\\n  workflows/train_entity_sync.py \\\n  deploy/slurm/x.sbatch",
        # 提交信息（heredoc 正文）里出现 pytest / train 不该触发
        "git commit -F - <<'MSG'\n- workflows/train_entity_sync.py: pytest cargo\nMSG\necho done",
        # 只读命令
        "squeue -u hengzhi",
        "tail -30 artifacts/runs/x/history.jsonl",
        "sha256sum model.pt",
        # 提交作业是对的；srun 才是禁止的
        "sbatch deploy/slurm/league_stage2_48h.sbatch",
        # 字符串里提到 pip install 不算执行
        'echo "pip install is fine in a string"',
        # 空脚本 / 只有注释
        "",
        "# just a comment\n\n",
    ],
)
def test_guard_allows(script: str) -> None:
    assert check_heavy(script) == []


# ─────────────────── 守卫：必须拦下 ───────────────────

@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (".venv/bin/python -m pytest tests/x.py", "pytest"),
        ("OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q", "pytest"),
        (".venv/bin/python workflows/train_entity_sync.py", "training"),
        ("cargo build --release", "cargo"),
        ("pip install -r requirements.txt", "pip install"),
        ("srun --gres=gpu:2 hostname", "srun"),        # heredoc 结束后面的真实调用仍然要拦
        ("cat <<'EOF'\nhi\nEOF\n.venv/bin/python -m pytest -q", "pytest"),
    ],
)
def test_guard_blocks(script: str, expected: str) -> None:
    hits = check_heavy(script)
    # 标签可能带解释（"srun (use sbatch)"），所以按子串匹配
    assert any(expected in hit for hit in hits), f"{expected!r} not in {hits} for {script!r}"


def test_guard_blocks_torch_import() -> None:
    assert "torch import" in check_heavy("python -c 'import torch; print(1)'")


def test_guard_is_line_scoped_not_whole_script() -> None:
    """同一脚本里既有合法命令又有重活时，必须只因为重活而被拦。"""
    script = "squeue -u $USER\n.venv/bin/python -m pytest tests -q\n"
    assert check_heavy(script) == ["pytest"]


# ─────────────────── 文档检查器 ───────────────────

def _load_check_docs():
    sys.path.insert(0, str(ROOT / "ops"))
    import check_docs  # noqa: WPS433

    return check_docs


def test_check_docs_live_tree_is_clean() -> None:
    """现行文档树不能有坏链接或指向已退休路径的活引用。"""
    mod = _load_check_docs()
    problems = []
    for path in mod.iter_markdown():
        problems += mod.check_links(path)
        problems += mod.check_retired(path)
    assert not problems, "\n".join(f"{p.kind} {p.path}:{p.line} {p.detail}" for p in problems)


def test_check_docs_detects_broken_link(tmp_path: Path) -> None:
    mod = _load_check_docs()
    doc = tmp_path / "doc.md"
    doc.write_text("see [missing](./nope.md)\n", encoding="utf-8")
    problems = mod.check_links(doc)
    assert len(problems) == 1
    assert problems[0].kind == "broken-link"


def test_check_docs_ignores_links_inside_code_fences(tmp_path: Path) -> None:
    mod = _load_check_docs()
    doc = tmp_path / "doc.md"
    doc.write_text("```\n[missing](./nope.md)\n```\n", encoding="utf-8")
    assert mod.check_links(doc) == []


def test_check_docs_ignores_external_and_anchor_links(tmp_path: Path) -> None:
    mod = _load_check_docs()
    doc = tmp_path / "doc.md"
    doc.write_text("[a](https://example.com/x) [b](#section)\n", encoding="utf-8")
    assert mod.check_links(doc) == []


def test_check_docs_flags_links_into_machine_local_trees(tmp_path: Path) -> None:
    """指向 records/ 或 artifacts/ 的链接换台机器就断，本机存在也算错。"""
    mod = _load_check_docs()
    probe = tmp_path / "doc.md"
    probe.write_text("see [note](../records/deadbeef/note.md)\n", encoding="utf-8")

    # 用 REPO_ROOT 之外的 doc 测不到（解析结果也不在仓库内），所以直接测判定函数
    assert mod.is_machine_local(mod.REPO_ROOT / "records" / "x" / "note.md") is True
    assert mod.is_machine_local(mod.REPO_ROOT / "artifacts" / "runs" / "x") is True
    assert mod.is_machine_local(mod.REPO_ROOT / "docs" / "README.md") is False
    assert mod.is_machine_local(tmp_path / "whatever.md") is False
    # 临时目录里的相对链接不该被误判成"机器本地"，只会被当成普通坏链接
    kinds = {p.kind for p in mod.check_links(probe)}
    assert "link-into-machine-local" not in kinds
    assert kinds == {"broken-link"}


# ─────────────────── PowerShell 编码守卫 ───────────────────

def _ps1_files():
    roots = [ROOT / "ops", ROOT / "client"]
    out = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(root.rglob("*.ps1")))
    return out


def test_ps1_files_with_non_ascii_must_be_utf8_with_bom() -> None:
    """Windows PowerShell 5.1 按系统 ANSI 代码页读没有 BOM 的脚本。

    含中文的 .ps1 若不带 BOM，中文会被错误解码并**打乱行结构**——2026-11-12 就因此
    让 ops/tunnel-auth.ps1 里的 C# here-string 整段被当成 PowerShell 代码，
    `using System;` 报"缺少 using 指令"。这条测试把该格式钉死。
    """
    bad = []
    for path in _ps1_files():
        raw = path.read_bytes()
        has_bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            raw.decode("ascii")
            is_ascii = True
        except UnicodeDecodeError:
            is_ascii = False
        if not is_ascii and not has_bom:
            bad.append(str(path.relative_to(ROOT)))
    assert not bad, (
        "these .ps1 files contain non-ASCII but lack a UTF-8 BOM, so Windows "
        "PowerShell 5.1 will mis-parse them: " + ", ".join(bad)
    )


def test_ps1_files_exist_and_parse_as_utf8() -> None:
    files = _ps1_files()
    assert files, "expected at least one PowerShell script under ops/ or client/"
    for path in files:
        text = path.read_bytes().decode("utf-8-sig")  # strips the BOM when present
        assert text.strip(), f"{path} is empty"
