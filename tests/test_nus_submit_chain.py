"""``--phase forced-chain``：一次提交"GPU 验收 → afterok 长训"两条作业。

动机（2026-11-08）：重跑 GPU 验收在 ``gpu`` 分区排了近两小时队，长训要等验收通过才能提交，
用户只能盯着。链式提交把两个作业一次交上去，长训由 Slurm 的 ``afterok`` 依赖自动启动，
验收失败时长训被标成 ``DependencyNeverSatisfied``，不会白跑 12 小时。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.nus_submit as nus


def test_parse_parsable_handles_plain_and_multi_cluster():
    assert nus._parse_parsable("12345\n") == "12345"
    assert nus._parse_parsable("12345;clusterA\n") == "12345"
    assert nus._parse_parsable("  67890  ") == "67890"


@pytest.mark.parametrize("bad", ["", "\n", "sbatch: error: something", "12a45"])
def test_parse_parsable_rejects_garbage(bad):
    with pytest.raises(ValueError):
        nus._parse_parsable(bad)


def test_chain_dirs_requires_explicit_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HSAI_P50_AUDIT", raising=False)
    monkeypatch.delenv("HSAI_LONG_OUT", raising=False)
    with pytest.raises(ValueError, match="HSAI_P50_AUDIT"):
        nus.chain_dirs(tmp_path)
    monkeypatch.setenv("HSAI_P50_AUDIT", str(tmp_path / "a"))
    with pytest.raises(ValueError, match="HSAI_LONG_OUT"):
        nus.chain_dirs(tmp_path)


def test_chain_dirs_rejects_same_directory(monkeypatch, tmp_path):
    same = str(tmp_path / "same")
    monkeypatch.setenv("HSAI_P50_AUDIT", same)
    monkeypatch.setenv("HSAI_LONG_OUT", same)
    with pytest.raises(ValueError, match="不能相同"):
        nus.chain_dirs(tmp_path)


def test_chain_dirs_rejects_non_empty_dirs(monkeypatch, tmp_path):
    audit, long = tmp_path / "audit", tmp_path / "long"
    audit.mkdir()
    (audit / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HSAI_P50_AUDIT", str(audit))
    monkeypatch.setenv("HSAI_LONG_OUT", str(long))
    with pytest.raises(ValueError, match="验收输出目录已存在且非空"):
        nus.chain_dirs(tmp_path)


def test_chain_dirs_rejects_pending_stop(monkeypatch, tmp_path):
    audit, long = tmp_path / "audit", tmp_path / "long"
    long.mkdir()
    (long / "STOP_REQUESTED").write_text("stop", encoding="utf-8")
    monkeypatch.setenv("HSAI_P50_AUDIT", str(audit))
    monkeypatch.setenv("HSAI_LONG_OUT", str(long))
    with pytest.raises(ValueError, match="输出目录已存在且非空"):
        nus.chain_dirs(tmp_path)


def test_chain_dirs_accepts_two_fresh_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("HSAI_P50_AUDIT", str(tmp_path / "audit"))
    monkeypatch.setenv("HSAI_LONG_OUT", str(tmp_path / "long"))
    dirs = nus.chain_dirs(tmp_path)
    assert dirs["audit"].endswith("audit") and dirs["long"].endswith("long")


def test_chain_helpers_are_exposed():
    for name in ("chain_precheck", "submit_chain", "chain_dirs", "_parse_parsable"):
        assert callable(getattr(nus, name))


def test_chain_dependency_string_is_afterok():
    """依赖语义必须是 afterok（验收 ExitCode 0 才启动长训），不能是 afterany。"""
    src = (ROOT / "tools/nus_submit.py").read_text(encoding="utf-8")
    assert "--dependency=afterok:" in src
    assert "--dependency=afterany" not in src


def test_chain_phase_fails_cleanly_without_server_inputs(tmp_path, monkeypatch):
    """本地没有 7357 来源时，--phase forced-chain 必须以 PRECHECK FAILED 退出（2），不提交任何作业。"""
    monkeypatch.setenv("HSAI_P50_AUDIT", str(tmp_path / "audit"))
    monkeypatch.setenv("HSAI_LONG_OUT", str(tmp_path / "long"))
    monkeypatch.setenv("HSAI_SOURCE_7357", str(tmp_path / "missing-source"))
    old_cwd = os.getcwd()
    try:
        monkeypatch.setattr(sys, "argv",
                            ["nus_submit.py", "--phase", "forced-chain", "--check-only"])
        with pytest.raises(SystemExit) as exc:
            nus.main()
    finally:
        os.chdir(old_cwd)
    assert exc.value.code == 2
