"""``--phase forced-chain``：一次提交"GPU 验收 → afterok 长训"两条作业。

动机（2026-11-08）：重跑 GPU 验收在 ``gpu`` 分区排了近两小时队，长训要等验收通过才能提交，
用户只能盯着。链式提交把两个作业一次交上去，长训由 Slurm 的 ``afterok`` 依赖自动启动，
验收失败时长训被标成 ``DependencyNeverSatisfied``，不会白跑 12 小时。
"""

from __future__ import annotations

import json
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


def test_chain_dirs_defaults_to_timestamped_dirs(monkeypatch, tmp_path):
    """不给环境变量也要能用：默认生成两个带时间戳的新目录，并写回环境供两个作业继承。"""
    monkeypatch.delenv("HSAI_P50_AUDIT", raising=False)
    monkeypatch.delenv("HSAI_LONG_OUT", raising=False)
    dirs = nus.chain_dirs(tmp_path)
    assert "p50-forced-tier-preflight-" in dirs["audit"]
    assert "p50-forced-tier-long-" in dirs["long"]
    assert dirs["audit"] != dirs["long"]
    assert Path(os.environ["HSAI_P50_AUDIT"]).name == Path(dirs["audit"]).name
    assert Path(os.environ["HSAI_LONG_OUT"]).name == Path(dirs["long"]).name


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


# ── 训练库自动定位（按曲线身份，不靠手填路径）────────────────────────────
def _bank_tree(root, name, *, profile, curve_sha):
    d = root / "artifacts/libraries" / name
    d.mkdir(parents=True)
    (d / "report.json").write_text(
        json.dumps({"calibration_profile": profile, "curve_sha256": curve_sha,
                    "splits": {"train": {"digest": "x"}}}), encoding="utf-8")
    return d


def test_resolve_bank_finds_the_round_curve(monkeypatch, tmp_path):
    curve = tmp_path / "config/curve.json"
    curve.parent.mkdir(parents=True)
    curve.write_text(json.dumps({"calibration_profile": "p50-v2-weaker"}), encoding="utf-8")
    monkeypatch.setenv("HSAI_P50_CURVE", str(curve))
    monkeypatch.delenv("HSAI_P50_BANK", raising=False)
    want = nus._sha_lf(curve)
    old = _bank_tree(tmp_path, "bot-p50-3702-v1", profile=None, curve_sha="other")
    # 旧库没有 profile（None -> 写成 null），新库匹配当前曲线
    (old / "report.json").write_text(json.dumps({"splits": {}}), encoding="utf-8")
    new = _bank_tree(tmp_path, "bot-p50-3702-v2-weaker", profile="p50-v2-weaker",
                     curve_sha=want)
    assert nus.resolve_bank(tmp_path) == new


def test_resolve_bank_explains_when_nothing_matches(monkeypatch, tmp_path):
    curve = tmp_path / "config/curve.json"
    curve.parent.mkdir(parents=True)
    curve.write_text(json.dumps({"calibration_profile": "p50-v2-weaker"}), encoding="utf-8")
    monkeypatch.setenv("HSAI_P50_CURVE", str(curve))
    monkeypatch.delenv("HSAI_P50_BANK", raising=False)
    _bank_tree(tmp_path, "bot-p50-3702-v1", profile=None, curve_sha="other")
    with pytest.raises(ValueError) as exc:
        nus.resolve_bank(tmp_path)
    msg = str(exc.value)
    assert "找不到" in msg and "bot-p50-3702-v1" in msg and "HSAI_P50_BANK" in msg


def test_resolve_bank_prefers_explicit_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HSAI_P50_BANK", str(tmp_path / "hand-picked"))
    assert nus.resolve_bank(tmp_path) == tmp_path / "hand-picked"


def test_resolve_bank_refuses_ambiguity(monkeypatch, tmp_path):
    curve = tmp_path / "config/curve.json"
    curve.parent.mkdir(parents=True)
    curve.write_text(json.dumps({"calibration_profile": "p50-v2-weaker"}), encoding="utf-8")
    monkeypatch.setenv("HSAI_P50_CURVE", str(curve))
    monkeypatch.delenv("HSAI_P50_BANK", raising=False)
    want = nus._sha_lf(curve)
    _bank_tree(tmp_path, "bank-a", profile="p50-v2-weaker", curve_sha=want)
    _bank_tree(tmp_path, "bank-b", profile="p50-v2-weaker", curve_sha=want)
    with pytest.raises(ValueError, match="多个库"):
        nus.resolve_bank(tmp_path)


def test_submit_chain_returns_both_job_ids(monkeypatch, tmp_path):
    """回归：submit_chain 必须把两个作业号返回出来（曾因 dict(**info) 重复键 TypeError，
    而两个 sbatch 已经提交，作业号没打印 ⇒ 半夜误判成"没提交"）。"""
    monkeypatch.setattr(nus, "chain_precheck",
                        lambda root=None: dict(passed=True, phase="forced-chain",
                                               dirs={"audit": "a", "long": "l"},
                                               scripts={"gpu": "gpu-script", "long": "long-script"}))
    calls = []

    class _R:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def fake_run(argv, **kw):
        calls.append(list(argv))
        if "--parsable" in argv:
            return _R(0, "1111\n" if argv[-1] == "gpu-script" else "2222\n")
        return _R(0, "")

    monkeypatch.setattr(nus.subprocess, "run", fake_run)
    out = nus.submit_chain(tmp_path)
    assert out["passed"] is True and out["phase"] == "forced-chain"
    assert out["gpu_job"] == "1111" and out["long_job"] == "2222"
    assert out["dependency"] == "afterok:1111"
    assert any("--dependency=afterok:1111" in argv for argv in calls
               if "--parsable" in argv and "long-script" in argv)
