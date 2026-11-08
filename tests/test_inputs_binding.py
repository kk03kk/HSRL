"""长训启动链路：``inputs()`` 必须绑定到**本入口**，不能靠参数嗅探。

事故（2026-11-08，作业 1093247）：强制升本长训 9 秒失败，
``run_7357_long.py:32 ValueError: 库清单不符:train-start``。
根因链：``run_sustained_learning.parent_inputs`` 按 ``--source-7357`` 嗅探 →
强制升本入口同样带这个参数 ⇒ 选中**旧的** ``run_7357_long.inputs`` ⇒ 它把曲线摘要
写死成 ``config/opponent-p50-next.planned.json`` ⇒ 拿新库（``p50-v2-weaker``）去比旧曲线，
必然失败；即便不失败，也会冻结错父状态、用错课程表（旧 0/10/20/30M）。

修法（不动 ``run_forced_tier_long.py``，从而**保住已通过的 GPU 验收 1093246**）：
启动器按 task→runner 写入 ``HSAI_INPUTS_RUNNER=workflows.<runner>``，
``parent_inputs`` 优先采用它；sbatch 再显式 export 一份；登录节点预检静态核对三处接线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.nus_submit as nus
from workflows.launch_nus_refinement import inputs_runner_env


# ── 启动器：task→runner 的绑定字符串 ────────────────────────────────────
def test_launcher_binds_each_task_to_its_own_runner():
    assert inputs_runner_env("run_forced_tier_long.py") == "workflows.run_forced_tier_long"
    assert inputs_runner_env("run_7357_long.py") == "workflows.run_7357_long"
    assert inputs_runner_env(Path("workflows/run_forced_tier_long.py")) == \
        "workflows.run_forced_tier_long"


def test_launcher_exec_env_carries_the_binding():
    src = (ROOT / "workflows/launch_nus_refinement.py").read_text(encoding="utf-8")
    assert "HSAI_INPUTS_RUNNER=inputs_runner_env(runner)" in src
    assert "os.execve(" in src            # 仍在 exec 前注入，Torch 继承同一环境


# ── harness：显式绑定优先，嗅探只作回退并必须告警 ───────────────────────
def test_parent_inputs_honours_explicit_binding(monkeypatch):
    from workflows import run_sustained_learning as harness

    calls = []

    class _Stub:
        @staticmethod
        def inputs(args):
            calls.append(args)
            return "forced-inputs"

    monkeypatch.setitem(sys.modules, "workflows._stub_forced_runner", _Stub)
    monkeypatch.setenv("HSAI_INPUTS_RUNNER", "workflows._stub_forced_runner")
    assert harness.parent_inputs(object()) == "forced-inputs"
    assert len(calls) == 1


def test_parent_inputs_rejects_unimportable_binding(monkeypatch):
    from workflows import run_sustained_learning as harness
    monkeypatch.setenv("HSAI_INPUTS_RUNNER", "workflows.does_not_exist")
    with pytest.raises(ValueError, match="无法导入"):
        harness.parent_inputs(object())


def test_parent_inputs_rejects_module_without_inputs(monkeypatch):
    from workflows import run_sustained_learning as harness

    class _NoInputs:
        pass

    monkeypatch.setitem(sys.modules, "workflows._stub_no_inputs", _NoInputs)
    monkeypatch.setenv("HSAI_INPUTS_RUNNER", "workflows._stub_no_inputs")
    with pytest.raises(ValueError, match="没有 inputs"):
        harness.parent_inputs(object())


def test_parent_inputs_warns_when_falling_back(monkeypatch, capsys):
    from workflows import run_sustained_learning as harness
    monkeypatch.delenv("HSAI_INPUTS_RUNNER", raising=False)

    class _Args:
        source_7357 = None
        source_4159_audit = None
        bot_3702 = None
        curriculum_parent = None

    with pytest.raises(Exception):
        harness.parent_inputs(_Args())
    assert "未设置 HSAI_INPUTS_RUNNER" in capsys.readouterr().out


# ── 旧 checker：曲线摘要必须可配置（默认仍是旧曲线）────────────────────
def test_legacy_checker_uses_env_curve(tmp_path, monkeypatch):
    import json
    from workflows.run_7357_long import check_inputs

    bank = tmp_path / "bank"
    (bank / "report.json").parent.mkdir(parents=True)
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({"calibration_profile": "x"}), encoding="utf-8")
    want = nus._sha_lf(curve)
    splits = {}
    for key in ("train-start", "train-ramp1", "train-ramp2", "train", "dev", "test"):
        manifest = {"driver": "bot_firestone_p50_3702", "curve_sha256": want}
        splits[key] = manifest
        (bank / key).mkdir(parents=True)
        (bank / key / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (bank / key / "snap_00000001.json.gz").write_bytes(b"x")
    (bank / "report.json").write_text(json.dumps({"complete": True, "splits": splits}),
                                      encoding="utf-8")
    monkeypatch.setenv("HSAI_P50_CURVE", str(curve))

    class _Args:
        bot_3702 = str(bank)
        source_run = str(tmp_path)
        source_7357 = str(tmp_path)

    # 走到库检查时不应再报"曲线摘要不符"（后面会因为缺 7357 元数据而失败，属预期）
    with pytest.raises(Exception) as exc:
        check_inputs(_Args(), require_gpu=False)
    assert "曲线" not in str(exc.value) and "库清单不符" not in str(exc.value)


# ── 登录节点预检：三处接线任一缺失都必须拒绝提交 ────────────────────────
def _wiring_tree(tmp_path, *, drop=None):
    for rel in ("deploy/slurm", "workflows"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    copies = {
        "deploy/slurm/p50_forced.sbatch": ROOT / "deploy/slurm/p50_forced.sbatch",
        "workflows/launch_nus_refinement.py": ROOT / "workflows/launch_nus_refinement.py",
        "workflows/run_sustained_learning.py": ROOT / "workflows/run_sustained_learning.py",
    }
    for rel, src in copies.items():
        text = src.read_text(encoding="utf-8")
        if drop == rel:
            text = text.replace("export HSAI_INPUTS_RUNNER=workflows.run_forced_tier_long", "")
            text = text.replace("inputs_runner_env", "NOT_BOUND")
            text = text.replace("HSAI_INPUTS_RUNNER", "NOT_BOUND")
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return tmp_path


def test_wiring_check_accepts_the_repo(tmp_path):
    assert nus.check_inputs_wiring(_wiring_tree(tmp_path))


def test_wiring_check_rejects_missing_sbatch_export(tmp_path):
    root = _wiring_tree(tmp_path)
    sbatch = root / "deploy/slurm/p50_forced.sbatch"
    sbatch.write_text(sbatch.read_text(encoding="utf-8").replace(
        "export HSAI_INPUTS_RUNNER=workflows.run_forced_tier_long", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="HSAI_INPUTS_RUNNER"):
        nus.check_inputs_wiring(root)


def test_wiring_check_rejects_launcher_without_binding(tmp_path):
    root = _wiring_tree(tmp_path)
    launcher = root / "workflows/launch_nus_refinement.py"
    launcher.write_text(launcher.read_text(encoding="utf-8").replace(
        "inputs_runner_env", "something_else"), encoding="utf-8")
    with pytest.raises(ValueError, match="启动器"):
        nus.check_inputs_wiring(root)


def test_wiring_check_rejects_harness_without_support(tmp_path):
    root = _wiring_tree(tmp_path)
    harness = root / "workflows/run_sustained_learning.py"
    harness.write_text(harness.read_text(encoding="utf-8").replace(
        "HSAI_INPUTS_RUNNER", "SOMETHING"), encoding="utf-8")
    with pytest.raises(ValueError, match="run_sustained_learning"):
        nus.check_inputs_wiring(root)
