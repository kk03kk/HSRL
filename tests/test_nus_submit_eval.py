"""``tools/nus_submit.py --phase eval`` 的登录节点预检单测。

这些检查是**提交前的唯一门禁**（登录节点不能 import torch/引擎），所以每条都要有
反向用例：协议被改、权重被换、权重是同一次、输出目录已存在，都必须显式失败。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.nus_submit import eval_precheck, precheck

ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fake_tree(tmp_path: Path, *, games: int = 128, swap_sha: bool = False,
                   same_weights: bool = False, decode: str = "greedy") -> Path:
    """搭一个最小的 HSAI 目录树：配置 + 两份假权重 + 必需部署文件。"""
    root = tmp_path
    data_a = b"weights-of-7357"
    data_b = data_a if same_weights else b"weights-of-10004"
    for rel, data in (("artifacts/models/duel/7357/model.pt", data_a),
                      ("artifacts/models/duel/10004/model.pt", data_b)):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    config = {
        "schema": "hsai.eval-duel-config/v1",
        "models": {
            "A": {"id": "entity16m-nus-7357", "path": "artifacts/models/duel/7357/model.pt",
                  "file_sha256": sha(data_a), "bytes": len(data_a)},
            "B": {"id": "entity16m-p50-10004-820m",
                  "path": "artifacts/models/duel/10004/model.pt",
                  "file_sha256": sha(data_a) if swap_sha else sha(data_b),
                  "bytes": len(data_b)},
        },
        "games": games, "seed": 73000000, "decode": decode,
        "max_actions_per_turn": 80, "move_cap": 5, "move_repeat_limit": 2,
    }
    cfg_path = root / "config/eval-duel-7357-vs-10004.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    for rel in ("deploy/slurm/p50_eval.sbatch", "deploy/slurm/common.sh",
                "workflows/launch_nus_refinement.py",
                "workflows/eval_two_models_six_bots.py", "tools/stage_duel_models.py"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in ("HSAI_EVAL_CONFIG", "HSAI_EVAL_MODEL_A", "HSAI_EVAL_MODEL_B",
                 "HSAI_EVAL_OUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HSAI_EVAL_OUT", str(tmp_path / "out-not-exists"))


def test_clean_tree_passes(tmp_path):
    root = make_fake_tree(tmp_path)
    result = eval_precheck(root)
    assert result["passed"] is True
    assert result["games"] == 128 and result["seed"] == 73000000
    assert result["script"].endswith("p50_eval.sbatch")
    assert set(result["models"]) == {"A", "B"}


def test_precheck_dispatches_eval(tmp_path):
    root = make_fake_tree(tmp_path)
    assert precheck("eval", root=root)["phase"] == "eval"


def test_rejects_changed_rule(tmp_path):
    root = make_fake_tree(tmp_path, games=64)
    with pytest.raises(ValueError, match="games=64"):
        eval_precheck(root)


def test_rejects_changed_decode(tmp_path):
    root = make_fake_tree(tmp_path, decode="sample")
    with pytest.raises(ValueError, match="decode"):
        eval_precheck(root)


def test_rejects_swapped_weights(tmp_path):
    root = make_fake_tree(tmp_path, swap_sha=True)
    with pytest.raises(ValueError, match="SHA256 不符"):
        eval_precheck(root)


def test_rejects_identical_weights(tmp_path):
    root = make_fake_tree(tmp_path, same_weights=True)
    with pytest.raises(ValueError, match="同一份权重"):
        eval_precheck(root)


def test_rejects_existing_output_dir(tmp_path, monkeypatch):
    root = make_fake_tree(tmp_path)
    out = tmp_path / "busy-out"
    out.mkdir()
    (out / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HSAI_EVAL_OUT", str(out))
    with pytest.raises(ValueError, match="非空"):
        eval_precheck(root)


def test_accepts_empty_output_dir(tmp_path, monkeypatch):
    root = make_fake_tree(tmp_path)
    out = tmp_path / "fresh-out"
    out.mkdir()
    monkeypatch.setenv("HSAI_EVAL_OUT", str(out))
    assert eval_precheck(root)["passed"] is True


def test_rejects_missing_weight(tmp_path, monkeypatch):
    root = make_fake_tree(tmp_path)
    (root / "artifacts/models/duel/10004/model.pt").unlink()
    with pytest.raises(ValueError, match="缺少模型 B 权重"):
        eval_precheck(root)


def test_model_path_override(tmp_path, monkeypatch):
    root = make_fake_tree(tmp_path)
    moved = tmp_path / "elsewhere/model.pt"
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_bytes((root / "artifacts/models/duel/7357/model.pt").read_bytes())
    monkeypatch.setenv("HSAI_EVAL_MODEL_A", str(moved))
    assert eval_precheck(root)["models"]["A"]["file_sha256"] == sha(b"weights-of-7357")


def test_repo_config_matches_shipped_entry():
    """仓库里的真实协议配置必须仍然是用户批准的 128 局 / 80 动作 / 5 换位。"""
    cfg = json.loads((ROOT / "config/eval-duel-7357-vs-10004.json").read_text(encoding="utf-8"))
    assert cfg["games"] == 128
    assert cfg["max_actions_per_turn"] == 80
    assert cfg["move_cap"] == 5 and cfg["move_repeat_limit"] == 2
    assert cfg["models"]["A"]["file_sha256"] != cfg["models"]["B"]["file_sha256"]
