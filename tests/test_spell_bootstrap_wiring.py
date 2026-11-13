"""法术自举补贴的**接线守卫**：四层都要在，缺一层就红。

为什么单独写一条：这个项目已经两次栽在"配了没生效"上——
① `board_target` 配了但没接进采集，静默跑完一整段（RUNS §5.34）；
② L3 的 `lr 4e-6` 被 trainer state 覆盖，实际仍是 1.667e-6（D-048）。
所以凡是"新增一个训练开关"，必须有测试同时钉住
**配置白名单 → argparse → CollectConfig → 采集侧调用点**。

真正的行为测试在 `tests/test_spell_bootstrap.py`（7 条）；这里只查"线有没有断"。
"""
from __future__ import annotations

from pathlib import Path

from hsbg.az.train import personal
from hsbg.az.train.ppo import CollectConfig

ROOT = Path(__file__).resolve().parents[1]


def test_personal_options_accept_both_keys() -> None:
    """`personal.validate_config` 只认 OPTIONS 里的键，漏加就 `未知配置项` 崩。"""
    assert "spell-bootstrap-coef" in personal.OPTIONS
    assert "spell-bootstrap-per-game" in personal.OPTIONS


def test_collect_config_exposes_fields() -> None:
    fields = CollectConfig.__dataclass_fields__
    assert "spell_bootstrap_coef" in fields
    assert "spell_bootstrap_per_game" in fields
    cfg = CollectConfig(spell_bootstrap_coef=0.05, spell_bootstrap_per_game=3)
    assert cfg.spell_bootstrap_coef == 0.05
    assert cfg.spell_bootstrap_per_game == 3


def test_train_entry_wires_args_into_collect_config() -> None:
    """argparse 有、且真的塞进 CollectConfig；启动必须打印一行 `[spell-bootstrap]`。"""
    src = (ROOT / "workflows" / "train_entity_sync.py").read_text(encoding="utf-8")
    assert '"--spell-bootstrap-coef"' in src
    assert '"--spell-bootstrap-per-game"' in src
    assert "spell_bootstrap_coef=float(" in src
    assert "spell_bootstrap_per_game=int(" in src
    assert "[spell-bootstrap]" in src


def test_collect_paths_call_attach_spell() -> None:
    """lockstep 与逐局两条采集路径都要挂上（否则换路径就静默失效）。"""
    src = (ROOT / "hsbg" / "az" / "train" / "parallel_collect.py").read_text(
        encoding="utf-8")
    assert src.count("shaper.attach_spell") >= 2
