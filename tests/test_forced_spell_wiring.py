"""强制买法术的**接线守卫**：四层都要在，缺一层就红。

为什么单独写一条：这个项目已经三次栽在"配了没生效"上——
① `board_target` 配了但没接进采集，静默跑完一整段（RUNS §5.34）；
② L3 的 `lr 4e-6` 被 trainer state 覆盖，实际仍是 1.667e-6（D-048）；
③ `spell-bootstrap` 的奖励项确实挂上了，但零频动作拿不到它，21 轮 `shaping_mean` 恒 0
   （RUNS §5.79）。

所以凡是"新增一个训练开关"，必须有测试同时钉住
**配置白名单 → argparse → CollectConfig → 采集侧调用点**。
行为测试在 `tests/test_forced_spell.py`（19 条）；这里只查"线有没有断"。
"""
from __future__ import annotations

from pathlib import Path

from hsbg.az.train import personal
from hsbg.az.train.ppo import CollectConfig

ROOT = Path(__file__).resolve().parents[1]


def test_personal_options_accept_all_keys() -> None:
    """`personal.validate_config` 只认 OPTIONS 里的键，漏加就'未知配置项'崩。"""
    for key in ("forced-spell-per-game", "forced-spell-turn-min",
                "forced-spell-prob", "forced-spell-teacher"):
        assert key in personal.OPTIONS, key


def test_collect_config_defaults_are_off() -> None:
    """默认必须关闭：per_game=0 是"逐位等价旧路径"的唯一保证。"""
    cfg = CollectConfig()
    assert cfg.forced_spell_per_game == 0
    assert cfg.forced_spell_turn_min == 6
    assert cfg.forced_spell_teacher is False
    cfg2 = CollectConfig(forced_spell_per_game=3, forced_spell_turn_min=6,
                         forced_spell_prob=1.0, forced_spell_teacher=True)
    assert cfg2.forced_spell_per_game == 3
    assert cfg2.forced_spell_teacher is True


def test_train_entry_wires_args_into_collect_config() -> None:
    """argparse 有、且真的塞进 CollectConfig；启动必须打印一行 `[forced-spell]`。"""
    src = (ROOT / "workflows" / "train_entity_sync.py").read_text(encoding="utf-8")
    from hsbg.az.train.cli import build_parser
    args = build_parser().parse_args(['--forced-spell-per-game', '3', '--forced-spell-turn-min', '7',
                                     '--forced-spell-prob', '0.5', '--forced-spell-teacher', '1'])
    assert (args.forced_spell_per_game, args.forced_spell_turn_min,
            args.forced_spell_prob, args.forced_spell_teacher) == (3, 7, 0.5, 1)
    assert "forced_spell_per_game=int(" in src
    assert "[forced-spell]" in src


def test_policy_honours_the_tracker_in_the_finish_path() -> None:
    """强制必须落在 `_finish`（decide 与 decide_batch 共用的收尾），否则换路径就失效。"""
    src = (ROOT / "hsbg" / "az" / "train" / "ppo.py").read_text(encoding="utf-8")
    assert "self._spell = (ForcedSpellTracker(" in src
    assert "self._spell.decide(" in src
    assert "_script_action = spell_decision.action if spell_forced else UPGRADE_ACTION" in src
    assert "note_policy_action" in src


def test_collectors_aggregate_both_paths() -> None:
    """lockstep 与逐局两条采集路径都要写出逐局统计（否则冒烟看不到 hits）。"""
    src = (ROOT / "hsbg" / "az" / "train" / "parallel_collect.py").read_text(
        encoding="utf-8")
    assert src.count("forced_spell_game_meta(") >= 2
    assert src.count("aggregate_forced_spell(") >= 2
    pool = (ROOT / "hsbg" / "az" / "train" / "process_pool.py").read_text(
        encoding="utf-8")
    assert "aggregate_forced_spell(" in pool
    inline = (ROOT / "hsbg" / "az" / "train" / "ppo.py").read_text(encoding="utf-8")
    assert "aggregate_forced_spell(games_meta)" in inline
