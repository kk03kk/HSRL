"""接线回归（D-071）：lobby 种族排除必须真的接到**训练与评测的建局点**上。

只用静态断言（读源码）+ 一次真实引擎的端到端小样本（2 局），确认：
1. 训练/评测 CLI 都有 `--lobby-exclude`，且被传进 CollectConfig / EvalConfig；
2. 串行采集、lockstep 采集、评测三处建局都走 `make_filtered_game`；
3. 打开过滤后，**评测真的只见到被允许的种族**，且 `seed_sequence` 记录的是实际种子。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "workflows" / "train_entity_sync.py").read_text(encoding="utf-8")
EVAL = (ROOT / "workflows" / "eval_human_gap.py").read_text(encoding="utf-8")
POO = (ROOT / "hsbg" / "az" / "train" / "ppo.py").read_text(encoding="utf-8")
PC = (ROOT / "hsbg" / "az" / "train" / "parallel_collect.py").read_text(encoding="utf-8")
EV = (ROOT / "hsbg" / "az" / "train" / "evaluate.py").read_text(encoding="utf-8")


def test_cli_flags_exist():
    from hsbg.az.train.cli import build_parser
    assert build_parser().parse_args(['--lobby-exclude', 'DRAGON']).lobby_exclude == 'DRAGON'
    assert '"--lobby-exclude"' in EVAL


def test_flag_is_passed_into_configs():
    assert 'lobby_exclude=parse_race_list(getattr(args, "lobby_exclude", ""))' in TRAIN
    assert 'lobby_exclude=parse_race_list(getattr(args, "lobby_exclude", ""))' in EVAL


def test_all_game_creation_sites_use_the_filter():
    """三处建局都必须走 make_filtered_game（漏一处就等于过滤没生效）。"""
    assert 'make_filtered_game' in POO          # 串行采集
    assert PC.count("make_filtered_game") >= 3  # import + lockstep + serial
    assert "make_filtered_game" in EV           # 评测
    # 不允许再有裸的 E.Game(seed=...) 建局残留
    for name, text in (("ppo.py", POO), ("parallel_collect.py", PC),
                       ("evaluate.py", EV)):
        assert "E.Game(seed=int(cfg.seed)" not in text, name
        assert "E.Game(seed=int(config.seed)" not in text, name


def test_configs_carry_the_field():
    from hsbg.az.train.evaluate import EvalConfig
    from hsbg.az.train.ppo import CollectConfig
    assert CollectConfig().lobby_exclude == ()
    assert EvalConfig().lobby_exclude == ()


# ────────────────────────── 端到端（真实引擎）──────────────────────────
def test_every_eval_driver_passes_the_flag():
    """**所有**调用 ``eval_human_gap.py`` 的 sbatch 都必须把 ``--lobby-exclude`` 传下去。

    2026-11-17 真实事故：``eval_shard_parallel.sbatch``（512 局那两次评测用的驱动）
    只传了 ``--freeze-cap``，没传 ``--lobby-exclude`` ⇒ 即使设了
    ``HSAI_GAP_LOBBY_EXCLUDE`` 也会被**静默忽略**，跑出来照旧带龙/机械。
    """
    sbatch_dir = ROOT / "deploy" / "slurm"
    drivers = []
    for p in sorted(sbatch_dir.glob("*.sbatch")):
        text = p.read_text(encoding="utf-8")
        if "eval_human_gap.py" not in text:
            continue
        drivers.append(p.name)
        assert "--lobby-exclude" in text, (
            f"{p.name} 调用 eval_human_gap.py 却没传 --lobby-exclude"
            "（HSAI_GAP_LOBBY_EXCLUDE 会被静默忽略）")
        assert 'HSAI_GAP_LOBBY_EXCLUDE' in text, (
            f"{p.name} 没有把 HSAI_GAP_LOBBY_EXCLUDE 透传进 --lobby-exclude")
    assert drivers, "没有找到任何评测驱动脚本"


def test_eval_with_exclusion_only_sees_allowed_races(tmp_path):
    """跑 2 局真实对局：被禁种族绝不能出现在 lobby；seed_sequence 记录实际种子。"""
    pytest.importorskip("hsbg_engine")
    pytest.importorskip("torch")
    from workflows import eval_human_gap as G
    from hsbg.az.train import lobby_filter as LF
    from types import SimpleNamespace

    args = SimpleNamespace(
        device="cpu", games=2, seed=930400, max_steps=300,
        max_actions_per_turn=100, forbid_unfreeze=1, forbid_cancel=1,
        move_guard=1, move_cap=2, move_repeat_limit=2, freeze_cap=1,
        ban_trinkets=1, no_cancel_choose=0, tp_forbid_cancel=0,
        keep_start_board=1, forced_tier_plan="", forced_tier_prob=0.0,
        human_opponents="", human_opponent_sample="auto",
        mirror_snaps="", mirror_sample="p50band", mirror_strict=0,
        mirror_seats=7, mirror_p=1.0, lobby_exclude="MECHANICAL,DRAGON",
        dump_rows=0, out=str(tmp_path / "p" / "panel.json"))
    (tmp_path / "p").mkdir(parents=True, exist_ok=True)
    ckpt = (ROOT / "artifacts" / "models" / "league-final-20261111-step014858"
            / "model.pt")
    if not ckpt.is_file():
        pytest.skip(f"本地没有可用的 checkpoint：{ckpt}")
    entry = G.run_checkpoint("f", ckpt, args,
                            protocol=G.protocol_from_args(args))
    seeds = entry["seed_sequence"]
    assert len(seeds) == 2
    import hsbg_engine as E
    for s in seeds:
        g = E.Game(seed=int(s))
        f, _c = g.observe(0)
        names = LF.lobby_races_from_features(f)
        assert "MECHANICAL" not in names and "DRAGON" not in names, (s, names)
    # 至少有一局被过滤换过种子（否则说明过滤没生效）
    assert any(int(s) != 930400 + i for i, s in enumerate(seeds))
