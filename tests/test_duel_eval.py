"""2 模型 + 6 bot 配对评测的口径与门禁单测（不需要 torch/GPU）。

覆盖：
- **P50 口径**：P50(Σatk+Σhp) 不等于「攻击中位数 + 生命中位数」，报告主参考必须是前者；
- 逐回合表：``frames`` / ``games_at_turn`` / ``survivors`` 并列，空回合给 ``None``；
- 名次与配对差分：符号方向、平局、以对局为重采样的 bootstrap 可复现；
- 达标回合 / 最终状态 / 最佳局选择顺序；
- 协议：座位轮换、``protocol_sha256`` 稳定且随规则改变；A/B 同权重被拒绝；
- 报告门禁：非法动作 / 截断 / 局数不足 / 名次越界 -> ``valid=False``；
- markdown 能渲染（含最佳局战团表）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsbg.eval.duel_metrics import (
    board_totals,
    final_state_summary,
    paired_delta,
    percentile,
    rank_summary,
    reach_tier_stats,
    select_best_games,
    strength_summary,
    turn_table,
)
from workflows.eval_two_models_six_bots import (
    build_protocol,
    keyword_names,
    load_config,
    protocol_sha256,
    render_markdown,
    seats_for_game,
)
from workflows.eval_two_models_six_bots import build_report


# ── 构造样本 ─────────────────────────────────────────────────────────────
def frame(game: int, turn: int, *, tier: int = 3, atk: int = 10, hp: int = 10,
          health: int = 20, gold: int = 0, size: int = 2) -> dict:
    return {"game": int(game), "turn": int(turn), "tier": int(tier), "gold": int(gold),
            "health": int(health), "alive": health > 0, "size": int(size),
            "atk": int(atk), "hp": int(hp), "body": int(atk + hp)}


def model_audit(model_id: str, sha: str) -> dict:
    return {"id": model_id, "file_sha256": sha, "weights_sha256": None, "params": 16436034,
            "arch": "entity", "meta": {}, "forward_ok": True, "card_table_ok": True,
            "torch": "test", "device": "cpu"}


def base_cfg(**overrides) -> dict:
    cfg = {
        "models": {
            "A": {"id": "model-a", "path": "a/model.pt", "file_sha256": "aa" * 32},
            "B": {"id": "model-b", "path": "b/model.pt", "file_sha256": "bb" * 32},
        },
        "games": 2, "seed": 73000000, "max_actions_per_turn": 80, "move_cap": 5,
        "move_repeat_limit": 2, "max_steps": 50000, "best_games": 2, "top_power_games": 1,
    }
    cfg.update(overrides)
    return cfg


def synthetic_run(*, illegal: int = 0, truncated: int = 0, engine_bad: int = 0,
                  games: int = 2) -> dict:
    """与 ``run_games`` 同结构的假数据（2 局，A/B 各一座，逐局交换）。"""
    rows, frames = [], {"A": [], "B": []}
    for gi in range(games):
        seats = seats_for_game(gi)
        rank = {"A": 1 + gi % 2, "B": 2 - gi % 2}
        placements = [0] * 8
        placements[seats["A"]] = rank["A"]
        placements[seats["B"]] = rank["B"]
        for offset, seat in enumerate([s for s in range(8) if s not in seats.values()]):
            placements[seat] = 3 + offset
        final = {}
        for label in ("A", "B"):
            final[label] = frame(gi, 3, tier=4 + gi % 2, atk=100 + gi, hp=90 + gi,
                                 health=force_health(gi, label))
            frames[label].append(frame(gi, 1, tier=1, atk=4, hp=4))
            frames[label].append(frame(gi, 2, tier=2, atk=30, hp=25))
            frames[label].append(dict(final[label]))
        rows.append({
            "game": gi, "seed": 73000000 + gi, "seats": dict(seats), "turns": 3,
            "steps": 40, "decisions": {"A": 20, "B": 21}, "illegal": illegal,
            "over": truncated == 0 or gi > 0, "rank": rank, "placements": placements,
            "final": {label: dict(final[label]) for label in ("A", "B")},
            "final_board": {
                label: [{"position": 0, "card": 2897, "card_id": "BG31_329_G",
                         "name": "野猪玩家", "name_en": "Hog Player", "card_tier": 3,
                         "atk": 255, "hp": 244, "max_hp": 244, "golden": True,
                         "keywords": ["TAUNT"]}] for label in ("A", "B")},
            "guard": {label: {"decisions": 20, "moves": 1, "blocked_inverse": 2,
                              "blocked_repeat": 3, "blocked_cap": 49, "forced": 0}
                      for label in ("A", "B")},
            "engine_diagnostics": {"truncated_combats": 0, "truncated_recruit_contexts": 0,
                                   "turn_limit_end": 0, "engine_stopped": 0},
            "seconds": 1.0,
        })
    return {"frames": frames, "bot_frames": [frame(0, 1, atk=3, hp=3),
                                             frame(0, 2, atk=20, hp=15)],
            "games": rows, "illegal": illegal, "truncated_games": truncated,
            "engine_bad": engine_bad, "bot_eot_skipped": 0, "seconds": 2.0,
            "rule_counters": {"A": {"forced_cap80": 0}, "B": {"forced_cap80": 1}},
            "guard_totals": {label: {} for label in ("A", "B")}}


def force_health(gi: int, label: str) -> int:
    return 20 if (gi + (label == "B")) % 2 == 0 else 0


# ── P50 口径 ─────────────────────────────────────────────────────────────
def test_board_totals_body_is_sum_of_components():
    board = [{"atk": 10, "hp": 5}, {"atk": 3, "hp": 7}]
    assert board_totals(board) == {"size": 2, "atk": 13, "hp": 12, "body": 25}
    assert board_totals([]) == {"size": 0, "atk": 0, "hp": 0, "body": 0}


def test_p50_of_sums_is_not_sum_of_marginal_medians():
    """AGENTS.md 硬性口径：主参考是 P50(Σatk+Σhp)，不是 atk P50 + hp P50。"""
    frames = [
        frame(0, 5, atk=200, hp=0),     # body 200
        frame(1, 5, atk=10, hp=10),     # body 20
        frame(2, 5, atk=0, hp=400),     # body 400
    ]
    table = turn_table(frames, turns=[5])["5"]
    atk_p50 = percentile([f["atk"] for f in frames], 50)
    hp_p50 = percentile([f["hp"] for f in frames], 50)
    assert atk_p50 == 10.0 and hp_p50 == 10.0
    assert table["body_p50"] == 200.0          # 正确口径
    assert table["atk_p50"] == 10.0 and table["hp_p50"] == 10.0    # 辅助量，分开报告
    assert table["body_p50"] != atk_p50 + hp_p50


def test_strength_summary_empty_and_quantiles():
    empty = strength_summary([])
    assert empty["frames"] == 0 and empty["p50"] is None
    summary = strength_summary([1, 2, 3, 4])
    assert summary["frames"] == 4
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p25"] == pytest.approx(1.75)
    assert summary["p75"] == pytest.approx(3.25)


def test_turn_table_counts_frames_games_and_survivors():
    frames = [frame(0, 1), frame(1, 1), frame(0, 2, health=0), frame(1, 2)]
    table = turn_table(frames, turns=[1, 2, 3])
    assert table["1"]["frames"] == 2 and table["1"]["games_at_turn"] == 2
    assert table["1"]["survivors"] == 2
    assert table["2"]["frames"] == 2 and table["2"]["survivors"] == 1
    assert table["3"]["frames"] == 0 and table["3"]["body_p50"] is None
    assert table["3"]["tier_ge5"] is None


# ── 名次 / 配对 ──────────────────────────────────────────────────────────
def test_rank_summary_basic_metrics():
    ranks = [1, 2, 3, 4, 5, 6, 7, 8]
    summary = rank_summary(ranks)
    assert summary["games"] == 8 and summary["games_ranked"] == 8
    assert summary["mean_rank"] == pytest.approx(4.5)
    assert summary["top4_rate"] == pytest.approx(0.5)
    assert summary["win_rate"] == pytest.approx(0.125)
    assert summary["rank_counts"]["1"] == 1 and summary["rank_counts"]["8"] == 1
    lo, hi = summary["mean_rank_ci95"]
    assert lo <= summary["mean_rank"] <= hi


def test_rank_summary_ignores_unranked():
    summary = rank_summary([1, 2, 0])
    assert summary["games"] == 3 and summary["games_ranked"] == 2
    assert summary["mean_rank"] == pytest.approx(1.5)
    assert sum(summary["rank_counts"].values()) == 2   # 未完成/未排名不入分布


def test_paired_delta_direction_and_ties():
    result = paired_delta([1, 2, 3, 4], [3, 2, 1, 8])
    assert result["pairs"] == 4
    assert (result["a_better"], result["b_better"], result["ties"]) == (2, 1, 1)
    assert result["mean_delta"] == pytest.approx((-2 + 0 + 2 - 4) / 4)
    assert result["a_better_rate"] == pytest.approx(0.5)


def test_paired_delta_bootstrap_is_reproducible():
    a = list(range(1, 9)) * 3
    b = [r + 4 for r in a]
    first = paired_delta(a, b, bootstrap=500, seed=7)
    second = paired_delta(a, b, bootstrap=500, seed=7)
    assert first["mean_delta_ci95"] == second["mean_delta_ci95"]
    assert first["mean_delta"] == pytest.approx(-4.0)


def test_paired_delta_skips_missing_pairs():
    result = paired_delta([1, 0, 3], [2, 2, 0])
    assert result["pairs"] == 1
    assert result["mean_delta"] == pytest.approx(-1.0)


# ── 等级 / 最终状态 / 最佳局 ─────────────────────────────────────────────
def test_reach_tier_stats_first_turn_per_game():
    frames = [
        frame(0, 1, tier=1), frame(0, 2, tier=3), frame(0, 4, tier=4), frame(0, 6, tier=5),
        frame(1, 1, tier=1), frame(1, 2, tier=2), frame(1, 3, tier=3),
    ]
    stats = reach_tier_stats(frames)
    assert stats["4"]["games_reached"] == 1 and stats["4"]["turn_p50"] == 4.0
    assert stats["4"]["share_reached"] == pytest.approx(0.5)
    assert stats["5"]["turn_min"] == 6.0
    assert stats["6"]["games_reached"] == 0


def test_final_state_summary_distribution():
    rows = [frame(0, 5, tier=3, atk=10, hp=10), frame(1, 5, tier=6, atk=100, hp=100),
            frame(2, 5, tier=6, atk=50, hp=50)]
    summary = final_state_summary(rows)
    assert summary["games"] == 3
    assert summary["tier_p50"] == 6.0
    assert summary["tier_ge5_rate"] == pytest.approx(2 / 3)
    assert summary["body_p50"] == 100.0     # body: 20 / 200 / 100 -> 中位 100
    assert summary["tier_counts"]["6"] == 2


def test_select_best_games_prefers_rank_then_body_then_seed():
    rows = [
        {"rank": 2, "final_body": 500, "seed": 5},
        {"rank": 1, "final_body": 100, "seed": 9},
        {"rank": 1, "final_body": 900, "seed": 3},
        {"rank": 1, "final_body": 900, "seed": 2},
    ]
    best = select_best_games(rows, limit=3, order="rank")
    assert [r["seed"] for r in best] == [2, 3, 9]
    power = select_best_games(rows, limit=2, order="body")
    assert [r["seed"] for r in power] == [2, 3]
    with pytest.raises(ValueError):
        select_best_games(rows, limit=1, order="unknown")


# ── 协议 ─────────────────────────────────────────────────────────────────
def test_seats_alternate_each_game():
    assert seats_for_game(0) == {"A": 0, "B": 1}
    assert seats_for_game(1) == {"A": 1, "B": 0}
    assert seats_for_game(2) == {"A": 0, "B": 1}


def test_protocol_sha_is_stable_and_rule_sensitive(tmp_path: Path):
    cfg = base_cfg()
    audits = {"A": model_audit("model-a", "aa" * 32), "B": model_audit("model-b", "bb" * 32)}
    kwargs = dict(fingerprint={"engine": "x"}, source_sha="s" * 64, model_audits=audits,
                  runner_sha="r" * 64)
    first = build_protocol(cfg, **kwargs)
    second = build_protocol(base_cfg(), **kwargs)
    assert protocol_sha256(first) == protocol_sha256(second)
    cfg_cap = base_cfg(move_cap=3)
    assert protocol_sha256(build_protocol(cfg_cap, **kwargs)) != protocol_sha256(first)
    cfg_games = base_cfg(games=64)
    assert protocol_sha256(build_protocol(cfg_games, **kwargs)) != protocol_sha256(first)
    # CPU 与 GPU 不是同一个可复现协议（greedy argmax 可能在浮点并列处翻转）
    cpu = build_protocol(cfg, runtime={"torch": "2.10.0", "device_type": "cpu"}, **kwargs)
    gpu = build_protocol(cfg, runtime={"torch": "2.10.0", "device_type": "cuda"}, **kwargs)
    assert protocol_sha256(cpu) != protocol_sha256(gpu)
    assert protocol_sha256(build_protocol(cfg, **kwargs)) != protocol_sha256(cpu)


def test_load_config_rejects_same_weights(tmp_path: Path):
    cfg = base_cfg()
    cfg["models"]["B"]["file_sha256"] = cfg["models"]["A"]["file_sha256"]
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="相同"):
        load_config(path)


def test_load_config_requires_rules(tmp_path: Path):
    cfg = base_cfg()
    cfg.pop("max_actions_per_turn")
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="max_actions_per_turn"):
        load_config(path)


def test_real_repo_config_is_the_approved_duel():
    path = Path(__file__).resolve().parents[1] / "config/eval-duel-7357-vs-10004.json"
    cfg = load_config(path)
    assert cfg["games"] == 128
    assert cfg["max_actions_per_turn"] == 80
    assert cfg["move_cap"] == 5
    assert cfg["move_repeat_limit"] == 2
    assert cfg["models"]["A"]["id"] == "entity16m-nus-7357"
    assert cfg["models"]["B"]["id"] == "entity16m-p50-10004-820m"


def test_keyword_decode():
    assert keyword_names(0) == []
    assert keyword_names(1 | 4) == ["TAUNT", "WINDFURY"]
    assert "DIVINE_SHIELD" in keyword_names(1 << 1)


# ── 报告门禁 ─────────────────────────────────────────────────────────────
def _report(run: dict, cfg: dict | None = None) -> dict:
    cfg = cfg or base_cfg()
    audits = {"A": model_audit("model-a", "aa" * 32), "B": model_audit("model-b", "bb" * 32)}
    protocol = build_protocol(cfg, fingerprint={}, source_sha="s" * 64,
                              model_audits=audits, runner_sha="r" * 64)
    return build_report(cfg=cfg, protocol=protocol, audits=audits, run=run,
                        best_k=2, power_k=1)


def test_report_is_valid_for_clean_run():
    report = _report(synthetic_run())
    assert report["valid"], report["reasons"]
    assert report["results"]["A"]["rank"]["games"] == 2
    assert report["paired"]["pairs"] == 2
    assert report["results"]["A"]["best_games"]
    assert report["results"]["A"]["turns"]["1"]["body_p50"] == 8.0
    markdown = render_markdown(report)
    assert "配对评测报告" in markdown and "野猪玩家" in markdown


@pytest.mark.parametrize("kwargs,needle", [
    (dict(illegal=1), "非法动作"),
    (dict(truncated=1), "截断"),
    (dict(engine_bad=1), "引擎非正常结束"),
])
def test_report_rejects_dirty_run(kwargs, needle):
    report = _report(synthetic_run(**kwargs))
    assert not report["valid"]
    assert any(needle in reason for reason in report["reasons"])


def test_report_rejects_wrong_game_count():
    cfg = base_cfg(games=4)
    report = _report(synthetic_run(games=2), cfg=cfg)
    assert not report["valid"]
    assert any("完成局数" in reason for reason in report["reasons"])
