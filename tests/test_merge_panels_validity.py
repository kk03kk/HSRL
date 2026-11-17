"""合并工具的正确性测试：``tools/merge_panels.py`` 与 ``tools/merge_panels_multi.py``。

对应审计 P0-3（多模型 `rows.jsonl.gz` 互相覆盖 / 有效性计数只覆盖 1/N 分片）、
P1-6（配对键没有 seed）、P0-2（合并结果必须带协议指纹）：
这些错误都不会让评测"报错"，只会让**数字静默变错**，所以必须有测试钉住。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tools import merge_panels as MP
from tools import merge_panels_multi as MPM

N_GAMES = 12
N_TURNS = 2
PROTO = "a" * 64


def _rows(seed0: int, n_games: int = N_GAMES) -> list:
    rows = []
    for gi in range(n_games):
        for turn in range(1, N_TURNS + 1):
            rows.append({
                "seed": seed0 + gi, "turn": turn, "tier": 5, "gold": 4,
                "health": 30, "hand_n": 1, "actions": 5,
                "eot_attack": 10 * turn, "eot_board_hp": 10 * turn,
                "eot_board": [{"card_id": "BG26_135", "atk": 10, "hp": 10,
                               "position": 0}],
                "board_stats": {"BG26_135": [10, 10]},
                "hero": "H", "trinkets": [],
            })
    return rows


def _extra(name: str, seed0: int, n_games: int = N_GAMES, *,
           illegal: int = 0, truncated: int = 0, decisions: int = 1000,
           sha: str = PROTO, with_seeds: bool = True) -> dict:
    ranks = ([1, 2, 3, 4, 5, 6, 7, 8] * 4)[:n_games]
    d = {
        "name": name,
        "outcome": {"mean_rank": sum(ranks) / n_games,
                    "top4_rate": 0.5, "win_rate": 0.125,
                    "illegal_actions": illegal, "truncated_games": truncated,
                    "decisions": decisions,
                    "decisions_per_game": decisions / n_games},
        "rank_sequence": ranks,
        "action_hist": {"0": 5, "3": 1},
        "combat_exposure": [],
        "games": n_games,
        "seed": seed0,
        "protocol_sha256": sha,
        "field_stats": {"mirror_attempt": 100, "mirror_applied": 100},
    }
    if with_seeds:
        d["seed_sequence"] = [seed0 + i for i in range(n_games)]
    return d


def _write_shard(root: Path, tag: str, models: dict, *,
                 with_seeds: bool = True, sha: str = PROTO) -> Path:
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    for i, (name, seed0) in enumerate(models.items()):
        with gzip.open(d / f"rows-{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in _rows(seed0):
                fh.write(json.dumps(r) + "\n")
        (d / f"extra-{name}.json").write_text(
            json.dumps(_extra(name, seed0, illegal=(i % 2), truncated=0,
                              decisions=1000 + i, sha=sha, with_seeds=with_seeds)),
            encoding="utf-8")
    return d


# ────────────────────────── tools/merge_panels.py ──────────────────────────
def test_multi_model_shards_are_merged_per_model(tmp_path):
    """P0-3 核心：一个分片里 2 个模型、共 2 个分片 → 合并结果必须有 2 个模型。"""
    s0 = _write_shard(tmp_path, "p0", {"vA": 930000, "vB": 940000})
    s1 = _write_shard(tmp_path, "p1", {"vA": 931000, "vB": 941000})
    doc = MP.merge([s0, s1], tmp_path / "panel.json")
    names = [m["name"] for m in doc["models"]]
    assert names == ["vA", "vB"]           # 不再是"只剩最后一个模型"
    for m in doc["models"]:
        assert m["outcome"]["games"] == 2 * N_GAMES
        assert len(m["rank_sequence"]) == 2 * N_GAMES
        assert len(m["seed_sequence"]) == 2 * N_GAMES
        assert m["seed_sequence"][N_GAMES] == (931000 if m["name"] == "vA" else 941000)


def test_validity_counters_are_summed_across_shards(tmp_path):
    """P0-3：``illegal_actions``/``truncated_games`` 必须跨分片求和（不是取第 0 片）。"""
    s0 = _write_shard(tmp_path, "p0", {"vA": 930000, "vB": 940000})
    s1 = _write_shard(tmp_path, "p1", {"vA": 931000, "vB": 941000})
    # vA 在 p1 里带 3 个非法动作
    (s1 / "extra-vA.json").write_text(
        json.dumps(_extra("vA", 931000, illegal=3, decisions=2000)), encoding="utf-8")
    doc = MP.merge([s0, s1], tmp_path / "panel.json")
    va = next(m for m in doc["models"] if m["name"] == "vA")
    # p0 的 vA 是第 0 个模型 → illegal=0；p1 是第 0 个 → 但被覆盖成 3
    assert va["outcome"]["illegal_actions"] == 3
    assert va["outcome"]["decisions"] == 1000 + 2000


def test_protocol_is_recorded_and_must_match(tmp_path):
    s0 = _write_shard(tmp_path, "p0", {"vA": 930000})
    s1 = _write_shard(tmp_path, "p1", {"vA": 931000}, sha="b" * 64)
    with pytest.raises(SystemExit, match="不一致"):
        MP.merge([s0, s1], tmp_path / "panel.json")
    s1b = _write_shard(tmp_path, "p1b", {"vA": 931000})
    doc = MP.merge([s0, s1b], tmp_path / "panel2.json")
    assert doc["protocol"]["protocol_sha256"] == PROTO


def test_legacy_shard_without_protocol_is_refused(tmp_path):
    s0 = _write_shard(tmp_path, "p0", {"vA": 930000}, sha="")
    with pytest.raises(SystemExit, match="protocol_sha256"):
        MP.merge([s0], tmp_path / "panel.json")


def test_missing_model_in_one_shard_is_refused(tmp_path):
    """缺模型不静默降级（否则会拿"少一个分片"的样本冒充全量）。"""
    s0 = _write_shard(tmp_path, "p0", {"vA": 930000, "vB": 940000})
    s1 = _write_shard(tmp_path, "p1", {"vA": 931000})
    with pytest.raises(SystemExit, match="缺失"):
        MP.merge([s0, s1], tmp_path / "panel.json")


def test_expect_games_guard_catches_missing_shard(tmp_path):
    s0 = _write_shard(tmp_path, "p0", {"vA": 930000})
    with pytest.raises(SystemExit, match="名次样本"):
        MP.merge([s0], tmp_path / "panel.json", expect_games=2 * N_GAMES)


def test_legacy_layout_single_shard_still_works(tmp_path):
    """老分片（只有 rows.jsonl.gz/extra.json）必须仍能读。"""
    d = tmp_path / "legacy"
    d.mkdir()
    with gzip.open(d / "rows.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in _rows(930000):
            fh.write(json.dumps(r) + "\n")
    (d / "extra.json").write_text(json.dumps(_extra("old", 930000)),
                                  encoding="utf-8")
    doc = MP.merge([d], tmp_path / "panel.json")
    assert [m["name"] for m in doc["models"]] == ["old"]


# ────────────────────────── tools/merge_panels_multi.py ──────────────────────────
def _sub_panel(name: str, seed0: int, *, proto=PROTO, with_seeds=False) -> dict:
    trace = []
    for gi in range(N_GAMES):
        for turn in range(1, N_TURNS + 1):
            trace.append({
                "seed": seed0 + gi, "turn": turn, "tier": 5, "gold": 4,
                "health": 30, "hand_n": 1, "actions": 5, "strength": 20 * turn,
                "board_stats": {"BG26_135": [10, 10]},
                "roster": [{"pos": 0, "card_id": "BG26_135", "atk": 10, "hp": 10}],
                "hero": "H", "trinkets": [],
            })
    entry = {
        "name": name,
        "games": N_GAMES,
        "outcome": {"mean_rank": 4.5, "top4_rate": 0.5, "win_rate": 0.125,
                    "illegal_actions": 2, "truncated_games": 1, "decisions": 500,
                    "decisions_per_game": 500 / N_GAMES, "games": N_GAMES},
        "rank_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
        "trace": trace,
        "action_mix_by_turn": {"1": {"games": N_GAMES, "mean": {"buy": 2.0},
                                     "used_rate": {"buy": 0.5},
                                     "mean_used": {"buy": 4.0}}},
        "combat_by_turn": {"1": {"combats": 10, "win": 0.5, "tie": 0.0,
                                 "damage_taken_mean": 4.0, "ghost": 0.0,
                                 "mirror_applied": 1.0}},
        "opponents_by_turn": {"1": {"n": 10, "mean": 100.0, "p50": 100.0,
                                    "p10": 90.0, "p90": 110.0}},
        "action_hist": {"0": 10},
    }
    if with_seeds:
        entry["seed_sequence"] = [seed0 + i for i in range(N_GAMES)]
    doc = {"models": [entry], "panel": {"games": N_GAMES, "seed": seed0}}
    if proto:
        doc["protocol"] = {"protocol_sha256": proto}
    return doc


def test_multi_merge_sums_validity_counters(tmp_path):
    """P0-3：旧实现取 parts[0]，于是 illegal/truncated 只覆盖第 0 个分片。"""
    p0 = tmp_path / "s0" / "panel.json"
    p1 = tmp_path / "s1" / "panel.json"
    for p, seed0 in ((p0, 930000), (p1, 931000)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_sub_panel("vA", seed0)), encoding="utf-8")
    doc = MPM.merge([p0, p1], tmp_path / "out.json")
    o = doc["models"][0]["outcome"]
    assert o["illegal_actions"] == 4          # 2 + 2，不是 2
    assert o["truncated_games"] == 2          # 1 + 1
    assert o["decisions"] == 1000
    assert o["games"] == 2 * N_GAMES
    assert len(doc["models"][0]["rank_sequence"]) == 2 * N_GAMES


def test_multi_merge_survival_schema_matches_single(tmp_path):
    """P0-3：``survival`` 必须是 {games,max_turn_p50,reached}，不是 {turn: count}。"""
    p = tmp_path / "s0" / "panel.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_sub_panel("vA", 930000)), encoding="utf-8")
    doc = MPM.merge([p], tmp_path / "out.json")
    sv = doc["models"][0]["survival"]
    assert set(sv) == {"games", "max_turn_p50", "reached"}
    assert sv["games"] == N_GAMES


def test_multi_merge_action_mix_is_exact(tmp_path):
    """``mean/used_rate/mean_used`` 三个口径都能从分片精确还原。"""
    p0 = tmp_path / "s0" / "panel.json"
    p1 = tmp_path / "s1" / "panel.json"
    for p, seed0, games, mean, rate in ((p0, 930000, 10, 2.0, 0.5),
                                        (p1, 931000, 30, 4.0, 1.0)):
        p.parent.mkdir(parents=True, exist_ok=True)
        docp = _sub_panel("vA", seed0)
        docp["models"][0]["action_mix_by_turn"]["1"] = {
            "games": games, "mean": {"buy": mean}, "used_rate": {"buy": rate},
            "mean_used": {"buy": 4.0}}
        p.write_text(json.dumps(docp), encoding="utf-8")
    doc = MPM.merge([p0, p1], tmp_path / "out.json")
    mix = doc["models"][0]["action_mix_by_turn"]["1"]
    assert mix["games"] == 40
    assert mix["mean"]["buy"] == pytest.approx((2.0 * 10 + 4.0 * 30) / 40, abs=0.01)
    assert mix["used_rate"]["buy"] == pytest.approx((0.5 * 10 + 1.0 * 30) / 40, abs=0.001)
    assert mix["mean_used"]["buy"] == pytest.approx(4.0, abs=0.01)


def test_multi_merge_combat_and_opponents(tmp_path):
    p0 = tmp_path / "s0" / "panel.json"
    p1 = tmp_path / "s1" / "panel.json"
    for p, seed0 in ((p0, 930000), (p1, 931000)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_sub_panel("vA", seed0)), encoding="utf-8")
    doc = MPM.merge([p0, p1], tmp_path / "out.json")
    cb = doc["models"][0]["combat_by_turn"]["1"]
    assert cb["combats"] == 20 and cb["win"] == pytest.approx(0.5)
    opp = doc["models"][0]["opponents_by_turn"]["1"]
    assert opp["n"] == 20 and opp["mean"] == pytest.approx(100.0)
    assert opp["p50"] == 100.0            # 两个分片一致 → 保留
    assert opp["p10"] is None             # 无法还原 → None（不编数）


def test_multi_merge_marks_legacy_without_protocol(tmp_path):
    p = tmp_path / "s0" / "panel.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_sub_panel("vA", 930000, proto=None)), encoding="utf-8")
    doc = MPM.merge([p], tmp_path / "out.json")
    assert doc["protocol"]["protocol_sha256"] is None
    assert doc["protocol"]["legacy"] is True


def test_multi_merge_rejects_protocol_mismatch(tmp_path):
    p0 = tmp_path / "s0" / "panel.json"
    p1 = tmp_path / "s1" / "panel.json"
    for p, sha in ((p0, "a" * 64), (p1, "b" * 64)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_sub_panel("vA", 930000, proto=sha)), encoding="utf-8")
    with pytest.raises(SystemExit, match="不一致"):
        MPM.merge([p0, p1], tmp_path / "out.json")


def test_multi_merge_carries_seed_sequence(tmp_path):
    p = tmp_path / "s0" / "panel.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_sub_panel("vA", 930000, with_seeds=True)),
                 encoding="utf-8")
    doc = MPM.merge([p], tmp_path / "out.json")
    assert doc["models"][0]["seed_sequence"][0] == 930000
