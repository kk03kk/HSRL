"""``workflows/eval_human_gap.run_checkpoint`` 的回归测试（不需要引擎与 torch）。

覆盖审计里最重要的一条 P0：``2acfebf`` 把 ``PPOEvaluator`` 的构造删掉却留下
``if evaluator is None:`` ⇒ 所有**自家** checkpoint 都 ``UnboundLocalError``。
这条路径以前**没有任何测试**碰过，所以回归一整轮都没被发现。

同时覆盖：
* P0-2：面板/落盘文件必须带 ``protocol_sha256``；
* P0-3：多模型同进程时 ``rows-<name>.jsonl.gz`` / ``extra-<name>.json`` 不被覆盖，
  同时仍写一份旧名文件保证老合并脚本可用；
* 契约漂移（``--freeze-cap 0``）必须**直接拒绝出分**，而不是静默降级。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import eval_human_gap as G


# ────────────────────────── 造一个"能跑完"的假结果 ──────────────────────────
N_GAMES = 12          # `economy_curves` 有 "该回合局数 < 10 就跳过" 的门槛
N_TURNS = 3


def _fake_rows(n_games: int = N_GAMES, n_turns: int = N_TURNS):
    rows = []
    for gi in range(n_games):
        seed = 930000 + gi
        for turn in range(1, n_turns + 1):
            rows.append({
                "seed": seed, "turn": turn, "tier": 4, "gold": 5, "health": 30,
                "hand_n": 2, "actions": 6,
                "eot_attack": 10 * turn, "eot_board_hp": 10 * turn,
                "eot_board": [{"card_id": "BG26_135", "atk": 5, "hp": 5, "position": 0}],
                "board_stats": {"BG26_135": [5, 5]},
                "hero": "TEST_HERO", "trinkets": [],
            })
    return rows


_RANK_CYCLE = [1, 4, 8, 2, 3, 5, 6, 7, 1, 2, 3, 4]


def _fake_eval_result(n_games: int = N_GAMES):
    ranks = (_RANK_CYCLE * 4)[:n_games]
    return {
        "end_turn_economy": _fake_rows(n_games),
        "action_hist": {"0": 10, "3": 2},
        "rank_sequence": ranks,
        "seed_sequence": [930000 + i for i in range(n_games)],
        "mean_rank": sum(ranks) / len(ranks),
        "top4_rate": sum(1 for r in ranks if r <= 4) / len(ranks),
        "win_rate": sum(1 for r in ranks if r == 1) / len(ranks),
        "illegal_actions": 0,
        "truncated_games": 0,
        "decisions": 1234,
        "decisions_per_game": 1234 / n_games,
        "combat_exposure": [],
        "mirror": {"attempt": 100, "applied": 99, "missing": 1, "rejected": 0},
        "mirror_pending_resolved": 3,
        "mirror_skipped_ended": 0,
        "forced_tier_actions": 0,
    }


class _StubPolicy:
    name = "stub"

    def reset(self):
        pass


@pytest.fixture()
def patched(monkeypatch):
    """把引擎/权重相关的一切换掉，只留被测的编排逻辑。"""
    monkeypatch.setattr(G, "sha256_file", lambda p: "0" * 64)
    monkeypatch.setattr(G, "evaluate_policy", lambda policy, cfg: _fake_eval_result())
    monkeypatch.setattr(G, "make_eval_policy",
                        lambda spec, protocol=None, device="cpu": {
                            "policy": _StubPolicy(), "model": object(), "meta": {},
                            "kind": "own", "real_path": str(spec)})
    return monkeypatch


def _args(tmp_path: Path, **kw):
    base = dict(
        device="cpu", games=N_GAMES, seed=930000, max_steps=100,
        max_actions_per_turn=100, forbid_unfreeze=1, forbid_cancel=1,
        move_guard=1, move_cap=2, move_repeat_limit=2, freeze_cap=1,
        ban_trinkets=1, no_cancel_choose=0, tp_forbid_cancel=0,
        keep_start_board=1, forced_tier_plan="", forced_tier_prob=0.0,
        human_opponents="", human_opponent_sample="auto",
        mirror_snaps="", mirror_sample="p50band", mirror_strict=0,
        mirror_seats=7, mirror_p=1.0,
        dump_rows=1, out=str(tmp_path / "shard" / "panel.json"),
    )
    base.update(kw)
    (tmp_path / "shard").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(**base)


# ────────────────────────── P0-1：不再崩溃、结果结构完整 ──────────────────────────
def test_own_checkpoint_entry_is_built(patched, tmp_path):
    args = _args(tmp_path)
    entry = G.run_checkpoint("v1.3_r3", Path("artifacts/models/x/model.pt"), args)
    assert entry["name"] == "v1.3_r3"
    assert entry["games"] == N_GAMES
    assert entry["seed_start"] == 930000
    assert entry["seed_end"] == 930000 + N_GAMES - 1
    assert entry["outcome"]["illegal_actions"] == 0
    assert entry["rank_sequence"][:3] == [1, 4, 8]
    assert entry["seed_sequence"][:3] == [930000, 930001, 930002]
    # 过程刻度必须按 "先求和再取中位" 算（T1 = 10 + 10 = 20）
    assert entry["curves"]["1"]["strength_p50"] == 20
    assert entry["curves"]["1"]["n"] == N_GAMES


def test_field_stats_reports_injection_rate(patched, tmp_path):
    entry = G.run_checkpoint("m", Path("a.pt"), _args(tmp_path))
    fs = entry["field_stats"]
    assert fs["mirror_attempt"] == 100 and fs["mirror_applied"] == 99
    assert fs["mirror_apply_rate"] == 0.99
    assert fs["mirror_pending_resolved"] == 3


# ────────────────────────── P0-3：多模型不互相覆盖 ──────────────────────────
def test_dump_files_are_per_model_and_legacy_compatible(patched, tmp_path):
    args = _args(tmp_path)
    G.run_checkpoint("v1.2", Path("a.pt"), args)
    G.run_checkpoint("v1.3_r3", Path("b.pt"), args)
    shard = tmp_path / "shard"
    # 每个模型自己的文件都在，且内容各是各的（不被后者覆盖）
    for name in ("v1.2", "v1.3_r3"):
        p = shard / f"extra-{name}.json"
        assert p.is_file(), f"缺少 {p.name}"
        assert json.loads(p.read_text(encoding="utf-8"))["name"] == name
        assert (shard / f"rows-{name}.jsonl.gz").is_file()
    # 旧名文件仍在（兼容只认旧名的合并脚本），内容是最后一个模型
    legacy = json.loads((shard / "extra.json").read_text(encoding="utf-8"))
    assert legacy["name"] == "v1.3_r3"
    with gzip.open(shard / "rows.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.strip()) == N_GAMES * N_TURNS


def test_extra_json_carries_protocol_sha(patched, tmp_path):
    args = _args(tmp_path)
    G.run_checkpoint("v1.2", Path("a.pt"), args)
    extra = json.loads((tmp_path / "shard" / "extra-v1.2.json").read_text(encoding="utf-8"))
    assert len(extra["protocol_sha256"]) == 64
    assert extra["seed_sequence"][0] == 930000


# ────────────────────────── 契约漂移必须 fail-closed ──────────────────────────
@pytest.mark.parametrize("kw,needle", [
    ({"freeze_cap": 0}, "freeze_cap"),
    ({"forbid_cancel": 0}, "forbid_cancel"),
    ({"max_actions_per_turn": 40}, "max_actions_per_turn"),
    ({"ban_trinkets": 0}, "ban_trinkets"),
])
def test_contract_drift_refuses_to_score(patched, tmp_path, kw, needle):
    args = _args(tmp_path, **kw)
    with pytest.raises(ValueError, match=needle):
        G.run_checkpoint("x", Path("a.pt"), args)


# ────────────────────────── 对手场标签不许再说谎 ──────────────────────────
def test_opponent_label_is_truthful():
    snap = SimpleNamespace(human_opponents="", mirror_snaps="lib/x",
                           mirror_sample="p50band", mirror_strict=0,
                           mirror_seats=7, mirror_p=1.0)
    assert "snapshot_library" in G._opponent_label(snap)
    assert "engine_bot" in G._opponent_label(SimpleNamespace(
        human_opponents="", mirror_snaps="", mirror_seats=7))
    assert "human_board_pool" in G._opponent_label(SimpleNamespace(
        human_opponents="bank.jsonl", human_opponent_sample="auto",
        mirror_seats=7))


def test_safe_name_strips_path_separators():
    assert G._safe_name("v1.3_r3") == "v1.3_r3"
    assert G._safe_name("a/b:c") == "a_b_c"
    assert G._safe_name("") == ""
