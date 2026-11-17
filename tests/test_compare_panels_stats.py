"""``tools/compare_panels.py`` 的统计正确性测试。

对应审计 P1-6（按 seed 配对 vs 按下标）、P2-5（共同存活子集 vs 条件中位数）、
P0-2（协议校验）：这些都是"数字看起来正常但结论错了"的类别，必须有回归。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import compare_panels as C

PROTO = "c" * 64


def _model(name: str, ranks, seeds=None, *, strengths=None,
           illegal=0, truncated=0) -> dict:
    turns = sorted({int(t) for d in (strengths or {}).values() for t in d})
    curves = {str(t): {"n": 100, "strength_p50": 100.0 * t, "tier_mean": 5.0}
              for t in turns}
    m = {
        "name": name,
        "outcome": {"mean_rank": sum(r for r in ranks if r) / max(
                        1, sum(1 for r in ranks if r)),
                    "top4_rate": 0.5, "win_rate": 0.1,
                    "illegal_actions": illegal, "truncated_games": truncated,
                    "decisions": 1000, "decisions_per_game": 10.0},
        "rank_sequence": list(ranks),
        "curves": curves,
        "strength_by_seed": strengths or {},
    }
    if seeds is not None:
        m["seed_sequence"] = list(seeds)
    return m


def _panel(models, *, sha=PROTO) -> dict:
    doc = {"models": models, "panel": {"games": 8}}
    if sha:
        doc["protocol"] = {"protocol_sha256": sha}
    return doc


# ────────────────────────── P1-6：seed 配对 ──────────────────────────
def test_pairing_by_seed_is_order_independent():
    """候选的局序被打乱时，按 seed 配对必须与按序一致；按下标则会错。"""
    a = _model("A", [1, 2, 3, 4], seeds=[10, 11, 12, 13])
    b = _model("B", [4, 3, 2, 1], seeds=[13, 12, 11, 10])   # 逆序
    stat = C.paired_rank_diff(a["rank_sequence"], b["rank_sequence"],
                              rounds=200,
                              a_seeds=a["seed_sequence"], b_seeds=b["seed_sequence"])
    assert stat["pairing"] == "seed"
    assert stat["n"] == 4
    assert stat["ties"] == 4          # 每个 seed 两名次相同 → 全平
    assert stat["diff"] == 0.0


def test_pairing_by_seed_detects_shifted_missing_shard():
    """模拟"缺中间分片"：候选少了一个中间 seed 段 ⇒ 按下标会得到假差值，按 seed 不会。"""
    a_ranks = [1, 1, 1, 1, 8, 8, 8, 8]
    a_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    b_ranks = [8, 8, 8, 8]                # 候选只有后一半，且与 a 的同 seed 名次相反
    b_seeds = [4, 5, 6, 7]
    stat = C.paired_rank_diff(a_ranks, b_ranks, rounds=200,
                              a_seeds=a_seeds, b_seeds=b_seeds)
    assert stat["pairing"] == "seed"
    assert stat["n"] == 4                 # 只用交集
    assert stat["diff"] == 0.0            # seed 4..7: 8 vs 8 → 差 0
    assert stat["a_only"] == 4 and stat["b_only"] == 0
    # 按下标会拿 a[0..3]=1 减 b[0..3]=8 → 假的 -7
    naive = C.paired_rank_diff(a_ranks, b_ranks, rounds=50)
    assert naive["pairing"] == "index"
    assert naive["diff"] == pytest.approx(-7.0)


def test_pairing_falls_back_to_index_with_warning_flag():
    a = _model("A", [1, 2], )
    b = _model("B", [2, 1], )
    stat = C.paired_rank_diff(a["rank_sequence"], b["rank_sequence"], rounds=50)
    assert stat["pairing"] == "index"


def test_verdict_mentions_pairing_mode():
    a = _model("A", [1, 2, 3, 4], seeds=[0, 1, 2, 3])
    b = _model("B", [2, 1, 4, 3], seeds=[0, 1, 2, 3])
    models = {"A": a, "B": b}
    res = C.verdict(models, "A", "B", 0.05, 8, 0.05,
                    rows=C.per_turn(models, "A", "B", [1]))
    assert "按 seed 配对" in " ".join(res["reasons"])


# ────────────────────────── P2-5：共同存活子集 M1 ──────────────────────────
def test_per_turn_reports_paired_subset_median():
    """冠军活得更久 ⇒ 条件中位数会被拉低；共同存活子集必须不受影响。"""
    champ = _model("A", [1, 2], seeds=[0, 1], strengths={
        0: {"3": 30.0, "4": 40.0},
        1: {"3": 10.0, "4": 10.0},          # 弱局也活到 T4
    })
    cand = _model("B", [1, 2], seeds=[0, 1], strengths={
        0: {"3": 30.0, "4": 40.0},
        1: {"3": 10.0},                     # 弱局 T4 前就死了
    })
    models = {"A": champ, "B": cand}
    rows = C.per_turn(models, "A", "B", [3, 4])
    t4 = next(r for r in rows if r["turn"] == 4)
    # 条件中位数：冠军 25、候选 40（候选的 T4 只剩强局 → 看起来更强）
    assert t4["A"]["m1"] == 400.0 and t4["B"]["m1"] == 400.0   # curves 是构造值
    # 共同存活子集只有 seed 0 → 两者都 40
    assert t4["paired"]["n"] == 1
    assert t4["paired"]["A"] == 40.0 and t4["paired"]["B"] == 40.0


def test_verdict_prefers_paired_subset_for_m1_regression():
    """条件中位数说"退化 12%"，共同存活子集说"没退化" ⇒ 不得判 SUSPICIOUS。"""
    champ = _model("A", [1, 2, 3, 4], seeds=[0, 1, 2, 3], strengths={
        0: {"9": 100.0}, 1: {"9": 100.0}, 2: {"9": 100.0}, 3: {"9": 100.0}})
    cand = _model("B", [1, 1, 1, 1], seeds=[0, 1, 2, 3], strengths={
        0: {"9": 100.0}, 1: {"9": 100.0}, 2: {"9": 100.0}, 3: {"9": 100.0}})
    # 人为把 curves 的 T9 条件中位数改成候选更低
    champ["curves"]["9"] = {"n": 400, "strength_p50": 100.0, "tier_mean": 6.0}
    cand["curves"]["9"] = {"n": 200, "strength_p50": 80.0, "tier_mean": 6.0}
    models = {"A": champ, "B": cand}
    rows = C.per_turn(models, "A", "B", [9])
    res = C.verdict(models, "A", "B", 0.05, 8, 0.05, rows=rows)
    assert res["m1_regressions"] == []
    assert res["m1_paired_subset_turns"] == 1


def test_verdict_invalid_on_illegal_or_truncated():
    a = _model("A", [1], seeds=[0], illegal=3)
    b = _model("B", [1], seeds=[0])
    res = C.verdict({"A": a, "B": b}, "A", "B", 0.05, 8, 0.05)
    assert res["verdict"] == "INVALID"


# ────────────────────────── P0-2：协议校验 ──────────────────────────
def test_compare_refuses_mixed_protocols(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(_panel([_model("A", [1], seeds=[0])])), encoding="utf-8")
    p2.write_text(json.dumps(_panel([_model("B", [2], seeds=[0])], sha="d" * 64)),
                  encoding="utf-8")
    with pytest.raises(SystemExit, match="不一致"):
        C.main([str(p1), str(p2), "--champion", "A", "--candidate", "B"])


def test_compare_refuses_missing_protocol_by_default(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(_panel([_model("A", [1], seeds=[0])], sha=None)),
                  encoding="utf-8")
    p2.write_text(json.dumps(_panel([_model("B", [2], seeds=[0])], sha=None)),
                  encoding="utf-8")
    with pytest.raises(SystemExit, match="protocol_sha256"):
        C.main([str(p1), str(p2), "--champion", "A", "--candidate", "B"])


def test_compare_legacy_flag_allows_old_panels(tmp_path, capsys):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(_panel([_model("A", [1, 2], seeds=[0, 1])], sha=None)),
                  encoding="utf-8")
    p2.write_text(json.dumps(_panel([_model("B", [2, 1], seeds=[0, 1])], sha=None)),
                  encoding="utf-8")
    rc = C.main([str(p1), str(p2), "--champion", "A", "--candidate", "B",
                 "--legacy-no-protocol"])
    assert rc == 0
    assert "UNVERIFIED" in capsys.readouterr().out
