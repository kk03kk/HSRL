"""``tools/accept_gap_panel.py`` 验收器的 fail-closed 行为测试。

对应审计 P2-7：旧实现在 ``pass=None``（"未评估"）时既不算通过也不算失败，于是
**四项检查全 SKIP 也能 overall=PASS**；而且完全没有 ``docs/EVAL_PROTOCOL.md`` §5
第 0 步（illegal/truncated）的有效性门。
"""

from __future__ import annotations

from tools import accept_gap_panel as A


def _rank(mean_rank=4.5, top4=0.5, illegal=0, truncated=0, *, seeds=True):
    r = {
        "games": 40,
        "mean_rank": mean_rank,
        "top4_rate": top4,
        "win_rate": 0.125,
        "survive_t12": 0.7,
        "survive_t14": 0.35,
        "max_turn_p50": 13,
        "illegal_actions": illegal,
        "truncated_games": truncated,
        "rank_sequence": [1, 2, 3, 4] * 10,
    }
    if seeds:
        r["seed_sequence"] = list(range(930000, 930040))
    return r


def _model(rank=None, *, strength=None, tier=None, boards=True):
    return {
        "rank": rank or _rank(),
        "board": {"strength_p50": strength or {"8": 200.0, "12": 1400.0,
                                               "14": 3000.0, "16": 5000.0}},
        "tempo": {"tier_mean": tier or {"6": 4.0, "8": 5.0, "10": 6.0}},
        "coverage": ({"any_core_by_tier": {"S": 0.5}, "assembled_at_least": {"2": 0.6}}
                     if boards else None),
    }


def _judge(cand=None, base=None):
    return A.judge(cand or _model(), base or _model(), human_p50={})


# ────────────────────────── 第 0 步：有效性门 ──────────────────────────
def test_illegal_actions_blocks_acceptance():
    out = _judge(cand=_model(_rank(illegal=2)))
    assert out["validity_illegal_actions"]["pass"] is False
    assert out["overall"]["pass"] is False
    assert "validity_illegal_actions" in out["overall"]["missing_required"]


def test_truncated_games_blocks_acceptance():
    out = _judge(cand=_model(_rank(truncated=1)))
    assert out["validity_truncated_games"]["pass"] is False
    assert out["overall"]["pass"] is False


# ────────────────────────── None（未评估）不再等于通过 ──────────────────────────
def test_missing_coverage_does_not_count_as_pass():
    """旧实现：coverage.pass=None → 不在 failed 里 → overall PASS。"""
    out = _judge(cand=_model(boards=False), base=_model(boards=False))
    assert out["coverage"]["pass"] is None
    assert out["overall"]["pass"] is False
    assert "coverage" in out["overall"]["not_evaluated"]
    assert "coverage" in out["overall"]["missing_required"]


def test_overall_lists_required_checks():
    # 候选在 T12/T14/T16 各提升 ≥5% 且名次不退 → 应当通过
    cand = _model(strength={"8": 200.0, "12": 1600.0, "14": 3400.0, "16": 5800.0})
    out = _judge(cand=cand)
    assert set(out["overall"]["required"]) == set(A.REQUIRED_CHECKS)
    assert out["overall"]["pass"] is True          # 全部达标时能通过
    assert out["overall"]["not_evaluated"] == []


# ────────────────────────── 配对失效不再静默降级 ──────────────────────────
def test_unusable_pairing_fails_instead_of_using_point_estimate():
    """rank_sequence 缺失 → 旧实现退回 d_rank<=0 的点估计（可能放行）。"""
    bad = _model(_rank())
    bad["rank"]["rank_sequence"] = None
    out = _judge(cand=bad)
    assert out["rank"]["paired_usable"] is False
    assert out["rank"]["pass"] is False
    assert out["overall"]["pass"] is False


def test_short_pairing_fails():
    short = _model(_rank())
    short["rank"]["rank_sequence"] = [1, 2, 3]     # 有效对 < 20
    out = _judge(cand=short)
    assert out["rank"]["paired_usable"] is False
    assert out["rank"]["pass"] is False


# ────────────────────────── seed 配对 ──────────────────────────
def test_paired_delta_pairs_by_seed_when_available():
    base = _model(_rank())
    cand = _model(_rank())
    # 打乱候选的局序但保持 seed 对齐：按 seed 配对后差值应为 0
    order = list(range(40))[::-1]
    cand["rank"]["rank_sequence"] = [base["rank"]["rank_sequence"][i] for i in order]
    cand["rank"]["seed_sequence"] = [base["rank"]["seed_sequence"][i] for i in order]
    st = A.paired_rank_delta(cand["rank"]["rank_sequence"],
                             base["rank"]["rank_sequence"],
                             a_seeds=cand["rank"]["seed_sequence"],
                             b_seeds=base["rank"]["seed_sequence"])
    assert st["pairing"] == "seed"
    assert st["mean_delta"] == 0.0


def test_paired_delta_falls_back_to_index():
    st = A.paired_rank_delta([1, 2] * 15, [1, 2] * 15)
    assert st["pairing"] == "index"
