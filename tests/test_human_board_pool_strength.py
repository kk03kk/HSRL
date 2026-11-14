"""``human_board_pool`` 的绝对强度刻度（动态难度课程的基础设施）。

只测**合成池**（不读真实战团库、不依赖引擎），保证 CI 里能跑：
- 分位查询在稀疏高回合要借用相邻回合（T18 只有 4 条真人样本时不拿噪声当难度目标）；
- 按强度选板要能在窗口里找到接近目标的**真实**战团，且不缩放身材。
"""

from __future__ import annotations

from hsbg.az.train.human_board_pool import HumanBoard, HumanBoardPool


def _pool() -> HumanBoardPool:
    """turn 10 有 100 条（强度 100..199）、turn 12 有 40 条（400..439）、turn 18 只有 3 条。"""
    p = HumanBoardPool()
    for i in range(100):
        p.index.setdefault(10, []).append(
            HumanBoard(turn=10, strength=100 + i, board=[[1, 50, 50 + i]]))
    for i in range(40):
        p.index.setdefault(12, []).append(
            HumanBoard(turn=12, strength=400 + i, board=[[1, 200, 200 + i]]))
    for i in range(3):
        p.index.setdefault(18, []).append(
            HumanBoard(turn=18, strength=9000 + i, board=[[1, 4500, 4500 + i]]))
    p.kept = 143
    return p


def test_strength_quantile_is_monotone_and_in_range():
    p = _pool()
    lo = p.strength_quantile(10, 0.10)
    mid = p.strength_quantile(10, 0.50)
    hi = p.strength_quantile(10, 0.90)
    assert lo is not None and mid is not None and hi is not None
    assert lo <= mid <= hi
    assert 100 <= lo and hi <= 199


def test_strength_quantile_borrows_neighbour_when_bucket_is_sparse():
    """T18 只有 3 条 < min_n=30：必须借用样本足够的回合，而不是用 n=3 的分位。"""
    p = _pool()
    v = p.strength_quantile(18, 0.5)
    assert v is not None and v < 1000, f"should borrow a dense bucket, got {v}"
    # 关掉样本下限时才会用 T18 自己的 3 条
    v_tail = p.strength_quantile(18, 0.5, min_n=0)
    assert v_tail is not None and v_tail > 8000


def test_curve_band_bounds_covers_window():
    """分段的强度上下限取"代表回合 ±span"内所有回合的 min(p10)..max(p90)。

    只用代表回合那一张表会把控制器卡住：13 回合真正可用的强度下界其实是 11 回合的 p10。
    """
    from hsbg.az.train.human_curve import HumanCurve

    curve = HumanCurve({"curve": {
        "11": {"p10": 400, "p50": 1900, "p90": 8800},
        "13": {"p10": 1500, "p50": 8500, "p90": 33000},
        "15": {"p10": 2700, "p50": 22000, "p90": 84000},
    }})
    assert curve.ok
    lo, hi = curve.band_bounds("late", span=2)
    assert lo == 400.0        # 来自 11 回合的 p10
    assert hi == 84000.0      # 来自 15 回合的 p90
    assert curve.quantile(13, 8500) == 0.5
    assert curve.p50(13) == 8500.0


def test_sample_at_strength_returns_real_board_near_target():
    p = _pool()
    pick = p.sample_at_strength_pick(12, 420)
    assert pick is not None
    assert abs(pick.strength - 420) <= max(1.0, 420 * 0.08)
    # 返回的是库里那条战团的**原样**身材，不做任何缩放
    assert pick.board == [[1, 200, pick.board[0][2]]]


def test_sample_at_strength_widens_window_for_high_turns():
    """T18 目标 160（远低于该回合分布）→ 只能借用更早回合的真实战团。"""
    p = _pool()
    pick = p.sample_at_strength_pick(18, 160)
    assert pick is not None
    assert pick.turn <= 12
    assert abs(pick.strength - 160) <= 40


def test_effective_quantile_roundtrip():
    p = _pool()
    q = p.effective_quantile(10, 150)
    assert q is not None and 0.3 < q < 0.7


def test_sample_at_strength_on_empty_pool_is_none():
    p = HumanBoardPool()
    assert p.strength_quantile(10, 0.5) is None
    assert p.sample_at_strength_pick(10, 100.0) is None
