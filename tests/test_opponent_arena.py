"""``opponent_arena``（闭环对手难度控制器）的契约测试。

纯标准库、纯数值：不依赖引擎 / 网络 / GPU，也不 import 仓库里任何重包。

覆盖：``band_of`` 边界与夹紧、band 中心插值与 bounds 夹紧、deadband、升降方向、
单步上限、bounds 夹紧、``min_combats`` 跳过、脏 stats 不抛且强度恒有限正数、
持久化 round-trip、损坏文件回退、reach 驱动前期强度、棘轮收紧与 floor、
``parse_spec`` 正常/非法输入、``enabled=False``、history 列名契约。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "hsbg" / "az" / "train" / "opponent_arena.py"

#: 摘要里必须出现的 history 列名（下游直接写曲线，不能改名）。
HISTORY_KEYS = (
    "arena_early_str", "arena_mid_str", "arena_late_str",
    "arena_early_win", "arena_mid_win", "arena_late_win",
    "arena_reach9", "arena_reach12", "arena_it", "arena_reach_target",
    "arena_ratchet",
)

INIT = {"early": 1000.0, "mid": 2000.0, "late": 3000.0}


def _load_by_path():
    """按文件路径加载模块：既验证「可单独 import」，也避开包 ``__init__`` 的 torch 依赖。"""
    name = "opponent_arena_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass 需要 cls.__module__ 已在 sys.modules 里
    spec.loader.exec_module(module)
    return module


oa = _load_by_path()


def band(combats: int, losses: int) -> dict:
    """按采集器契约造一个 band 统计（控制器只读 combats/losses）。"""
    return {"combats": combats, "losses": losses,
            "own_str_sum": 0, "own_str_n": 0, "opp_str_sum": 0, "opp_str_n": 0}


def stats(*, episodes: int = 40, reach: dict | None = None,
          early: dict | None = None, mid: dict | None = None,
          late: dict | None = None) -> dict:
    """按契约拼一个 stats dict；默认 reach 满员，缺席的 band 不参与（=> 该 band 不动）。"""
    if reach is None:
        reach = {str(oa.REACH_TURNS[0]): episodes, str(oa.REACH_TURNS[1]): episodes}
    bands = {name: value for name, value in
             (("early", early), ("mid", mid), ("late", late)) if value is not None}
    return {"episodes": episodes, "reach": dict(reach), "bands": bands}


def controller(**kw):
    """默认开启、bounds 为空（不夹紧，便于观察纯步长效果）的控制器。"""
    return oa.ArenaController(oa.ArenaSpec(enabled=True, **kw), dict(INIT))


# ── band_of ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("turn,expected", [
    (0, "early"), (1, "early"), (7, "early"), (8, "mid"), (11, "mid"),
    (12, "late"), (30, "late"), (999, "late"), (1000, "late"),
])
def test_band_of_boundaries_and_clamping(turn, expected):
    assert oa.band_of(turn) == expected


def test_band_of_tolerates_junk_turn():
    assert oa.band_of(-5) == "early"
    assert oa.band_of(None) == "early"
    assert oa.band_of(float("nan")) == "early"
    assert oa.band_of("7") == "early"  # 数字字符串仍按数值解析


# ── target_for_turn：中心代表值 + 线性插值 + 夹紧 ────────────────────
def test_target_at_band_centers_is_band_strength():
    c = controller()
    assert c.target_for_turn(1) == pytest.approx(1000.0)   # early 中心左侧 = 常数
    assert c.target_for_turn(4) == pytest.approx(1000.0)   # early 中心
    assert c.target_for_turn(13) == pytest.approx(3000.0)  # late 中心
    assert c.target_for_turn(30) == pytest.approx(3000.0)  # late 中心右侧 = 常数


def test_target_interpolates_linearly_between_band_centers():
    c = controller()
    # early 中心 4 -> mid 中心 9.5
    assert c.target_for_turn(7) == pytest.approx(1000.0 + 1000.0 * 3.0 / 5.5)
    # mid 中心 9.5 -> late 中心 13
    assert c.target_for_turn(12) == pytest.approx(2000.0 + 1000.0 * 2.5 / 3.5)


def test_target_is_monotone_in_turn():
    c = controller()
    values = [c.target_for_turn(turn) for turn in range(1, 31)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))


def test_target_clamped_to_band_bounds():
    c = controller(bounds={"mid": (50.0, 1500.0), "early": (5000.0, 9000.0)})
    assert c.target_for_turn(9) == pytest.approx(1500.0)   # 上界
    assert c.target_for_turn(1) == pytest.approx(5000.0)   # 下界
    # 非法 bounds 只记 note，不抛、不夹
    bad = controller(bounds={"mid": (10.0, 1.0), "late": ("x", 3.0)})
    assert bad.target_for_turn(9) == pytest.approx(1909.0909, rel=1e-6)
    assert bad.notes


def test_disabled_controller_never_targets():
    off = oa.ArenaController(oa.ArenaSpec(), dict(INIT))
    assert off.target_for_turn(1) is None
    assert off.target_for_turn(13) is None
    on = controller()
    assert on.target_for_turn(5) == pytest.approx(1000.0 + 1000.0 / 5.5)


# ── update：deadband / 方向 / 步长上限 / bounds ─────────────────────
def test_deadband_keeps_strength():
    c = controller()
    before = dict(c.state.strength)
    out = c.update(stats(mid=band(40, 17), late=band(40, 19)))  # win 0.575 / 0.525
    assert c.state.strength["mid"] == pytest.approx(before["mid"])
    assert c.state.strength["late"] == pytest.approx(before["late"])
    assert out["arena_mid_win"] == pytest.approx(0.575)
    assert out["arena_it"] == 1


def test_high_win_raises_strength():
    c = controller()
    out = c.update(stats(mid=band(40, 8), late=band(40, 20)))  # mid win 0.80
    assert c.state.strength["mid"] == pytest.approx(2000.0 * 1.08)
    assert out["arena_mid_str"] == pytest.approx(2000.0 * 1.08)


def test_low_win_lowers_strength():
    c = controller()
    out = c.update(stats(mid=band(40, 32), late=band(40, 20)))  # mid win 0.20
    assert c.state.strength["mid"] == pytest.approx(2000.0 * 0.92)
    assert out["arena_mid_str"] == pytest.approx(2000.0 * 0.92)


def test_single_step_never_exceeds_max_step():
    c = controller(eta=10.0, max_step=0.08)
    c.update(stats(mid=band(40, 0), late=band(40, 0)))  # 两个 band 都是 100% 胜率
    assert c.state.last["step"]["mid"] == pytest.approx(0.08)
    assert c.state.strength["mid"] == pytest.approx(2000.0 * 1.08)
    assert c.state.strength["late"] == pytest.approx(3000.0 * 1.08)
    tiny = controller(eta=10.0, max_step=0.02)
    tiny.update(stats(mid=band(40, 0), late=band(40, 20)))
    assert tiny.state.strength["mid"] == pytest.approx(2000.0 * 1.02)


def test_bounds_clamp_updates():
    c = controller(bounds={"mid": (1500.0, 2050.0)})
    c.update(stats(mid=band(40, 0), late=band(40, 20)))  # 想涨到 2160，被夹到 2050
    assert c.state.strength["mid"] == pytest.approx(2050.0)
    for _ in range(5):
        c.update(stats(mid=band(40, 40), late=band(40, 20)))  # 一路降，撞下限
    assert c.state.strength["mid"] == pytest.approx(1500.0)


def test_min_combats_skips_band():
    c = controller()
    out = c.update(stats(mid=band(5, 0), late=band(11, 0)))
    assert c.state.strength["mid"] == pytest.approx(2000.0)
    assert c.state.strength["late"] == pytest.approx(3000.0)
    assert out["arena_mid_win"] == -1.0
    assert out["arena_late_win"] == -1.0
    assert any("min_combats" in note for note in c.notes)


# ── 脏数据：不抛、强度恒有限正数 ─────────────────────────────────────
BAD_STATS = [
    {},
    None,
    [],
    {"episodes": "x"},
    {"episodes": float("nan")},
    {"episodes": float("inf")},
    {"bands": 3},
    {"bands": {"mid": None}},
    {"bands": {"mid": {"combats": 40, "losses": float("inf")}}},
    {"bands": {"mid": {"combats": float("nan"), "losses": 1}}},
    {"bands": {"mid": {"combats": 40, "losses": None}}},
    {"episodes": 40, "reach": "oops", "bands": {"early": band(40, 20)}},
    {"episodes": 40, "reach": {"9": "many", "12": None}},
    {"episodes": 40, "reach": {"9": float("nan")}, "bands": {"early": {"combats": 40}}},
]


@pytest.mark.parametrize("bad", BAD_STATS)
def test_bad_stats_never_raise_and_stay_finite(bad):
    c = controller()
    out = c.update(bad)
    assert set(HISTORY_KEYS) <= set(out)
    assert out["arena_it"] == 1
    for name in oa.BAND_NAMES:
        value = c.state.strength[name]
        assert math.isfinite(value) and value > 0.0
    assert c.notes  # 脏数据必须留下说明，不能静默跳过


def test_strength_stays_positive_under_hostile_spec():
    c = controller(eta=1000.0, max_step=5.0, deadband=float("nan"), bounds={"mid": (0.0, 0.0)})
    for _ in range(3):
        c.update(stats(mid=band(40, 40), late=band(40, 0)))
    for name in oa.BAND_NAMES:
        value = c.state.strength[name]
        assert math.isfinite(value) and value > 0.0


def test_incomplete_init_strength_is_backfilled():
    c = oa.ArenaController(oa.ArenaSpec(enabled=True), {"mid": 2000.0})
    assert c.state.strength == pytest.approx({"early": 2000.0, "mid": 2000.0, "late": 2000.0})
    assert c.notes
    assert c.target_for_turn(3) == pytest.approx(2000.0)
    junk = oa.ArenaController(oa.ArenaSpec(enabled=True),
                              {"early": 0.0, "mid": float("nan"), "late": "x"})
    assert all(math.isfinite(v) and v > 0.0 for v in junk.state.strength.values())


# ── 持久化 ──────────────────────────────────────────────────────────
def test_state_roundtrip_through_disk(tmp_path):
    path = str(tmp_path / "arena.json")
    c = oa.ArenaController(oa.ArenaSpec(enabled=True), dict(INIT), path=path)
    for _ in range(2):
        c.update(stats(mid=band(40, 8), late=band(40, 20)))
    assert pathlib.Path(path).exists()
    assert not pathlib.Path(path + ".tmp").exists()  # 原子写不留 tmp
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    assert raw["version"] == 1

    fresh = oa.ArenaController(oa.ArenaSpec(enabled=True),
                              {"early": 7.0, "mid": 8.0, "late": 9.0}, path=path)
    assert fresh.state.strength == pytest.approx(c.state.strength)
    assert fresh.state.iters == c.state.iters == 2
    assert fresh.state.reach_target == pytest.approx(c.state.reach_target)
    assert fresh.state.streak == c.state.streak
    assert fresh.state.ratchet_level == c.state.ratchet_level
    assert fresh.target_for_turn(9) == pytest.approx(c.target_for_turn(9))


def test_ratchet_state_survives_restart(tmp_path):
    path = str(tmp_path / "arena.json")
    c = oa.ArenaController(oa.ArenaSpec(enabled=True, ratchet_k=2), dict(INIT), path=path)
    for _ in range(2):
        c.update(stats(mid=band(40, 0), late=band(40, 0)))
    assert c.state.ratchet_level == 1
    assert c.state.reach_target == pytest.approx(0.85)

    again = oa.ArenaController(oa.ArenaSpec(enabled=True, ratchet_k=2), dict(INIT), path=path)
    assert again.state.reach_target == pytest.approx(0.85)
    assert again.state.ratchet_level == 1
    assert again.state.streak == 0


def test_missing_state_file_is_silent(tmp_path):
    c = oa.ArenaController(oa.ArenaSpec(enabled=True), dict(INIT),
                           path=str(tmp_path / "nope.json"))
    assert c.state.iters == 0
    assert c.notes == []
    assert not (tmp_path / "nope.json").exists()  # 初始化只读不建


CORRUPT_PAYLOADS = [
    "{not json",
    "[]",
    '{"version": 1}',
    '{"strength": {"early": 1.0, "mid": 2.0}}',                 # 缺 late
    '{"strength": {"early": 1.0, "mid": 0.0, "late": 3.0}}',    # 非正数
    '{"strength": {"early": 1.0, "mid": "2.0", "late": 3.0}}',  # 非数值
    '{"strength": {"early": 1.0, "mid": 2.0, "late": null}}',
    '{"strength": "nope"}',
]


@pytest.mark.parametrize("payload", CORRUPT_PAYLOADS)
def test_corrupt_state_falls_back_to_init(tmp_path, payload):
    path = tmp_path / "arena.json"
    path.write_text(payload, encoding="utf-8")
    c = oa.ArenaController(oa.ArenaSpec(enabled=True), dict(INIT), path=str(path))
    assert c.state.strength == pytest.approx(INIT)  # 续训难度退回 init 而不是崩
    assert c.notes
    out = c.update(stats(mid=band(40, 8), late=band(40, 20)))  # 继续训练并自愈
    assert out["arena_it"] == 1
    healed = json.loads(path.read_text(encoding="utf-8"))
    assert healed["strength"]["mid"] == pytest.approx(2160.0)


def test_save_failure_is_recorded_not_raised(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")  # 让父目录是个文件 -> 读写都会失败
    c = oa.ArenaController(oa.ArenaSpec(enabled=True), dict(INIT),
                           path=str(blocker / "arena.json"))
    out = c.update(stats(mid=band(40, 8), late=band(40, 20)))
    assert out["arena_it"] == 1
    assert any(("读取" in note or "写入" in note) for note in c.notes)


# ── 前期段：reach 事件驱动 ──────────────────────────────────────────
def test_reach_target_drives_early_strength_down_and_up():
    low = controller()
    out = low.update(stats(episodes=20, reach={"9": 6, "12": 2}))
    assert low.state.strength["early"] == pytest.approx(1000.0 * 0.92)  # 活得太少 => 降 8%
    assert out["arena_reach9"] == pytest.approx(0.3)
    assert out["arena_reach12"] == pytest.approx(0.1)

    high = controller()
    high.update(stats(episodes=20, reach={"9": 20, "12": 20}))
    assert high.state.strength["early"] == pytest.approx(1000.0 * 1.06)  # eta*0.1 < 8% 上限


def test_early_band_uses_reach_not_win_rate():
    c = controller()
    # 前期战斗全败，但到达率达标：early 只按 reach 判定，所以强度反而升
    c.update(stats(episodes=20, reach={"9": 20, "12": 20}, early=band(40, 40)))
    assert c.state.strength["early"] == pytest.approx(1000.0 * 1.06)
    assert c.state.last["win"]["early"] == pytest.approx(0.0)


def test_reach_missing_or_zero_episodes_skips_early():
    c = controller()
    c.update(stats(episodes=0))
    assert c.state.strength["early"] == pytest.approx(1000.0)
    c.update({"episodes": 40, "bands": {}})
    assert c.state.strength["early"] == pytest.approx(1000.0)


# ── 棘轮：连续达标收紧前期目标，但不低于 floor ──────────────────────
def test_ratchet_tightens_reach_target_until_floor():
    c = controller(ratchet_k=3, reach_target=0.9, reach_floor=0.8)
    ok = lambda: stats(mid=band(40, 0), late=band(40, 0))  # noqa: E731 - 双 100% 胜率 + reach 满员

    c.update(ok())
    c.update(ok())
    assert c.state.streak == 2 and c.state.ratchet_level == 0
    c.update(ok())
    assert c.state.ratchet_level == 1
    assert c.state.reach_target == pytest.approx(0.85)
    assert c.state.streak == 0

    for _ in range(6):
        c.update(ok())
    assert c.state.ratchet_level == 2
    assert c.state.reach_target == pytest.approx(0.80)  # 被 floor 兜住，不会更低
    assert c.state.reach_target >= c.spec.reach_floor


def test_ratchet_requires_both_bands_and_reach():
    c = controller(ratchet_k=2)
    c.update(stats(mid=band(40, 0), late=band(40, 20)))  # late 0.5 未超过 0.55
    assert c.state.streak == 0
    c.update(stats(mid=band(40, 0), late=band(40, 0), reach={"9": 0, "12": 0}))
    assert c.state.streak == 0
    assert c.state.ratchet_level == 0
    assert c.state.reach_target == pytest.approx(0.9)


def test_ratchet_note_is_recorded():
    c = controller(ratchet_k=1)
    c.update(stats(mid=band(40, 0), late=band(40, 0)))
    assert c.state.ratchet_level == 1
    assert any("棘轮" in note for note in c.notes)


# ── history 摘要契约 ────────────────────────────────────────────────
def test_history_keys_types_and_missing_values():
    c = controller()
    out = c.update(stats(episodes=20, reach={"9": 10, "12": 5},
                         early=band(40, 20), mid=band(40, 8)))
    assert set(HISTORY_KEYS) <= set(out)
    for key in ("arena_early_str", "arena_mid_str", "arena_late_str",
                "arena_early_win", "arena_mid_win", "arena_late_win",
                "arena_reach9", "arena_reach12", "arena_reach_target"):
        assert isinstance(out[key], float), key
    assert isinstance(out["arena_it"], int) and not isinstance(out["arena_it"], bool)
    assert isinstance(out["arena_ratchet"], int)
    assert out["arena_mid_win"] == pytest.approx(0.8)
    assert out["arena_late_win"] == -1.0     # bands 里没有 late -> 样本不足
    assert out["arena_early_win"] == pytest.approx(0.5)
    assert out["arena_reach9"] == pytest.approx(0.5)
    assert out["arena_reach12"] == pytest.approx(0.25)
    assert out["arena_it"] == 1
    assert out["arena_ratchet"] == 0


def test_last_records_round_inputs_and_steps():
    c = controller()
    c.update(stats(episodes=20, reach={"9": 10, "12": 5},
                   mid=band(40, 8), late=band(40, 32)))
    last = c.state.last
    assert set(last["win"]) == set(last["step"]) == set(last["strength"]) == set(oa.BAND_NAMES)
    assert last["win"]["mid"] == pytest.approx(0.8)
    assert last["step"]["mid"] == pytest.approx(0.08)
    assert last["step"]["late"] == pytest.approx(-0.08)
    assert last["reach_rate"] == pytest.approx(0.5)
    assert last["strength"]["mid"] == pytest.approx(2160.0)
    assert last["iters"] == 1


def test_summary_reports_live_state():
    c = controller()
    c.update(stats(mid=band(40, 8), late=band(40, 20)))
    summary = c.summary()
    assert summary["enabled"] is True and summary["iters"] == 1
    assert summary["strength"] == pytest.approx(c.state.strength)
    assert summary["reach_target"] == pytest.approx(0.9)
    assert summary["notes"] == list(c.notes)
    summary["strength"]["mid"] = -1.0  # 返回副本，不污染内部状态
    assert c.state.strength["mid"] > 0.0


# ── parse_spec ──────────────────────────────────────────────────────
def test_parse_spec_applies_known_fields():
    notes: list = []
    spec = oa.parse_spec(
        "enabled=1,eta=0.8,reach_target=0.88,min_combats=20,reach_turn=10,"
        "ratchet_k=5,deadband=0.1,max_step=0.05,reach_floor=0.75,reach_deadband=0.04",
        note=notes)
    assert spec.enabled is True
    assert spec.eta == pytest.approx(0.8)
    assert spec.reach_target == pytest.approx(0.88)
    assert spec.min_combats == 20 and spec.reach_turn == 10 and spec.ratchet_k == 5
    assert spec.deadband == pytest.approx(0.1)
    assert spec.max_step == pytest.approx(0.05)
    assert spec.reach_floor == pytest.approx(0.75)
    assert spec.reach_deadband == pytest.approx(0.04)
    assert notes == []
    assert oa.parse_spec("enabled=no").enabled is False
    assert oa.parse_spec("enabled=TRUE").enabled is True
    assert oa.parse_spec("enabled=false").enabled is False
    assert oa.parse_spec("enabled=yes").enabled is True


def test_parse_spec_ignores_dicts_unknowns_and_bad_values():
    notes: list = []
    spec = oa.parse_spec(
        "target_win=0.9,bounds=1,bogus=2,enabled=maybe,eta=abc,"
        "min_combats=2.5,deadband=,no_equals", note=notes)
    defaults = oa.ArenaSpec()
    assert spec.target_win == defaults.target_win  # dict 字段不从 kv 解析
    assert spec.bounds == defaults.bounds
    assert spec.enabled == defaults.enabled
    assert spec.eta == defaults.eta
    assert spec.min_combats == defaults.min_combats
    assert spec.deadband == defaults.deadband
    joined = " | ".join(notes)
    for token in ("target_win", "bounds", "bogus", "enabled", "eta",
                  "min_combats", "deadband", "no_equals"):
        assert token in joined, token


def test_parse_spec_empty_is_default():
    notes: list = []
    assert oa.parse_spec("", note=notes) == oa.ArenaSpec()
    assert notes == []
    assert oa.parse_spec(None).enabled is False
    assert oa.parse_spec("  ,  ").enabled is False


def test_spec_defaults_are_not_shared():
    first, second = oa.ArenaSpec(), oa.ArenaSpec()
    first.target_win["mid"] = 0.0
    first.bounds["mid"] = (1.0, 2.0)
    assert second.target_win["mid"] == pytest.approx(0.6)
    assert second.bounds == {}


# ── 硬约束：纯标准库、可单独 import ──────────────────────────────────
def test_module_imports_only_stdlib():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "json", "math", "os", "dataclasses", "typing"}, roots


def test_importable_through_package_path():
    module = pytest.importorskip("hsbg.az.train.opponent_arena")
    assert module.band_of(8) == "mid"
    assert module.ArenaSpec().enabled is False
    assert module.ArenaController(module.ArenaSpec(), dict(INIT)).target_for_turn(5) is None


def test_notes_are_deduplicated():
    c = controller()
    c.update({})
    after_first = len(c.notes)
    assert after_first > 0
    for _ in range(5):
        c.update({})
    assert len(c.notes) == after_first  # 同类说明去重，不随回合刷屏
    assert c.state.iters == 6
    assert len(c.notes) <= oa.MAX_NOTES
