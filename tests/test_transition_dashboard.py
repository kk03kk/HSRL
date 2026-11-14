"""``tools/transition_dashboard.py`` 的回归：读盘容错、抽样、真人比值、棘轮判定、``--json``。

为什么测这些
------------
面板是"看训练有没有变强"的最后一道人眼入口，它出错的形态不是崩，而是**悄悄算错**
（把 -1 当 0、把抽样末行丢掉、把难度上涨读成棋力上涨）。所以用例集中在：
1. 脏数据不抛、不冒充 0；
2. ``--last`` / ``--every`` 的抽样边界（末行必须在）；
3. 嵌套 ``stat`` / ``stats`` / ``meta`` 与顶层的优先级；
4. 人曲线缺/坏时降级成 ``-`` 并告警；
5. 棘轮三态（上升 / 未上升 / 数据不足）与 2% 门槛；
6. ``--json`` 可被 ``json.loads``，表格列宽不会因超宽数字错位。

全部用 ``tmp_path`` 造临时 history，不触碰任何真实 run 目录。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.transition_dashboard import (  # noqa: E402
    _fmt_value, _get, _num, _text_width, _trend, main, read_history, resolve_history,
)

Row = Union[Dict[str, Any], str]


# ── 造数据 ─────────────────────────────────────────────────────────────────
def write_jsonl(path: Path, rows: Sequence[Row]) -> Path:
    """写临时 history：dict 走 ``json.dumps``，str 原样落盘（用来造坏行/半行）。"""
    lines = [row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
             for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_rows(count: int, mid_start: int = 500, step: int = 12) -> List[Dict[str, Any]]:
    """造 ``count`` 行"稳步变强"的 history；arena 强度可按 ``step`` 调快慢。"""
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        rows.append({
            "it": 100 + index,
            "elapsed": 600 + 120 * index,
            "tel_tier_6": 4.9, "tel_tier_8": 5.8, "tel_tier_10": 7.1,
            "tel_own_str_8": 300 + 6 * index,
            "tel_own_str_10": 1400 + 20 * index,
            "tel_own_str_12": 1434 + 21 * index,
            "tel_dmg_9": 21.4, "tel_dmg_12": 33.8,
            "tel_reach9": 0.412, "tel_reach12": 0.208,
            "rg_gain_10": 9.2, "rg_gain_11": 10.1,
            "comp_s_hits": 0.312, "rank_mean": 4.71,
            "arena_early_str": 400 + 3 * index,
            "arena_mid_str": mid_start + step * index,
            "arena_late_str": 730 + step * index,
            "arena_early_win": 0.612, "arena_mid_win": 0.503, "arena_late_win": 0.281,
            "arena_reach9": 0.405, "arena_reach12": 0.192, "arena_reach_target": 0.5,
            "approx_kl": 0.0117, "entropy": 1.4723, "anchor_kl": 0.0042,
            "loss": 0.0873, "adv_std": 0.6641,
        })
    return rows


def run_cli(capsys: pytest.CaptureFixture[str], *args: Any) -> tuple:
    """跑一次 CLI，返回 ``(退出码, stdout, stderr)``。"""
    code = main([str(arg) for arg in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(capsys: pytest.CaptureFixture[str], *args: Any) -> Dict[str, Any]:
    """跑 ``--json`` 并解析 stdout；退出码非 0 直接失败（附 stderr）。"""
    code, out, err = run_cli(capsys, *args, "--json")
    assert code == 0, err
    return json.loads(out)


# ── 取值：本层 / 子字典 / 脏值 ─────────────────────────────────────────────
def test_get_prefers_top_level_then_nested_in_order():
    """同一个键写了几个位置：本层优先，其次 stat → stats → meta。"""
    assert _get({"tel_own_str_12": 1434}, "tel_own_str_12") == 1434
    assert _get({"stat": {"tel_own_str_12": 1602}}, "tel_own_str_12") == 1602
    assert _get({"stats": {"tel_own_str_12": 1700}}, "tel_own_str_12") == 1700
    assert _get({"meta": {"tel_own_str_12": 1800}}, "tel_own_str_12") == 1800
    both = {"stat": {"tel_own_str_12": 1602}, "stats": {"tel_own_str_12": 1700},
            "meta": {"tel_own_str_12": 1800}}
    assert _get(both, "tel_own_str_12") == 1602
    assert _get(dict(both, tel_own_str_12=1434), "tel_own_str_12") == 1434
    # 本层留了 null 占位时不能让子字典里的真值被挡住
    assert _get({"tel_own_str_12": None, "meta": {"tel_own_str_12": 1800}},
                "tel_own_str_12") == 1800
    assert _get({}, "tel_own_str_12", "-") == "-"
    assert _get({"stat": None}, "tel_own_str_12", "-") == "-"


@pytest.mark.parametrize("value", [-1, None, "abc", "", "-1", "  ", float("nan"),
                                   float("inf"), float("-inf"), "-", [1, 2]])
def test_no_sample_values_are_missing_not_zero(value: Any):
    """``-1``/``null``/字符串/NaN 都是"无样本"：显示 ``-``，绝不当 0 平均进面板。"""
    assert _num(value) is None
    assert _fmt_value(_num(value), 3) == "-"


def test_zero_and_numeric_strings_are_real_values():
    """0 与数字字符串是真数据（0 命中率不等于没样本）。"""
    assert _num(0) == 0.0
    assert _fmt_value(_num(0), 3) == "0.000"
    assert _num("358") == 358.0
    assert _fmt_value(358.4, 0) == "358"


# ── 读盘：坏行只计数 ───────────────────────────────────────────────────────
def test_read_history_skips_bad_lines_and_keeps_line_numbers(tmp_path: Path):
    """坏 JSON / 非对象行 / 空行：只算进计数，好的行连物理行号一起留下。"""
    path = write_jsonl(tmp_path / "history.jsonl",
                       ['{"it": 1}', "", "这不是 JSON", "[1, 2, 3]", '{"it": 2, "half":',
                        '{"it": 3}'])
    rows, stats = read_history(path)
    assert stats == {"total": 5, "valid": 2, "bad": 3}
    assert [line_no for line_no, _ in rows] == [1, 6]
    assert [row["it"] for _, row in rows] == [1, 3]


def test_dirty_run_still_prints_tables_and_warns(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]):
    """掺了坏行的 run 也必须打完整表格，并提示跳过了几行。"""
    path = write_jsonl(tmp_path / "history.jsonl", [make_rows(2)[0], "坏行", '{"it": 2}'])
    code, out, _ = run_cli(capsys, path)
    assert code == 0
    assert "[对局结构]" in out and "[难度与健康]" in out and "[判读]" in out
    assert "警告" in out and "坏" in out


# ── 嵌套键 ─────────────────────────────────────────────────────────────────
def test_metrics_nested_in_stat_stats_meta_are_read(tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]):
    """指标藏在 stat / stats / meta 里也要能读到（多代写盘格式共存）。"""
    path = write_jsonl(tmp_path / "history.jsonl", [{
        "it": 7,
        "stat": {"tel_own_str_12": 1434, "tel_reach12": 0.208},
        "stats": {"arena_mid_str": 549},
        "meta": {"approx_kl": 0.0117},
    }])
    record = run_json(capsys, path)["records"][0]
    assert record["tel_own_str_12"] == 1434
    assert record["tel_reach12"] == 0.208
    assert record["arena_mid_str"] == 549
    assert record["approx_kl"] == 0.0117


def test_missing_and_sentinel_values_render_as_dash(tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]):
    """缺键、``-1``、``null``、字符串在表格里都是 ``-``，不是 0。"""
    path = write_jsonl(tmp_path / "history.jsonl", [{
        "it": 1, "tel_own_str_12": -1, "tel_dmg_9": None, "arena_mid_str": "缺",
        "tel_reach12": "null", "comp_s_hits": 0.312,
    }])
    record = run_json(capsys, path)["records"][0]
    for key in ("tel_own_str_12", "tel_dmg_9", "arena_mid_str", "tel_reach12"):
        assert record[key] is None, key
    assert record["comp_s_hits"] == 0.312
    _, out, _ = run_cli(capsys, path)
    row = next(line for line in out.splitlines() if line.strip().startswith("1 "))
    assert "-" in row and "0.000" not in row


# ── 抽样：--last / --every ─────────────────────────────────────────────────
def test_last_and_every_pick_the_expected_rows(tmp_path: Path,
                                               capsys: pytest.CaptureFixture[str]):
    """``--last 6 --every 3`` = 末 6 行里每 3 取 1，再补上末行。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(12))
    payload = run_json(capsys, path, "--last", 6, "--every", 3)
    assert payload["stats"]["window"] == 6
    assert payload["stats"]["shown"] == 3
    assert [record["it"] for record in payload["records"]] == [106, 109, 111]


def test_every_keeps_last_row_when_off_stride(tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]):
    """抽样步长对不齐时末行仍要出现，否则判读会停在中途的旧数据上。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(10))
    payload = run_json(capsys, path, "--last", 0, "--every", 4)
    assert [record["it"] for record in payload["records"]] == [100, 104, 108, 109]


def test_bad_sampling_arguments_exit_with_error(tmp_path: Path):
    """``--every 0`` 这种参数要报错退出，而不是悄悄按 1 处理。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(2))
    with pytest.raises(SystemExit):
        main([str(path), "--every", "0"])


# ── 真人曲线比值 ───────────────────────────────────────────────────────────
def test_human_curve_ratio_of_t8_is_two(tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]):
    """T8 自建板面 358 / 人 p50 179 = 2.00（表里两位小数，JSON 里是原值）。"""
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({"8": {"p50": 179, "p75": 266}}), encoding="utf-8")
    path = write_jsonl(tmp_path / "history.jsonl", [{"it": 1, "tel_own_str_8": 358}])
    payload = run_json(capsys, path, "--human-curve", curve)
    assert round(payload["records"][0]["ratio_8"], 2) == 2.0
    assert payload["records"][0]["ratio_10"] is None      # 曲线没有 T10 → 不硬算
    _, out, _ = run_cli(capsys, path, "--human-curve", curve)
    assert "2.00" in out and "警告" not in out


def test_broken_human_curve_warns_and_shows_dash(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]):
    """人曲线是坏 JSON：一行警告 + 比值列全 ``-``，退出码仍是 0。"""
    curve = tmp_path / "curve.json"
    curve.write_text("not json at all", encoding="utf-8")
    path = write_jsonl(tmp_path / "history.jsonl", [{"it": 1, "tel_own_str_8": 358}])
    code, out, _ = run_cli(capsys, path, "--human-curve", curve)
    assert code == 0
    assert "警告" in out and "human-curve" in out
    assert "2.00" not in out
    payload = run_json(capsys, path, "--human-curve", curve)
    assert payload["human_curve"]["error"]
    assert payload["records"][0]["ratio_8"] is None


def test_missing_human_curve_file_warns(tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]):
    """人曲线文件不存在同样是"警告 + 降级"，不该抛 FileNotFoundError。"""
    path = write_jsonl(tmp_path / "history.jsonl", [{"it": 1, "tel_own_str_8": 358}])
    code, out, _ = run_cli(capsys, path, "--human-curve", tmp_path / "nope.json")
    assert code == 0 and "警告" in out and "读不了" in out


# ── 结构化输出 ─────────────────────────────────────────────────────────────
def test_json_output_is_parsable_and_has_expected_keys(tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]):
    """``--json`` 必须是能被其它脚本消费的整块 JSON（警告走 stderr）。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(6))
    code, out, err = run_cli(capsys, path, "--json")
    assert code == 0 and err == ""
    payload = json.loads(out)
    assert {"run", "history", "stats", "records", "verdict", "ratchet",
            "human_curve", "warnings"} <= set(payload)
    assert payload["stats"] == {"total": 6, "valid": 6, "bad": 0, "window": 6, "shown": 6}
    assert len(payload["records"]) == 6
    item = payload["verdict"]["metrics"]["tel_own_str_12"]
    assert (item["first"], item["last"], item["delta"]) == (1434, 1539, 105)
    assert item["dir"] == "up"
    assert payload["verdict"]["metrics"]["arena_mid_str"]["dir"] == "up"


# ── 难度棘轮：三态 ─────────────────────────────────────────────────────────
def test_ratchet_reports_up(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """中/后期 arena 强度连续 5 行上升且涨幅 > 2%：棘轮上升。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(6))
    assert run_json(capsys, path)["ratchet"]["state"] == "up"
    _, out, _ = run_cli(capsys, path)
    assert "难度棘轮：上升中（↑）" in out


def test_ratchet_reports_flat_when_not_monotone(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]):
    """中间掉过一次就不是"连续上升"：判未上升。"""
    rows = make_rows(6)
    rows[-2]["arena_mid_str"] -= 40
    path = write_jsonl(tmp_path / "history.jsonl", rows)
    assert run_json(capsys, path)["ratchet"]["state"] == "flat"
    _, out, _ = run_cli(capsys, path)
    assert "难度棘轮：未上升" in out


def test_ratchet_reports_insufficient_with_short_history(tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]):
    """不满 5 行不硬猜趋势：判数据不足并报 n/5 行。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(3))
    payload = run_json(capsys, path)
    assert payload["ratchet"]["state"] == "insufficient"
    assert payload["ratchet"]["mid"]["samples"] == 3
    _, out, _ = run_cli(capsys, path)
    assert "难度棘轮：数据不足" in out and "3/5 行" in out


def test_ratchet_needs_more_than_two_percent(tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]):
    """单调但只涨 0.5%：不算棘轮（2% 门槛）。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(6, mid_start=1000, step=1))
    assert run_json(capsys, path)["ratchet"]["state"] == "flat"


def test_trend_unit_edges():
    """``_trend`` 本体：够样本单调升 = up，样本不足 = insufficient，抖动 = flat。"""
    assert _trend([100, 102, 104, 106, 108])["state"] == "up"
    assert _trend([100, 102])["state"] == "insufficient"
    assert _trend([100, 104, 102, 106, 108])["state"] == "flat"
    assert _trend([100, 100, 100, 100, 100])["state"] == "flat"


# ── 入口与排版 ─────────────────────────────────────────────────────────────
def test_run_dir_argument_and_missing_history(tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]):
    """给 run 目录能自己找 history.jsonl；目录里没有就退出码 2 并说清楚。"""
    run_dir = tmp_path / "self-play-20261114-001"
    run_dir.mkdir()
    write_jsonl(run_dir / "history.jsonl", make_rows(2))
    code, out, _ = run_cli(capsys, run_dir)
    assert code == 0 and "self-play-20261114-001" in out
    assert resolve_history(run_dir) == run_dir / "history.jsonl"

    empty = tmp_path / "no-history"
    empty.mkdir()
    assert resolve_history(empty) is None
    code, _, err = run_cli(capsys, empty)
    assert code == 2 and "history.jsonl" in err


def test_table_columns_stay_aligned_with_wide_numbers(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]):
    """超宽数字只加宽这一列，不让整行错位（中文按 2 列算）。"""
    rows = make_rows(3)
    rows[1]["elapsed"] = 1234567
    path = write_jsonl(tmp_path / "history.jsonl", rows)
    _, out, _ = run_cli(capsys, path)
    block = out.split("[对局结构]")[1].split("[难度与健康]")[0]
    lines = [line for line in block.strip().splitlines()[1:] if line.strip()]
    assert len(lines) >= 5
    assert len({_text_width(line) for line in lines}) == 1


def test_judgement_block_reports_arrows_and_current_values(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """``[判读]`` 要给出方向箭头、变化量、两个当前值与棘轮结论。"""
    path = write_jsonl(tmp_path / "history.jsonl", make_rows(6))
    _, out, _ = run_cli(capsys, path)
    verdict = out.split("[判读]")[1]
    assert "自建板面 T12: 1434 → 1539 (+105) ↑" in verdict
    assert "当前值：到达 12 回合占比 0.208" in verdict
    assert "arena 目标胜率 0.500" in verdict
    assert "难度棘轮：" in verdict


# ── 静态约束：只用标准库、不写死本机路径 ───────────────────────────────────
def test_tool_imports_only_stdlib():
    """面板必须只用标准库：不许 import 仓库内模块（也不能靠 numpy/torch）。"""
    tree = ast.parse((ROOT / "tools" / "transition_dashboard.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names), names


def test_tool_has_no_hardcoded_checkout_path():
    """不能写死本机绝对路径：run 目录必须从命令行来。"""
    source = (ROOT / "tools" / "transition_dashboard.py").read_text(encoding="utf-8")
    assert "D:\\" not in source and "C:\\" not in source and "/home/" not in source
