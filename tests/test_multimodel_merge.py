"""``tools/merge_panels.py`` 的核心行为：按模型名分组 + 按输入顺序拼接。

原来这个文件是"脚本式测试"（模块级 ``raise SystemExit``），会让 ``pytest tests``
直接 ``INTERNALERROR``（审计 P1-5）。现在改成真正的 pytest，并补上 2026-11-17 起
的新契约：**分片必须带 ``protocol_sha256``**（否则无法证明同榜，审计 P0-2）。

不需要 GPU / 引擎。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from tools.merge_panels import merge

MODEL = "v1.2"
PROTO = "e" * 64


def _rows_for(seed: int):
    out = []
    for t in range(1, 7):
        out.append({
            "seed": seed, "turn": t, "tier": 6, "gold": 0, "health": 20,
            "hand_n": 1, "actions": 5,
            "eot_attack": 10 * t, "eot_board_hp": 10 * t,
            "eot_board": [{"position": 0, "card_id": "BG36_366", "name": "x",
                           "atk": 10 * t, "hp": 10 * t, "golden": False,
                           "keywords": [], "tier": 5}],
            "board": [], "board_stats": {}, "hero": 1, "trinkets": [],
            "rewards": None, "consecutive_repeats": 0,
            "gold_events": [{"action": 8, "kind": "buy", "before": 3,
                             "after": 0, "net_change": -3}],
        })
    return out


def _write_shard(d: Path, seed0: int, n_games: int, *, name: str = MODEL,
                 per_model_files: bool = False):
    d.mkdir(parents=True, exist_ok=True)
    allrows = []
    for g in range(n_games):
        allrows.extend(_rows_for(seed0 + g))
    stem = f"rows-{name}" if per_model_files else "rows"
    with gzip.open(d / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in allrows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    extra = {
        "name": name,
        "outcome": {"decisions": 100, "illegal_actions": 0,
                    "truncated_games": 0, "decisions_per_game": 16.7},
        # rank_sequence / seed_sequence 必须与 games **一一对应**，否则合并时
        # `_seed_sequence` 会拒绝产出（这正是我们想抓的不一致）。
        "rank_sequence": [(i % 4) + 1 for i in range(n_games)],
        "seed_sequence": [seed0 + g for g in range(n_games)],
        "action_hist": {}, "combat_exposure": [], "seconds": 1.0,
        "games": n_games, "seed": seed0,
        "protocol_sha256": PROTO,
    }
    fname = f"extra-{name}.json" if per_model_files else "extra.json"
    (d / fname).write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")


def test_two_shards_are_concatenated_in_input_order(tmp_path):
    a, b = tmp_path / "n0p0", tmp_path / "n0p1"
    _write_shard(a, 930000, 2)
    _write_shard(b, 930002, 3)
    doc = merge([a, b], tmp_path / "panel.json")
    assert len(doc["models"]) == 1
    m = doc["models"][0]
    assert m["games"] == 5
    assert len(m["trace"]) == 5 * 6                 # 5 局 × 6 回合
    assert m["rank_sequence"] == [1, 2, 1, 2, 3]
    assert m["seed_sequence"][0] == 930000 and m["seed_sequence"][-1] == 930004
    assert doc["protocol"]["protocol_sha256"] == PROTO


def test_per_model_filenames_also_work(tmp_path):
    """新落盘布局（``rows-<name>`` / ``extra-<name>``）必须与旧布局等价。"""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_shard(a, 930000, 2, per_model_files=True)
    _write_shard(b, 930002, 3, per_model_files=True)
    doc = merge([a, b], tmp_path / "panel.json")
    assert doc["models"][0]["games"] == 5
    assert len(doc["models"][0]["trace"]) == 30