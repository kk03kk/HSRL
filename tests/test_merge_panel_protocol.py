"""``merge_panels`` 必须把**完整**协议段带进合并面板（不只 sha256）。

2026-11-17 的真实故障：``eval_shard_parallel`` 生成的面板 ``protocol`` 里只剩
``protocol_sha256``，``contract`` / ``field`` / ``engine`` 全丢 —— 一个"能比"的面板
无法自证"比的是什么"。原因：分片 ``extra-*.json`` 只记 sha，合并时没去读分片的
``panel.json``。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from tools.merge_panels import merge
from hsbg.az.train.eval_policy import protocol_sha256_of

PROTO = "f" * 64
FULL = {
    "schema": "hsai.eval_protocol/v1",
    "protocol_sha256": PROTO,
    "contract": {"max_actions_per_turn": 100, "move_cap": 2, "freeze_cap": 1},
    "field": {"mirror_snaps": "lib/x", "mirror_sample": "p50band"},
    "engine": {"game_version": "37.0.2.256589"},
}
PROTO = protocol_sha256_of(FULL)
FULL['protocol_sha256'] = PROTO


def _write_shard(d: Path, seed0: int, n: int = 12) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "rows-va.jsonl.gz", "wt", encoding="utf-8") as fh:
        for g in range(n):
            for t in (1, 2):
                fh.write(json.dumps({
                    "seed": seed0 + g, "turn": t, "tier": 5, "gold": 3,
                    "health": 30, "hand_n": 1, "actions": 4,
                    "eot_attack": 10, "eot_board_hp": 10,
                    "eot_board": [], "board_stats": {}, "hero": "H",
                    "trinkets": [],
                }) + "\n")
    extra = {
        "name": "va", "games": n,
        "outcome": {"illegal_actions": 0, "truncated_games": 0, "decisions": 100},
        "rank_sequence": [1 + (i % 4) for i in range(n)],
        "seed_sequence": [seed0 + i for i in range(n)],
        "action_hist": {}, "combat_exposure": [],
        "protocol_sha256": PROTO,
    }
    (d / "extra-va.json").write_text(json.dumps(extra), encoding="utf-8")
    # 分片自己的 panel.json 带完整 protocol
    (d / "panel.json").write_text(json.dumps({"schema": "hsai.human_gap_panel/v1",
                                              "protocol": dict(FULL),
                                              "models": [{"name": "va"}]}),
                                  encoding="utf-8")
    return d


def test_merged_panel_carries_full_protocol(tmp_path):
    a = _write_shard(tmp_path / "s0", 930000)
    b = _write_shard(tmp_path / "s1", 930100)
    doc = merge([a, b], tmp_path / "panel.json")
    proto = doc["protocol"]
    assert proto["protocol_sha256"] == PROTO
    assert proto["contract"]["max_actions_per_turn"] == 100
    assert proto["field"]["mirror_snaps"] == "lib/x"
    assert proto["engine"]["game_version"] == "37.0.2.256589"


def test_merged_panel_falls_back_to_sha_when_shard_panel_missing(tmp_path):
    a = _write_shard(tmp_path / "s0", 930000)
    (a / "panel.json").unlink()
    doc = merge([a], tmp_path / "panel.json")
    assert doc["protocol"] == {"protocol_sha256": PROTO}
