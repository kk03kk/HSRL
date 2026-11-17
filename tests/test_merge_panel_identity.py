"""合并面板必须带上**模型身份**（checkpoint / sha256 / arch / seed 区间）。

2026-11-17 验证实际评测时发现：合并后的 panel.json 里 `checkpoint_sha256` 是空的 ——
分片 ``extra-*.json`` 不含身份字段，``_model_entry`` 也不去分片 ``panel.json`` 里找。
一个判胜面板如果说不清"评的是哪个权重"，就等于不可审计。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from tools.merge_panels import merge

PROTO = "a" * 64
SHA = "1808f178" + "0" * 56


def _shard(d: Path, seed0: int, n: int = 12, *, with_ident_in_extra: bool) -> Path:
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
        "name": "va", "games": n, "seconds": 10.0,
        "outcome": {"illegal_actions": 0, "truncated_games": 0, "decisions": 100},
        "rank_sequence": [1 + (i % 4) for i in range(n)],
        "seed_sequence": [seed0 + i for i in range(n)],
        "action_hist": {}, "combat_exposure": [], "protocol_sha256": PROTO,
    }
    if with_ident_in_extra:
        extra.update({"checkpoint": "artifacts/models/x/model.pt",
                      "checkpoint_sha256": SHA, "arch": "entity",
                      "seed_start": seed0, "seed_end": seed0 + n - 1})
    (d / "extra-va.json").write_text(json.dumps(extra), encoding="utf-8")
    (d / "panel.json").write_text(json.dumps({
        "protocol": {"protocol_sha256": PROTO},
        "models": [{"name": "va", "checkpoint": "artifacts/models/x/model.pt",
                    "checkpoint_sha256": SHA, "arch": "entity",
                    "seed_start": seed0, "seed_end": seed0 + n - 1}]}),
        encoding="utf-8")
    return d


def test_identity_is_taken_from_shard_panel_when_extra_lacks_it(tmp_path):
    a = _shard(tmp_path / "s0", 930000, with_ident_in_extra=False)
    b = _shard(tmp_path / "s1", 930100, with_ident_in_extra=False)
    doc = merge([a, b], tmp_path / "panel.json")
    m = doc["models"][0]
    assert m["checkpoint_sha256"] == SHA
    assert m["checkpoint"].endswith("model.pt")
    assert m["arch"] == "entity"
    assert m["seed_start"] == 930000
    assert m["seed_end"] == 930111          # 两个 shard 合并后的末 seed（不是 s0 的）


def test_identity_is_taken_from_extra_when_present(tmp_path):
    a = _shard(tmp_path / "s0", 930000, with_ident_in_extra=True)
    doc = merge([a], tmp_path / "panel.json")
    assert doc["models"][0]["checkpoint_sha256"] == SHA
    assert doc["models"][0]["seconds"] == 10.0


def test_seed_range_spans_all_shards(tmp_path):
    """合并面板的 seed 区间必须是全部 shard 的 min/max（不能抄第一个 shard）。"""
    a = _shard(tmp_path / "s0", 930000, with_ident_in_extra=False)
    b = _shard(tmp_path / "s1", 930100, with_ident_in_extra=False)
    doc = merge([a, b], tmp_path / "panel.json")
    m = doc["models"][0]
    assert m["seed_start"] == 930000
    assert m["seed_end"] == 930111           # s1 的末 seed
    assert m["shard_count"] == 2


def test_seconds_is_summed_across_shards(tmp_path):
    a = _shard(tmp_path / "s0", 930000, with_ident_in_extra=True)
    b = _shard(tmp_path / "s1", 930100, with_ident_in_extra=True)
    doc = merge([a, b], tmp_path / "panel.json")
    assert doc["models"][0]["seconds"] == 20.0
