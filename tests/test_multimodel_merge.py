"""离线验证 merge_panels 对「一个面板里含多个模型」的处理：
- 按模型名分组
- 同名模型的 trace / rank_sequence / strength_by_seed 按输入顺序拼接
- rank_sequence 顺序在四模型之间保持一致（配对前提）

不需要 GPU / 引擎。
"""
import gzip
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"D:\HSAI")
sys.path.insert(0, str(ROOT))
from tools.merge_panels import merge  # noqa

NAMES = ["v1.2", "v1.3_r1", "v1.3_r2", "v1.3_r3"]


def rows_for(seed):
    out = []
    for t in range(1, 7):
        out.append({
            "seed": seed, "turn": t, "tier": 6, "gold": 0, "health": 20, "hand_n": 1,
            "actions": 5, "eot_attack": 10 * t, "eot_board_hp": 10 * t,
            "eot_board": [{"position": 0, "card_id": "BG36_366", "name": "x",
                           "atk": 10 * t, "hp": 10 * t, "golden": False,
                           "keywords": [], "tier": 5}],
            "board": [], "board_stats": {}, "hero": 1, "trinkets": [], "rewards": None,
            "consecutive_repeats": 0,
            "gold_events": [{"action": 8, "kind": "buy", "before": 3, "after": 0, "net_change": -3}],
            "name": "m", "outcome": {"decisions": 100, "illegal_actions": 0,
                                     "truncated_games": 0, "decisions_per_game": 16.7},
            "rank_sequence": [1, 2, 3, 4],
        })
    return out


def write_shard(d: Path, seed0: int, n_games: int):
    d.mkdir(parents=True, exist_ok=True)
    allrows = []
    for g in range(n_games):
        allrows.extend(rows_for(seed0 + g))
    with gzip.open(d / "rows.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in allrows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # extra.json 只记一个模型的元信息（真实场景每个子面板只有一个模型名）
    (d / "extra.json").write_text(json.dumps({
        "name": NAMES[0], "outcome": {"decisions": 100, "illegal_actions": 0,
                                      "truncated_games": 0, "decisions_per_game": 16.7},
        "rank_sequence": [1, 2, 3, 4], "action_hist": {}, "combat_exposure": [],
        "seconds": 1.0, "games": n_games, "seed": seed0}, ensure_ascii=False),
        encoding="utf-8")


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    a = td / "n0p0"
    b = td / "n0p1"
    write_shard(a, 930000, 2)
    write_shard(b, 930002, 3)
    doc = merge([a, b], td / "panel.json")

m = doc["models"][0]
games = m.get("games")
print("merged games =", games, " (期望 5)")
print("trace 行数 =", len(m.get("trace") or []), " (期望 5 局 × 6 回合 = 30)")
rs = m.get("rank_sequence") or []
print("rank_sequence =", rs, " (期望 [1,2,3,4,1,2,3,4])")
ok = (games == 5 and len(m.get("trace") or []) == 30 and rs == [1, 2, 3, 4, 1, 2, 3, 4])
print("MULTI_OK" if ok else "MULTI_FAIL")
raise SystemExit(0 if ok else 1)
