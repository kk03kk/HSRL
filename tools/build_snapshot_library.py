#!/usr/bin/env python
"""把"真人战团库"转换成 **快照库**（``hsbg.mirror_snapshots`` v1），以复用 36h 冠军那套注入。

为什么这么做
------------
36h 冠军用的是 `mirror_snapshots`：整席位快照注入 → 引擎置 ``ended_turn=true`` →
该席位**只带这块板子进战斗**，且回合开始时引擎会给所有活着的席位复位（``flow.rs:381``），
所以每回合都能重新注入。真人库那条 `inject_board` 路径没有这个复位语义，实测只能注入一次
（31%–40% 落地率），场里大半是对手的旧板子/bot —— 这也是"平均名次 1.068"的原因。

本工具只做**格式转换**：不缩放身材、不改强度，用的是 ``inject_quasi_state`` 同一套帧构造
（``dry_run=True``），所以转出来的帧与生产注入逐位同构。

用法::

    python tools/build_snapshot_library.py \
        --rows artifacts/libraries/arena-cal-v1/boards.jsonl \
        --out-dir artifacts/libraries/arena-cal-v1-snap
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

SCHEMA = "hsbg.mirror_snapshots"
SCHEMA_VERSION = 1
STAGE_EOT = "recruitment_end_before_combat"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="human board rows -> mirror snapshot library")
    ap.add_argument("--rows", default="artifacts/libraries/arena-cal-v1/boards.jsonl")
    ap.add_argument("--out-dir", default="artifacts/libraries/arena-cal-v1-snap")
    ap.add_argument("--fidelity", default="full")
    ap.add_argument("--seat", type=int, default=0,
                    help="写进帧的 seat（只用一个值时快照池允许跨座位回落）")
    a = ap.parse_args(argv)

    import hsbg_engine as E                      # noqa: PLC0415 - 需要引擎（服务器上跑）
    from hsbg.az.train.human_board_pool import HumanBoardPool
    from hsbg.az.train.human_obs_full import inject_quasi_state

    # 用**生产那条路**解析：HumanBoardPool 载入时就把每行转成准完整 payload，
    # 训练侧的 `--human-board-fidelity full` 用的就是它。不要另写解析器。
    pool = HumanBoardPool(a.rows, sample="p50")
    if not pool.kept:
        print(f"[error] pool empty: {a.rows}", file=sys.stderr)
        return 2
    print(f"boards = {pool.kept}  payload = {pool.payload_kept}  "
          f"turns = {pool.turns()[0]}..{pool.turns()[-1]}")

    tpl_game = E.Game(seed=12345)                # 只当"帧模板"用（读 player_json）
    # 模板必须走过"选英雄/第 1 回合"：turn 0 时席位有 pending_choice，注入会被引擎拒绝
    # （2026-11-14 实测：所有帧 applied=False，reason=pending_choice）。
    _steps = 0
    while int(tpl_game.turn) < 2 and not tpl_game.is_over() and _steps < 4000:
        _acted = False
        for _s in range(8):
            if tpl_game.is_over() or int(tpl_game.turn) >= 2:
                break
            if tpl_game.needs_action(_s):
                _m = tpl_game.legal_mask(_s)
                _a = next((i for i, b in enumerate(_m) if b), 0)
                tpl_game.step(_s, int(_a))
                _acted = True
                _steps += 1
        if not _acted:
            break
    print(f"template game advanced to turn {int(tpl_game.turn)}")
    frames: List[Dict[str, Any]] = []
    bad = 0
    per_turn: Dict[int, int] = {}
    for t in pool.turns():
        for hb in pool.index.get(t) or []:
            payload = pool.payload_from_pick(hb, int(t))
            if not isinstance(payload, dict):
                bad += 1
                continue
            try:
                rep = inject_quasi_state(tpl_game, 1, payload, str(a.fidelity),
                                         reset_ended=False, dry_run=True)
            except BaseException as exc:         # noqa: BLE001
                bad += 1
                if bad <= 5:
                    print(f"  [warn] turn={t} failed: {exc!r}"[:200])
                continue
            frame = rep.get("frame")
            if not isinstance(frame, dict) or "board" not in frame:
                bad += 1
                if bad <= 5:
                    print(f"  [warn] turn={t} no frame: applied={rep.get('applied')} "
                          f"reason={rep.get('reason')}"[:220])
                continue
            src_turn = int(getattr(hb, "turn", t) or t)
            frames.append({
                "turn": int(t),
                "seat": int(a.seat),
                "player": frame,
                "stage": STAGE_EOT,
                "source_turn": src_turn,
                "source": "arena_cal_v1",
                "frame_sha256": hashlib.sha256(
                    json.dumps(frame, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            })
            per_turn[int(t)] = int(per_turn.get(int(t), 0)) + 1

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "snap_cal.json.gz")
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "meta": {"source_rows": a.rows, "fidelity": str(a.fidelity),
                 "frames_total": len(frames), "bad_rows": bad,
                 "note": "converted from human board library; no stat scaling"},
        "frames": frames,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": SCHEMA, "schema_version": SCHEMA_VERSION,
                   "file": os.path.basename(path), "frames": len(frames),
                   "per_turn": {str(k): v for k, v in sorted(per_turn.items())},
                   "bad_rows": bad}, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")

    print(f"\nwrote {path}  frames={len(frames)}  bad={bad}")
    print(f"{'turn':>4} {'frames':>7}")
    for t in sorted(per_turn):
        print(f"{t:>4} {per_turn[t]:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
