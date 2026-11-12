#!/usr/bin/env python3
"""从已提取的真人动作流生成 ``bc_samples.jsonl.gz``（BC / board-problems 的前置）。

**这曾经是整条"专家教学"链路的断点**：``hsbg/az/train/bc_aux.load_bc_samples``、
``board_problems.build_reference_table``、``dpo_upgrade`` 全都读
``data/firestone_bc/bc_samples.jsonl.gz``，而代码树里**只有消费者、没有生产者**。
``workflows/firestone_bc_parse.py`` 期望的是另一种输入（每局 ``events.jsonl.gz``），
与本项目实际拿到的 ``xmlz/*.xml.zip`` 对不上，所以它从未跑出过任何东西。

实测发现：``data/real_library/*/actions.jsonl.gz`` 里 297,884 条动作记录**已经**是
loader 认得的形状——``action_type``、``action_index``、``card_id``、``turn``、``gold``、
``tier``、``health``、``armor``、``shop_cards``、``hand_cards``、``board_cards`` 全都有，
而且 96% 落在 9 类白名单里。真正缺的只有两样：

1. **``mmr``** —— 动作文件里没有，必须按 ``game_id`` 从 ``library.jsonl`` 关联（不伪造）；
2. **``hero``** —— 同样按局从 ``library.jsonl`` 关联。

本脚本只做这两步关联与字段改名，**不改写任何数值、不填补缺失、不生成新样本**。
缺 mmr 的记录直接丢弃并计数。

用法::

    PYTHONPATH=engine/python:. python tools/build_bc_samples.py \
        --actions data/real_library/firestone-20261112-1000/actions.jsonl.gz \
        --library data/real_library/firestone-20261112-1000/library.jsonl \
        --out data/firestone_bc/bc_samples.jsonl.gz --verify
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

#: 输出里保留的字段（loader 认的名字）
KEEP = ("action_type", "action_index", "card_id", "turn", "gold", "tier",
        "health", "armor", "shop_cards", "hand_cards", "board_cards",
        "game_id", "source")


def game_meta(library: Path) -> Dict[str, Dict[str, Any]]:
    """game_id -> {mmr, hero}。

    **必须只取 ``perspective == 'self'`` 的行**：``library.jsonl`` 每局同时有 self 与
    opponent 行，对手的英雄卡不同。早先不分视角直接取第一行，导致 10,776 局的 hero
    与动作流不属于同一玩家（会把错误英雄喂给英雄嵌入）。mmr 两侧相同，但仍以 self 为准。
    """
    out: Dict[str, Dict[str, Any]] = {}
    conflicts = collections.Counter()
    for row in _iter_jsonl(library):
        if row.get("perspective") != "self":
            continue
        gid = row.get("game_id")
        if not gid:
            continue
        item = {"mmr": row.get("mmr"), "hero": row.get("hero_card_id")}
        cur = out.get(gid)
        if cur is None:
            out[gid] = item
        else:
            for k, v in item.items():
                if v is not None and cur.get(k) is not None and cur[k] != v:
                    conflicts[k] += 1
    if conflicts:
        print(f"WARN: per-game self-perspective conflicts (kept first): {dict(conflicts)}",
              file=sys.stderr)
    return out


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 bc_samples.jsonl.gz")
    ap.add_argument("--actions", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", default="data/firestone_bc/bc_samples.jsonl.gz")
    ap.add_argument("--verify", action="store_true", help="写完后用 load_bc_samples 复核")
    args = ap.parse_args(argv)

    meta = game_meta(Path(args.library))
    print(f"games in library : {len(meta)}")
    with_mmr = sum(1 for m in meta.values() if isinstance(m.get("mmr"), int))
    with_hero = sum(1 for m in meta.values() if isinstance(m.get("hero"), str))
    print(f"  with mmr       : {with_mmr}")
    print(f"  with hero      : {with_hero}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    classes = collections.Counter()

    with gzip.open(args.actions, "rt", encoding="utf-8") as src, \
            gzip.open(out_path, "wt", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            r = json.loads(line)
            gid = r.get("game_id")
            m = meta.get(gid) or {}
            mmr = m.get("mmr")
            if not isinstance(mmr, int):
                stats["drop_no_mmr"] += 1
                continue
            rec = {k: r.get(k) for k in KEEP if k in r}
            rec["mmr"] = int(mmr)
            hero = m.get("hero")
            if isinstance(hero, str):
                rec["hero"] = hero
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["written"] += 1
            classes[str(r.get("action_type"))] += 1

    print()
    print(f"read    : {stats['read']}")
    print(f"written : {stats['written']}")
    print(f"no-mmrr : {stats['drop_no_mmr']}")
    print(f"output  : {out_path} ({out_path.stat().st_size} bytes)")
    print("action_type distribution (written):")
    for k, v in classes.most_common():
        print(f"  {k}: {v}")

    if args.verify:
        from hsbg.az.train.bc_aux import load_bc_samples
        pool = load_bc_samples(str(out_path), mmr_min=0, mmr_full=8000)
        print()
        print(f"load_bc_samples @ mmr_min=0 : n={len(pool)}")
        for k, v in sorted(pool.class_distribution().items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")
        default = load_bc_samples(str(out_path))
        print(f"load_bc_samples @ default   : n={len(default)} (mmr_min filter applied)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
