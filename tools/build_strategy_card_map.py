#!/usr/bin/env python3
"""把《酒馆战棋流派攻略》变成机器可用的卡表映射。

问题：攻略是中文散文（``docs/battlegrounds_strategy_guide.md``，33 个流派，
5 个 S 级），而模型、引擎、回放数据里全是 ``BG36_345`` 这样的 card_id。两边对不上，
"模型会不会玩 S 级阵容"就没法量化。

本脚本做桥接：中文名 → card_id（``data/card_zhcn.json`` 反查）→ 种族/酒馆等级
（``data/bg_cards.json``），产出 ``config/strategy_cards.json``。

产物用途：
1. **度量**：判定一局终局板面是否含有某个流派的"核心卡"，从而比较
   真人胜者 vs 我们的模型在阵容选择上的差距（见 ``workflows/eval_human_gap.py``）。
2. **训练信号**：给辅助头/软偏置提供标签来源（``hsbg/az/train/archetype_bias.py``
   的 preference table 同源）。
3. **审计**：把"攻略里有、但卡表里找不到"的卡名显式列出来——那通常意味着攻略
   与当前卡池版本不一致，必须知道而不是猜。

**不做的事**：不生成策略、不改引擎、不碰训练。纯文本 → JSON。

用法::

    python tools/build_strategy_card_map.py            # 写 config/strategy_cards.json
    python tools/build_strategy_card_map.py --check    # 只报告，不写文件
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

GUIDE = ROOT / "docs" / "battlegrounds_strategy_guide.md"
ZHCN = ROOT / "data" / "card_zhcn.json"
CARDS = ROOT / "data" / "bg_cards.json"
OUT = ROOT / "config" / "strategy_cards.json"

#: ``card_race`` 数值 → 种族名。
#:
#: **不是猜的**：用攻略自己的 33 个流派标题（"元素 - 活化艾泽里特"）反推出来的，
#: 每条核心卡投票给"它所属种族"，取众数，33 个流派零冲突。见
#: docs/analysis/STRATEGY_KNOWLEDGE_20261112.md。
#: 早先手写的一版把 14/15/17/20/23/24 全部标错，导致 21 个流派被误报为"跨种族"。
RACE_NAMES = {
    11: "undead", 14: "murloc", 15: "demon", 17: "mechanical", 18: "elemental",
    20: "beast", 23: "pirate", 24: "dragon", 43: "quilboar", 92: "naga",
    126: "all",
}

#: 攻略标题里的中文种族 → 期望的 tribe 名（用 RACE_NAMES 的同一套词）。
TRIBE_ZH = {
    "野兽": "beast", "恶魔": "demon", "龙": "dragon", "元素": "elemental",
    "机械": "mechanical", "鱼人": "murloc", "海盗": "pirate", "野猪人": "quilboar",
    "亡灵": "undead", "纳迦": "naga", "畸变怪": "aberration",
    "中立": None, "混合": None,
}

SECTION_RE = {
    "core": "核心卡牌",
    "extra": "附加卡牌",
    "how": "玩法说明",
    "formed": "成型随从组合",
    "transition": "常用过渡随从",
}


def parse_guide(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^## ", text, flags=re.M)[1:]
    comps = []
    for block in blocks:
        lines = block.splitlines()
        name = lines[0].strip()
        body = "\n".join(lines[1:])
        m = re.search(r"\*\*难度：(.+?) \| 强度：(.+?)\*\*", body)
        diff, tier = (m.group(1).strip(), m.group(2).strip()) if m else ("?", "?")

        def section(title: str) -> List[str]:
            mm = re.search(rf"### {title}\n\n(.*?)(?=\n### |\Z)", body, re.S)
            if not mm:
                return []
            return [x.strip() for x in mm.group(1).strip().splitlines() if x.strip()]

        comps.append({
            "name": name,
            "difficulty": diff,
            "tier": tier,
            "core_names": section(SECTION_RE["core"]),
            "extra_names": section(SECTION_RE["extra"]),
            "formed": section(SECTION_RE["formed"]),
            "transition_names": section(SECTION_RE["transition"]),
            "how": " ".join(section(SECTION_RE["how"])),
        })
    return comps


def load_tables() -> tuple[Dict[str, Any], Dict[str, Any]]:
    zhcn = json.loads(ZHCN.read_text(encoding="utf-8"))
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cards if isinstance(c, dict) and "id" in c}
    return zhcn, by_id


def build_name_index(zhcn: Dict[str, Any], by_id: Dict[str, Any]) -> Dict[str, List[str]]:
    """中文名 → 候选 card_id 列表（按"更像基础版"排序）。"""
    index: Dict[str, List[str]] = defaultdict(list)
    for card_id, meta in zhcn.items():
        if not isinstance(meta, dict):
            continue
        name = (meta.get("n") or "").strip()
        if not name:
            continue
        index[name].append(card_id)

    def rank(card_id: str) -> tuple:
        # 优先 BG/BG3 系列、非金色(_G)、卡表里存在的
        return (
            0 if card_id in by_id else 1,
            1 if card_id.endswith("_G") else 0,
            0 if card_id.startswith("BG") else 1,
            len(card_id),
            card_id,
        )

    for name in index:
        index[name].sort(key=rank)
    return index


def resolve(names: List[str], index: Dict[str, List[str]],
            by_id: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    resolved, missing = [], []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        cands = index.get(name) or []
        cand = next((c for c in cands if c in by_id), None)
        if cand is None:
            missing.append(name)
            continue
        card = by_id[cand]
        race = card.get("card_race")
        resolved.append({
            "zh": name,
            "card_id": cand,
            "en": card.get("name"),
            "tribe": RACE_NAMES.get(race, race),
            "tribe_id": race,
            "tavern_tier": card.get("tech_level"),
        })
    return resolved, missing


def expected_tribe(comp_name: str):
    """从 "元素 - 活化艾泽里特" 这种标题里取出期望种族。"""
    head = comp_name.split("-")[0].strip()
    return TRIBE_ZH.get(head, "UNKNOWN")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="攻略 → 卡表映射")
    ap.add_argument("--check", action="store_true", help="只报告，不写文件")
    args = ap.parse_args(argv)

    for p in (GUIDE, ZHCN, CARDS):
        if not p.is_file():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    comps = parse_guide(GUIDE)
    zhcn, by_id = load_tables()
    index = build_name_index(zhcn, by_id)

    out_comps = []
    total_missing = 0
    for c in comps:
        core, miss_core = resolve(c["core_names"], index, by_id)
        extra, miss_extra = resolve(c["extra_names"], index, by_id)
        trans, miss_trans = resolve(c["transition_names"], index, by_id)
        total_missing += len(miss_core) + len(miss_extra) + len(miss_trans)
        want = expected_tribe(c["name"])
        core_tribes = [x["tribe"] for x in core if isinstance(x["tribe"], str)]
        # 只把"整个流派一张本种族核心卡都没有"当作可疑解——跨种族的经济引擎
        # （铜须、巴琳达、罗杰斯、私掠者）是攻略明确写的，不算错。
        suspect = bool(want) and want != "UNKNOWN" and core and want not in core_tribes
        foreigners = [{"zh": x["zh"], "card_id": x["card_id"], "tribe": x["tribe"]}
                      for x in core
                      if want and isinstance(x["tribe"], str) and x["tribe"] != want]
        out_comps.append({
            "name": c["name"],
            "tier": c["tier"],
            "difficulty": c["difficulty"],
            "expected_tribe": want,
            "tribe_suspect": suspect,
            "foreign_core": foreigners,
            "core": core,
            "extra": extra,
            "transition": trans,
            "formed_note": " / ".join(c["formed"]),
            "core_card_ids": [x["card_id"] for x in core],
            "tribes": sorted(set(core_tribes)),
            "unresolved": {"core": miss_core, "extra": miss_extra,
                           "transition": miss_trans},
            "how": c["how"],
        })

    order = {"S": 0, "A": 1, "B": 2, "C": 3}
    out_comps.sort(key=lambda c: (order.get(c["tier"], 9), c["difficulty"], c["name"]))

    tiers = defaultdict(int)
    for c in out_comps:
        tiers[c["tier"]] += 1

    print(f"compositions        : {len(out_comps)}  tiers={dict(sorted(tiers.items()))}")
    print(f"card table          : {len(by_id)} engine cards, {len(zhcn)} zh-CN names")
    print(f"unresolved names    : {total_missing}")
    if total_missing:
        print("  (these guide names do not exist in the current card table — the guide")
        print("   may predate or postdate the build; do NOT silently drop them)")
        for c in out_comps:
            u = c["unresolved"]
            if any(u.values()):
                print(f"  {c['tier']} {c['name']}: "
                      f"core={u['core']} extra={u['extra']} trans={u['transition']}")

    print()
    bad = [c for c in out_comps if c["tribe_suspect"]]
    print(f"resolution suspects : {len(bad)} comps  (no core card of the comp's own tribe)")
    for c in bad:
        ids = ", ".join(f"{m['zh']}->{m['card_id']}({m['tribe']})"
                        for m in c["foreign_core"])
        print(f"  {c['tier']} {c['name']}  want={c['expected_tribe']}: {ids}")
    if bad:
        print("  !! treat these card_ids as UNVERIFIED — the Chinese name probably")
        print("     resolves to the wrong printing of the card.")

    print()
    print("-- S tier resolved cores --")
    for c in out_comps:
        if c["tier"] != "S":
            continue
        print(f"  {c['name']}  ({c['difficulty']})")
        for card in c["core"]:
            print(f"      {card['zh']:<14} -> {card['card_id']:<12} "
                  f"{str(card['en'])[:22]:<22} tribe={card['tribe']} t{card['tavern_tier']}")

    if args.check:
        print("\n--check: not writing")
        return 0

    payload = {
        "schema": "hsai.strategy_cards/v1",
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": {
            "guide": str(GUIDE.relative_to(ROOT)).replace("\\", "/"),
            "zhcn": str(ZHCN.relative_to(ROOT)).replace("\\", "/"),
            "cards": str(CARDS.relative_to(ROOT)).replace("\\", "/"),
            "engine": __import__("json").loads(
                (ROOT / "config" / "engine-upstream-lock.json").read_text(encoding="utf-8"))
            if (ROOT / "config" / "engine-upstream-lock.json").is_file() else None,
        },
        "tier_counts": dict(sorted(tiers.items())),
        "unresolved_total": total_missing,
        "comps": out_comps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
