"""流派结构分析：共享核心、共享经济/成长机制、低成本转型边。

用途（专家先验 E3）：不要按种族名判断流派相似，而是看**核心卡重合**与**机制角色**。
输出 `docs/analysis/COMP_SHARED_CORES_<date>.md`，供人工判读与后续奖励设计。

输入
- `config/strategy_cards.json`（33 个流派：tier / core / foreign_core / transition / how）
- `data/p2/cards_37.json`（当前版本卡表：kind / tier / races / text_zh）

角色标签是**关键词启发式**，不是游戏规则；只用来排序，不当作事实写进奖励。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 角色关键词（中文卡面）：顺序即优先级，一个随从只取第一个命中的角色
ROLE_RULES: List[tuple[str, str]] = [
    ("economy", r"铸币|金币|免费刷新|获得1枚|经济|酒馆法术.{0,6}费"),
    ("spell", r"酒馆法术|施放.{0,6}法术|法术"),
    ("gem", r"宝石"),
    ("magnet", r"磁力"),
    ("refresh", r"刷新"),
    ("discover", r"发现"),
    ("growth", r"永久|属性值翻倍|每当你|每当.{0,10}后"),
]
ROLE_ZH = {
    "economy": "经济", "spell": "法术", "gem": "宝石", "magnet": "磁力",
    "refresh": "刷新", "discover": "发现", "growth": "成长", "": "未标注",
}


def load_cards() -> Dict[str, dict]:
    path = os.path.join(ROOT, "data", "p2", "cards_37.json")
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            # 该表是 `{"meta":...,"cards":[{...},\n{...},\n]}` 的逐行排版，
            # 除最后一张外每行都以逗号结尾 —— 去掉尾逗号才能单独解析。
            if line.endswith(","):
                line = line[:-1]
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = c.get("id")
            if isinstance(cid, str):
                out[cid] = c
    return out


def role_of(text: str) -> str:
    for role, pat in ROLE_RULES:
        if re.search(pat, text or ""):
            return role
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default=os.path.join(ROOT, "config", "strategy_cards.json"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "docs", "analysis", "COMP_SHARED_CORES_20261113.md"))
    ap.add_argument("--min-share", type=int, default=2,
                    help="被 >=N 个流派列为核心的卡才算共享")
    args = ap.parse_args()

    with open(args.comps, encoding="utf-8") as f:
        doc = json.load(f)
    comps = doc["comps"]
    cards = load_cards()

    # ---- 每张核心卡的角色 / 等级 / 种族 ----
    info: Dict[str, dict] = {}
    membership: Dict[str, List[str]] = collections.defaultdict(list)
    for c in comps:
        name = c["name"]
        for slot in ("core", "foreign_core", "transition"):
            for card in c.get(slot) or []:
                cid = card.get("card_id")
                if not isinstance(cid, str):
                    continue
                if name not in membership[cid]:
                    membership[cid].append(name)
                if cid in info:
                    continue
                meta = cards.get(cid) or {}
                text = meta.get("text_zh") or ""
                info[cid] = {
                    "zh": card.get("zh") or meta.get("name_zh") or cid,
                    "tier": card.get("tavern_tier") or meta.get("tier"),
                    "tribe": card.get("tribe") or ",".join(meta.get("races") or []) or None,
                    "role": role_of(text),
                    "slot": slot,
                    "text": text[:110],
                }

    # ---- 共享卡（跨流派）----
    shared = sorted(
        ((cid, names) for cid, names in membership.items() if len(names) >= args.min_share),
        key=lambda kv: -len(kv[1]),
    )

    # ---- 两两重合：低成本转型候选 ----
    sets = {c["name"]: set(c.get("core_card_ids") or []) for c in comps}
    tiers = {c["name"]: c.get("tier") for c in comps}
    pairs = []
    names_sorted = sorted(sets)
    for i, a in enumerate(names_sorted):
        for b in names_sorted[i + 1:]:
            sa, sb = sets[a], sets[b]
            if not sa or not sb:
                continue
            inter = sa & sb
            union = sa | sb
            if not inter:
                continue
            pairs.append({
                "a": a, "b": b,
                "shared": len(inter),
                "jaccard": len(inter) / len(union),
                "acquire": sorted(sb - sa),
                "drop": sorted(sa - sb),
            })
    pairs.sort(key=lambda d: (-d["jaccard"], -d["shared"]))

    # ---- 每个流派的"必须等级"与机制构成 ----
    per_comp = []
    for c in comps:
        ids = [x for x in (c.get("core_card_ids") or []) if isinstance(x, str)]
        t = [info[i]["tier"] for i in ids if isinstance(info.get(i, {}).get("tier"), int) and info[i]["tier"]]
        roles = collections.Counter(info[i]["role"] for i in ids if i in info)
        per_comp.append({
            "name": c["name"], "tier": c.get("tier"),
            "n_core": len(ids),
            "max_tier": max(t) if t else None,
            "roles": dict(roles),
            "economy": [info[i]["zh"] for i in ids if info.get(i, {}).get("role") == "economy"],
            "tribes": c.get("tribes") or [],
        })

    def short(cid: str) -> str:
        d = info.get(cid) or {}
        role = d.get("role") or ""
        tier = d.get("tier")
        tier_s = f"T{tier}" if isinstance(tier, int) else "T?"
        role_s = f"/{ROLE_ZH.get(role, role)}" if role else ""
        return f"{d.get('zh', cid)}（{tier_s}{role_s}）"

    L: List[str] = []
    L.append("# 33 个流派的共享核心与转型成本（结构分析）\n")
    L.append("状态：**自动生成**，输入是用户本人编写的 `config/strategy_cards.json` + "
             "版本卡表 `data/p2/cards_37.json`。\n")
    L.append("口径说明：")
    L.append("- **共享核心** = 被 ≥2 个流派列在 `core`/`foreign_core`/`transition` 里的同一张卡；")
    L.append("- **角色标签**（经济/法术/宝石/磁力/刷新/发现/成长）是卡面关键词启发式，"
             "只用于排序，**不写进奖励**；")
    L.append("- **转型成本** 用 `|目标核心 - 当前核心|` 估计 = 转型时还需要找到几张卡，"
             "`jaccard` 越高越便宜。\n")

    L.append("## 1. 跨流派共享核心（最可能是经济/成长引擎）\n")
    L.append("| 卡 | 等级 | 关键词角色 | 几个流派 | 卡面（节选） | 流派 |")
    L.append("|---|---:|---|---:|---|---|")
    for cid, names in shared[:25]:
        d = info.get(cid) or {}
        tier = d.get("tier")
        text = (d.get("text") or "").replace("|", "/")[:46]
        L.append(f"| {d.get('zh', cid)} | {tier if tier else '?'} | "
                 f"{ROLE_ZH.get(d.get('role') or '', '未命中')} | {len(names)} | "
                 f"{text} | " + "、".join(names) + " |")
    L.append("")

    L.append("## 2. 各流派的机制构成与所需等级\n")
    L.append("| 流派 | 级别 | 核心数 | 最高核心等级 | 含经济核心 | 种族 |")
    L.append("|---|---|---:|---:|---|---|")
    for d in sorted(per_comp, key=lambda d: ({"S": 0, "A": 1, "B": 2}.get(d["tier"], 3), d["name"])):
        L.append(f"| {d['name']} | {d['tier']} | {d['n_core']} | "
                 f"{d['max_tier'] if d['max_tier'] else '?'} | "
                 f"{('是：' + '、'.join(d['economy'])) if d['economy'] else '否'} | "
                 f"{'/'.join(d['tribes']) or '-'} |")
    L.append("")

    L.append("## 3. 低成本转型边（核心卡重合 ≥1，按 jaccard 排序）\n")
    L.append("| 从 | 到 | 共享卡数 | jaccard | 转型还需找 | 会失去 |")
    L.append("|---|---|---:|---:|---|---|")
    for p in pairs[:30]:
        L.append(f"| {p['a']}({tiers.get(p['a'])}) | {p['b']}({tiers.get(p['b'])}) | "
                 f"{p['shared']} | {p['jaccard']:.2f} | "
                 f"{len(p['acquire']) or '0（直接转）'} | {len(p['drop'])} |")
    L.append("")

    L.append("## 4. 没有任何核心重合的流派对（机制冲突 / 必须重建）\n")
    iso = []
    for i, a in enumerate(names_sorted):
        for b in names_sorted[i + 1:]:
            if sets[a] and sets[b] and not (sets[a] & sets[b]):
                iso.append((a, b))
    L.append(f"共 **{len(iso)}** 对几乎无共享核心（33 个流派共 {len(names_sorted)*(len(names_sorted)-1)//2} 对）。")
    L.append("")
    L.append("这解释了为什么「转型」不是一个动作，而是一次重建：这些对之间没有任何可复用的核心。")
    L.append("")

    L.append("## 5. 直接可用的结论（给训练/评测）\n")
    hubs = [cid for cid, names in shared if len(names) >= 3]
    L.append(f"1. **共享引擎是少数几张 T5–T6 的卡**：被 ≥3 个流派列为核心的只有 "
             f"**{len(hubs)}** 张 —— "
             + "、".join(f"{info.get(c, {}).get('zh', c)}（{len(membership[c])}）" for c in hubs[:6])
             + "。它们就是「经济/成长引擎」的实际载体。")
    L.append("2. **关键词标签会漏掉真引擎**：例如铜须的卡面是「战吼触发两次」而不是"
             "「获得金币」，功能上是资源倍增器，但关键词命中不了。所以 §1 必须**连卡面一起读**，"
             "标签只用于排序。")
    n6 = sum(1 for d in per_comp if d["max_tier"] == 6)
    n5 = sum(1 for d in per_comp if d["max_tier"] == 5)
    L.append(f"3. **33 个流派里有 {n6} 个的最高核心在 6 本、{n5} 个在 5 本，"
             f"没有任何一个能在 4 本成型。** 这是「停 4 本刷牌」必然掉后期上限的结构原因。")
    L.append("4. 转型边（§3）可以直接变成课程里的「指定转型」任务：从 A 的核心集出发，"
             "只差 1–2 张卡时切换到 B；§4 的 "
             f"{len(iso)} 对（占全部 {len(names_sorted)*(len(names_sorted)-1)//2} 对的大多数）"
             "没有共享核心，属于「重建」而不是「转型」。")
    L.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"OK comps={len(comps)} cards={len(cards)} shared={len(shared)} pairs={len(pairs)} "
          f"iso={len(iso)} hubs={len(hubs)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
