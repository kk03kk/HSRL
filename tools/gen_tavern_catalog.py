import collections
import io
import json
import os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cs = json.load(open(os.path.join(D, "engine/crates/hsbg-engine/data/cards.json"), encoding="utf-8"))["cards"]
p2p = os.path.join(D, "data/p2/cards_37.json")
P2 = {}
if os.path.isfile(p2p):
    p2 = json.load(open(p2p, encoding="utf-8"))
    P2 = {c.get("id"): c for c in (p2.get("cards") or []) if isinstance(c, dict)}

def nm(c):
    return (c.get("name_zh") or (P2.get(c.get("id")) or {}).get("name_zh")
            or c.get("name") or (P2.get(c.get("id")) or {}).get("name") or c.get("id"))

def races(c):
    return ",".join(c.get("races") or []) or "—"

kind = collections.Counter(c.get("kind") for c in cs)
minions = [c for c in cs if "minion" in (c.get("pools") or [])]
heroes = [c for c in cs if c.get("kind") == "hero"]
tl = [c for c in cs if c.get("kind") == "trinket_lesser"]
tg = [c for c in cs if c.get("kind") == "trinket_greater"]

L = []
A = L.append
A("# 酒馆全卡表参考（37.0.2.256589）")
A("")
A("> 自动生成：`tools/gen_tavern_catalog.py`（本地 cards.json + p2/cards_37.json 补名）")
A("> 引擎卡表指纹以 `config/engine-upstream-lock.json` 为准；本文件只是**查阅用**参考。")
A("")
A("## 0. 总量")
A("")
A("| 类别 | 数量 |")
A("|---|---:|")
for k, v in kind.most_common():
    A("| %s | %d |" % (k, v))
A("| **卡表总数** | **%d** |" % len(cs))
A("")
A("## 1. 当前牌池随从（%d 个）" % len(minions))
A("")
byt = collections.defaultdict(list)
for c in minions:
    byt[c.get("tier")].append(c)
for t in sorted(byt, key=lambda x: (x is None, x)):
    lst = byt[t]
    A("### 等级 %s —— %d 个" % (t, len(lst)))
    A("")
    A("| 卡牌 id | 名字 | 种族 | 基础身材 |")
    A("|---|---|---|---|")
    for c in sorted(lst, key=lambda c: (nm(c) or "")):
        A("| `%s` | %s | %s | %d/%d |" % (c["id"], nm(c), races(c),
                                          c.get("atk") or 0, c.get("hp") or 0))
    A("")
A("## 2. 英雄（%d 个）" % len(heroes))
A("")
A("| 卡牌 id | 名字 |")
A("|---|---|")
for c in sorted(heroes, key=lambda c: (nm(c) or "")):
    A("| `%s` | %s |" % (c["id"], nm(c)))
A("")
A("## 3. 饰品：小饰品 %d 个" % len(tl))
A("")
A("| 卡牌 id | 名字 | 效果 |")
A("|---|---|---|")
for c in sorted(tl, key=lambda c: (nm(c) or "")):
    txt = (c.get("text_zh") or c.get("text") or "").replace("|", "/").replace("\n", " ")
    A("| `%s` | %s | %s |" % (c["id"], nm(c), txt))
A("")
A("## 4. 饰品：大饰品 %d 个" % len(tg))
A("")
A("| 卡牌 id | 名字 | 效果 |")
A("|---|---|---|")
for c in sorted(tg, key=lambda c: (nm(c) or "")):
    txt = (c.get("text_zh") or c.get("text") or "").replace("|", "/").replace("\n", " ")
    A("| `%s` | %s | %s |" % (c["id"], nm(c), txt))
A("")

out = os.path.join(D, "docs/analysis/TAVERN_CATALOG_20261117.md")
os.makedirs(os.path.dirname(out), exist_ok=True)
io.open(out, "w", encoding="utf-8").write("\n".join(L))
print("wrote", out, len(L), "lines")
print("minions", len(minions), "heroes", len(heroes), "trinket_lesser", len(tl), "trinket_greater", len(tg))
for t in sorted(byt, key=lambda x: (x is None, x)):
    print("  tier", t, "->", len(byt[t]))
