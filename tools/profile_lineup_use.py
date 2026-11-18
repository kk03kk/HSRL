"""Join the user's guide, sealed EOT rosters and action ledgers by policy/seed."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_research_panels import load_sealed_panel


def read_cards():
    cards = json.loads((ROOT/"engine/crates/hsbg-engine/data/cards.json").read_text(encoding="utf-8"))["cards"]
    return {c["id"]: c for c in cards}


def read_guide(cards):
    names = defaultdict(set)
    for c in cards.values():
        if not c.get("normal"):
            names[c["name_zh"]].add(c["id"])
    entries = []
    for text in (ROOT/"docs/battlegrounds_strategy_guide.md").read_text(encoding="utf-8").split("\n## ")[1:]:
        core_text = text.split("### 核心卡牌")[1].split("### ")[0]
        ids = set()
        for name in filter(None, map(str.strip, core_text.splitlines())):
            if len(names[name]) != 1:
                raise ValueError(f"ambiguous or missing core: {name}")
            ids.update(names[name])
        entries.append((text.splitlines()[0],text.split("强度：")[1].split("**")[0],ids))
    return entries


def profile(panel, ledger):
    db = read_cards()
    normal = {k:c.get("normal") or k for k,c in db.items()}
    comps = read_guide(db)
    models = {m["name"]:m for m in panel["models"]}
    boards = {(m["name"],r["seed"],r["turn"]):Counter(normal[c["card_id"]] for c in r["roster"])
              for m in models.values() for r in m["trace"]}
    ranks = {(m["name"],s):v for m in models.values() for s,v in zip(m["seed_sequence"],m["rank_sequence"])}
    counts = defaultdict(lambda:defaultdict(Counter))
    details = {}
    seen = set()
    total_actions = Counter()
    pirates = {"BG36_523","BG33_823","BG33_825"}
    with gzip.open(ledger,"rt") as stream:
        for line in stream:
            game = json.loads(line)
            model, seed = key = game["model"],game["seed"]
            if key in seen or game["rank"] != ranks[key]:
                raise ValueError("duplicate or mismatched ledger")
            seen.add(key)
            played = set()
            offers = set()
            prev = Counter()
            flags = defaultdict(set)
            economic = Counter()
            verbs = Counter()
            for turn, t in sorted(((int(k),v) for k,v in game["turns"].items())):
                have = boards.get((*key,turn), Counter())
                offers.update(normal[k] for k in t["offers"])
                for a,kind,card,target,g0,g1,gained in t["actions"]:
                    total_actions[model] += 1
                    if 46 <= a < 196 and kind == "Minion":
                        played.add(normal[card])
                    if a == 1: verbs["refresh"] += 1
                    if 8 <= a < 16: verbs["buy_"+kind] += 1
                    if 16 <= a < 23: verbs["sell"] += 1
                    if 46 <= a < 126: verbs["play_"+kind] += 1
                    if 126 <= a < 196: verbs["magnetize"] += 1
                    if 46 <= a < 126 and card == "BG33_815" and g1 > g0:
                        economic["wealth_casts"] += 1
                        economic["wealth_gold"] += g1-g0
                    if 46 <= a < 126 and card in {"BG33_811","BG33_812","BG33_813","BG33_814","BG33_815"}:
                        economic["bounties"] += 1
                if turn >= 9:
                    economic["max_pirate_parts"] = max(economic["max_pirate_parts"],len(pirates & have.keys()))
                    for name,_,ids in comps:
                        same = ids & have.keys() & prev.keys()
                        pair = len(ids & have.keys()) >= 2
                        persistent = len(same) >= 2
                        if len(ids)==1:
                            k = next(iter(ids))
                            pair = have[k]>=2
                            persistent = pair and prev[k]>=2
                        if pair: flags[name].add("pair")
                        if turn >= 10 and persistent and (same & played): flags[name].add("persistent")
                prev = have
            for name,_,ids in comps:
                if offers & ids: flags[name].add("offered")
                if played & ids: flags[name].add("deployed")
                counts[model][name].update(flags[name])
            details.setdefault(model,{})[seed] = {
                "persistent":[name for name,_,_ in comps if "persistent" in flags[name]],
                "pirate_deployed":sorted(pirates & played), "economic":dict(economic),"verbs":dict(verbs)}
    if seen != set(ranks) or any(total_actions[k]!=m["outcome"]["decisions"] for k,m in models.items()):
        raise ValueError("incomplete panel join")
    return {"actions":total_actions,"guide":counts,"games":details}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("panel","ledger","output"):
        ap.add_argument(key,type=Path)
    args = ap.parse_args()
    report = profile(load_sealed_panel(args.panel),args.ledger)
    with args.output.open("x",encoding="utf-8") as out:
        json.dump(report,out,ensure_ascii=False)
