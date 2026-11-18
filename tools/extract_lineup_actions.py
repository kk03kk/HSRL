"""Reduce sealed decision evidence to a card/action ledger, without inference."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_research_panels import load_sealed_panel


def reduce_trace(ref):
    import hsbg_engine as engine
    cards = json.loads((ROOT / "engine/crates/hsbg-engine/data/cards.json").read_text(encoding="utf-8"))["cards"]
    ids = [engine.card_info(i+1)[0] for i in range(len(cards))]
    def item(slot):
        kind, value = next(iter(slot["item"].items()))
        return kind, ids[value["card"]]
    path = Path(ref["directory"]) / ref["file"]
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != ref["sha256"]:
            raise ValueError("trace identity mismatch")
    games = []
    with gzip.open(path, "rt") as stream:
        if json.loads(next(stream))["identity"] != ref["identity"]:
            raise ValueError("wrong policy")
        for line in stream:
            r = json.loads(line)
            event = r["event"]
            if event == "game_start":
                game = {"model":ref["identity"]["name"], "seed":r["seed"], "turns":{}}
                games.append(game)
            elif event == "game_end":
                game["rank"] = r["rank"]
                if sum(len(t["actions"]) for t in game["turns"].values()) != r["actions"]:
                    raise ValueError("incomplete game")
            elif event == "action":
                p = r["state_before"]["players"][r["seat"]]
                q = r["state_after"]["players"][r["seat"]]
                t = game["turns"].setdefault(r["turn"], {"actions":[], "offers":set()})
                a = r["action"]
                kind, card, target = "", "", ""
                if 8 <= a < 16:
                    kind, card = item(p["shop"][a-8])
                elif 16 <= a < 23:
                    card = ids[p["board"][a-16]["card"]]
                elif 46 <= a < 196:
                    slot = (a-46)//8 if a < 126 else (a-126)//7
                    kind, card = item(p["hand"][slot])
                    if a >= 126:
                        target = ids[p["board"][(a-126)%7]["card"]]
                before = Counter(item(h)[1] for h in p["hand"])
                after = Counter(item(h)[1] for h in q["hand"])
                t["actions"].append([a,kind,card,target,p["gold"],q["gold"],list((after-before).elements())])
                if game["seed"] != r["seed"]:
                    raise ValueError("unexpected seed")
                for j,slot in enumerate(p["shop"][:8]):
                    if r["legal_mask"][8+j]:
                        t["offers"].add(item(slot)[1])
                if "lobby" not in game:
                    game["lobby"] = r["state_before"]["lobby_races"]
    n = sum(len(t["actions"]) for g in games for t in g["turns"].values())
    if n != ref["actions"] or [g["seed"] for g in games] != ref["seeds"]:
        raise ValueError("incomplete trace")
    for g in games:
        for t in g["turns"].values():
            t["offers"] = sorted(t["offers"])
    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    panel = load_sealed_panel(args.panel)
    refs = [ref for m in panel["models"] for ref in m["decision_traces"]]
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=8) as pool, gzip.open(args.output/"actions.jsonl.gz","xt") as out:
        for games in pool.map(reduce_trace, refs):
            for g in games:
                out.write(json.dumps(g, separators=(",",":"))+"\n")
    print("complete", len(refs), sum(r["actions"] for r in refs), flush=True)
