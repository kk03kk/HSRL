"""Extract exact pre-action boards and entity changes for selected sealed games."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.compare_research_panels import load_sealed_panel


def inspect(task):
    import hsbg_engine as engine
    ref, seeds = task
    cards = json.loads((ROOT/"engine/crates/hsbg-engine/data/cards.json").read_text(encoding="utf-8"))["cards"]
    ids = [engine.card_info(i+1)[0] for i in range(len(cards))]
    path = Path(ref["directory"])/ref["file"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != ref["sha256"]:
            raise ValueError("source hash mismatch")
    def board(p):
        return [[m["eid"],ids[m["card"]],m["atk"],m["hp"],m["gift_data"],m["kw"],m["bonus_kw"]] for m in p["board"]]
    selected = {s:[] for s in seeds}
    prefixes = [b'"seed":'+str(s).encode()+b',' for s in seeds]
    finished = set()
    with gzip.open(path,"rb") as stream:
        for line in stream:
            if not any(mark in line for mark in prefixes):
                continue
            r = json.loads(line)
            s = r.get("seed")
            if s not in selected:
                continue
            if r["event"]=="game_end":
                if len(selected[s]) != r["actions"]:
                    raise ValueError("incomplete selected game")
                finished.add(s)
                if finished == set(seeds):
                    break
            elif r["event"]=="action":
                p = r["state_before"]["players"][r["seat"]]
                q = r["state_after"]["players"][r["seat"]]
                selected[s].append(dict(t=r["turn"],i=r["action_index"],a=r["action"],
                                        b=board(p),after=board(q),pending=p["pending"],
                                        trinkets=p["trinkets"]))
    if finished != set(seeds):
        raise ValueError("missing selected seed")
    return ref["identity"]["name"],selected


if __name__=="__main__":
    panel = load_sealed_panel(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    requests = [(name,int(seed)) for name,seed in (spec.split("=") for spec in sys.argv[3:])]
    tasks = []
    for model in panel["models"]:
        for ref in model["decision_traces"]:
            seeds = [s for n,s in requests if n==model["name"] and s in ref["seeds"]]
            if seeds:
                tasks.append((ref,seeds))
    if sum(len(s) for _,s in tasks)!=len(requests):
        raise ValueError("unmatched requests")
    out.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=4) as pool, gzip.open(out/"selected.jsonl.gz","xt") as stream:
        for name, games in pool.map(inspect,tasks):
            for s,records in games.items():
                stream.write(json.dumps(dict(name=name,seed=s,actions=records),separators=(",",":"))+"\n")
