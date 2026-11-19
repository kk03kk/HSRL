"""Join two peer panels and the independent 3265 comparison by game seed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from hsbg.runtime.artifacts import atomic_json
from tools.summarize_peer_finetune import body, estimate, p50, prepare, utility_limits


def sharded(directory, prefix):
    return [r for i in range(8) for r in json.loads((directory / f"{prefix}-{i}.json").read_bytes())]


def index(rows):
    result = {r["seed"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate seed")
    return result


def describe(games):
    turns = {}
    for t in range(6,19):
        values = [v for g in games if (v:=body(g,t)) is not None]
        turns[t] = dict(n=len(values),p50=p50(values))
    return dict(turns=turns, mean_rank=np.mean([g["metrics"]["rank"] for g in games]),
        counts={k:sum(g["metrics"][k] for g in games)
                for k in ("win","top2","top4","pig_chain","elem_chain","mech","dragon")})


def delta(rows, left, right):
    result = {k:estimate([r[right]["metrics"][k]-r[left]["metrics"][k] for r in rows])
              for k in ("rank","utility","win","top2","top4","pig_chain","elem_chain","reach12")}
    bounds=[(utility_limits(r[left]),utility_limits(r[right])) for r in rows]
    result["strict_effect_bounds"]=[np.mean([b[0]-a[1] for a,b in bounds]),np.mean([b[1]-a[0] for a,b in bounds])]
    for t in (12,14,15):
        both=[r for r in rows if body(r[left],t) is not None and body(r[right],t) is not None]
        end=[np.log1p(body(r[right],t))-np.log1p(body(r[left],t)) for r in both]
        result[f"log{t}"]=estimate(end)
        result[f"growth9to{t}"]=estimate([v-np.log1p(body(r[right],9))+np.log1p(body(r[left],9)) for v,r in zip(end,both)])
    return result


def report(pilot, confirmation, legacy):
    first = sharded(pilot, "panel")
    fresh = sharded(confirmation, "panel")
    base = index(sharded(legacy,"base"))
    seen = set()
    for cohort, start, n in ((first,189430000,384),(fresh,190260000,512)):
        if {r["parent"]["seed"] for r in cohort} != set(range(start,start+n)) or len(cohort) != n:
            raise ValueError("incomplete preregistered panel")
        for row in cohort:
            seed = row["parent"]["seed"]
            if seed in seen:
                raise ValueError("overlapping evaluation panels")
            seen.add(seed)
            if cohort is fresh:
                row["base"] = base[seed]
            if len({(a["seed"],a["lobby"]) for a in row.values()})!=1:
                raise ValueError("mismatched lobby or seed")
            for a in row.values():
                a["metrics"].update(prepare(a),reach12=int(body(a,12) is not None))
    if set(base)!=set(range(190260000,190260512)):
        raise ValueError("unexpected baseline games")
    return dict(fresh={k:describe([r[k] for r in fresh]) for k in ("base","parent","candidate")},
        candidate_vs_base=delta(fresh,"base","candidate"),candidate_vs_peer=delta(fresh,"parent","candidate"),
        pooled_vs_peer=delta(first+fresh,"parent","candidate"),
        fresh_by_lobby={label:dict(n=len(part),utility=estimate([r["candidate"]["metrics"]["utility"]-r["parent"]["metrics"]["utility"] for r in part]))
                  for label,part in (("pig",[r for r in fresh if r["parent"]["lobby"] & 128]),
                                     ("no_pig",[r for r in fresh if not r["parent"]["lobby"] & 128]))})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","confirmation","legacy","out"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    atomic_json(args.out,report(args.pilot,args.confirmation,args.legacy))
