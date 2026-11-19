"""Independently verify paired shard identities, ranks and late-game outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.summarize_peer_finetune import UTILITY, body, estimate


def seal(run, dest):
    read = lambda p: json.loads(p.read_bytes())
    result, plan = (read(run/name) for name in ("complete.json","plan.json"))
    files = [run/f"panel-{i}.json" for i in range(8)]
    rows = sorted((r for p in files for r in read(p)),key=lambda r:r["parent"]["seed"])
    n, start = result["n"], result["seed"]
    if (plan["seed"] != start or plan["games"] != n or len(rows) != n
            or [r["parent"]["seed"] for r in rows] != list(range(start, start+n))):
        raise ValueError("incorrect seed coverage")
    ranks = []
    for r in rows:
        a,b = r["parent"],r["candidate"]
        if (a["seed"],a["lobby"]) != (b["seed"],b["lobby"]):
            raise ValueError("lobby or seed pairing failed")
        pair = [g["metrics"]["rank"] for g in (a,b)]
        if any(not 1<=p<=8 or g["metrics"]["utility"] != UTILITY[p-1] for p,g in zip(pair,(a,b))):
            raise ValueError("recorded rank utility mismatch")
        ranks.append(pair)
    ranks = np.asarray(ranks, dtype=int)
    a,b = ranks.T
    u = np.asarray(UTILITY)
    gain = estimate(u[b-1] - u[a-1])
    if abs(gain["mean"]-result["delta"]["utility"]["mean"])>1e-12:
        raise ValueError("recorded mean utility mismatch")
    distribution = {}
    for k in range(1,8):
        distribution[k] = dict(parent=int((a<=k).sum()),candidate=int((b<=k).sum()),
            difference=estimate((b<=k).astype(float)-(a<=k)))
    late = {}
    for turn in (9,10,11,12,13,14,15,16):
        own = [body(r["parent"],turn) for r in rows]
        new = [body(r["candidate"],turn) for r in rows]
        # Both eliminated = tie; single survivor wins; two survivors compare their complete board.
        sign = []
        for l,r in zip(own,new):
            if l is None or r is None:
                sign.append(int(r is not None)-int(l is not None))
            else:
                sign.append(int(r>l)-int(r<l))
        def summary(v):
            reached=[x for x in v if x is not None]
            return dict(n=len(reached),median=float(np.median(reached)) if reached else None)
        late[turn] = dict(parent=summary(own),candidate=summary(new),
            both_dead=sum(l is None and r is None for l,r in zip(own,new)),
            wins=sign.count(1),ties=sign.count(0),losses=sign.count(-1),net=estimate(sign))
    identities = {p.name:file_identity(p) for p in files+[run/"complete.json", run/"plan.json"]}
    atomic_json(dest, dict(n=n, seed=start, utility=gain, rank_cdf=distribution, late=late, files=identities,
        rank_pair=dict(better=int((b<a).sum()), equal=int((b==a).sum()), worse=int((b>a).sum())),
        protocol=plan["protocol"], field_bounds=result["exact_field_utility_bounds"],
        uncertainty="pointwise paired by natural game seed, conditional on frozen opponent field"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("dest",type=Path)
    args=parser.parse_args()
    seal(args.run,args.dest)
