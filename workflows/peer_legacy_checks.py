"""Benchmark the native fork against 3265 and replay the fixed D-151 midfield cases."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows import peer_ppo_panel as P
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, PROTOCOL
from workflows.peer_distillation import opening, rollout, MATCH_SEED
from tools.summarize_peer_finetune import UTILITY, estimate

OUT = Path("artifacts/runs/nus-peer-legacy-20261119-1")
ARCHIVE = Path("artifacts/runs/nus-peer-distill-assess-20261119-2")
SEAL = Path("artifacts/runs/nus-peer-heads-20261119-1/trained.json")
START = 190260000
GAMES = 512


def read(path):
    return json.loads(path.read_bytes())


def shard(i):
    torch.set_num_threads(1)
    proto, pool = P.Q.config("hold", allow_tail_variant=True)
    seal = read(SEAL)
    if (proto.sha256()!=PROTOCOL or file_identity(P.Q.BASE)["sha256"]!=P.Q.PLAN["base_sha"]
            or file_identity(INITIAL)["sha256"]!=INITIAL_SHA or file_identity(seal["model"])!=seal["files"]["model.pt"]):
        raise ValueError("legacy comparison assets changed")
    policies = {k: P.Q.policy(v, proto) for k,v in dict(base=P.Q.BASE,peer=INITIAL,candidate=seal["model"]).items()}
    natural = [P.trial(policies["base"], pool, seed, event_turns=True) for seed in range(START+i, START+GAMES, 8)]
    atomic_json(OUT / f"base-{i}.json", natural)
    matches = []
    for archived in read(ARCHIVE / f"matched-{i}.json"):
        j = archived["j"]
        seed = MATCH_SEED + j
        fork = opening(policies["base"], pool, seed, 9+j%4)
        if fork.state is None or fork.opening != archived["opening"]:
            raise ValueError("failed to reproduce protected state")
        bounds = {}
        ranks = {}
        for k, pol in policies.items():
            upper = []
            ranks[k] = rollout(pol, pool, seed, state=fork.state, context=fork.strict, bounds=upper)
            bounds[k] = upper[0]
        if any(ranks[k]!=archived["ranks"][k] or bounds[k]!=archived["tail_rank_upper"][k] for k in ("base","peer")):
            raise ValueError("protected reference rank differs")
        matches.append(dict(seed=seed, opening=fork.opening, groups=archived["groups"], ranks=ranks, bounds=bounds))
    atomic_json(OUT / f"matched-{i}.json", matches)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / "plan.json", dict(reference=P.Q.PLAN["base_sha"], source_seal=file_identity(SEAL),
        protocol=PROTOCOL, seed=START, games=GAMES, matched_archive=str(ARCHIVE)))
    def launch(i):
        with (OUT / f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(i)], check=True, stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(launch, range(8)))
    natural = [v for i in range(8) for v in read(OUT/f"base-{i}.json")]
    if sorted(r["seed"] for r in natural) != list(range(START, START+GAMES)):
        raise ValueError("incomplete natural coverage")
    matches = [v for i in range(8) for v in read(OUT/f"matched-{i}.json")]
    if len({v["seed"] for v in matches}) != len(matches):
        raise ValueError("overlapping protection states")
    result = {}
    for group in ("low","mech","dragon"):
        subset = [m for m in matches if group in m["groups"]]
        result[group] = dict(n=len(subset), top4={k:sum(m["ranks"][k]<=4 for m in subset) for k in ("base","peer","candidate")},
            utility={k:estimate([UTILITY[m["ranks"]["candidate"]-1]-UTILITY[m["ranks"][k]-1] for m in subset]) for k in ("base","peer")})
    atomic_json(OUT / "complete.json", dict(matched=result, n=GAMES, extrapolated_matches=sum(any(v["bounds"].values()) for v in matches)))


if __name__ == "__main__":
    run() if len(sys.argv)==1 else shard(int(sys.argv[1]))
