"""Offline bot-library generation; no learner optimization or game input."""
import argparse
import json
from local_offline import prepare

if __name__ == '__main__':
    handles = prepare()
    from taverngym.research.library_build import generate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--games', type=int, default=32)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--split', choices=['train','validation','test'], required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.config,a.out,games=a.games,seed=a.seed,split=a.split)))
