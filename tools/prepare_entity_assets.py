"""Build fingerprinted offline rule vectors once, before server training."""
import argparse
import json
from local_offline import prepare

if __name__ == '__main__':
    handles = prepare()
    from taverngym.rl.rule_assets import build_assets
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(build_assets(a.out), ensure_ascii=False))
