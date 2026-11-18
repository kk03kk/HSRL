"""Cross-fit the two sale models and measure repeatability of their bootstrapped targets."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json
from workflows.assess_purchase_consequences import interval
from workflows.learn_sale_afterstates import ROOT


def main():
    heads = json.loads((ROOT/'heads.json').read_text())
    rows = [r for s in range(16) for r in json.loads((ROOT/f'source-{s:02}'/'complete.json').read_text())['rows']]
    changes, gains, covariance, variance = [], [], [], []
    branch_count = 0
    for r in rows:
        if not r['opportunity']:
            continue
        # Fit never used this training arm's source games.
        score = np.asarray(r['x']) @ heads[1-r['i']//6%2]['weight']
        base = r['sales'].index(r['default'])
        pick = int(score.argmax())
        if score[pick]-score[base] <= .035:
            pick = base
        y = np.asarray(r['outcomes'])
        branch_count += y.size
        changes.append(int(pick!=base))
        gains.append(float((y[:,pick]-y[:,base]).mean()))
        z = y-y.mean(axis=1,keepdims=True)
        cov = np.mean([np.mean(z[a]*z[b]) for a,b in ((0,1),(1,2),(2,0))])
        covariance.append(cov)
        variance.append(float(np.mean(z*z)))
    atomic_json(ROOT/'signal.json', dict(games=len(rows), opportunities=len(gains), native_branches=branch_count,
                changed=sum(changes), crossfit_critic_gain=interval(gains),
                repeatable_fraction=np.mean(covariance)/np.mean(variance)))


if __name__ == '__main__':
    main()
