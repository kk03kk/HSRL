"""Compare terminal and growth preferences among sampled paths from identical openings."""
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.env.obs_std import GLOBAL_FIELD_INDEX, MINION_ATK, MINION_F, MINION_HP, N_BOARD, SLICE_BOARD
from hsbg.runtime.artifacts import atomic_json
from workflows.assess_purchase_consequences import interval
from workflows.group_relative_recruit import ROOT, SHARDS, output
from workflows.group_relative_recruit import GroupPolicy
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from tools.assess_investment_timing import UTILITY
from workflows.d132_paired_plans import check
from workflows.probe_investment_timing import MODEL


def main():
    groups=available=ties=non_ties=0
    gaps, concordance, within_tie=[],[],[]
    for arm in range(2):
        for s in range(SHARDS):
            d=output(arm,0)/f'rollout-{s:02}'
            for r in json.loads((d/'groups.json').read_text()):
                groups+=1
                if not sum(r['actions']):
                    continue
                with np.load(d/f"group-{r['i']}.npz") as data:
                    end=np.flatnonzero(data['a']==0)
                    f=data['f'][end]
                turns=np.rint(20*f[:,GLOBAL_FIELD_INDEX['turn']]).astype(int)
                board=f[:,SLICE_BOARD].reshape(-1,N_BOARD,MINION_F)
                power=np.rint(10*(board[:,:,MINION_ATK]+board[:,:,MINION_HP]).sum(axis=1))
                offset=0; growth=[]
                for n in r['actions']:
                    mask=(end>=offset)&(end<offset+n)
                    eot=dict(zip(turns[mask],power[mask]))
                    growth.append(np.log1p(eot[12])-np.log1p(eot[9]) if 9 in eot and 12 in eot else None)
                    offset+=n
                avail=[i for i,g in enumerate(growth) if g is not None]
                available+=len(avail)
                if len(avail)<2:
                    continue
                gaps.append(max(growth[i] for i in avail)-min(growth[i] for i in avail))
                same,aligned=[],[]
                for a,b in combinations(avail,2):
                    d_rank=r['ranks'][b]-r['ranks'][a]
                    d_growth=growth[a]-growth[b]
                    if not d_rank:
                        ties+=1;same.append(abs(d_growth))
                    elif d_growth:
                        non_ties+=1;aligned.append(float(d_rank*d_growth>0))
                if same:
                    within_tie.append(np.mean(same))
                if aligned:
                    concordance.append(np.mean(aligned))
    atomic_json(ROOT/'baseline-growth.json',dict(groups=groups,paths_with_both_eot=available,equal_rank_pairs=ties,
                ordered_rank_pairs=non_ties,within_source_growth_range=interval(gaps),
                same_rank_absolute_growth_gap=interval(within_tie),rank_growth_concordance=interval(concordance)))


def greedy(r):
    w=GroupPolicy('development',MODEL,7)
    result=evaluate_policy(w,EvalConfig(games=1,seed=r['seed'],mirror_pool=w.pool))
    check(result)
    rank=result['rank_sequence'][0]
    utility=np.array(UTILITY)[np.array(r['ranks'])-1]
    return dict(seed=r['seed'],base=rank,sampled_mean=float(utility.mean()-UTILITY[rank-1]),
                hindsight_best=float(utility.max()-UTILITY[rank-1]),wins=int(sum(utility>UTILITY[rank-1])))


if __name__=='__main__':
    if sys.argv[1:] == ['greedy']:
        groups=[r for a in (0,1) for s in range(SHARDS) for r in
                json.loads((output(a,0)/f'rollout-{s:02}'/'groups.json').read_text())]
        with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
            result=list(pool.map(greedy,groups))
        atomic_json(ROOT/'sampling-gap.json',dict(rows=result,mean=interval([r['sampled_mean'] for r in result]),
                    best=interval([r['hindsight_best'] for r in result]),better_paths=sum(r['wins'] for r in result)))
    else:
        main()
