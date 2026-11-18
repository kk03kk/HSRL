"""Locate the exact public decisions at which matched on-policy paths first separate."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json
from tools.assess_investment_timing import UTILITY
from workflows.group_relative_recruit import checkpoint, evaluator, output
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import MODEL


ROOT = Path('artifacts/runs/nus-recruit-branch-credit-20261118-1')


def forks(lengths, acts, cards, masks, features):
    offsets = np.r_[0,np.cumsum(lengths)]
    todo = [(list(range(len(lengths))),0)]
    while todo:
        members, depth = todo.pop()
        groups = defaultdict(list)
        for m in members:
            if depth < lengths[m]:
                k = offsets[m]+depth
                # Equal observed history and choice set at this prefix, with a shared simulator seed.
                key = (cards[k].tobytes(), masks[k].tobytes(), features[k].tobytes())
                groups[key].append(m)
        for group in groups.values():
            if len(group) < 2:
                continue
            options = defaultdict(list)
            for m in group:
                options[int(acts[offsets[m]+depth])].append(m)
            if len(options) > 1:
                yield offsets[group[0]]+depth, depth, options
            todo.extend((v,depth+1) for v in options.values() if len(v)>1)


def extract(task):
    arm,it,s = task
    native('development')
    peer = 1-arm
    actors = [evaluator('development',p) for p in (MODEL,checkpoint(peer,3),checkpoint(peer,4))]
    path = output(arm,it)/f'rollout-{s:02}'
    rows = json.loads((path/'groups.json').read_text())
    nodes = []
    for row in rows:
        counts = row['actions']
        if not sum(counts):
            continue
        with np.load(path/f"group-{row['i']}.npz") as batch:
            acts,cards,masks,features = [batch[k] for k in ('a','c','m','f')]
        if sum(counts)!=len(acts):
            raise ValueError('incomplete source')
        for k,depth,choices in forks(counts,acts,cards,masks,features):
            mask = np.unpackbits(masks[k])[:245].astype(bool)
            picks = [int(actor(features[k],cards[k].astype(np.int64),mask,0)[0].argmax()) for actor in actors]
            nodes.append(dict(source=row['seed'],depth=depth,turn=row['start'],picks=picks,
                              outcomes={a:[row['ranks'][m] for m in ids] for a,ids in choices.items()}))
    return nodes


def extract_iteration(arm,it):
    dest=ROOT/f'arm-{arm}-{it}'
    dest.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
        parts=list(pool.map(extract,[(arm,it,s) for s in range(8)]))
    atomic_json(dest/'nodes.json',[node for part in parts for node in part])


def compare(nodes,index,agreement=False):
    values=defaultdict(list)
    proposed=0
    for node in nodes:
        p=node['picks']
        alt=p[index];base=p[0]
        if alt==base or (agreement and alt!=p[1]):
            continue
        proposed+=1
        y=node['outcomes']
        if str(base) in y and str(alt) in y:
            means=[np.mean([UTILITY[r-1] for r in y[str(a)]]) for a in (base,alt)]
            values[node['source']].append(float(means[1]-means[0]))
    per_source=np.array([np.mean(v) for v in values.values()])
    interval=None
    if len(per_source):
        rng=np.random.default_rng(192811)
        interval=np.quantile(rng.choice(per_source,(5000,len(per_source))).mean(1),[.025,.975]).tolist()
    return dict(proposed=proposed,observed_pairs=sum(map(len,values.values())),sources=len(values),
                utility=float(per_source.mean()) if len(values) else None,ci95=interval)


def assess():
    report=[]
    for arm in (0,1):
        nodes=[n for it in range(4) for n in json.loads((ROOT/f'arm-{arm}-{it}'/'nodes.json').read_text())]
        report.append(dict(data_arm=arm,actor_arm=1-arm,nodes=len(nodes),sources=len({n['source'] for n in nodes}),
                           end=compare(nodes,2),stable=compare(nodes,2,agreement=True),
                           median_depth=float(np.median([n['depth'] for n in nodes]))))
    atomic_json(ROOT/'assessment.json',report)


if __name__=='__main__':
    if sys.argv[1]=='extract':
        k=int(sys.argv[2]);extract_iteration(k//4,k%4)
    elif sys.argv[1]=='assess':
        assess()
    else:
        raise ValueError('unknown stage')
