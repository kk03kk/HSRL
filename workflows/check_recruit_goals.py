"""Screen measurable two-recruit option targets using existing training trajectories only."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.env.obs_std import GLOBAL_FIELD_INDEX, GLOBAL_FIELD_SCALE, MINION_ATK, MINION_F, MINION_HP, N_BOARD, SLICE_BOARD
from hsbg.runtime.artifacts import atomic_json
from tools.assess_investment_timing import UTILITY
from workflows.assess_purchase_consequences import interval
from workflows.group_relative_recruit import ROOT, SHARDS, UPDATES, output


def segment(arr, counts, t):
    actions = arr['a']
    f = arr['f']
    turns = np.rint(20*f[:,GLOBAL_FIELD_INDEX['turn']]).astype(int)
    endpoints = np.flatnonzero((turns==t+1)&(actions==0))
    board = f[endpoints,SLICE_BOARD].reshape(-1,N_BOARD,MINION_F)
    strength=np.log1p(np.rint(10*(board[:,:,MINION_ATK]+board[:,:,MINION_HP]).sum(1)))
    hp=f[:,GLOBAL_FIELD_INDEX['health']]*GLOBAL_FIELD_SCALE['health']
    armor=f[:,GLOBAL_FIELD_INDEX['armor']]*GLOBAL_FIELD_SCALE['armor']
    offsets=np.r_[0,np.cumsum(counts)]
    rows=[]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        end=np.flatnonzero((endpoints>=lo)&(endpoints<hi))
        next_start=np.flatnonzero(turns[lo:hi]==t+2)
        size=int(np.sum(turns[lo:hi]<=t+1))
        if len(end)!=1 or not len(next_start):
            rows.append(None)
        else:
            after=lo+next_start[0]
            rows.append(dict(grow=float(strength[end[0]]),health=float(hp[after]+armor[after]),
                             start_hp=float(hp[lo]+armor[lo]),size=size))
    return rows


def main():
    groups=[]
    for arm in (0,1):
        for it in range(UPDATES):
            for s in range(SHARDS):
                d=output(arm,it)/f'rollout-{s:02}'
                for r in json.loads((d/'groups.json').read_text()):
                    if not sum(r['actions']):
                        continue
                    with np.load(d/f"group-{r['i']}.npz") as arr:
                        options=segment(arr,r['actions'],r['start'])
                    ids=[k for k,row in enumerate(options) if row]
                    if len(ids)<2:
                        continue
                    u=np.asarray([UTILITY[r['ranks'][i]-1] for i in ids])
                    record=dict(arm=arm,start_hp=options[ids[0]]['start_hp'])
                    for key in ('health','grow'):
                        v=np.asarray([options[i][key] for i in ids])
                        record[key]=float(u[v>=v.max()-1e-7].mean()-u.mean())
                        record[key+'_range']=float(v.max()-v.min())
                    groups.append(record)
    results={}
    for label,selected in [('all',groups),('low_hp',[g for g in groups if g['start_hp']<=20]),
                           ('high_hp',[g for g in groups if g['start_hp']>20])]:
        results[label]={k:interval([g[k] for g in selected]) for k in ('grow','health','grow_range','health_range')}
    atomic_json(ROOT/'goal-screen.json',results)


if __name__=='__main__':
    main()
