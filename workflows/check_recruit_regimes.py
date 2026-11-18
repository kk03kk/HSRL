"""Test if a publicly visible pre-intervention board predicts when to deploy a broad actor."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.assess_investment_timing import UTILITY
from workflows.group_relative_recruit import ROOT as TRAIN
from workflows.learn_recruit_residual import paired_endpoints
from workflows.validate_stable_recruits import ROOT as HOLD,FIRST


ROOT=Path('artifacts/runs/nus-recruit-regimes-20261118-1')


def cohort(root):
    return [g for i in range(8) for g in json.loads((root/f'panel-{i:02}'/'rows.json').read_text())]


def features(games):
    at=lambda g,t: g['eot'].get(str(t),[0.,0.,1.])
    e4,e5,e6=[np.array([at(g,t) for g in games],float) for t in (4,5,6)]
    p4,p5,p6=[np.log1p(s[:,0]) for s in (e4,e5,e6)]
    return np.column_stack((p6,e6[:,1],e6[:,2],p6-p5,p5-p4,e6[:,1]-e4[:,1],e6[:,2]-e4[:,2]))


def fit():
    ROOT.mkdir(parents=True,exist_ok=False)
    rows=cohort(TRAIN)
    if [c[0]['seed'] for c in rows]!=list(range(87900000,87900256)):
        raise ValueError('wrong development panel')
    x=features([r[0] for r in rows])
    u=np.array([[UTILITY[g['rank']-1] for g in r] for r in rows])
    y=u[:,1:].mean(1)-u[:,0]
    mean=x.mean(0);scale=np.maximum(.01,x.std(0))
    z=(x-mean)/scale
    w=np.linalg.solve(z.T@z+128*np.eye(z.shape[1]),z.T@(y-y.mean()))
    atomic_json(ROOT/'gate.json',dict(mean=mean.tolist(),scale=scale.tolist(),weight=w.tolist(),
                intercept=y.mean(),n=len(rows),inputs=[file_identity(TRAIN/f'panel-{i:02}'/'rows.json') for i in range(8)]))


def assess():
    gate=json.loads((ROOT/'gate.json').read_text())
    rows=cohort(HOLD)
    for row in rows:
        for turn in ('4','5','6'):
            if any(r['eot'].get(turn)!=row[0]['eot'].get(turn) for r in row):
                raise ValueError('early boards differ between policies')
    x=features([r[0] for r in rows])
    pred=gate['intercept']+(x-gate['mean'])/gate['scale']@gate['weight']
    chosen=[[r[0],*(r[3:] if p>0 else [r[0],r[0]])] for r,p in zip(rows,pred)]
    report=paired_endpoints([g for r in chosen for g in r],FIRST,256)
    report['assignments']=int(sum(pred>0));report['gate']=file_identity(ROOT/'gate.json')
    for j in (0,1):
        records=[g for r,c in zip(rows,chosen) for g in (r[3+j],c[1+j],c[1+j])]
        report.setdefault('over_full_deployment',[]).append(paired_endpoints(records,FIRST,256)['report'][1])
    atomic_json(ROOT/'assessment.json',report)


if __name__=='__main__':
    if sys.argv[1]=='fit':
        fit()
    elif sys.argv[1]=='assess':
        assess()
    else:
        raise ValueError('unknown stage')
